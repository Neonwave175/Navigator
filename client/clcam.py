import gc
import subprocess as sub
import time as t
import os
import sys
import asyncio
import threading
import queue

import cv2
import ncnn
import numpy as np
import pygame
import websockets

from rich.console import Console
from rich.table import Table
from rich.live import Live

# Add Server directory to sys.path to import encrypt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Server")))
import encrypt as e

import click

KEYSEED = 2342344554623453242345234634577674354634563456567467564674567456
key = e.buildkey(KEYSEED)
console = Console()
input_queue = queue.Queue()

def input_listener():
    while True:
        try:
            # click.getchar reads a single keypress without echoing to the terminal
            ch = click.getchar()
            if ch:
                input_queue.put(ch.lower())
        except Exception:
            break

async def send_image(path):
    try:
        encrypted_bytes = e.encrypt_image(path, key)
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await ws.send(encrypted_bytes)
            response = await ws.recv()
            return f"[Server Response]: {response}"
    except Exception as err:
        return f"[Error sending image]: {err}"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

from rich.text import Text

def generate_status(detections, fps, volts, status="Scanning"):
    text = Text()
    text.append("📸 clcam | ", style="bold cyan")
    text.append(f"FPS: {fps:.1f} | Volt: {volts} | Status: {status} | ", style="dim")
    
    if detections:
        det_str = ", ".join([f"{label} ({conf:.2f})" for label, conf, _ in detections])
        text.append(f"Detections: {det_str} | ", style="bold green")
    else:
        text.append("Detections: None | ", style="dim")
        
    text.append("[s] Send | [q] Exit", style="italic white")
    return text

def main():
    # Start input listener thread
    t_input = threading.Thread(target=input_listener, daemon=True)
    t_input.start()

    # Load Camera
    CAMERA_INDEX = 0
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        console.print(f"[bold red]LOL I cant open the camea: {CAMERA_INDEX}[/bold red]")
        exit(1)

    # Load Model
    net = ncnn.Net()
    net.opt.num_threads = 2
    net.opt.use_vulkan_compute = False
    net.load_param("yolov8n_ncnn_model/model.ncnn.param")
    net.load_model("yolov8n_ncnn_model/model.ncnn.bin")
    target = 320
    CONF_THRESH = 0.4
    NMS_THRESH = 0.45

    # Timing
    tspeed = t.perf_counter()
    tick = 0
    volts = "N/A"
    current_status = "Scanning environment..."

    temp_img_path = "captured_vlm.jpg"

    with Live(generate_status([], 0.0, volts, current_status), refresh_per_second=10, console=console) as live:
        while True:
            try:
                tick += 1
                elapsed = t.perf_counter() - tspeed
                fps = tick / elapsed

                ret_cap, img = cap.read()
                if not ret_cap or img is None:
                    current_status = "Error: Camera disconnected"
                    live.update(generate_status([], fps, volts, current_status))
                    t.sleep(0.5)
                    continue

                h, w = img.shape[:2]

                mat_in = ncnn.Mat.from_pixels_resize(
                    img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, target, target
                )
                mean = [0, 0, 0]
                norm = [1 / 255.0, 1 / 255.0, 1 / 255.0]
                mat_in.substract_mean_normalize(mean, norm)

                try:
                    ex = net.create_extractor()
                    ex.input("in0", mat_in)
                    ret, mat_out = ex.extract("out0")
                except Exception as e_inf:
                    current_status = f"Inference error: {e_inf}"
                    live.update(generate_status([], fps, volts, current_status))
                    with open("Fpslog.txt", "a") as log:
                        log.write(f"FPS: {fps:.2f}\n")
                    break

                out = np.array(mat_out).copy()
                out = out.transpose(1, 0)

                boxes = out[:, :4]
                scores = out[:, 4:]
                class_ids = np.argmax(scores, axis=1)
                confidences = np.max(scores, axis=1)

                mask = confidences > CONF_THRESH
                boxes = boxes[mask]
                confidences = confidences[mask]
                class_ids = class_ids[mask]

                scale_x = w / target
                scale_y = h / target
                x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
                y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
                x2 = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x
                y2 = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y

                nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
                indices = cv2.dnn.NMSBoxes(
                    nms_boxes, confidences.tolist(), CONF_THRESH, NMS_THRESH
                )
                prev_area = 0
                area_i = None

                detections = []
                for idx in indices:
                    idx = idx if isinstance(idx, (int, np.integer)) else idx[0]

                    width = x2[idx] - x1[idx]
                    height = y2[idx] - y1[idx]
                    area = width * height

                    label = COCO_CLASSES[class_ids[idx]]
                    conf = confidences[idx]
                    detections.append((label, conf, area))

                    if area >= prev_area:
                        prev_area = area
                        area_i = idx

                # Clean Memory
                del ex, mat_in, mat_out, out
                if tick % 50 == 0:
                    gc.collect()
                    result = sub.run(
                        ["vcgencmd", "measure_volts"], capture_output=True, text=True
                    )
                    volts = result.stdout.strip() if result.stdout else "N/A"

                live.update(generate_status(detections, fps, volts, current_status))

                # Check for background terminal input
                if not input_queue.empty():
                    cmd = input_queue.get_nowait()
                    if cmd == 's':
                        current_status = "Sending image to VLM..."
                        live.update(generate_status(detections, fps, volts, current_status))
                        cv2.imwrite(temp_img_path, img)
                        
                        # run async function synchronously
                        vlm_result = asyncio.run(send_image(temp_img_path))
                        
                        if os.path.exists(temp_img_path):
                            try:
                                os.remove(temp_img_path)
                            except OSError:
                                pass
                        
                        # Print result above the table
                        live.console.print(f"\n[bold green]{vlm_result}[/bold green]\n")
                        current_status = "Scanning environment..."
                        live.update(generate_status(detections, fps, volts, current_status))
                    elif cmd == 'q':
                        current_status = "Exiting..."
                        live.update(generate_status(detections, fps, volts, current_status))
                        break

            except KeyboardInterrupt:
                elapsed = t.perf_counter() - tspeed
                console.print(f"\n[bold red]Average FPS = {tick / elapsed:.2f}[/bold red]")
                with open("FpslogRealtime.txt", "a") as log:
                    log.write(f"FPS: {tick / elapsed:.2f}\n")
                break

    cap.release()

if __name__ == "__main__":
    main()
