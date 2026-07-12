import gc
import subprocess as sub
import time as t
import os
import sys
import asyncio
import termios
import tty
import select

import cv2
import ncnn
import numpy as np
import websockets

# Add Server directory to sys.path to import encrypt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Server")))
import encrypt as e

KEYSEED = 2342344554623453242345234634577674354634563456567467564674567456
key = e.buildkey(KEYSEED)

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


async def send_image(path):
    try:
        encrypted_bytes = e.encrypt_image(path, key)
        async with websockets.connect("ws://192.168.68.59:8765") as ws:
            await ws.send(encrypted_bytes)
            response = await ws.recv()
            return f"[Server Response]: {response}"
    except Exception as err:
        return f"[Error sending image]: {err}"


def get_key_nonblocking():
    """Return a single lowercase char if one is waiting on stdin, else None.
    Non-blocking, checked once per loop iteration -- no separate thread,
    so nothing else is touching the terminal at the same time."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1).lower()
    return None


import textwrap
import shutil

RESULT_LINES = 8  # how many lines are reserved for the result text


def render_block(fps, volts, status, detections, last_result, prev_line_count):
    """Redraw a fixed-height block in place: RESULT_LINES lines of
    wrapped result text (padded with blanks if short) + 1 status
    line. Moving the cursor up by exactly the number of lines we
    printed last time, then rewriting every line, keeps the block a
    constant size -- no resizing, no leftover fragments, no flicker.
    Returns the new line count so the caller can pass it in next time."""
    width = shutil.get_terminal_size((100, 20)).columns

    wrapped = textwrap.wrap(last_result, width=width) or [""]
    wrapped = wrapped[:RESULT_LINES]
    wrapped += [""] * (RESULT_LINES - len(wrapped))

    det_str = ", ".join(f"{lbl} ({c:.2f})" for lbl, c in detections) or "None"
    status_line = f"clcam | FPS {fps:.1f} | Volt {volts} | {status} | Det: {det_str} | [s]end [q]uit"

    lines = wrapped + [status_line]

    if prev_line_count:
        sys.stdout.write(f"\033[{prev_line_count}A")  # move cursor up to start of block
    for line in lines:
        sys.stdout.write("\r\033[K" + line + "\n")  # clear line, print, move to next
    sys.stdout.flush()

    return len(lines)


def main():
    # Put the terminal in cbreak mode ONCE, up front, so keys can be
    # read one at a time without waiting for Enter. This replaces the
    # old approach of toggling raw mode on every single keypress in a
    # background thread, which was racing with the display code.
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return

    net = ncnn.Net()
    net.opt.num_threads = 2
    net.opt.use_vulkan_compute = False
    net.load_param("yolov8n_ncnn_model/model.ncnn.param")
    net.load_model("yolov8n_ncnn_model/model.ncnn.bin")
    target = 320
    CONF_THRESH = 0.4
    NMS_THRESH = 0.45

    tspeed = t.perf_counter()
    tick = 0
    volts = "N/A"
    status = "Scanning"
    last_result = "(none yet)"
    last_print_time = 0.0
    prev_line_count = 0
    temp_img_path = "captured_vlm.jpg"

    try:
        while True:
            tick += 1
            elapsed = t.perf_counter() - tspeed
            fps = tick / elapsed

            ret_cap, img = cap.read()
            if not ret_cap or img is None:
                status = "Error: camera disconnected"
                t.sleep(0.5)
                continue

            h, w = img.shape[:2]
            mat_in = ncnn.Mat.from_pixels_resize(
                img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, target, target
            )
            mat_in.substract_mean_normalize([0, 0, 0], [1 / 255.0, 1 / 255.0, 1 / 255.0])

            ex = net.create_extractor()
            ex.input("in0", mat_in)
            ret, mat_out = ex.extract("out0")

            out = np.array(mat_out).copy().transpose(1, 0)
            boxes = out[:, :4]
            scores = out[:, 4:]
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

            mask = confidences > CONF_THRESH
            boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

            scale_x, scale_y = w / target, h / target
            x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
            y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
            x2 = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x
            y2 = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y

            nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
            indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), CONF_THRESH, NMS_THRESH)

            detections = []
            for idx in indices:
                idx = idx if isinstance(idx, (int, np.integer)) else idx[0]
                label = COCO_CLASSES[class_ids[idx]]
                detections.append((label, confidences[idx]))

            del ex, mat_in, mat_out, out
            if tick % 50 == 0:
                gc.collect()
                result = sub.run(["vcgencmd", "measure_volts"], capture_output=True, text=True)
                volts = result.stdout.strip() if result.stdout else "N/A"

            # Only repaint a few times a second -- no need to hammer the
            # terminal every single frame.
            now = t.perf_counter()
            if now - last_print_time > 0.2:
                prev_line_count = render_block(fps, volts, status, detections, last_result, prev_line_count)
                last_print_time = now

            key_press = get_key_nonblocking()
            if key_press == 's':
                status = "Sending..."
                cv2.imwrite(temp_img_path, img)
                last_result = asyncio.run(send_image(temp_img_path))
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                status = "Scanning"
                prev_line_count = render_block(fps, volts, status, detections, last_result, prev_line_count)
                last_print_time = t.perf_counter()
            elif key_press == 'q':
                break

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        cap.release()
        print()  # move off the overwritten line


if __name__ == "__main__":
    main()
