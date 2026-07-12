import gc
import subprocess as sub
import time as t

import cv2
import ncnn
import numpy as np
import pygame

# pygame.mixer.init()
# alert_sound = pygame.mixer.Sound("alert.ogg")

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Load Camera
CAMERA_INDEX = 0
# Camera Number
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"LOL I cant open the camea: {CAMERA_INDEX}")
    exit(1)


# Load Model
net = ncnn.Net()
net.opt.num_threads = 2
# Runs the best at 2 threads -> Lower power consumption and better FPS
net.opt.use_vulkan_compute = False
net.load_param("yolov8n_ncnn_model/model.ncnn.param")
net.load_model("yolov8n_ncnn_model/model.ncnn.bin")
target = 320
CONF_THRESH = 0.4
NMS_THRESH = 0.45

# Timing
tspeed = t.perf_counter()
tick = 0

while True:
    try:
        tick += 1

        ret_cap, img = cap.read()
        if not ret_cap or img is None:
            print("Cant see anything! A you dude, FIX IT NOW!!")
            print("OR you are dumb, you didn't CONNECT THE CAMERA")
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
        except Exception as e:
            # Kill gracefully, display FPS and record it
            print("Inference error:", e)
            elapsed = t.perf_counter() - tspeed
            print(f"Average FPS = {tick / elapsed:.2f}")
            with open("Fpslog.txt", "a") as log:
                log.write(f"FPS: {tick / elapsed:.2f}\n")
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

        print(f"Detected {len(indices)} objects:")
        for i in indices:
            i = i if isinstance(i, (int, np.integer)) else i[0]

            width = x2[i] - x1[i]
            height = y2[i] - y1[i]
            area = width * height

            label = COCO_CLASSES[class_ids[i]]
            conf = confidences[i]
            print(f"  {label}: {conf:.2f}")

            if area >= prev_area:
                prev_area = area
                area_i = i

        if area_i is not None:
            print(
                f"  Largest!! {COCO_CLASSES[class_ids[area_i]]}  area={prev_area:.0f}px²"
            )
            # alert_sound.play()
        else:
            print("  Largest!! (no detections)")
        # Clean Memory
        del ex, mat_in, mat_out, out
        if tick % 50 == 0:
            gc.collect()
            result = sub.run(
                ["vcgencmd", "measure_volts"], capture_output=True, text=True
            )
            print(result.stdout)

    except KeyboardInterrupt:
        # Kill gracefully, display FPS and record it
        elapsed = t.perf_counter() - tspeed
        print(f"Average FPS = {tick / elapsed:.2f}")
        with open("FpslogRealtime.txt", "a") as log:
            log.write(f"FPS: {tick / elapsed:.2f}\n")

        break

cap.release()
