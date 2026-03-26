#!/usr/bin/env python3
"""Headless RTSP camera detector

Usage:
    python rtsp_detector.py --url rtsp://... --model yolov8s.pt --conf 0.4

The script connects to a network camera stream (RTSP or HTTP), runs YOLOv8
on each frame (or every Nth frame), logs counts of person/car/truck to
`security_log.txt`, and optionally writes snapshots to `detections/`.
"""
import time
import argparse
from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).parent
DETECTIONS_DIR = BASE_DIR / "detections"
DETECTIONS_DIR.mkdir(exist_ok=True)
LOG_FILE = BASE_DIR / "security_log.txt"


ALLOWED_CLASSES = {"person", "car", "truck"}


def log_counts(counts: Counter, ts: float, frame=None, save_snapshot=True):
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    if not counts:
        line = f"[{timestr}] No objects detected"
    else:
        parts = [f"{k}:{v}" for k, v in counts.items()]
        line = f"[{timestr}] " + ", ".join(parts)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if save_snapshot and frame is not None and counts:
        fname = DETECTIONS_DIR / f"rtsp_{int(ts)}.jpg"
        cv2.imwrite(str(fname), frame)
    print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="RTSP/HTTP camera url")
    parser.add_argument("--model", default="yolov8s.pt", help="path to weights")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--skip", type=int, default=0, help="skip N frames between detections")
    args = parser.parse_args()

    print(f"Loading model {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream {args.url}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("stream ended or cannot read frame")
                break
            frame_count += 1
            if args.skip and frame_count % (args.skip + 1) != 1:
                continue

            results = model(frame, conf=args.conf, verbose=False)[0]
            names = model.names if hasattr(model, "names") else {}
            classes = [names.get(int(b.cls[0]), str(int(b.cls[0]))) for b in results.boxes]
            filtered = [c for c in classes if c in ALLOWED_CLASSES]
            counts = Counter(filtered)
            ts = time.time()
            log_counts(counts, ts, frame=frame)

    finally:
        cap.release()


if __name__ == "__main__":
    main()
