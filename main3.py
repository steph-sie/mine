from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from pathlib import Path
from collections import Counter
import base64
import cv2
import numpy as np

from ultralytics import YOLO


BASE_DIR = Path(__file__).parent
DETECTIONS_DIR = BASE_DIR / "detections"
DETECTIONS_DIR.mkdir(exist_ok=True)
LOG_FILE = BASE_DIR / "security_log.txt"


app = FastAPI(title="YOLOv8 Detection API")

# Allow CORS for local development (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(path: str = "yolov8s.pt"):
    return YOLO(path)


MODEL = load_model()


def imdecode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def log_counts(counts: Counter, ts: float, save_snapshot: bool = True, frame=None):
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    if not counts:
        line = f"[{timestr}] No objects detected"
    else:
        parts = [f"{k}:{v}" for k, v in counts.items()]
        line = f"[{timestr}] " + ", ".join(parts)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    snapshot_b64 = None
    if save_snapshot and frame is not None and counts:
        fname = DETECTIONS_DIR / f"api_snapshot_{int(ts)}.jpg"
        cv2.imwrite(str(fname), frame)
        with open(fname, "rb") as f:
            snapshot_b64 = base64.b64encode(f.read()).decode("ascii")

    return line, snapshot_b64


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.4):
    try:
        data = await file.read()
        frame = imdecode_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")

    results = MODEL(frame, conf=conf, verbose=False)[0]
    names = MODEL.names if hasattr(MODEL, "names") else {}
    classes = [names.get(int(b.cls[0]), str(int(b.cls[0]))) for b in results.boxes]
    counts = Counter(classes)

    # draw boxes for snapshot and collect box metadata
    vis = frame.copy()
    boxes_out = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = names.get(cls, str(cls))
        confv = float(box.conf[0])
        color = (0, 255, 0)
        if label == "person":
            color = (0, 0, 255)
        elif label in ("car", "truck"):
            color = (255, 165, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{label} {confv:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        boxes_out.append({"label": label, "conf": confv, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    ts = time.time()
    line, snapshot_b64 = log_counts(counts, ts, save_snapshot=True, frame=vis)

    return JSONResponse({
        "timestamp": ts,
        "log_line": line,
        "counts": dict(counts),
        "boxes": boxes_out,
        "snapshot_base64": snapshot_b64,
    })


if __name__ == "__main__":
    uvicorn.run("main3:app", host="0.0.0.0", port=8000, reload=False)
