"""
FastAPI Detection Service with Entity Tracking.
Processes video files server-side with YOLOv8 ByteTrack,
maintains persistent entity registry, and serves annotated frames + state.
"""

import os
import time
import base64
import threading
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from ultralytics import YOLO
from entity_registry import EntityRegistry

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DETECTIONS_DIR = BASE_DIR / "detections"
DETECTIONS_DIR.mkdir(exist_ok=True)
LOG_FILE = BASE_DIR / "security_log.txt"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="CCTV Live Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────

MODEL = YOLO("yolov8s.pt")
registry = EntityRegistry(gone_timeout=5.0)

# Video processing state
video_lock = threading.Lock()
video_thread: Optional[threading.Thread] = None
video_running = False
latest_frame_jpeg: Optional[bytes] = None
latest_boxes: list = []
processing_fps: float = 0.0

# AI state
ai_enabled = False
ai_last_call = 0.0
ai_cooldown = 10.0
ai_summary_text = ""

ALLOWED_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat"}

# ── Drawing ───────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "person": (0, 0, 255),
    "car": (255, 165, 0),
    "truck": (255, 165, 0),
    "bus": (255, 165, 0),
    "motorcycle": (255, 165, 0),
    "bicycle": (200, 200, 0),
    "dog": (0, 200, 0),
    "cat": (0, 200, 0),
}


def draw_annotated_frame(frame, results, names, registry_ref):
    """Draw bounding boxes with display names from the entity registry."""
    vis = frame.copy()
    boxes_out = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = names.get(cls, str(cls))
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None

        if label not in ALLOWED_CLASSES:
            continue

        color = CLASS_COLORS.get(label, (0, 255, 0))

        # Get display name from registry
        display_name = label
        if track_id is not None and track_id in registry_ref.entities:
            display_name = registry_ref.entities[track_id].display_name

        tag = f"{display_name} {conf:.2f}"
        if track_id is not None:
            tag = f"{display_name} {conf:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Background for text
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        boxes_out.append({
            "label": label,
            "display_name": display_name,
            "track_id": track_id,
            "conf": round(conf, 2),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    return vis, boxes_out


# ── Video Processing Thread ───────────────────────────────────────────────────

def process_video(video_path: str, conf: float = 0.4, fps: int = 10):
    """Background thread: reads video, runs tracking, updates registry."""
    global video_running, latest_frame_jpeg, latest_boxes, processing_fps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        video_running = False
        return

    names = MODEL.names if hasattr(MODEL, "names") else {}
    frame_delay = 1.0 / fps

    # Log start
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] SYSTEM — Started processing: {video_path}\n")

    while video_running:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            # Loop video for demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # Run tracking
        results = MODEL.track(frame, persist=True, conf=conf, verbose=False)[0]

        now = time.time()

        # Update entity registry
        for box in results.boxes:
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is None:
                continue
            cls = int(box.cls[0])
            label = names.get(cls, str(cls))
            if label not in ALLOWED_CLASSES:
                continue
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)

            # Extract crop for thumbnail
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            registry.update(track_id, label, conf_val, bbox, crop, now)

        # Sweep gone entities
        registry.sweep_gone(now)

        # Draw annotated frame
        vis, boxes_out = draw_annotated_frame(frame, results, names, registry)

        # Encode to JPEG
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])

        with video_lock:
            latest_frame_jpeg = buf.tobytes()
            latest_boxes = boxes_out
            processing_fps = 1.0 / max(time.time() - t0, 0.001)

        # AI call if enabled
        if ai_enabled:
            _maybe_call_ai(now)

        # Throttle to target FPS
        elapsed = time.time() - t0
        sleep_time = max(0, frame_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    video_running = False

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] SYSTEM — Video processing stopped\n")


# ── AI (OpenRouter) ──────────────────────────────────────────────────────────

def _maybe_call_ai(now: float):
    """Call OpenRouter for enhanced narrative if cooldown has passed."""
    global ai_last_call, ai_summary_text

    if now - ai_last_call < ai_cooldown:
        return

    ai_last_call = now

    # Build context from registry
    state = registry.get_state()
    active = state["entities"]["active"]
    events = state["events"][-10:]
    summary = state["summary"]

    if not active and not events:
        return

    active_desc = ", ".join([f"{e['display_name']} ({e['label']}, {e['duration']}s)" for e in active])
    recent_events = "\n".join([f"- {e['time']} {e['message']}" for e in events])

    prompt = f"""You are a security camera AI analyst. Provide a brief (1-2 sentence) narrative summary of what's happening.

Currently visible: {active_desc or 'nothing'}
Total unique entities this session: {summary['total_unique']}
Recent events:
{recent_events or 'None'}

Respond with just the narrative summary, no JSON."""

    # Fire and forget in a thread
    threading.Thread(target=_call_openrouter, args=(prompt,), daemon=True).start()


def _call_openrouter(prompt: str):
    """Synchronous OpenRouter API call (runs in background thread)."""
    global ai_summary_text

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return

    import urllib.request
    import json

    payload = json.dumps({
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {"role": "system", "content": "You are a concise security camera analyst. Give brief, direct summaries."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 150,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            ai_summary_text = text

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] AI — {text}\n")

            registry.events.append({
                "time": time.strftime("%H:%M:%S"),
                "timestamp": time.time(),
                "type": "ai_analysis",
                "entity_id": None,
                "message": f"AI: {text}",
            })
    except Exception as e:
        print(f"OpenRouter error: {e}")


# ── Request Models ────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    video_path: str = "test2.mp4"
    conf: float = 0.4
    fps: int = 10


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return {
        "status": "ok",
        "video_running": video_running,
        "ai_enabled": ai_enabled,
        "fps": round(processing_fps, 1),
    }


@app.post("/start")
def start_video(req: StartRequest):
    global video_running, video_thread

    if video_running:
        return JSONResponse({"error": "Already running"}, status_code=400)

    # Resolve video path
    vpath = str(BASE_DIR / req.video_path) if not os.path.isabs(req.video_path) else req.video_path
    if not os.path.exists(vpath):
        raise HTTPException(status_code=404, detail=f"Video not found: {vpath}")

    video_running = True
    video_thread = threading.Thread(
        target=process_video,
        args=(vpath, req.conf, req.fps),
        daemon=True,
    )
    video_thread.start()

    return {"message": f"Started processing {req.video_path}", "fps": req.fps, "conf": req.conf}


@app.post("/stop")
def stop_video():
    global video_running
    video_running = False
    return {"message": "Stopped"}


@app.post("/reset")
def reset_all():
    global video_running, latest_frame_jpeg, latest_boxes, ai_summary_text
    video_running = False
    time.sleep(0.5)  # Let thread finish
    latest_frame_jpeg = None
    latest_boxes = []
    ai_summary_text = ""
    registry.reset()
    # Reset YOLO tracker state
    MODEL.predictor = None
    return {"message": "Reset complete"}


@app.get("/frame")
def get_frame():
    """Returns the latest annotated frame as JPEG."""
    if latest_frame_jpeg is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)
    return Response(content=latest_frame_jpeg, media_type="image/jpeg")


@app.get("/frame64")
def get_frame_base64():
    """Returns the latest annotated frame as base64 JSON (for easier frontend consumption)."""
    if latest_frame_jpeg is None:
        return {"frame": None}
    return {"frame": base64.b64encode(latest_frame_jpeg).decode("ascii")}


@app.get("/state")
def get_state():
    """Returns full entity state: active, gone, events, summary."""
    state = registry.get_state()
    state["video_running"] = video_running
    state["ai_enabled"] = ai_enabled
    state["ai_summary"] = ai_summary_text
    state["fps"] = round(processing_fps, 1)
    state["boxes"] = latest_boxes
    return state


@app.get("/entities")
def get_entities():
    return {
        "active": registry.get_active(),
        "gone": registry.get_gone(),
    }


@app.get("/events")
def get_events(n: int = 50):
    return {"events": registry.get_recent_events(n)}


@app.post("/toggle-ai")
def toggle_ai():
    global ai_enabled
    ai_enabled = not ai_enabled
    status_str = "enabled" if ai_enabled else "disabled"

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] SYSTEM — AI analysis {status_str}\n")

    return {"ai_enabled": ai_enabled, "message": f"AI analysis {status_str}"}


@app.get("/videos")
def list_videos():
    """List available video files for the demo."""
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [f.name for f in BASE_DIR.iterdir() if f.suffix.lower() in exts and f.is_file()]
    return {"videos": sorted(videos)}


# ── Legacy detect endpoint (for webcam mode) ─────────────────────────────────

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.4):
    """Legacy endpoint: accepts an image upload and returns detection results with tracking."""
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    results = MODEL.track(frame, persist=True, conf=conf, verbose=False)[0]
    names = MODEL.names if hasattr(MODEL, "names") else {}

    now = time.time()
    for box in results.boxes:
        track_id = int(box.id[0]) if box.id is not None else None
        if track_id is None:
            continue
        cls = int(box.cls[0])
        label = names.get(cls, str(cls))
        if label not in ALLOWED_CLASSES:
            continue
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        registry.update(track_id, label, conf_val, (x1, y1, x2, y2), crop, now)

    registry.sweep_gone(now)

    vis, boxes_out = draw_annotated_frame(frame, results, names, registry)
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
    snapshot_b64 = base64.b64encode(buf).decode("ascii")

    state = registry.get_state()
    return JSONResponse({
        "timestamp": now,
        "boxes": boxes_out,
        "snapshot_base64": snapshot_b64,
        **state,
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main3:app", host="0.0.0.0", port=8000, reload=False)
