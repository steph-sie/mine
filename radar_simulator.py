"""
Axis D2110-VE Radar Simulator
Publishes realistic radar detections via MQTT, synced to video file processing.
Converts YOLO camera detections into simulated radar coordinates.
"""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ── Radar Configuration (Axis D2110-VE specs) ────────────────────────────────

RADAR_SERIAL = "RADAR001"
RADAR_FOV_DEG = 180  # degrees horizontal
RADAR_MAX_RANGE_PERSON = 60  # meters
RADAR_MAX_RANGE_VEHICLE = 85  # meters
RADAR_MOUNTING_HEIGHT = 3.5  # meters

# Camera-to-radar calibration (simulated co-located sensors)
CAMERA_HFOV_DEG = 90  # approximate camera horizontal FOV
CAMERA_VFOV_DEG = 60  # approximate camera vertical FOV
SCENE_DEPTH_M = 40  # estimated max depth of camera scene in meters
SCENE_WIDTH_M = 30  # estimated width of camera scene at max depth

# Noise parameters
RANGE_NOISE_M = 0.5
ANGLE_NOISE_DEG = 2.0
SPEED_NOISE_MS = 0.3

# MQTT topics (Axis ADF format)
TOPIC_FRAME = f"axis/{RADAR_SERIAL}/axis.scene.frame_v1"
TOPIC_TRACK = f"axis/{RADAR_SERIAL}/axis.scene.object_track_v1"

# Class mapping: YOLO → Axis radar
YOLO_TO_RADAR = {
    "person": "Human",
    "car": "Vehicle",
    "truck": "Vehicle",
    "bus": "Vehicle",
    "motorcycle": "Vehicle",
    "bicycle": "Vehicle",
    "dog": "Unknown",
    "cat": "Unknown",
}

BASE_DIR = Path(__file__).parent


# ── Coordinate conversion ────────────────────────────────────────────────────

def pixel_to_radar(cx_norm: float, cy_norm: float, frame_w: int, frame_h: int):
    """
    Convert normalized pixel center (0-1) to simulated radar polar coordinates.

    Assumptions:
    - Camera center maps to radar angle 0 (straight ahead)
    - Left edge maps to -CAMERA_HFOV/2, right edge to +CAMERA_HFOV/2
    - Vertical position (cy) maps to range: top=far, bottom=near
    - Objects at bottom of frame are closer to the radar
    """
    # Angle: horizontal position → radar azimuth
    angle_deg = (cx_norm - 0.5) * CAMERA_HFOV_DEG

    # Range: vertical position → distance (bottom=close, top=far)
    # Use inverse relationship: objects at bottom (cy=1.0) are near, top (cy=0.0) are far
    min_range = 3.0  # minimum radar range
    range_m = min_range + (1.0 - cy_norm) * (SCENE_DEPTH_M - min_range)

    # Add realistic noise
    angle_deg += random.gauss(0, ANGLE_NOISE_DEG)
    range_m += random.gauss(0, RANGE_NOISE_M)
    range_m = max(min_range, range_m)

    # Convert to cartesian (x = lateral, y = forward from radar)
    angle_rad = math.radians(angle_deg)
    x_m = range_m * math.sin(angle_rad)
    y_m = range_m * math.cos(angle_rad)

    return angle_deg, range_m, x_m, y_m


class RadarTrack:
    """Maintains state for a simulated radar track."""

    def __init__(self, track_id: int, cls: str, angle: float, range_m: float, x: float, y: float):
        self.track_id = track_id
        self.cls = cls
        self.angle = angle
        self.range_m = range_m
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.path = [{"t": 0.0, "x": round(x, 2), "y": round(y, 2)}]
        self.speeds = []

    def update(self, angle: float, range_m: float, x: float, y: float, dt: float):
        self.prev_x, self.prev_y = self.x, self.y
        self.angle = angle
        self.range_m = range_m
        self.x = x
        self.y = y
        self.last_seen = time.time()

        # Calculate velocity
        if dt > 0:
            dx = x - self.prev_x
            dy = y - self.prev_y
            speed = math.sqrt(dx**2 + dy**2) / dt
            speed += random.gauss(0, SPEED_NOISE_MS)
            speed = max(0, speed)
            heading = math.degrees(math.atan2(dx, dy)) % 360
        else:
            speed = 0
            heading = 0

        self.speeds.append(speed)
        elapsed = time.time() - self.first_seen
        self.path.append({"t": round(elapsed, 2), "x": round(x, 2), "y": round(y, 2)})

        return speed, heading

    @property
    def duration(self):
        return self.last_seen - self.first_seen


class RadarSimulator:
    """Simulates Axis D2110-VE radar by processing video and publishing MQTT."""

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"radar_sim_{RADAR_SERIAL}")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.connected = False
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.tracks: dict[int, RadarTrack] = {}
        self.model = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[RADAR] Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            self.connected = True
        else:
            print(f"[RADAR] MQTT connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        print(f"[RADAR] Disconnected from MQTT broker")
        self.connected = False

    def connect(self):
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            time.sleep(1)
        except Exception as e:
            print(f"[RADAR] Failed to connect to MQTT broker: {e}")
            raise

    def disconnect(self):
        # Publish track summaries for all active tracks
        for track in self.tracks.values():
            self._publish_track_summary(track)
        self.client.loop_stop()
        self.client.disconnect()

    def _publish_frame(self, detections: list):
        """Publish per-frame radar detections."""
        payload = {
            "source": "radar",
            "device_serial": RADAR_SERIAL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detections": detections,
        }
        self.client.publish(TOPIC_FRAME, json.dumps(payload), qos=0)

    def _publish_track_summary(self, track: RadarTrack):
        """Publish track summary when a track ends."""
        speeds = track.speeds if track.speeds else [0]
        payload = {
            "source": "radar",
            "device_serial": RADAR_SERIAL,
            "track_id": track.track_id,
            "class": track.cls,
            "start_time": datetime.fromtimestamp(track.first_seen, timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(track.last_seen, timezone.utc).isoformat(),
            "duration_s": round(track.duration, 1),
            "avg_speed_ms": round(sum(speeds) / len(speeds), 2),
            "max_speed_ms": round(max(speeds), 2),
            "min_speed_ms": round(min(speeds), 2),
            "path": track.path[-50:],  # Last 50 path points
        }
        self.client.publish(TOPIC_TRACK, json.dumps(payload), qos=1)

    def run_synced(self, video_path: str, conf: float = 0.4, fps: float = 5.0):
        """Run radar simulator synced to a video file."""
        print(f"[RADAR] Loading YOLO model for radar simulation...")
        self.model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[RADAR] ERROR: Cannot open video: {video_path}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_delay = 1.0 / fps
        names = self.model.names if hasattr(self.model, "names") else {}

        print(f"[RADAR] Simulating Axis D2110-VE — FOV: {RADAR_FOV_DEG}°, Max range: {RADAR_MAX_RANGE_VEHICLE}m")
        print(f"[RADAR] Publishing to MQTT topic: {TOPIC_FRAME}")
        print(f"[RADAR] Video: {video_path} ({frame_w}x{frame_h}) at {fps} Hz")

        prev_time = time.time()
        active_track_ids = set()

        while True:
            t0 = time.time()
            dt = t0 - prev_time
            prev_time = t0

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            # Run lightweight detection (no tracking needed, just detections)
            results = self.model(frame, conf=conf, verbose=False)[0]

            detections = []
            current_ids = set()

            for i, box in enumerate(results.boxes):
                cls = int(box.cls[0])
                label = names.get(cls, str(cls))
                radar_cls = YOLO_TO_RADAR.get(label)
                if radar_cls is None:
                    continue

                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_norm = ((x1 + x2) / 2) / frame_w
                cy_norm = ((y1 + y2) / 2) / frame_h

                # Convert to radar coordinates
                angle_deg, range_m, x_m, y_m = pixel_to_radar(cx_norm, cy_norm, frame_w, frame_h)

                # Use detection index as track_id (stable for single detections per class)
                track_id = i + 1
                current_ids.add(track_id)

                # Estimate object size from bbox
                bbox_w = (x2 - x1) / frame_w
                bbox_h = (y2 - y1) / frame_h
                size_m = max(bbox_w, bbox_h) * SCENE_WIDTH_M * 0.3
                size_m = round(max(0.3, min(size_m, 5.0)), 1)

                # Update or create track
                if track_id in self.tracks:
                    track = self.tracks[track_id]
                    speed, heading = track.update(angle_deg, range_m, x_m, y_m, dt)
                else:
                    track = RadarTrack(track_id, radar_cls, angle_deg, range_m, x_m, y_m)
                    self.tracks[track_id] = track
                    speed, heading = 0.0, 0.0

                # Add radar confidence (slightly different from camera)
                radar_conf = min(0.99, conf_val * (0.85 + random.uniform(0, 0.15)))

                detection = {
                    "track_id": track_id,
                    "class": radar_cls,
                    "confidence": round(radar_conf, 2),
                    "polar": {
                        "angle_deg": round(angle_deg, 1),
                        "range_m": round(range_m, 1),
                    },
                    "cartesian": {
                        "x_m": round(x_m, 1),
                        "y_m": round(y_m, 1),
                    },
                    "velocity": {
                        "speed_ms": round(speed, 2),
                        "heading_deg": round(heading, 1),
                    },
                    "size_m": size_m,
                    "bbox_normalized": {
                        "cx": round(cx_norm, 3),
                        "cy": round(cy_norm, 3),
                    },
                }
                detections.append(detection)

            # Publish frame
            if detections or active_track_ids:
                self._publish_frame(detections)

            # Check for ended tracks
            ended_ids = active_track_ids - current_ids
            for tid in ended_ids:
                if tid in self.tracks:
                    self._publish_track_summary(self.tracks[tid])
                    del self.tracks[tid]

            active_track_ids = current_ids

            # Log periodically
            if int(t0) % 5 == 0 and detections:
                det_summary = ", ".join([f"{d['class']}@{d['polar']['range_m']}m" for d in detections])
                print(f"[RADAR] {len(detections)} detections: {det_summary}")

            # Throttle
            elapsed = time.time() - t0
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        print("[RADAR] Simulation ended")

    def run_standalone(self, fps: float = 2.0, num_targets: int = 3):
        """Run with synthetic moving targets (no video needed)."""
        print(f"[RADAR] Running standalone mode with {num_targets} synthetic targets")

        # Create synthetic targets
        targets = []
        for i in range(num_targets):
            cls = random.choice(["Human", "Human", "Vehicle"])
            angle = random.uniform(-80, 80)
            range_m = random.uniform(5, 50)
            speed = random.uniform(0.5, 3.0) if cls == "Human" else random.uniform(5, 15)
            heading = random.uniform(0, 360)
            targets.append({
                "id": i + 1,
                "class": cls,
                "angle": angle,
                "range": range_m,
                "speed": speed,
                "heading": heading,
            })

        while True:
            t0 = time.time()
            detections = []

            for t in targets:
                # Move target
                heading_rad = math.radians(t["heading"])
                dt = 1.0 / fps
                dx = t["speed"] * math.sin(heading_rad) * dt
                dy = t["speed"] * math.cos(heading_rad) * dt

                angle_rad = math.radians(t["angle"])
                x = t["range"] * math.sin(angle_rad) + dx
                y = t["range"] * math.cos(angle_rad) + dy

                t["range"] = math.sqrt(x**2 + y**2)
                t["angle"] = math.degrees(math.atan2(x, y))

                # Bounce at boundaries
                if t["range"] > 70 or t["range"] < 3:
                    t["heading"] = (t["heading"] + 180) % 360
                if abs(t["angle"]) > 85:
                    t["heading"] = (360 - t["heading"]) % 360

                # Add noise
                noisy_angle = t["angle"] + random.gauss(0, ANGLE_NOISE_DEG)
                noisy_range = t["range"] + random.gauss(0, RANGE_NOISE_M)

                x_m = noisy_range * math.sin(math.radians(noisy_angle))
                y_m = noisy_range * math.cos(math.radians(noisy_angle))

                detection = {
                    "track_id": t["id"],
                    "class": t["class"],
                    "confidence": round(random.uniform(0.75, 0.98), 2),
                    "polar": {
                        "angle_deg": round(noisy_angle, 1),
                        "range_m": round(noisy_range, 1),
                    },
                    "cartesian": {
                        "x_m": round(x_m, 1),
                        "y_m": round(y_m, 1),
                    },
                    "velocity": {
                        "speed_ms": round(t["speed"] + random.gauss(0, SPEED_NOISE_MS), 2),
                        "heading_deg": round(t["heading"], 1),
                    },
                    "size_m": 0.5 if t["class"] == "Human" else 2.0,
                    "bbox_normalized": {
                        "cx": round(0.5 + noisy_angle / CAMERA_HFOV_DEG, 3),
                        "cy": round(1.0 - noisy_range / SCENE_DEPTH_M, 3),
                    },
                }
                detections.append(detection)

            self._publish_frame(detections)

            elapsed = time.time() - t0
            time.sleep(max(0, (1.0 / fps) - elapsed))


def main():
    parser = argparse.ArgumentParser(description="Axis D2110-VE Radar Simulator")
    parser.add_argument("--video", type=str, default=None, help="Video file for synced mode")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence")
    parser.add_argument("--fps", type=float, default=5.0, help="Publish rate (Hz)")
    parser.add_argument("--standalone", action="store_true", help="Run with synthetic targets")
    args = parser.parse_args()

    sim = RadarSimulator(broker_host=args.broker, broker_port=args.port)

    try:
        sim.connect()

        if args.standalone:
            sim.run_standalone(fps=args.fps)
        elif args.video:
            video_path = str(BASE_DIR / args.video) if not args.video.startswith("/") else args.video
            sim.run_synced(video_path, conf=args.conf, fps=args.fps)
        else:
            # Default: try test2.mp4
            default_video = str(BASE_DIR / "test2.mp4")
            if Path(default_video).exists():
                sim.run_synced(default_video, conf=args.conf, fps=args.fps)
            else:
                print("[RADAR] No video file specified, running standalone mode")
                sim.run_standalone(fps=args.fps)
    except KeyboardInterrupt:
        print("\n[RADAR] Shutting down...")
    finally:
        sim.disconnect()


if __name__ == "__main__":
    main()
