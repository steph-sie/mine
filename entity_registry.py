"""
Entity Registry — Stateful tracker for persistent entity identification.
Tracks people/objects across frames with dedup, enter/exit detection, and narrative logging.
"""

import time
import base64
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from threading import Lock

LOG_FILE = Path(__file__).parent / "security_log.txt"


@dataclass
class TrackedEntity:
    id: int
    label: str
    display_name: str
    first_seen: float
    last_seen: float
    status: str = "active"  # "active" | "gone"
    best_snapshot_b64: Optional[str] = None
    best_conf: float = 0.0
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    frame_count: int = 0
    gone_time: Optional[float] = None

    @property
    def duration(self) -> float:
        end = self.gone_time if self.gone_time else self.last_seen
        return end - self.first_seen

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "display_name": self.display_name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "status": self.status,
            "duration": round(self.duration, 1),
            "best_conf": round(self.best_conf, 2),
            "last_bbox": list(self.last_bbox),
            "frame_count": self.frame_count,
            "thumbnail_b64": self.best_snapshot_b64,
            "gone_time": self.gone_time,
        }


class EntityRegistry:
    def __init__(self, gone_timeout: float = 5.0):
        self.entities: Dict[int, TrackedEntity] = {}
        self.class_counters: Dict[str, int] = {}
        self.events: List[dict] = []
        self.gone_timeout = gone_timeout
        self.session_start: float = time.time()
        self._lock = Lock()

    def _make_display_name(self, label: str) -> str:
        count = self.class_counters.get(label, 0) + 1
        self.class_counters[label] = count
        return f"{label.capitalize()} #{count}"

    def _log_to_file(self, event_type: str, message: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {event_type} — {message}"
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _encode_crop(self, crop: np.ndarray) -> Optional[str]:
        if crop is None or crop.size == 0:
            return None
        try:
            _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 60])
            return base64.b64encode(buf).decode("ascii")
        except Exception:
            return None

    def update(
        self,
        track_id: int,
        label: str,
        conf: float,
        bbox: Tuple[int, int, int, int],
        frame_crop: Optional[np.ndarray],
        timestamp: float,
    ) -> Optional[dict]:
        with self._lock:
            if track_id in self.entities:
                entity = self.entities[track_id]
                entity.last_seen = timestamp
                entity.last_bbox = bbox
                entity.frame_count += 1
                if entity.status == "gone":
                    entity.status = "active"
                    entity.gone_time = None
                    event = {
                        "time": time.strftime("%H:%M:%S", time.localtime(timestamp)),
                        "timestamp": timestamp,
                        "type": "reappeared",
                        "entity_id": track_id,
                        "message": f"{entity.display_name} reappeared",
                    }
                    self.events.append(event)
                    self._log_to_file("REAPPEARED", event["message"])
                    return event
                if conf > entity.best_conf:
                    entity.best_conf = conf
                    entity.best_snapshot_b64 = self._encode_crop(frame_crop)
                return None
            else:
                display_name = self._make_display_name(label)
                snapshot_b64 = self._encode_crop(frame_crop)
                entity = TrackedEntity(
                    id=track_id,
                    label=label,
                    display_name=display_name,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    best_snapshot_b64=snapshot_b64,
                    best_conf=conf,
                    last_bbox=bbox,
                    frame_count=1,
                )
                self.entities[track_id] = entity
                event = {
                    "time": time.strftime("%H:%M:%S", time.localtime(timestamp)),
                    "timestamp": timestamp,
                    "type": "appeared",
                    "entity_id": track_id,
                    "message": f"{display_name} appeared",
                }
                self.events.append(event)
                self._log_to_file("APPEARED", f"{display_name} entered the scene")
                return event

    def sweep_gone(self, current_time: Optional[float] = None) -> List[dict]:
        now = current_time or time.time()
        new_events = []
        with self._lock:
            for entity in self.entities.values():
                if entity.status == "active" and (now - entity.last_seen) > self.gone_timeout:
                    entity.status = "gone"
                    entity.gone_time = now
                    dur = round(entity.duration, 1)
                    event = {
                        "time": time.strftime("%H:%M:%S", time.localtime(now)),
                        "timestamp": now,
                        "type": "left",
                        "entity_id": entity.id,
                        "message": f"{entity.display_name} left after {dur}s",
                    }
                    new_events.append(event)
                    self.events.append(event)
                    self._log_to_file("LEFT", f"{entity.display_name} departed after {dur}s")
        return new_events

    def get_active(self) -> List[dict]:
        with self._lock:
            return [e.to_dict() for e in self.entities.values() if e.status == "active"]

    def get_gone(self) -> List[dict]:
        with self._lock:
            return [e.to_dict() for e in self.entities.values() if e.status == "gone"]

    def get_recent_events(self, n: int = 50) -> List[dict]:
        with self._lock:
            return list(self.events[-n:])

    def generate_summary(self) -> dict:
        with self._lock:
            active = [e for e in self.entities.values() if e.status == "active"]
            gone = [e for e in self.entities.values() if e.status == "gone"]
            total = len(self.entities)

            class_counts = {}
            for e in active:
                class_counts[e.label] = class_counts.get(e.label, 0) + 1

            parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(class_counts.items())]
            active_desc = ", ".join(parts) if parts else "nothing"

            total_by_class = {}
            for e in self.entities.values():
                total_by_class[e.label] = total_by_class.get(e.label, 0) + 1
            total_parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(total_by_class.items())]

            session_dur = time.time() - self.session_start
            if session_dur < 60:
                session_str = f"{int(session_dur)}s"
            else:
                session_str = f"{int(session_dur // 60)}m {int(session_dur % 60)}s"

            text = f"Currently tracking {active_desc}. {total} unique entities seen ({', '.join(total_parts)}). Session: {session_str}."

            return {
                "active_count": len(active),
                "gone_count": len(gone),
                "total_unique": total,
                "session_duration": round(session_dur, 1),
                "session_duration_str": session_str,
                "text": text,
            }

    def get_state(self) -> dict:
        return {
            "entities": {
                "active": self.get_active(),
                "gone": self.get_gone(),
            },
            "events": self.get_recent_events(),
            "summary": self.generate_summary(),
        }

    def reset(self):
        with self._lock:
            self.entities.clear()
            self.class_counters.clear()
            self.events.clear()
            self.session_start = time.time()
        self._log_to_file("RESET", "Entity registry cleared")
