"""
Entity Registry — Stateful tracker with Re-ID appearance matching.
Tracks people/objects across frames with dedup, enter/exit detection,
and MobileNetV2 embeddings for re-identifying returning entities.
"""

import time
import base64
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from threading import Lock

import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

LOG_FILE = Path(__file__).parent / "security_log.txt"

# ── Re-ID Embedding Model ────────────────────────────────────────────────────

_reid_model = None
_reid_transform = None
_reid_device = None


def _get_reid_model():
    """Lazy-load MobileNetV2 feature extractor (singleton)."""
    global _reid_model, _reid_transform, _reid_device
    if _reid_model is None:
        _reid_device = "mps" if torch.backends.mps.is_available() else "cpu"
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        # Remove classifier, keep feature extractor (outputs 1280-dim vector)
        model.classifier = torch.nn.Identity()
        model.eval()
        model.to(_reid_device)
        _reid_model = model
        _reid_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),  # Standard ReID input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _reid_model, _reid_transform, _reid_device


def extract_embedding(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract a 1280-dim appearance embedding from a BGR crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
        return None
    try:
        model, transform, device = _get_reid_model()
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(crop_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


# ── Entity Data ───────────────────────────────────────────────────────────────

@dataclass
class TrackedEntity:
    id: int
    label: str
    display_name: str
    first_seen: float
    last_seen: float
    status: str = "active"  # "active" | "gone"
    source: str = "camera"  # "camera" | "radar" | "fused"
    best_snapshot_b64: Optional[str] = None
    best_conf: float = 0.0
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    frame_count: int = 0
    gone_time: Optional[float] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # Track all YOLO track_ids that mapped to this entity (for merges)
    merged_track_ids: List[int] = field(default_factory=list)
    # Radar metadata (populated when fused or radar-only)
    radar_data: Optional[dict] = field(default=None, repr=False)

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
            "source": self.source,
            "duration": round(self.duration, 1),
            "best_conf": round(self.best_conf, 2),
            "last_bbox": list(self.last_bbox),
            "frame_count": self.frame_count,
            "thumbnail_b64": self.best_snapshot_b64,
            "gone_time": self.gone_time,
            "reidentified": len(self.merged_track_ids) > 1,
            "radar_data": self.radar_data,
        }


# ── Entity Registry ──────────────────────────────────────────────────────────

class EntityRegistry:
    def __init__(self, gone_timeout: float = 5.0, reid_threshold: float = 0.70, reid_enabled: bool = True):
        self.entities: Dict[int, TrackedEntity] = {}
        self.class_counters: Dict[str, int] = {}
        self.events: List[dict] = []
        self.gone_timeout = gone_timeout
        self.reid_threshold = reid_threshold
        self.reid_enabled = reid_enabled
        self.session_start: float = time.time()
        self._lock = Lock()
        # Maps YOLO track_id → canonical entity id (for merged entities)
        self._track_id_map: Dict[int, int] = {}

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

    def _find_reid_match(self, label: str, embedding: np.ndarray, timestamp: float) -> Optional[int]:
        """Find a recently-gone entity that matches this embedding."""
        if embedding is None:
            return None

        best_id = None
        best_sim = self.reid_threshold

        for eid, entity in self.entities.items():
            # Only match against gone entities of the same class
            if entity.status != "gone" or entity.label != label:
                continue
            # Only match against recently-gone entities (within 60s)
            if entity.gone_time and (timestamp - entity.gone_time) > 60.0:
                continue
            if entity.embedding is None:
                continue

            sim = cosine_similarity(embedding, entity.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = eid

        return best_id

    def _resolve_track_id(self, track_id: int) -> int:
        """Resolve a YOLO track_id to the canonical entity id."""
        return self._track_id_map.get(track_id, track_id)

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
            # Check if this track_id was merged into an existing entity
            canonical_id = self._track_id_map.get(track_id)
            if canonical_id is not None and canonical_id in self.entities:
                entity = self.entities[canonical_id]
                entity.last_seen = timestamp
                entity.last_bbox = bbox
                entity.frame_count += 1
                if entity.status == "gone":
                    entity.status = "active"
                    entity.gone_time = None
                if conf > entity.best_conf:
                    entity.best_conf = conf
                    entity.best_snapshot_b64 = self._encode_crop(frame_crop)
                    if self.reid_enabled:
                        emb = extract_embedding(frame_crop)
                        if emb is not None:
                            entity.embedding = emb
                return None

            # Known entity with same track_id
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
                    if self.reid_enabled:
                        emb = extract_embedding(frame_crop)
                        if emb is not None:
                            entity.embedding = emb
                return None

            # ── New track_id: try Re-ID match against gone entities ──
            embedding = None
            if self.reid_enabled:
                embedding = extract_embedding(frame_crop)

            matched_id = self._find_reid_match(label, embedding, timestamp) if self.reid_enabled else None

            if matched_id is not None:
                # Re-ID match found! Merge: reuse existing entity
                entity = self.entities[matched_id]
                entity.last_seen = timestamp
                entity.last_bbox = bbox
                entity.frame_count += 1
                entity.status = "active"
                entity.gone_time = None
                entity.merged_track_ids.append(track_id)
                if conf > entity.best_conf:
                    entity.best_conf = conf
                    entity.best_snapshot_b64 = self._encode_crop(frame_crop)
                if embedding is not None:
                    entity.embedding = embedding
                # Map new track_id → canonical entity
                self._track_id_map[track_id] = matched_id

                event = {
                    "time": time.strftime("%H:%M:%S", time.localtime(timestamp)),
                    "timestamp": timestamp,
                    "type": "reidentified",
                    "entity_id": matched_id,
                    "message": f"{entity.display_name} re-identified (was gone)",
                }
                self.events.append(event)
                self._log_to_file("RE-ID", f"{entity.display_name} re-identified after departure")
                return event

            # ── Truly new entity ──
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
                embedding=embedding,
                merged_track_ids=[track_id],
            )
            self.entities[track_id] = entity
            self._track_id_map[track_id] = track_id

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
            reids = sum(1 for e in self.entities.values() if len(e.merged_track_ids) > 1)

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

            reid_str = f" {reids} re-identified." if reids else ""
            text = f"Currently tracking {active_desc}. {total} unique entities seen ({', '.join(total_parts)}).{reid_str} Session: {session_str}."

            return {
                "active_count": len(active),
                "gone_count": len(gone),
                "total_unique": total,
                "reidentified_count": reids,
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
            self._track_id_map.clear()
            self.session_start = time.time()
        self._log_to_file("RESET", "Entity registry cleared")
