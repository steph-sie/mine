import os
import time
from collections import Counter
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
import html
import base64

from ultralytics import YOLO


BASE_DIR = Path(__file__).parent
DETECTIONS_DIR = BASE_DIR / "detections"
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_DIR / "security_log.txt"


def load_historical_summaries():
    summaries = []
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        intervals = {}  # key: (start_minute, end_minute), value: list of Counter
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and ']' in line:
                timestamp_str = line[1:line.find(']')]
                try:
                    ts = time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    ts_float = time.mktime(ts)
                    content = line[line.find(']')+1:].strip()
                    counts = Counter()
                    if content != "No objects detected":
                        parts = content.split(', ')
                        for part in parts:
                            obj, num = part.split(':')
                            counts[obj] = int(num)
                    # 5-minute intervals
                    minutes_since_epoch = int(ts_float // 60)
                    interval_start_min = (minutes_since_epoch // 5) * 5
                    interval_end_min = interval_start_min + 5
                    key = (interval_start_min, interval_end_min)
                    if key not in intervals:
                        intervals[key] = []
                    intervals[key].append(counts)
                except:
                    pass  # skip malformed lines
        # compute max counts per interval
        for key, count_list in sorted(intervals.items()):
            if count_list:
                max_counts = Counter()
                for c in count_list:
                    for k, v in c.items():
                        max_counts[k] = max(max_counts.get(k, 0), v)
                start_ts = key[0] * 60
                end_ts = key[1] * 60
                start_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(start_ts))
                end_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(end_ts))
                if max_counts:
                    parts = [f"{v} {k}" for k, v in sorted(max_counts.items())]
                    summary_line = f"{start_str}-{end_str}: " + ", ".join(parts)
                else:
                    summary_line = f"{start_str}-{end_str}: No objects detected"
                summaries.append(summary_line)
    return summaries


@st.cache_resource
def load_model(weights_path: str = "yolov8s.pt"):
    return YOLO(weights_path)


def draw_boxes(frame, results, names, tracking_enabled=False):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = names.get(cls, str(cls))
        conf = float(box.conf[0])
        color = (0, 255, 0)
        if label == "person":
            color = (0, 0, 255)
        elif label in ("car", "truck"):
            color = (255, 165, 0)

        id_str = ""
        if tracking_enabled and hasattr(box, "id"):
            id_val = int(box.id[0])
            id_str = f" ID{id_val}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}{id_str} {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def log_counts(counts: Counter, ts: float):
    """Write counts to log file; returns formatted line."""
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    if not counts:
        line = f"[{timestr}] No objects detected"
    else:
        parts = [f"{k}:{v}" for k, v in counts.items()]
        line = f"[{timestr}] " + ", ".join(parts)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    return line


def main():
    st.set_page_config(page_title="CCTV Object Detection & Security Log", layout="wide")
    st.title("[Prototype] CCTV Object Detection & Security Log")

    # Load historical summaries
    historical_summaries = load_historical_summaries()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        source_option = st.selectbox("Video source", ["Sample file", "Upload file", "Webcam"], index=0)
        confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.4)
        fps = st.slider("Target FPS", 1, 30, 8)
        save_snapshots = st.checkbox("Save snapshots on detections", value=True)
        tracking_enabled = st.checkbox("Enable tracking (assign IDs)", value=False)
        # Clear security log button
        if st.button("Clear security log"):
            try:
                open(LOG_FILE, "w", encoding="utf-8").close()
            except Exception:
                pass
            st.session_state.log_lines = []
            st.session_state.summary_lines = []
            # clear snapshots state (optionally delete files)
            st.session_state.snapshots = []
            # reset tracking state
            st.session_state.last_snapshot_time_by_id = {}
            st.success("Security log cleared")

        if source_option == "Sample file":
            video_path = st.text_input("Path to video file", "test2.mp4")
            upload_file = None
        elif source_option == "Upload file":
            upload_file = st.file_uploader("Upload video (mp4, avi)", type=["mp4", "avi", "mov"])
            video_path = None
        else:
            video_path = 0
            upload_file = None

    model = load_model("yolov8s.pt")

    # UI columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty()

    with col2:
        st.subheader("Event Log")
        # Scrollable log area: fixed height with vertical scroll
        st.markdown(
            """
            <style>
            .log-box {
                height: 480px;
                overflow-y: auto;
                background: #0f172a;
                color: #e6eef8;
                padding: 12px;
                border-radius: 6px;
                font-family: monospace;
                font-size: 13px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        log_box = st.empty()

    # session state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []
    if "last_log_time" not in st.session_state:
        st.session_state.last_log_time = 0.0
    if "summary_lines" not in st.session_state:
        st.session_state.summary_lines = historical_summaries.copy()
    if "interval_start" not in st.session_state:
        st.session_state.interval_start = 0.0
    if "interval_counts" not in st.session_state:
        st.session_state.interval_counts = []
    if "last_snapshot_time" not in st.session_state:
        st.session_state.last_snapshot_time = 0.0
    if "last_snapshot_time_by_id" not in st.session_state:
        st.session_state.last_snapshot_time_by_id = {}
    if "snapshots" not in st.session_state:
        # load existing snapshot paths sorted by modified time
        files = list(DETECTIONS_DIR.glob("snapshot_*.jpg")) + list(DETECTIONS_DIR.glob("crop_*.jpg"))
        files = sorted(files, key=lambda p: p.stat().st_mtime)
        st.session_state.snapshots = [(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)), str(p)) for p in files]

    start = st.button("Start")
    stop = st.button("Stop")

    if start:
        st.session_state.running = True
        st.session_state.interval_start = time.time()
        st.session_state.interval_counts = []
    if stop:
        st.session_state.running = False

    # Summary section
    st.subheader("Summary")
    st.markdown(
        """
        <style>
        .summary-box {
            height: 200px;
            overflow-y: auto;
            background: #0f172a;
            color: #e6eef8;
            padding: 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 13px;
        }
        .gallery {
            display: flex;
            flex-wrap: wrap;
        }
        .thumb {
            margin: 4px;
            border: 1px solid #444;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    summary_placeholder = st.empty()
    gallery_placeholder = st.empty()

    # Display initial historical summaries
    summary_lines = st.session_state.summary_lines[-20:]
    summary_content = "<div class='summary-box'>" + "<br>".join([html.escape(s) for s in summary_lines[::-1]]) + "</div>"
    summary_placeholder.markdown(summary_content, unsafe_allow_html=True)

    # show existing gallery thumbnails
    if st.session_state.snapshots:
        recent = st.session_state.snapshots[-20:]
        gallery_html = "<div class='gallery'>"
        for ts, path in recent[::-1]:
            try:
                with open(path, "rb") as imgf:
                    b64 = base64.b64encode(imgf.read()).decode()
                gallery_html += f"<a href='data:image/jpeg;base64,{b64}' target='_blank'><img class='thumb' src='data:image/jpeg;base64,{b64}' width='120' title='{ts}'/></a>"
            except Exception:
                continue
        gallery_html += "</div>"
        gallery_placeholder.markdown(gallery_html, unsafe_allow_html=True)

    if upload_file is not None:
        # save uploaded to a temp file
        temp_path = BASE_DIR / "_uploaded_video"
        with open(temp_path, "wb") as f:
            f.write(upload_file.getbuffer())
        video_path = str(temp_path)

    # Open capture
    cap = None
    if video_path is not None and st.session_state.running:
        try:
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            st.error(f"Cannot open video: {e}")

    # Run loop
    try:
        while st.session_state.running:
            if cap is None or not cap.isOpened():
                time.sleep(0.5)
                st.session_state.running = False
                break

            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                # video ended
                st.session_state.running = False
                break

            # run detection (optionally track)
            if tracking_enabled:
                results = model.track(frame, conf=confidence, verbose=False)[0]
            else:
                results = model(frame, conf=confidence, verbose=False)[0]
            names = model.names if hasattr(model, "names") else {}

            # count objects (only person, car, truck)
            raw_classes = []
            track_ids = []
            for b in results.boxes:
                raw_classes.append(names.get(int(b.cls[0]), str(int(b.cls[0]))))
                if tracking_enabled and hasattr(b, "id"):
                    track_ids.append(int(b.id[0]))
                else:
                    track_ids.append(None)
            allowed = {"person", "car", "truck"}
            classes = [c for c in raw_classes if c in allowed]
            counts = Counter(classes)

            # draw
            vis = draw_boxes(frame.copy(), results, names, tracking_enabled=tracking_enabled)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(vis_rgb)
            # use explicit width (use_column_width deprecated)
            video_placeholder.image(img, width=960)

            # logging every ~1 second
            now = time.time()
            if now - st.session_state.last_log_time >= 1.0:
                # log counts
                line = log_counts(counts, now)
                st.session_state.log_lines.append(line)
                st.session_state.last_log_time = now

                if save_snapshots and frame is not None and counts:
                    # throttle snapshots to once every 5 seconds (or per ID)
                    if tracking_enabled:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                        for i, box in enumerate(results.boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            label = names.get(cls, str(cls))
                            if label not in ("person", "car", "truck"):
                                continue
                            track_id = track_ids[i] if i < len(track_ids) else None
                            last = st.session_state.last_snapshot_time_by_id.get(track_id, 0)
                            if track_id is not None and now - last < 5.0:
                                continue
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue
                            fname = DETECTIONS_DIR / f"crop_{int(now)}_{label}_{track_id}_{i}.jpg"
                            cv2.imwrite(str(fname), crop)
                            st.session_state.snapshots.append((ts, str(fname)))
                            if track_id is not None:
                                st.session_state.last_snapshot_time_by_id[track_id] = now
                        st.session_state.snapshots = st.session_state.snapshots[-100:]
                    else:
                        if now - st.session_state.last_snapshot_time >= 5.0:
                            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                            for i, box in enumerate(results.boxes):
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cls = int(box.cls[0])
                                label = names.get(cls, str(cls))
                                if label not in ("person", "car", "truck"):
                                    continue
                                crop = frame[y1:y2, x1:x2]
                                if crop.size == 0:
                                    continue
                                fname = DETECTIONS_DIR / f"crop_{int(now)}_{label}_{i}.jpg"
                                cv2.imwrite(str(fname), crop)
                                st.session_state.snapshots.append((ts, str(fname)))
                            # keep only last 100 thumbnails
                            st.session_state.snapshots = st.session_state.snapshots[-100:]
                            st.session_state.last_snapshot_time = now

                st.session_state.interval_counts.append(counts)

                # check for summary interval
                if now - st.session_state.interval_start >= 300:  # 5 minutes
                    if st.session_state.interval_counts:
                        max_counts = Counter()
                        for c in st.session_state.interval_counts:
                            for k, v in c.items():
                                max_counts[k] = max(max_counts.get(k, 0), v)
                        start_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.session_state.interval_start))
                        end_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
                        if max_counts:
                            parts = [f"{v} {k}" for k, v in sorted(max_counts.items())]
                            summary_line = f"{start_str}-{end_str}: " + ", ".join(parts)
                        else:
                            summary_line = f"{start_str}-{end_str}: No objects detected"
                        st.session_state.summary_lines.append(summary_line)
                    # reset
                    st.session_state.interval_start = now
                    st.session_state.interval_counts = []

            # update log UI (keep last 200 lines). Render inside scrollable div.
            lines = st.session_state.log_lines[-200:]
            # escape log lines to safe HTML
            content = "<div class='log-box'>" + "<br>".join([html.escape(s) for s in lines[::-1]]) + "</div>"
            log_box.markdown(content, unsafe_allow_html=True)

            # update summary
            summary_lines = st.session_state.summary_lines[-20:]
            summary_content = "<div class='summary-box'>" + "<br>".join([html.escape(s) for s in summary_lines[::-1]]) + "</div>"
            summary_placeholder.markdown(summary_content, unsafe_allow_html=True)

            # update gallery
            if st.session_state.snapshots:
                recent = st.session_state.snapshots[-20:]
                gallery_html = "<div class='gallery'>"
                for ts, path in recent[::-1]:
                    try:
                        with open(path, "rb") as imgf:
                            b64 = base64.b64encode(imgf.read()).decode()
                        gallery_html += f"<a href='data:image/jpeg;base64,{b64}' target='_blank'><img class='thumb' src='data:image/jpeg;base64,{b64}' width='120' title='{ts}'/></a>"
                    except Exception:
                        continue
                gallery_html += "</div>"
                gallery_placeholder.markdown(gallery_html, unsafe_allow_html=True)

            # throttle to target FPS
            elapsed = time.time() - start_time
            to_wait = max(0, (1.0 / fps) - elapsed)
            time.sleep(to_wait)

    finally:
        if cap is not None:
            cap.release()

    # finalize last interval if any
    if st.session_state.interval_counts:
        max_counts = Counter()
        for c in st.session_state.interval_counts:
            for k, v in c.items():
                max_counts[k] = max(max_counts.get(k, 0), v)
        start_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.session_state.interval_start))
        end_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time()))
        if max_counts:
            parts = [f"{v} {k}" for k, v in sorted(max_counts.items())]
            summary_line = f"{start_str}-{end_str}: " + ", ".join(parts)
        else:
            summary_line = f"{start_str}-{end_str}: No objects detected"
        st.session_state.summary_lines.append(summary_line)
        st.session_state.interval_counts = []
        # update summary one last time
        summary_lines = st.session_state.summary_lines[-20:]
        summary_content = "<div class='summary-box'>" + "<br>".join([html.escape(s) for s in summary_lines[::-1]]) + "</div>"
        summary_placeholder.markdown(summary_content, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
