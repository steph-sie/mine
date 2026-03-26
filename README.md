
Home Security CCTV Monitor

YOLOv8 + OpenAI GPT + Tkinter GUI for real-time home surveillance.
Detects people, vehicles, and suspicious activity with automatic classification into NORMAL, CAUTION, and THREAT events.

YOLO handles the heavy lifting of object detection, while potential threats are analyzed and summarized by ChatGPT.

Features
YOLOv8 object detection (people, cars, trucks, etc).
Behavior-based logic:
Walking past → CAUTION
Approaching house or camera → THREAT
Walking away → CAUTION
Loitering near or interacting with cars → THREAT
AI-powered event summaries via OpenAI Chat API.
Timeline logging with memory (last 10 alerts).
Automatic snapshots saved to detections/.
Tkinter GUI with live video feed, status panel, and log window.

<img width="1125" height="1071" alt="image" src="https://github.com/user-attachments/assets/43189b47-324e-45ef-b063-7a126eaa6298" />

<img width="1123" height="1071" alt="image" src="https://github.com/user-attachments/assets/a7c16dfa-62f8-41a6-89b0-82cf488b46ae" />

<img width="1125" height="1071" alt="image" src="https://github.com/user-attachments/assets/961a6548-56df-4f17-af9b-ca2f2124d3e5" />

---

## Updated Interfaces

Beyond the original Tkinter GUI, the repository now includes:

* **Streamlit dashboard** (`main2.py`) for lightweight local operation.
* **FastAPI detection service** (`main3.py`) exposing `/detect` for image uploads.
* **Static web client** (`frontend/static_client.html`) and a full Next.js/Tailwind
  prototype under `frontend/` (requires Node/npm).
* **RTSP detector script** (`rtsp_detector.py`) that can consume an IP camera
  stream directly.

### Running the new tools

Install dependencies once:

```bash
pip install -r requirements.txt
```

#### Streamlit dashboard

```bash
streamlit run main2.py
```

#### API server

```bash
python main3.py
# or: uvicorn main3:app --host 0.0.0.0 --port 8000
```

POST an image to test:

```bash
curl -X POST "http://localhost:8000/detect?conf=0.4" -F file=@frame.jpg
```

#### Static browser client (no Node)

1. `cd frontend`
2. Serve files: `python -m http.server 5500`
3. Open `http://localhost:5500/static_client.html` and allow camera access.

#### RTSP/IP camera stream

```bash
python rtsp_detector.py --url rtsp://USER:PASS@CAM-IP/stream --conf 0.4
```

Logs and snapshots are written to `security_log.txt` and `detections/`.

---

*For advanced usage, see comments in each script or ask for tailoring to your
setup.*
