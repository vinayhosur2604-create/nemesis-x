# NEMESIS-X backend (Stage 2)

FastAPI + YOLO/ByteTrack + SQLite. Runs fully local; **no API key needed**.

## Setup
1. Python 3.10+.
2. Install PyTorch first (CUDA build for RTX 3050/4050): https://pytorch.org/get-started/locally/
   (CPU-only also works, just slower.)
3. Then:
   ```
   python -m venv .venv
   # Windows: .venv\Scripts\activate      Ubuntu: source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --port 8000
   ```
4. Open http://localhost:8000 (first analysis downloads the YOLO weights, so it needs internet once).

## Config (env vars)
NEMESIS_MODEL (default yolov8n.pt) · NEMESIS_CONF (0.25) · NEMESIS_STRIDE (1)

## API
GET /api/health · /api/system · POST /api/videos · GET /api/videos · GET /api/videos/{id}
POST /api/videos/{id}/analyze · GET .../tracks · .../tracks/{tid} · .../frame?t=SEC · .../file
POST /api/chat {question, video_id?}

## Notes
- Ultralytics is AGPL-3.0; check its licence for your use.
- Coordinates are image-plane pixels, not metric 3D.
- Not yet implemented: event candidates, anomaly scoring, re-identification, and wiring the frontend panels to these endpoints.
