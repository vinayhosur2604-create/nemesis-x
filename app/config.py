import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
DB = DATA / "nemesis.db"
MODEL = os.getenv("NEMESIS_MODEL", "yolov8n.pt")   # downloaded on first run
CONF = float(os.getenv("NEMESIS_CONF", "0.25"))
STRIDE = int(os.getenv("NEMESIS_STRIDE", "1"))     # analyse every Nth frame
ALLOWED = {".mp4", ".avi", ".mov"}
UPLOADS.mkdir(parents=True, exist_ok=True)
