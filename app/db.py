import sqlite3
from .config import DB

def conn():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    return c

def init():
    c = conn()
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript("""
    CREATE TABLE IF NOT EXISTS videos(
      id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, path TEXT,
      status TEXT DEFAULT 'uploaded', stage TEXT DEFAULT 'not analysed',
      progress REAL DEFAULT 0, fps REAL, frames INTEGER, duration REAL,
      width INTEGER, height INTEGER, error TEXT,
      created TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS detections(
      video_id INTEGER, frame INTEGER, t REAL, track_id INTEGER, cls TEXT,
      conf REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL);
    CREATE INDEX IF NOT EXISTS ix_det ON detections(video_id, track_id, frame);
    """)
    c.commit(); c.close()
