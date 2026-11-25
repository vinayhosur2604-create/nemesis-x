"""Detection + multi-object tracking (YOLO + ByteTrack). Only real model output is stored."""
import time
import cv2
from . import db
from .config import CONF, MODEL, STRIDE

LIVE = {"video_id": None, "fps": None}

def device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"

def _flush(c, rows):
    c.executemany("INSERT INTO detections VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    rows.clear()

def run(vid):
    LIVE.update(video_id=vid, fps=None)
    c = db.conn()
    try:
        path = c.execute("SELECT path FROM videos WHERE id=?", (vid,)).fetchone()["path"]
        cap = cv2.VideoCapture(path)
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not fps or n <= 0:
            raise RuntimeError("Could not read video metadata (unsupported codec?)")
        c.execute("DELETE FROM detections WHERE video_id=?", (vid,))
        c.execute("UPDATE videos SET status='processing',stage='loading model',progress=0,error=NULL,"
                  "fps=?,frames=?,duration=?,width=?,height=? WHERE id=?", (fps, n, n / fps, w, h, vid))
        c.commit()
        from ultralytics import YOLO
        model = YOLO(MODEL)
        c.execute("UPDATE videos SET stage='detection + tracking' WHERE id=?", (vid,)); c.commit()
        rows, t0 = [], time.time()
        stream = model.track(source=path, stream=True, persist=True, tracker="bytetrack.yaml",
                             conf=CONF, vid_stride=STRIDE, device=device(), verbose=False)
        for i, r in enumerate(stream):
            f, b = i * STRIDE, r.boxes
            if b is not None and b.id is not None:   # untracked boxes are skipped, never invented
                for xy, tid, cl, cf in zip(b.xyxy.tolist(), b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist()):
                    rows.append((vid, f, f / fps, tid, model.names[cl], cf, *xy))
            if i % 10 == 0:
                _flush(c, rows)
                LIVE["fps"] = round((i + 1) / max(time.time() - t0, 1e-6), 1)
                c.execute("UPDATE videos SET progress=? WHERE id=?", (min(1.0, f / n), vid)); c.commit()
        _flush(c, rows)
        c.execute("UPDATE videos SET status='done',stage='complete',progress=1 WHERE id=?", (vid,)); c.commit()
    except Exception as e:
        c.execute("UPDATE videos SET status='failed',stage='failed',error=? WHERE id=?", (str(e), vid)); c.commit()
    finally:
        LIVE.update(video_id=None, fps=None); c.close()
