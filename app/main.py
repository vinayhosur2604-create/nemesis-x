import re, shutil, threading, uuid
from pathlib import Path
import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import db, pipeline, system
from .config import ALLOWED, ROOT, UPLOADS

app = FastAPI(title="NEMESIS-X")
db.init()
_lock = threading.Lock()   # one analysis at a time (GPU friendly)

def video(c, vid):
    r = c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
    if not r: raise HTTPException(404, "Unknown video")
    return r

@app.get("/api/health")
def health():
    return {"status": "ok", "device": "cuda" if pipeline.device() == 0 else "cpu"}

@app.get("/api/system")
def sys_metrics(): return system.snapshot()

@app.post("/api/videos")
def upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED: raise HTTPException(400, f"Unsupported type {ext!r}; use {sorted(ALLOWED)}")
    dest = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)
    c = db.conn(); cur = c.execute("INSERT INTO videos(name,path) VALUES(?,?)", (file.filename, str(dest))); c.commit()
    return {"id": cur.lastrowid, "name": file.filename}

@app.get("/api/videos")
def list_videos():
    c = db.conn(); return [dict(r) for r in c.execute("SELECT id,name,status,stage,progress,duration FROM videos ORDER BY id DESC")]

@app.get("/api/videos/{vid}")
def status(vid: int): return {k: v for k, v in dict(video(db.conn(), vid)).items() if k != "path"}

@app.get("/api/videos/{vid}/file")
def file(vid: int): return FileResponse(video(db.conn(), vid)["path"])

@app.post("/api/videos/{vid}/analyze")
def analyze(vid: int):
    video(db.conn(), vid)
    def job():
        with _lock: pipeline.run(vid)
    threading.Thread(target=job, daemon=True).start()
    return {"queued": vid}

@app.get("/api/videos/{vid}/tracks")
def tracks(vid: int):
    c = db.conn(); video(c, vid)
    q = """SELECT track_id id, cls, COUNT(*) n, MIN(t) first_t, MAX(t) last_t, AVG(conf) mean_conf
           FROM detections WHERE video_id=? GROUP BY track_id, cls ORDER BY track_id"""
    return [dict(r) for r in c.execute(q, (vid,))]

@app.get("/api/videos/{vid}/tracks/{tid}")
def track(vid: int, tid: int):
    c = db.conn(); video(c, vid)
    rows = c.execute("SELECT frame,t,cls,conf,x1,y1,x2,y2 FROM detections WHERE video_id=? AND track_id=? ORDER BY frame", (vid, tid)).fetchall()
    if not rows: raise HTTPException(404, "No such track")
    return {"id": tid, "coordinate_space": "image-plane pixels (not metric 3D)", "points": [dict(r) for r in rows]}

@app.get("/api/videos/{vid}/frame")
def frame(vid: int, t: float):
    """Evidence frame extracted from the original file at time t (seconds)."""
    cap = cv2.VideoCapture(video(db.conn(), vid)["path"])
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, img = cap.read(); cap.release()
    if not ok: raise HTTPException(404, "Frame not available")
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")

class Q(BaseModel):
    question: str
    video_id: int | None = None

@app.post("/api/chat")
def chat(q: Q):
    """Answers come only from stored tracking observations; otherwise INSUFFICIENT EVIDENCE."""
    c = db.conn()
    r = (video(c, q.video_id) if q.video_id else c.execute("SELECT * FROM videos WHERE status='done' ORDER BY id DESC").fetchone())
    if not r or r["status"] != "done":
        return {"answer": "INSUFFICIENT EVIDENCE: no processed video is available.", "evidence": []}
    vid, s = r["id"], q.question.lower()
    m = re.search(r"(?:object|track|id)\s*#?(\d+)", s)
    if m:
        tid = int(m.group(1))
        x = c.execute("SELECT cls,COUNT(*) n,MIN(t) a,MAX(t) b FROM detections WHERE video_id=? AND track_id=?", (vid, tid)).fetchone()
        if not x["n"]: return {"answer": f"INSUFFICIENT EVIDENCE: track {tid} was not observed.", "evidence": []}
        return {"answer": f"Track {tid} ({x['cls']}) was observed in {x['n']} frames from {x['a']:.2f}s to {x['b']:.2f}s. Movement history: /api/videos/{vid}/tracks/{tid}.",
                "evidence": [{"video_id": vid, "start": x["a"], "end": x["b"], "track": tid}]}
    if re.search(r"what happened|summar|which objects|objects", s):
        rows = c.execute("SELECT cls,COUNT(DISTINCT track_id) n FROM detections WHERE video_id=? GROUP BY cls ORDER BY n DESC", (vid,)).fetchall()
        if not rows: return {"answer": "No tracked objects were detected in this video.", "evidence": []}
        return {"answer": "Tracked objects: " + ", ".join(f"{x['n']} {x['cls']}" for x in rows) +
                ". Event reasoning is not implemented yet, so interactions and unusual activity cannot be determined.", "evidence": []}
    return {"answer": "INSUFFICIENT EVIDENCE: this question cannot be answered from the stored tracking observations.", "evidence": []}

app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="ui")
