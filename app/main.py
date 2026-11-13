"""PromoFinder web API + static frontend.

Run locally:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from promofinder import __version__
from promofinder.candidates import Config
from promofinder.evaluate import evaluate
from promofinder.exporters import to_csv, to_edl, to_json
from promofinder.learn import LearnedRanker, featurize_payload, fit, training_rows
from promofinder.llm import llm_status, make_reranker_from_env
from promofinder import video as vid
from promofinder.parser import ParseError, iter_texts_from_upload, load_show_from_dir, parse_show
from promofinder.pipeline import analyze

from .storage import Store

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample_show"
SAMPLE_KEY = ROOT / "data" / "sample_answer_key.json"
SAMPLE_TITLE = "Naukrani Nahi, Rajkumari Hoon Main"
MAX_UPLOAD_BYTES = int(os.getenv("PROMOFINDER_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_VIDEO_BYTES = int(os.getenv("PROMOFINDER_MAX_VIDEO_MB", "500")) * 1024 * 1024
VIDEO_DIR = Path(os.getenv("PROMOFINDER_VIDEO_DIR", "/tmp/promofinder_media"))
VIDEO_TTL_SEC = int(os.getenv("PROMOFINDER_VIDEO_TTL_HOURS", "24")) * 3600
MODEL_PATH = os.getenv("PROMOFINDER_MODEL_PATH", str(ROOT / "data" / "model" / "ranker.json"))  # shipped model
MODEL_SAVE_PATH = os.getenv("PROMOFINDER_MODEL_SAVE_PATH", MODEL_PATH)  # where retrained models go (must be writable)
ADMIN_TOKEN = os.getenv("PROMOFINDER_ADMIN_TOKEN", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("promofinder")

app = FastAPI(title="PromoFinder", version=__version__,
              description="Finds 20–60s promo clips (hook + cliffhanger + standalone) in micro-drama screenplays.")
store = Store(os.getenv("PROMOFINDER_DB", str(ROOT / "promofinder.db")))
LLM_CACHE: dict = {}
_sample_show = None
RANKER = LearnedRanker.load(MODEL_SAVE_PATH) or LearnedRanker.load(MODEL_PATH)


def sample_show():
    global _sample_show
    if _sample_show is None:
        _sample_show = load_show_from_dir(str(SAMPLE_DIR), title=SAMPLE_TITLE)
    return _sample_show


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        log.info("%s %s -> %s in %.0fms", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - t0) * 1000)
    return response


@app.exception_handler(ParseError)
async def parse_error_handler(_, exc: ParseError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/config")
def config():
    return {"defaults": Config().__dict__, "llm": llm_status(), "sample": {"title": SAMPLE_TITLE, "has_key": SAMPLE_KEY.exists()},
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024), "max_video_mb": MAX_VIDEO_BYTES // (1024 * 1024),
            "video": vid.ffmpeg_available(), "learned_model": RANKER is not None}


@app.post("/api/analyze")
async def analyze_endpoint(
    files: Optional[List[UploadFile]] = File(None),
    use_sample: bool = Form(False),
    title: Optional[str] = Form(None),
    min_len: float = Form(20.0),
    max_len: float = Form(60.0),
    top_k: int = Form(20),
    allow_stitched: bool = Form(True),
    use_llm: bool = Form(False),
    ranking: str = Form("learned"),
):
    if not (5 <= min_len < max_len <= 180):
        raise HTTPException(422, "lengths must satisfy 5 <= min_len < max_len <= 180")
    if not (1 <= top_k <= 100):
        raise HTTPException(422, "top_k must be between 1 and 100")
    if use_sample:
        show, source = sample_show(), "sample"
    else:
        if not files:
            raise HTTPException(422, "upload a .zip of episodes or one or more .txt episode files, or use the sample show")
        texts, total = [], 0
        for f in files:
            data = await f.read()
            total += len(data)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
            texts.extend(iter_texts_from_upload(os.path.basename(f.filename or "upload.txt"), data))
        show, source = parse_show(texts, title=title or None), "upload"
    if title:
        show.title = title
    cfg = Config(min_len=min_len, max_len=max_len, top_k=top_k, allow_stitched=allow_stitched)
    reranker = None
    if use_llm:
        reranker = make_reranker_from_env(cache=LLM_CACHE)
        if reranker is None:
            raise HTTPException(400, "LLM re-ranking is not configured on this server")
    if ranking not in ("learned", "rules"):
        raise HTTPException(422, "ranking must be 'learned' or 'rules'")
    ranker = RANKER if ranking == "learned" else None
    result = analyze(show, cfg, reranker=reranker, ranker=ranker)
    result["source"] = source
    if ranker is not None:
        result["model"] = {"trained_on": ranker.meta.get("trained_on"), "trained_at": ranker.meta.get("trained_at"),
                           "votes_used": ranker.meta.get("votes", 0)}
    job_id = store.save_job(show.title, source, result)
    result["job_id"] = job_id
    return result


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    res = job["result"]
    res["job_id"] = job_id
    return res


@app.get("/api/jobs/{job_id}/export")
def export(job_id: str, format: str = "csv"):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    res = job["result"]
    stem = "".join(ch if ch.isalnum() else "_" for ch in res["show"]["title"])[:40] or "promos"
    if format == "csv":
        body, mime, ext = to_csv(res), "text/csv", "csv"
    elif format == "json":
        body, mime, ext = to_json(res), "application/json", "json"
    elif format == "edl":
        body, mime, ext = to_edl(res), "text/plain", "edl"
    else:
        raise HTTPException(422, "format must be csv, json or edl")
    return PlainTextResponse(body, media_type=mime,
                             headers={"Content-Disposition": f'attachment; filename="{stem}_promos.{ext}"'})


@app.post("/api/jobs/{job_id}/evaluate")
async def evaluate_job(job_id: str, answer_key: Optional[UploadFile] = File(None)):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if answer_key is not None:
        try:
            key = json.loads((await answer_key.read()).decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPException(422, "answer key must be a JSON file")
    elif job["source"] == "sample" and SAMPLE_KEY.exists():
        key = json.loads(SAMPLE_KEY.read_text(encoding="utf-8"))
    else:
        raise HTTPException(422, "upload an answer key JSON for this show")
    if not isinstance(key, dict) or "promos" not in key:
        raise HTTPException(422, 'answer key needs a "promos" list (see docs/answer_key_format.md)')
    metrics = evaluate(job["result"], key)
    if job["result"].get("ranking") == "learned" and job["source"] == "sample":
        metrics["note"] = ("The learned model was trained on this show's answer key, so these numbers are optimistic. "
                           "Held-out numbers (episodes the model never saw) are below.")
        metrics["held_out"] = (RANKER.meta.get("cv") if RANKER else None)
    if job["result"].get("baseline"):
        b = evaluate(job["result"]["baseline"], key)
        metrics["baseline"] = {"name": job["result"]["baseline"]["name"], "per_episode": b["per_episode"]}
    return metrics


class Feedback(BaseModel):
    job_id: str = Field(..., max_length=40)
    promo_id: str = Field(..., max_length=400)
    vote: int = Field(..., ge=-1, le=1)
    note: Optional[str] = Field(None, max_length=500)


@app.post("/api/feedback")
def feedback(fb: Feedback):
    if fb.vote == 0:
        raise HTTPException(422, "vote must be 1 or -1")
    if not store.get_job(fb.job_id):
        raise HTTPException(404, "job not found")
    fid = store.add_feedback(fb.job_id, fb.promo_id, fb.vote, fb.note)
    return {"ok": True, "id": fid}


@app.get("/api/feedback/summary")
def feedback_summary(job_id: Optional[str] = None):
    return store.feedback_summary(job_id)


# ---------------------------------------------------------------- video
def _job_or_404(job_id: str) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


def _job_media(job_id: str) -> Path:
    if not job_id.isalnum():
        raise HTTPException(400, "bad job id")
    return VIDEO_DIR / job_id


def _sources(job_id: str) -> dict:
    d = _job_media(job_id) / "sources.json"
    if not d.exists():
        return {}
    return {int(k): v for k, v in json.loads(d.read_text()).items()}


def _cleanup_media():
    if not VIDEO_DIR.exists():
        return
    now = time.time()
    for d in VIDEO_DIR.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > VIDEO_TTL_SEC:
                import shutil
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


@app.post("/api/jobs/{job_id}/videos")
async def upload_videos(job_id: str, files: List[UploadFile] = File(...)):
    job = _job_or_404(job_id)
    if not vid.ffmpeg_available():
        raise HTTPException(503, "video tools are not installed on this server")
    _cleanup_media()
    media = _job_media(job_id)
    media.mkdir(parents=True, exist_ok=True)
    known = {e["number"]: e["duration"] for e in job["result"]["show"]["episode_list"]}
    sources, report, total = _sources(job_id), [], 0
    for f in files:
        name = os.path.basename(f.filename or "")
        ext = os.path.splitext(name)[1].lower()
        ep = vid.episode_from_filename(name)
        if ext not in vid.VIDEO_EXTS:
            report.append({"file": name, "status": "skipped", "reason": "not a video file"}); continue
        if ep is None or ep not in known:
            report.append({"file": name, "status": "skipped",
                           "reason": "could not tell which episode this is; name it like episode-05.mp4"}); continue
        dest = media / f"ep{ep}{ext}"
        with open(dest, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    out.close(); dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"videos too large (limit {MAX_VIDEO_BYTES // (1024 * 1024)} MB per upload)")
                out.write(chunk)
        try:
            info = vid.probe(str(dest))
        except vid.VideoError as exc:
            dest.unlink(missing_ok=True)
            report.append({"file": name, "status": "rejected", "reason": str(exc)}); continue
        sources[ep] = {"path": str(dest), "mtime": dest.stat().st_mtime, **info}
        item = {"file": name, "episode": ep, "status": "linked", "duration": round(info["duration"], 1)}
        if known[ep] and abs(info["duration"] - known[ep]) / known[ep] > 0.05:
            item["warning"] = (f"video is {info['duration']:.0f}s but the script is {known[ep]:.0f}s; "
                               "cuts may land early or late")
        report.append(item)
    (media / "sources.json").write_text(json.dumps({str(k): v for k, v in sources.items()}))
    return {"linked": sorted(sources), "files": report}


@app.get("/api/jobs/{job_id}/videos")
def list_videos(job_id: str):
    _job_or_404(job_id)
    return {"linked": sorted(_sources(job_id)), "video": vid.ffmpeg_available()}


def _find_promo(result: dict, promo_id: str) -> Optional[dict]:
    for p in result["promos"]:
        if p["id"] == promo_id:
            return p
    for picks in result["per_episode"].values():
        for p in picks:
            if p["id"] == promo_id:
                return p
    return None


@app.get("/api/jobs/{job_id}/clip")
def clip(job_id: str, promo_id: str, vertical: bool = True, demo: bool = False, offset: float = 0.0):
    job = _job_or_404(job_id)
    promo = _find_promo(job["result"], promo_id)
    if promo is None:
        raise HTTPException(404, "promo not found in this job")
    if not vid.ffmpeg_available():
        raise HTTPException(503, "video tools are not installed on this server")
    if not -30 <= offset <= 30:
        raise HTTPException(422, "offset must be between -30 and 30 seconds")
    sources = {} if demo else _sources(job_id)
    if not demo:
        missing = [s["episode"] for s in promo["segments"] if s["episode"] not in sources]
        if missing:
            raise HTTPException(409, f"upload the video for episode {', '.join(map(str, missing))} first")
    offsets = {s["episode"]: offset for s in promo["segments"]} if offset else {}
    key = vid.cache_key(job_id, promo_id, vertical, demo, sources, offsets)
    out_dir = _job_media(job_id) / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{key}.mp4"
    if not out.exists():
        try:
            vid.render_promo(promo, str(out) + ".part.mp4", sources=sources, vertical=vertical, demo=demo, offsets=offsets)
        except vid.VideoError as exc:
            raise HTTPException(422, str(exc))
        os.replace(str(out) + ".part.mp4", out)
    eps = "-".join(str(s["episode"]) for s in promo["segments"])
    fname = f"promo_rank{promo['rank']}_ep{eps}{'_demo' if demo else ''}{'_9x16' if vertical else ''}.mp4"
    return FileResponse(out, media_type="video/mp4", filename=fname)


# ---------------------------------------------------------------- learned ranker
@app.get("/api/model")
def model_info():
    if RANKER is None:
        return {"available": False}
    return {"available": True, "meta": RANKER.meta, "top_weights": RANKER.top_weights(10)}


@app.post("/api/model/retrain")
def retrain(request: Request):
    """Retrain the ranker on the sample answer key plus every editor vote. Needs PROMOFINDER_ADMIN_TOKEN."""
    global RANKER
    if not ADMIN_TOKEN or request.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(403, "retraining needs the admin token")
    from promofinder.pipeline import score_all
    _, cands, _ = score_all(sample_show(), Config())
    key = json.loads(SAMPLE_KEY.read_text(encoding="utf-8"))
    X, y, names = training_rows(cands, key)
    w = [1.0] * len(X)
    votes = 0
    jobs: dict = {}
    for fb in store.all_feedback():
        if fb["job_id"] not in jobs:
            j = store.get_job(fb["job_id"])
            jobs[fb["job_id"]] = j["result"] if j else None
        res = jobs[fb["job_id"]]
        p = _find_promo(res, fb["promo_id"]) if res else None
        if not p or not p.get("features"):
            continue
        X.append(featurize_payload(p, names)); y.append(1 if fb["vote"] > 0 else 0); w.append(3.0)
        votes += 1
    model = fit(X, y, names, sample_weight=w, meta={
        "trained_on": SAMPLE_TITLE + (f" + {votes} editor votes" if votes else ""), "votes": votes,
        "rows": len(X), "positives": sum(y), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cv": RANKER.meta.get("cv") if RANKER else None})
    model.save(MODEL_SAVE_PATH)
    RANKER = model
    return {"ok": True, "votes_used": votes, "rows": len(X), "top_weights": model.top_weights(8)}


STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
