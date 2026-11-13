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
from promofinder.llm import llm_status, make_reranker_from_env
from promofinder.parser import ParseError, iter_texts_from_upload, load_show_from_dir, parse_show
from promofinder.pipeline import analyze

from .storage import Store

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample_show"
SAMPLE_KEY = ROOT / "data" / "sample_answer_key.json"
SAMPLE_TITLE = "Naukrani Nahi, Rajkumari Hoon Main"
MAX_UPLOAD_BYTES = int(os.getenv("PROMOFINDER_MAX_UPLOAD_MB", "25")) * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("promofinder")

app = FastAPI(title="PromoFinder", version=__version__,
              description="Finds 20–60s promo clips (hook + cliffhanger + standalone) in micro-drama screenplays.")
store = Store(os.getenv("PROMOFINDER_DB", str(ROOT / "promofinder.db")))
LLM_CACHE: dict = {}
_sample_show = None


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
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}


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
    result = analyze(show, cfg, reranker=reranker)
    result["source"] = source
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


STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
