import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["PROMOFINDER_DB"] = str(tmp_path_factory.mktemp("db") / "t.db")
    os.environ.pop("PROMOFINDER_LLM_PROVIDER", None)
    import importlib
    import app.main as m
    importlib.reload(m)
    return TestClient(m.app)


def test_health_and_config(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["llm"]["available"] is False and cfg["defaults"]["min_len"] == 20


def test_sample_flow(client):
    r = client.post("/api/analyze", data={"use_sample": "true", "top_k": "10"})
    assert r.status_code == 200
    j = r.json()
    assert len(j["promos"]) == 10 and j["show"]["episodes"] == 35
    jid = j["job_id"]
    assert client.get(f"/api/jobs/{jid}").json()["job_id"] == jid
    ev = client.post(f"/api/jobs/{jid}/evaluate").json()
    assert ev["per_episode"]["episodes_evaluated"] == 35 and "baseline" in ev
    for fmt in ("csv", "json", "edl"):
        e = client.get(f"/api/jobs/{jid}/export?format={fmt}")
        assert e.status_code == 200 and "attachment" in e.headers["content-disposition"]
    fb = client.post("/api/feedback", json={"job_id": jid, "promo_id": j["promos"][0]["id"], "vote": 1})
    assert fb.json()["ok"]
    assert client.get(f"/api/feedback/summary?job_id={jid}").json()["total"]["votes"] >= 1


def test_upload_zip(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in sorted((ROOT / "data" / "sample_show").glob("episode-00[1-3].txt")):
            z.write(p, f"Mini Show/{p.name}")
    r = client.post("/api/analyze", files=[("files", ("mini.zip", buf.getvalue(), "application/zip"))])
    assert r.status_code == 200
    assert r.json()["show"]["title"] == "Mini Show" and r.json()["show"]["episodes"] == 3


def test_bad_inputs(client):
    assert client.post("/api/analyze", data={}).status_code == 422
    assert client.post("/api/analyze", files=[("files", ("a.txt", b"no script here", "text/plain"))]).status_code == 422
    assert client.post("/api/analyze", data={"use_sample": "true", "min_len": "70", "max_len": "60"}).status_code == 422
    assert client.post("/api/analyze", data={"use_sample": "true", "use_llm": "true"}).status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/feedback", json={"job_id": "nope", "promo_id": "x", "vote": 1}).status_code == 404


def test_v2_endpoints(client, tmp_path):
    import shutil, subprocess
    r = client.post("/api/analyze", data={"use_sample": "true", "top_k": "5"}).json()
    jid, p = r["job_id"], r["promos"][0]
    assert r["ranking"] == "learned"
    assert client.get("/api/model").json()["available"] is True
    assert client.post("/api/model/retrain").status_code == 403
    rules = client.post("/api/analyze", data={"use_sample": "true", "ranking": "rules"}).json()
    assert rules["ranking"] == "rules"
    if not shutil.which("ffmpeg"):
        return
    assert client.get(f"/api/jobs/{jid}/clip", params={"promo_id": p["id"]}).status_code == 409  # no video yet
    demo = client.get(f"/api/jobs/{jid}/clip", params={"promo_id": p["id"], "demo": "true"})
    assert demo.status_code == 200 and demo.headers["content-type"] == "video/mp4" and len(demo.content) > 10000
    ep = p["segments"][0]["episode"]
    src = tmp_path / f"episode-{ep:02d}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=25:duration=150", "-c:v", "libx264", "-preset", "ultrafast", "-y", str(src)], check=True)
    up = client.post(f"/api/jobs/{jid}/videos", files=[("files", (src.name, src.read_bytes(), "video/mp4")),
                                                       ("files", ("notes.txt", b"x", "text/plain"))]).json()
    assert ep in up["linked"] and any(f["status"] == "skipped" for f in up["files"])
    if all(s["episode"] == ep for s in p["segments"]):
        real = client.get(f"/api/jobs/{jid}/clip", params={"promo_id": p["id"], "vertical": "false"})
        assert real.status_code == 200 and len(real.content) > 10000
