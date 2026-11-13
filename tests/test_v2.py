"""Tests for v2: learned ranker and MP4 export."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from promofinder.candidates import Config
from promofinder.learn import LearnedRanker, fit, featurize_payload, label, training_rows
from promofinder.evaluate import load_key
from promofinder.parser import load_show_from_dir
from promofinder.pipeline import analyze, score_all
from promofinder import video as vid

ROOT = Path(__file__).resolve().parent.parent
KEY = json.loads((ROOT / "data" / "sample_answer_key.json").read_text(encoding="utf-8"))
MODEL = ROOT / "data" / "model" / "ranker.json"
needs_ffmpeg = pytest.mark.skipif(not vid.ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def show():
    return load_show_from_dir(str(ROOT / "data" / "sample_show"))


@pytest.fixture(scope="module")
def scored(show):
    return score_all(show, Config())


# ---------- learning to rank ----------
def test_every_candidate_has_features(scored):
    _, cands, _ = scored
    keys = set(cands[0].features)
    assert {"hook_visual", "ends_at_episode_end", "starts_mid_conversation"} <= keys
    assert all(set(c.features) == keys for c in cands[:500])


def test_label_requires_start_and_end_alignment(scored):
    _, cands, _ = scored
    gold, _ = load_key(KEY)
    pos = [c for c in cands if label(c, gold)[0]]
    assert 30 <= len(pos) <= 200          # sharp labels: only near-exact cuts count
    g = {(s[0], round(s[1], 1)) for gg in gold for s in gg["segs"][:1]}
    assert all(any(abs(c.segments[0].start - st) <= 2.0 for ep, st in g if ep == c.segments[0].episode) for c in pos)


def test_fit_and_roundtrip(scored, tmp_path):
    _, cands, _ = scored
    X, y, names = training_rows(cands, KEY, episodes=set(range(1, 18)))
    m = fit(X, y, names)
    p = tmp_path / "m.json"
    m.save(str(p))
    m2 = LearnedRanker.load(str(p))
    assert abs(m.prob(cands[0]) - m2.prob(cands[0])) < 1e-12
    w = dict((d["feature"], d["weight"]) for d in m.top_weights(len(names)))
    assert w["score_hook"] > 0 and w["score_cliff"] > 0     # learned weights agree with editing sense


def test_shipped_model_and_blend(show):
    m = LearnedRanker.load(str(MODEL))
    assert m is not None and m.meta["cv"]["blend"]["hit@3_strict"] >= m.meta["cv"]["rules"]["hit@3_strict"]
    res = analyze(show, ranker=m)
    assert res["ranking"] == "learned"
    p = res["promos"][0]
    assert 0 <= p["scores"]["learned"] <= 1 and p["features"]
    assert abs(featurize_payload(p, m.names)[0] - p["scores"]["hook"]) < 1e-3


# ---------- video ----------
def test_episode_from_filename():
    assert vid.episode_from_filename("episode-05.mp4") == 5
    assert vid.episode_from_filename("Naukrani_EP12_final.mov") == 12
    assert vid.episode_from_filename("3.mp4") == 3
    assert vid.episode_from_filename("trailer.mp4") is None


@needs_ffmpeg
def test_demo_and_real_clip(show, tmp_path):
    res = analyze(show)
    p = next(x for x in res["promos"] if x["type"] == "single")
    out = tmp_path / "demo.mp4"
    vid.render_promo(p, str(out), demo=True, vertical=True)
    info = vid.probe(str(out))
    assert (info["width"], info["height"]) == vid.VERTICAL and abs(info["duration"] - p["duration"]) < 0.3
    ep = p["segments"][0]["episode"]
    src = tmp_path / f"episode-{ep}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=640x360:rate=25:duration={int(p['segments'][0]['end']) + 2}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-y", str(src)], check=True)
    real = tmp_path / "real.mp4"
    vid.render_promo(p, str(real), sources={ep: {"path": str(src), **vid.probe(str(src))}}, vertical=False)
    ri = vid.probe(str(real))
    assert (ri["width"], ri["height"]) == vid.LANDSCAPE and ri["has_audio"]
    assert abs(ri["duration"] - p["duration"]) < 0.3


@needs_ffmpeg
def test_stitched_demo_clip(show, tmp_path):
    res = analyze(show)
    p = next((x for x in res["promos"] if x["type"] == "stitched"), None)
    if p is None:
        pytest.skip("no stitched promo in top list")
    out = tmp_path / "st.mp4"
    vid.render_promo(p, str(out), demo=True, vertical=False)
    assert abs(vid.probe(str(out))["duration"] - p["duration"]) < 0.5
