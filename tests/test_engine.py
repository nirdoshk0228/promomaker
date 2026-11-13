import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from promofinder.candidates import Config, single_candidates
from promofinder.evaluate import evaluate, iou
from promofinder.exporters import _tc_frames, to_csv, to_edl
from promofinder.features import EpisodeIndex
from promofinder.llm import LLMClient, Reranker, _extract_json_array
from promofinder.models import Candidate, Segment
from promofinder.parser import ParseError, iter_texts_from_upload, load_show_from_dir, parse_episode_text, parse_show
from promofinder.pipeline import analyze
from promofinder.scoring import cliff_score, hook_score
from promofinder.selection import candidate_overlap

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_show"
KEY = json.loads((ROOT / "data" / "sample_answer_key.json").read_text(encoding="utf-8"))

MINI = """EPISODE NUMBER: Episode 9
EPISODE LOGLINE: Test.
CONTINUITY NOTES: None.
--- SCRIPT START ---
SCENE 01
Scene heading: INT. HALL – NIGHT
**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Aerial wide shot of an empty palace at night.
**CAMERA:**
Start: (00:03.00) | End: (00:05.00)
Close-up as Rani's hand swings across Meera's face.
**SOUND FX:**
Start: (00:03.50) | End: (00:04.30)
A sharp SLAP.
**MEERA (Shocked, Trembling):**
Start: (00:05.00) | End: (00:07.00)
"Aapne mujhe kyun maara?"
**CAMERA:**
Start: (00:07.00) | End: (00:30.00)
Medium shot of the two women staring at each other.
**RANI (V.O., Cold):**
Start: (00:08.00) | End: (00:10.00)
"Kyunki tu is ghar ki beti nahi hai."
**CAMERA:**
Start: (00:30.00) | End: (00:33.00)
Extreme close-up on Meera, frozen.
**SOUND FX:**
Start: (00:30.50) | End: (00:32.50)
Ominous drone.
**CAMERA:**
Start: (00:33.00) | End: (00:34.00)
Cut to black.
--- SCRIPT END ---"""


@pytest.fixture(scope="module")
def show():
    return load_show_from_dir(str(SAMPLE))


@pytest.fixture(scope="module")
def result(show):
    return analyze(show)


# ---------- parser ----------
def test_parse_sample(show):
    assert len(show.episodes) == 35
    assert [e.number for e in show.episodes] == list(range(1, 36))
    assert all(e.shots for e in show.episodes)
    assert not show.warnings


def test_label_split_and_timecodes():
    ep, _ = parse_episode_text(MINI, "mini.txt")
    assert ep.number == 9
    v = [b for b in ep.blocks if b.speaker == "RANI"][0]
    assert v.modifiers == ["V.O."] and v.emotions == ["Cold"]
    assert v.text == "Kyunki tu is ghar ki beti nahi hai."
    assert (v.start, v.end) == (8.0, 10.0)
    assert len(ep.shots) == 5


def test_zip_upload_in_memory():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Show A/episode-009.txt", MINI)
        z.writestr("Show A/notes.md", "ignore me")
        z.writestr("__MACOSX/._episode-009.txt", "junk")
    files = list(iter_texts_from_upload("a.zip", buf.getvalue()))
    assert [f for f, _ in files] == ["Show A/episode-009.txt"]
    s = parse_show(files)
    assert s.title == "Show A" and len(s.episodes) == 1


def test_non_screenplay_rejected():
    with pytest.raises(ParseError):
        parse_show([("x.txt", "just some notes, no timecodes")])


# ---------- scoring ----------
def test_slap_hook_beats_establishing_shot():
    ep, _ = parse_episode_text(MINI, "mini.txt")
    idx = EpisodeIndex(ep)
    est = Segment(9, 0.0, 30.0, 0, 2)
    slap = Segment(9, 3.0, 33.0, 1, 3)
    assert hook_score(idx, slap).value > hook_score(idx, est).value + 0.2


def test_resolved_ending_scores_lower():
    ep, _ = parse_episode_text(MINI.replace("Ominous drone.", "Everyone laughs and hugs, relieved."), "m.txt")
    ep2, _ = parse_episode_text(MINI, "m.txt")
    seg = Segment(9, 3.0, 33.0, 1, 3)
    assert cliff_score(EpisodeIndex(ep), seg).value < cliff_score(EpisodeIndex(ep2), seg).value


def test_candidates_respect_length_and_cut_to_black(show):
    idxs = {e.number: EpisodeIndex(e) for e in show.episodes}
    cands = single_candidates(idxs, Config())
    assert cands
    for c in cands[:2000]:
        assert 20 <= c.duration <= 60
        idx = idxs[c.segments[0].episode]
        assert idx.shot_ok[c.segments[0].shot_to]


# ---------- pipeline ----------
def test_top_list_is_diverse(result):
    promos = result["promos"]
    assert len(promos) == 20
    cands = [Candidate([Segment(s["episode"], s["start"], s["end"], 0, 0) for s in p["segments"]]) for p in promos]
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            assert candidate_overlap(a, b) <= 0.3


def test_per_episode_picks_stay_in_episode(result):
    for ep, picks in result["per_episode"].items():
        for p in picks:
            assert {s["episode"] for s in p["segments"]} == {int(ep)}


def test_deterministic(show, result):
    again = analyze(show)
    assert [p["id"] for p in again["promos"]] == [p["id"] for p in result["promos"]]


# ---------- evaluation ----------
def test_iou_basics():
    assert iou([(1, 0, 10)], [(1, 0, 10)]) == 1.0
    assert iou([(1, 0, 10)], [(2, 0, 10)]) == 0.0
    assert abs(iou([(1, 0, 10)], [(1, 5, 15)]) - 5 / 15) < 1e-9


def test_quality_floor_on_sample(result):
    """Regression guard: these numbers are reported in the README."""
    m = evaluate(result, KEY)["per_episode"]
    assert m["hit@3_strict"] >= 0.75
    assert m["hit@1_loose"] >= 0.85
    assert m["top1_trap_rate"] <= 0.1


# ---------- LLM re-ranker ----------
def _fake_transport(verdict="use", fail=False):
    def handler(req):
        if fail:
            return httpx.Response(500, json={"error": "boom"})
        body = json.loads(req.content)
        user = body["messages"][-1]["content"]
        import re
        ids = re.findall(r"### Clip id: (\S+)", user)
        out = [{"id": i, "hook": 9, "cliffhanger": 8, "standalone": 7, "verdict": verdict, "reason": "stub"} for i in ids]
        return httpx.Response(200, json={"choices": [{"message": {"content": "```json\n" + json.dumps(out) + "\n```"}}]})
    return httpx.MockTransport(handler)


def test_json_extraction_handles_fences():
    assert _extract_json_array('Sure!\n```json\n[{"id": "a"}]\n```') == [{"id": "a"}]


def test_reranker_blends_scores(show):
    rr = Reranker(LLMClient("groq", "k", transport=_fake_transport()))
    res = analyze(show, reranker=rr)
    assert res["llm"]["judged"] > 0 and not res["llm"]["errors"]
    assert all(p.get("llm") for p in res["promos"])
    assert res["promos"][0]["reasons"][0].startswith("Editor AI:")


def test_reranker_failure_falls_back(show, monkeypatch):
    monkeypatch.setattr("promofinder.llm.time.sleep", lambda s: None)
    rr = Reranker(LLMClient("groq", "k", transport=_fake_transport(fail=True)))
    res = analyze(show, reranker=rr)
    assert res["llm"]["judged"] == 0 and res["llm"]["errors"]
    assert len(res["promos"]) == 20 and not res["promos"][0].get("llm")


# ---------- exporters ----------
def test_edl_and_csv(result):
    assert _tc_frames(61.6) == "00:01:01:15"
    edl = to_edl(result)
    assert edl.startswith("TITLE:") and "EP0" in edl
    csv_text = to_csv(result)
    assert csv_text.splitlines()[0].startswith("rank,promo_id")
