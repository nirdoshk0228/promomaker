"""Explainable heuristic scoring of promo candidates.

score = 0.30 hook + 0.30 cliffhanger + 0.20 standalone + 0.15 intensity + 0.05 duration fit
Each component is in [0, 1] and comes with the evidence (which line triggered it) so an editor can see *why*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .features import CONJUNCTION_START, EpisodeIndex
from .models import Block, Candidate, Segment

WEIGHTS = {"hook": 0.30, "cliff": 0.30, "standalone": 0.20, "intensity": 0.15, "duration": 0.05}
HOOK_WINDOW = 3.0
CLIFF_WINDOW = 4.5
STITCH_PENALTY = 0.08


@dataclass
class Evidence:
    value: float
    block: Block | None
    note: str


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _squash(x: float) -> float:
    """Map an unbounded evidence total to [0, 1) without hard saturation."""
    return 1.0 - math.exp(-1.3 * max(x, 0.0))


def _quote(b: Block | None, n: int = 90) -> str:
    if b is None:
        return ""
    t = b.text if len(b.text) <= n else b.text[: n - 1].rstrip() + "…"
    return f'{b.speaker}: "{t}"' if b.kind == "DIALOGUE" else t


def hook_score(idx: EpisodeIndex, seg: Segment) -> Evidence:
    blocks = idx.blocks_between(seg.start, min(seg.start + HOOK_WINDOW, seg.end))
    best, best_b, note = 0.0, None, ""
    for b in blocks:
        f = idx.bf[b.index]
        v = f.intensity + 0.25 * min(f.physical, 1) + (0.15 if f.energy > 0.5 and (f.exclaim or f.question) else 0)
        v *= 1.0 - 0.13 * max(0.0, b.start - seg.start)
        if v > best:
            best, best_b = v, b
    first = blocks[0] if blocks else None
    if first is not None:
        ff = idx.bf[first.index]
        if ff.establishing:
            best -= 0.25; note = "opens on an establishing shot"
        elif first.kind == "DIALOGUE" and ff.energy < 0:
            best -= 0.1; note = "opens on calm dialogue"
    if best_b is not None and not note:
        f = idx.bf[best_b.index]
        kind = "physical shock" if f.physical else "reveal" if f.reveal else "threat" if f.threat else \
            "charged line" if f.energy > 0 else "moment"
        note = f"{kind} at +{best_b.start - seg.start:.1f}s"
    return Evidence(_squash(best), best_b, note)


def cliff_score(idx: EpisodeIndex, seg: Segment) -> Evidence:
    t0 = max(seg.start, seg.end - CLIFF_WINDOW)
    blocks = idx.blocks_between(t0, seg.end)
    best, best_b, resolution = 0.0, None, 0
    for b in blocks:
        f = idx.bf[b.index]
        v = (0.5 * f.cliff_cue + 0.35 * f.question + 0.3 * f.trailing + 0.4 * f.threat + 0.35 * f.reveal
             + 0.3 * max(f.energy, 0) + 0.3 * min(f.physical, 1))
        v *= 0.6 + 0.4 * (b.start - t0) / max(seg.end - t0, 0.1)
        resolution += f.resolution
        if v > best:
            best, best_b = v, b
    notes = []
    if resolution:
        best -= 0.35 * min(resolution, 2); notes.append("ending feels resolved")
    if seg.shot_to >= idx.last_content_shot:
        best += 0.2; notes.append("cuts where the writers put the episode cliffhanger")
    if best_b is not None:
        f = idx.bf[best_b.index]
        kind = "question left hanging" if f.question else "threat" if f.threat else "reveal" if f.reveal else \
            "suspense cue" if f.cliff_cue else "charged beat"
        notes.insert(0, kind)
    return Evidence(_squash(best), best_b, "; ".join(notes))


def standalone_score(cand: Candidate, idxs: dict[int, EpisodeIndex]) -> tuple[float, list[str], list[str]]:
    v, good, bad = 0.75, [], []
    speakers, dialogues, total_expo, dream, flash, conflict, comedy = set(), [], 0, False, 0, False, False
    blocks_all = []
    for seg in cand.segments:
        idx = idxs[seg.episode]
        bl = idx.blocks_between(seg.start, seg.end)
        blocks_all.append(bl)
        for b in bl:
            f = idx.bf[b.index]
            if b.kind == "DIALOGUE":
                speakers.add(b.speaker); dialogues.append((b, f))
            total_expo += f.exposition
            dream |= f.dream
            flash += f.flashback
            conflict |= bool(f.physical or f.threat)
            comedy |= f.comedy
    first_seg = cand.segments[0]
    idx0 = idxs[first_seg.episode]
    if dialogues:
        b0, f0 = dialogues[0]
        if CONJUNCTION_START.match(b0.text) or f0.backref:
            v -= 0.15; bad.append("first line leans on earlier context")
    prev = idx0.previous_dialogue(first_seg.start)
    if prev is not None and first_seg.start - prev.end < 1.2:
        v -= 0.25; bad.append("starts mid-conversation")
    backrefs = sum(f.backref for _, f in dialogues)
    if backrefs:
        v -= min(0.2, 0.06 * backrefs); bad.append("refers back to earlier events")
    if dream:
        v -= 0.35; bad.append("contains a dream/sleep beat")
    nblocks = sum(len(b) for b in blocks_all) or 1
    if flash / nblocks > 0.4:
        v -= 0.15; bad.append("mostly flashback/voice-over")
    n = len(speakers)
    if n == 0:
        v -= 0.2; bad.append("no dialogue")
    elif n == 1:
        v -= 0.05
    elif n <= 4:
        v += 0.1; good.append(f"{n} speaking characters — easy to follow")
    else:
        v -= 0.1; bad.append(f"{n} speakers — crowded for a cold viewer")
    if conflict:
        v += 0.1; good.append("clear conflict on screen")
    if total_expo / max(cand.duration, 1) * 10 > 0.8:
        v -= 0.15; bad.append("exposition-heavy (money, legal or logistics talk)")
    if comedy:
        v -= 0.15; bad.append("comic side-beat")
    if len(cand.segments) > 1:
        names = [set(b.speaker for b in bl if b.kind == "DIALOGUE") for bl in blocks_all]
        shared = set.intersection(*names) if all(names) else set()
        texts = [" ".join(b.text for b in bl).lower() for bl in blocks_all]
        mentioned = any(sp.split()[0].lower() in texts[-1] for sp in names[0]) if names[0] else False
        if shared or mentioned:
            v += 0.05; good.append("stitched moments share characters")
        else:
            v -= 0.2; bad.append("stitched moments feel unrelated")
        v -= 0.05 * (len(cand.segments) - 1)
    return _clip(v), good, bad


def intensity_score(cand: Candidate, idxs: dict[int, EpisodeIndex]) -> float:
    vals = []
    for seg in cand.segments:
        idx = idxs[seg.episode]
        vals += [idx.bf[b.index].intensity for b in idx.blocks_between(seg.start, seg.end)]
    if not vals:
        return 0.0
    vals.sort(reverse=True)
    top = vals[: max(1, len(vals) // 2)]
    return _clip(0.6 * sum(top) / len(top) + 0.4 * vals[0])


def duration_score(d: float, lo: float = 20, hi: float = 60) -> float:
    if d < lo or d > hi:
        return 0.0
    if 25 <= d <= 45:
        return 1.0
    if d < 25:
        return 0.7 + 0.3 * (d - lo) / 5
    return 1.0 - 0.3 * (d - 45) / max(hi - 45, 1)


def score_candidate(cand: Candidate, idxs: dict[int, EpisodeIndex], min_len=20, max_len=60) -> Candidate:
    first, last = cand.segments[0], cand.segments[-1]
    hook = hook_score(idxs[first.episode], first)
    cliff = cliff_score(idxs[last.episode], last)
    stand, good, bad = standalone_score(cand, idxs)
    inten = intensity_score(cand, idxs)
    dur = duration_score(cand.duration, min_len, max_len)
    cand.scores = {"hook": hook.value, "cliff": cliff.value, "standalone": stand, "intensity": inten, "duration": dur}
    cand.score = sum(WEIGHTS[k] * cand.scores[k] for k in WEIGHTS)
    cand.score -= STITCH_PENALTY * (len(cand.segments) - 1)   # every extra cut costs editor time and coherence
    cand.reasons = []
    if hook.block is not None and not hook.note.startswith("opens on"):
        cand.reasons.append(f"Hook: {hook.note}")
    if cliff.block is not None and cliff.note:
        cand.reasons.append(f"Ending: {cliff.note}")
    cand.reasons += good
    cand.warnings = bad + ([hook.note] if hook.note.startswith("opens on") else [])
    cand.hook_block, cand.cliff_block = hook.block, cliff.block  # used by the API for previews
    cand.features = raw_features(cand, idxs)
    return cand


def raw_features(cand: Candidate, idxs: dict[int, EpisodeIndex]) -> dict:
    """Low-level signals behind the scores. The learned ranker uses these together with the component scores."""
    first, last = cand.segments[0], cand.segments[-1]
    i0, i1 = idxs[first.episode], idxs[last.episode]
    hook_bl = i0.blocks_between(first.start, min(first.start + HOOK_WINDOW, first.end))
    cliff_bl = i1.blocks_between(max(last.start, last.end - CLIFF_WINDOW), last.end)
    hf = [(b, i0.bf[b.index]) for b in hook_bl]
    cf = [(b, i1.bf[b.index]) for b in cliff_bl]
    dlg_h = [f for b, f in hf if b.kind == "DIALOGUE"]
    first_f = hf[0][1] if hf else None
    speakers, expo, dream, comedy, flash, nblocks, backref = set(), 0, 0, 0, 0, 0, 0
    first_dlg = None
    for seg in cand.segments:
        idx = idxs[seg.episode]
        for b in idx.blocks_between(seg.start, seg.end):
            f = idx.bf[b.index]
            nblocks += 1
            if b.kind == "DIALOGUE":
                speakers.add(b.speaker)
                backref += f.backref
                if first_dlg is None:
                    first_dlg = (b, f)
            expo += f.exposition; dream += f.dream; comedy += f.comedy; flash += f.flashback
    prev = i0.previous_dialogue(first.start)
    return {
        "hook_visual": max((f.physical for b, f in hf if b.kind != "DIALOGUE"), default=0),
        "hook_energy": max((f.energy for f in dlg_h), default=0.0),
        "hook_reveal": max((f.reveal for _, f in hf), default=0),
        "hook_threat": max((f.threat for _, f in hf), default=0),
        "hook_question": float(any(f.question for f in dlg_h)),
        "hook_peak": max((f.intensity for _, f in hf), default=0.0),
        "opens_establishing": float(bool(first_f and first_f.establishing)),
        "starts_mid_conversation": float(prev is not None and first.start - prev.end < 1.2),
        "first_line_needs_context": float(bool(first_dlg and (CONJUNCTION_START.match(first_dlg[0].text) or first_dlg[1].backref))),
        "cliff_cue": sum(f.cliff_cue for _, f in cf),
        "cliff_question": float(any(f.question or f.trailing for _, f in cf)),
        "cliff_threat": max((f.threat for _, f in cf), default=0),
        "cliff_reveal": max((f.reveal for _, f in cf), default=0),
        "cliff_resolution": sum(f.resolution for _, f in cf),
        "ends_at_episode_end": float(last.shot_to >= i1.last_content_shot),
        "speakers": min(len(speakers), 6),
        "backrefs": backref,
        "exposition_per_10s": expo / max(cand.duration, 1) * 10,
        "dream": float(dream > 0),
        "comedy": float(comedy > 0),
        "flashback_share": flash / max(nblocks, 1),
        "segments": len(cand.segments),
        "cross_episode": float(len(cand.episodes) > 1),
        "duration": cand.duration,
    }
