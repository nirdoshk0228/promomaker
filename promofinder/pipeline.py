"""End-to-end analysis: Show -> ranked promo list (+ per-episode picks)."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable, Optional

from .candidates import Config, single_candidates, stitched_candidates
from .features import EpisodeIndex
from .models import Candidate, Show
from .scoring import WEIGHTS, score_candidate
from .selection import select


def fmt_tc(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _block_dict(b):
    return {"kind": b.kind, "speaker": b.speaker, "emotions": b.emotions, "modifiers": b.modifiers,
            "text": b.text, "start": round(b.start, 2), "end": round(b.end, 2), "scene": b.scene_heading}


def candidate_payload(c: Candidate, idxs: dict[int, EpisodeIndex], rank: int) -> dict:
    d = c.to_dict()
    d["rank"] = rank
    d["id"] = "-".join(f"e{s.episode}_{s.start:.1f}_{s.end:.1f}" for s in c.segments)
    for seg, sd in zip(c.segments, d["segments"]):
        idx = idxs[seg.episode]
        sd["start_tc"], sd["end_tc"] = fmt_tc(seg.start), fmt_tc(seg.end)
        sd["episode_duration"] = round(idx.ep.duration, 2)
        sd["blocks"] = [_block_dict(b) for b in idx.blocks_between(seg.start, seg.end)]
    hb, cb = getattr(c, "hook_block", None), getattr(c, "cliff_block", None)
    d["opening"] = _block_dict(hb) if hb else None
    d["ending"] = _block_dict(cb) if cb else None
    return d


def analyze(show: Show, cfg: Optional[Config] = None, reranker: Optional[Callable] = None,
            progress: Optional[Callable[[str], None]] = None) -> dict:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    say = progress or (lambda m: None)
    idxs = {ep.number: EpisodeIndex(ep) for ep in show.episodes}
    say("generating candidates")
    singles = single_candidates(idxs, cfg)
    stitched = stitched_candidates(idxs, cfg) if cfg.allow_stitched and len(show.episodes) >= 1 else []
    cands = singles + stitched
    say(f"scoring {len(cands)} candidates")
    for c in cands:
        score_candidate(c, idxs, cfg.min_len, cfg.max_len)
    t_score = time.perf_counter()

    llm_info = None
    final_pool = cands
    if reranker is not None and cands:
        # shortlist = the show-level top 2K plus each episode's top 3 (wider than what we finally show,
        # so the LLM can promote a clip the heuristics under-ranked)
        pool = {id(c): c for c in select(cands, k=max(cfg.top_k * 2, 30), max_overlap=cfg.overlap_iou,
                                         max_per_episode=cfg.max_per_episode + 1)}
        for ep in show.episodes:
            for c in select(cands, k=cfg.per_episode + 1, max_overlap=cfg.overlap_iou, episode=ep.number,
                            only_episode=True):
                pool[id(c)] = c
        shortlist = list(pool.values())
        say(f"asking the LLM to judge {len(shortlist)} shortlisted promos")
        llm_info = reranker(shortlist, idxs)
        judged = [c for c in shortlist if c.llm]
        if judged:  # rank only clips that were judged, so blended and raw scores are never mixed
            final_pool = judged
    top = select(final_pool, k=cfg.top_k, max_overlap=cfg.overlap_iou, max_per_episode=cfg.max_per_episode)
    per_episode = {}
    for ep in show.episodes:
        # an episode's shortlist only uses footage from that episode; cross-episode stitches live in the show list
        picks = select(final_pool, k=cfg.per_episode, max_overlap=cfg.overlap_iou, episode=ep.number,
                       only_episode=True)
        if not picks and final_pool is not cands:
            picks = select(cands, k=cfg.per_episode, max_overlap=cfg.overlap_iou, episode=ep.number,
                           only_episode=True)
        per_episode[ep.number] = [candidate_payload(c, idxs, r) for r, c in enumerate(picks, 1)]
    from .evaluate import baseline_result  # naive "last ~35 s of each episode" rule, kept for comparison
    baseline = baseline_result(show, "episode_end")
    elapsed = time.perf_counter() - t0
    return {
        "show": {
            "title": show.title, "episodes": len(show.episodes), "runtime_sec": round(show.runtime, 1),
            "episode_list": [{"number": e.number, "logline": e.logline, "duration": round(e.duration, 1),
                              "shots": len(e.shots), "blocks": len(e.blocks)} for e in show.episodes],
            "warnings": show.warnings,
        },
        "config": asdict(cfg),
        "weights": WEIGHTS,
        "stats": {"candidates": len(cands), "single": len(singles), "stitched": len(stitched),
                  "score_seconds": round(t_score - t0, 3), "total_seconds": round(elapsed, 3)},
        "llm": llm_info,
        "promos": [candidate_payload(c, idxs, r) for r, c in enumerate(top, 1)],
        "per_episode": per_episode,
        "baseline": {"name": "Last ~35 s of each episode", "per_episode": baseline["per_episode"],
                     "promos": baseline["promos"]},
    }
