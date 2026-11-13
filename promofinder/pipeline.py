"""End-to-end analysis: Show -> ranked promo list (+ per-episode picks)."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable, Optional

from .candidates import Config, single_candidates, stitched_candidates
from .features import EpisodeIndex
from .models import Candidate, Show
from .scoring import WEIGHTS, score_candidate
from .selection import candidate_overlap, select


def fmt_tc(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _block_dict(b):
    return {"kind": b.kind, "speaker": b.speaker, "emotions": b.emotions, "modifiers": b.modifiers,
            "text": b.text, "start": round(b.start, 2), "end": round(b.end, 2), "scene": b.scene_heading}


def candidate_payload(c: Candidate, idxs: dict[int, EpisodeIndex], rank: int) -> dict:
    d = c.to_dict()
    d["rank"] = rank
    d["features"] = {k: round(float(v), 4) for k, v in getattr(c, "features", {}).items()}
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


def score_all(show: Show, cfg: Config):
    """Generate and heuristically score every candidate. Returns (episode indexes, candidates, counts)."""
    idxs = {ep.number: EpisodeIndex(ep) for ep in show.episodes}
    singles = single_candidates(idxs, cfg)
    stitched = stitched_candidates(idxs, cfg) if cfg.allow_stitched else []
    cands = singles + stitched
    for c in cands:
        score_candidate(c, idxs, cfg.min_len, cfg.max_len)
        c.scores["rules"] = c.score
    return idxs, cands, {"single": len(singles), "stitched": len(stitched)}


def analyze(show: Show, cfg: Optional[Config] = None, reranker: Optional[Callable] = None,
            progress: Optional[Callable[[str], None]] = None, ranker=None) -> dict:
    """ranker: optional LearnedRanker; when given, its probability replaces the rule-based score."""
    cfg = cfg or Config()
    t0 = time.perf_counter()
    say = progress or (lambda m: None)
    say("generating and scoring candidates")
    idxs, cands, counts = score_all(show, cfg)
    singles, stitched = counts["single"], counts["stitched"]
    if ranker is not None and cands:
        ranker.apply(cands)
    t_score = time.perf_counter()

    llm_info = None
    final_pool = cands
    if reranker is not None and cands:
        # shortlist, in priority order: each episode's best clip, the show-level top list, then each
        # episode's runners-up — capped so one run fits inside free-tier LLM rate limits
        cap = getattr(reranker, "max_clips", 100)
        per_ep = {ep.number: select(cands, k=cfg.per_episode, max_overlap=cfg.overlap_iou, episode=ep.number,
                                    only_episode=True) for ep in show.episodes}
        ordered = [p[0] for p in per_ep.values() if p]
        ordered += select(cands, k=cfg.top_k, max_overlap=cfg.overlap_iou, max_per_episode=cfg.max_per_episode)
        for rank in range(1, cfg.per_episode):
            ordered += [p[rank] for p in per_ep.values() if len(p) > rank]
        pool = {}
        for c in ordered:
            if len(pool) >= cap:
                break
            pool.setdefault(id(c), c)
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
        if len(picks) < cfg.per_episode and final_pool is not cands:
            # top up with clips the LLM did not judge (they keep their own score and carry no AI verdict)
            for c in sorted((c for c in cands if not c.llm and c.episodes == [ep.number]), key=lambda c: -c.score):
                if len(picks) >= cfg.per_episode:
                    break
                if all(candidate_overlap(c, p) <= cfg.overlap_iou for p in picks):
                    picks.append(c)
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
        "ranking": "learned" if ranker is not None else "rules",
        "stats": {"candidates": len(cands), "single": singles, "stitched": stitched,
                  "score_seconds": round(t_score - t0, 3), "total_seconds": round(elapsed, 3)},
        "llm": llm_info,
        "promos": [candidate_payload(c, idxs, r) for r, c in enumerate(top, 1)],
        "per_episode": per_episode,
        "baseline": {"name": "Last ~35 s of each episode", "per_episode": baseline["per_episode"],
                     "promos": baseline["promos"]},
    }
