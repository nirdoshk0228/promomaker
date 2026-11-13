"""Generate promo candidates: continuous single clips and stitched multi-moment promos."""
from __future__ import annotations

from dataclasses import dataclass

from .features import EpisodeIndex
from .models import Candidate, Segment
from .scoring import cliff_score, hook_score, intensity_score


@dataclass
class Config:
    min_len: float = 20.0
    max_len: float = 60.0
    allow_stitched: bool = True
    hook_seg: tuple = (3.0, 15.0)        # length range of the opening moment in a stitched promo
    cliff_seg: tuple = (6.0, 30.0)       # length range of the closing moment
    max_episode_gap: int = 6             # stitched promos may join moments up to N episodes apart
    pool_size: int = 40                  # top hook/cliff moments kept for stitching
    top_k: int = 20
    per_episode: int = 3
    max_per_episode: int = 2             # in the overall top list
    overlap_iou: float = 0.3


def _windows(idx: EpisodeIndex, lo: float, hi: float):
    """All (i, j) shot ranges whose span is within [lo, hi] and that cut cleanly at both ends."""
    shots = idx.ep.shots
    n = idx.last_content_shot + 1
    for i in range(n):
        if not idx.shot_ok[i] or not idx.clean_cut_at(shots[i].start):
            continue
        for j in range(i, n):
            if not idx.shot_ok[j]:
                break
            d = shots[j].end - shots[i].start
            if d > hi:
                break
            if d >= lo and idx.clean_cut_at(shots[j].end):
                yield i, j


def _seg(idx: EpisodeIndex, i: int, j: int) -> Segment:
    s = idx.ep.shots
    return Segment(episode=idx.ep.number, start=s[i].start, end=s[j].end, shot_from=i, shot_to=j)


def single_candidates(idxs: dict[int, EpisodeIndex], cfg: Config) -> list[Candidate]:
    out = []
    for idx in idxs.values():
        for i, j in _windows(idx, cfg.min_len, cfg.max_len):
            out.append(Candidate(segments=[_seg(idx, i, j)]))
    return out


def stitched_candidates(idxs: dict[int, EpisodeIndex], cfg: Config) -> list[Candidate]:
    hooks, cliffs = [], []
    for idx in idxs.values():
        for i, j in _windows(idx, *cfg.hook_seg):
            seg = _seg(idx, i, j)
            v = hook_score(idx, seg).value + 0.3 * intensity_score(Candidate([seg]), idxs)
            hooks.append((v, seg))
        for i, j in _windows(idx, *cfg.cliff_seg):
            seg = _seg(idx, i, j)
            cliffs.append((cliff_score(idx, seg).value, seg))
    hooks = _dedupe_pool(hooks, cfg.pool_size)
    cliffs = _dedupe_pool(cliffs, cfg.pool_size)
    out = []
    for _, h in hooks:
        for _, c in cliffs:
            if h.episode == c.episode:
                if c.start - h.end < 5.0:
                    continue
            elif not (0 < c.episode - h.episode <= cfg.max_episode_gap):
                continue
            total = h.duration + c.duration
            if cfg.min_len <= total <= cfg.max_len:
                out.append(Candidate(segments=[h, c]))
    return out


def _dedupe_pool(pool, k):
    """Keep the top-k moments, skipping ones that overlap an already-kept moment by >50%."""
    pool.sort(key=lambda t: -t[0])
    kept = []
    for v, s in pool:
        if all(s.episode != o.episode or _overlap(s, o) / min(s.duration, o.duration) < 0.5 for _, o in kept):
            kept.append((v, s))
        if len(kept) >= k:
            break
    return kept


def _overlap(a: Segment, b: Segment) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))
