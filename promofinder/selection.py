"""Pick a diverse, non-overlapping set of promos from scored candidates."""
from __future__ import annotations

from .models import Candidate


def candidate_overlap(a: Candidate, b: Candidate) -> float:
    """Largest overlap between any pair of segments, as a fraction of the shorter segment."""
    best = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode != sb.episode:
                continue
            inter = max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
            if inter:
                best = max(best, inter / min(sa.duration, sb.duration))
    return best


def select(cands: list[Candidate], k: int, max_overlap: float = 0.3, max_per_episode: int | None = None,
           episode: int | None = None, only_episode: bool = False) -> list[Candidate]:
    picked: list[Candidate] = []
    per_ep: dict[int, int] = {}
    for c in sorted(cands, key=lambda c: -c.score):
        if episode is not None and (episode not in c.episodes or (only_episode and c.episodes != [episode])):
            continue
        if max_per_episode and any(per_ep.get(e, 0) >= max_per_episode for e in c.episodes):
            continue
        if any(candidate_overlap(c, p) > max_overlap for p in picked):
            continue
        picked.append(c)
        for e in c.episodes:
            per_ep[e] = per_ep.get(e, 0) + 1
        if len(picked) >= k:
            break
    return picked
