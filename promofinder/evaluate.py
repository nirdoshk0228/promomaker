"""Evaluate predicted promos against a hand-labelled answer key.

Matching: temporal IoU between a prediction and a gold promo, computed over all their segments
(segments only overlap when they are in the same episode). IoU >= 0.5 is a strict match, >= 0.3 a loose one.
"""
from __future__ import annotations

import random
from typing import Iterable

from .parser import parse_time


def _segs(p: dict) -> list[tuple[int, float, float]]:
    out = []
    for s in p["segments"]:
        st, en = s["start"], s["end"]
        st = parse_time(st) if isinstance(st, str) else float(st)
        en = parse_time(en) if isinstance(en, str) else float(en)
        out.append((int(s["episode"]), st, en))
    return out


def iou(a: list, b: list) -> float:
    inter = 0.0
    for ea, sa, xa in a:
        for eb, sb, xb in b:
            if ea == eb:
                inter += max(0.0, min(xa, xb) - max(sa, sb))
    la = sum(x - s for _, s, x in a)
    lb = sum(x - s for _, s, x in b)
    union = la + lb - inter
    return inter / union if union > 0 else 0.0


def load_key(key: dict) -> tuple[list, list]:
    gold = [{"id": p.get("promo_id", f"G{i}"), "type": p.get("type", ""), "why": p.get("why", ""),
             "segs": _segs(p)} for i, p in enumerate(key.get("promos", []), 1)]
    traps = []
    for t in key.get("traps", []):
        st = parse_time(t["start"]) if isinstance(t["start"], str) else float(t["start"])
        en = parse_time(t["end"]) if isinstance(t["end"], str) else float(t["end"])
        traps.append({"episode": int(t["episode"]), "start": st, "end": en, "reason": t.get("reason", ""),
                      "text": t.get("text", "")})
    return gold, traps


def _hits_trap(segs, traps, hook_win=3.0, end_win=4.5) -> bool:
    e0, s0, _ = segs[0]
    e1, _, x1 = segs[-1]
    for t in traps:
        if t["episode"] == e0 and t["start"] < s0 + hook_win and t["end"] > s0:
            return True
        if t["episode"] == e1 and t["start"] < x1 and t["end"] > x1 - end_win:
            return True
    return False


def evaluate(result: dict, key: dict, strict: float = 0.5, loose: float = 0.3, edge_tol: float = 3.0) -> dict:
    gold, traps = load_key(key)
    preds = [{"rank": p["rank"], "id": p["id"], "segs": _segs(p), "type": p["type"]} for p in result["promos"]]

    def best_match(segs):
        best, gid = 0.0, None
        for g in gold:
            v = iou(segs, g["segs"])
            if v > best:
                best, gid = v, g["id"]
        return best, gid

    # overall top-K list
    rows = []
    for p in preds:
        v, gid = best_match(p["segs"])
        rows.append({"rank": p["rank"], "id": p["id"], "type": p["type"], "best_iou": round(v, 3),
                     "matched_gold": gid if v >= loose else None, "strict": v >= strict,
                     "trap": _hits_trap(p["segs"], traps)})
    k = len(preds)
    matched_strict = {r["matched_gold"] for r in rows if r["strict"]}
    matched_loose = {r["matched_gold"] for r in rows if r["matched_gold"]}
    overall = {
        "k": k,
        "precision_strict": _ratio(sum(r["strict"] for r in rows), k),
        "precision_loose": _ratio(sum(bool(r["matched_gold"]) for r in rows), k),
        "gold_covered_strict": _ratio(len(matched_strict), len(gold)),
        "gold_covered_loose": _ratio(len(matched_loose), len(gold)),
        "trap_rate": _ratio(sum(r["trap"] for r in rows), k),
    }

    # per-episode: does the editor's shortlist of 3 contain a gold promo for that episode?
    per_ep = result.get("per_episode", {})
    ep_rows, hit1, hit3, hit1l, hit3l, n_eval, edge_h, edge_c, n_edge, trap1 = [], 0, 0, 0, 0, 0, 0, 0, 0, 0
    for ep_str, picks in per_ep.items():
        ep = int(ep_str)
        g_ep = [g for g in gold if any(e == ep for e, _, _ in g["segs"])]
        if not g_ep or not picks:
            continue
        n_eval += 1
        ious = [max(iou(_segs(p), g["segs"]) for g in g_ep) for p in picks]
        h1, h3 = ious[0] >= strict, any(v >= strict for v in ious[:3])
        l1, l3 = ious[0] >= loose, any(v >= loose for v in ious[:3])
        hit1 += h1; hit3 += h3; hit1l += l1; hit3l += l3
        top = _segs(picks[0])
        trap1 += _hits_trap(top, traps)
        singles = [g for g in g_ep if len(g["segs"]) == 1]
        if singles:
            n_edge += 1
            gs = singles[0]["segs"][0]
            edge_h += abs(top[0][1] - gs[1]) <= edge_tol and top[0][0] == gs[0]
            edge_c += abs(top[-1][2] - gs[2]) <= edge_tol and top[-1][0] == gs[0]
        ep_rows.append({"episode": ep, "top1_iou": round(ious[0], 3), "best_of_3_iou": round(max(ious[:3]), 3),
                        "hit@1": h1, "hit@3": h3})
    per_episode = {
        "episodes_evaluated": n_eval,
        "hit@1_strict": _ratio(hit1, n_eval), "hit@3_strict": _ratio(hit3, n_eval),
        "hit@1_loose": _ratio(hit1l, n_eval), "hit@3_loose": _ratio(hit3l, n_eval),
        "hook_start_within_3s": _ratio(edge_h, n_edge), "cliff_end_within_3s": _ratio(edge_c, n_edge),
        "top1_trap_rate": _ratio(trap1, n_eval),
    }
    return {"overall": overall, "per_episode": per_episode, "rows": rows, "episode_rows": ep_rows,
            "gold_count": len(gold), "trap_count": len(traps),
            "thresholds": {"strict_iou": strict, "loose_iou": loose, "edge_tolerance_s": edge_tol}}


def _ratio(a, b):
    return round(a / b, 3) if b else None


# ---------- baselines (to show the scorer adds value over naive rules) ----------

def baseline_result(show, kind: str, length: float = 35.0, seed: int = 7) -> dict:
    """'episode_end': last `length` seconds before the episode's final content shot (what a rushed editor might do).
    'random': a random valid window per episode."""
    from .features import EpisodeIndex
    from .candidates import _windows
    rng = random.Random(seed)
    per_ep, promos = {}, []
    for ep in show.episodes:
        idx = EpisodeIndex(ep)
        wins = list(_windows(idx, 20, 60))
        if not wins:
            continue
        if kind == "episode_end":
            ends = [w for w in wins if w[1] == idx.last_content_shot] or wins
            i, j = min(ends, key=lambda w: abs((ep.shots[w[1]].end - ep.shots[w[0]].start) - length))
        else:
            i, j = rng.choice(wins)
        p = {"rank": 1, "id": f"{kind}-{ep.number}", "type": "single",
             "segments": [{"episode": ep.number, "start": ep.shots[i].start, "end": ep.shots[j].end}]}
        per_ep[ep.number] = [p]
        promos.append(p)
    return {"promos": promos, "per_episode": per_ep}


def summarize(metrics: dict) -> Iterable[str]:
    o, e = metrics["overall"], metrics["per_episode"]
    yield f"Per-episode shortlist (top-3): hit@3 strict {e['hit@3_strict']}  loose {e['hit@3_loose']}"
    yield f"Per-episode top pick:          hit@1 strict {e['hit@1_strict']}  loose {e['hit@1_loose']}"
    yield f"Top pick edges: hook start within 3s {e['hook_start_within_3s']}, cliff end within 3s {e['cliff_end_within_3s']}"
    yield f"Top pick trap rate: {e['top1_trap_rate']}"
    yield f"Overall top-{o['k']}: precision strict {o['precision_strict']} loose {o['precision_loose']}, trap rate {o['trap_rate']}"
