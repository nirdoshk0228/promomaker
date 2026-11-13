"""Export promo lists for editors: CSV (spreadsheet), JSON (automation) and CMX3600 EDL (Premiere/Resolve/FCP)."""
from __future__ import annotations

import csv
import io
import json

FPS = 25


def _tc_frames(sec: float, fps: int = FPS) -> str:
    total = int(round(sec * fps))
    f = total % fps
    s = total // fps
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}:{f:02d}"


def to_csv(result: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "promo_id", "type", "total_sec", "score", "hook", "cliffhanger", "standalone",
                "segment", "episode", "start", "end", "segment_sec", "opening_line", "ending_line",
                "llm_verdict", "why"])
    for p in result["promos"]:
        for n, s in enumerate(p["segments"], 1):
            w.writerow([p["rank"], p["id"], p["type"], p["duration"], p["score"], p["scores"]["hook"],
                        p["scores"]["cliff"], p["scores"]["standalone"], n, s["episode"], s["start_tc"], s["end_tc"],
                        round(s["end"] - s["start"], 2), (p.get("opening") or {}).get("text", ""),
                        (p.get("ending") or {}).get("text", ""), (p.get("llm") or {}).get("verdict", ""),
                        " | ".join(p["reasons"])])
    return buf.getvalue()


def to_json(result: dict) -> str:
    slim = []
    for p in result["promos"]:
        slim.append({"rank": p["rank"], "id": p["id"], "type": p["type"], "duration": p["duration"],
                     "score": p["score"], "scores": p["scores"], "reasons": p["reasons"], "warnings": p["warnings"],
                     "llm": p.get("llm"),
                     "segments": [{"episode": s["episode"], "start": s["start"], "end": s["end"],
                                   "start_tc": s["start_tc"], "end_tc": s["end_tc"]} for s in p["segments"]]})
    return json.dumps({"show": result["show"]["title"], "promos": slim}, indent=2, ensure_ascii=False)


def to_edl(result: dict, gap_sec: float = 2.0) -> str:
    """One EDL; promos are laid out back to back on the record timeline with a gap between them.
    Reel names are EP001, EP002 ... so editors can relink each reel to that episode's master file."""
    title = "".join(ch for ch in result["show"]["title"].upper() if ch.isalnum() or ch == " ")[:60] or "PROMOS"
    lines = [f"TITLE: {title} PROMOS", "FCM: NON-DROP FRAME", ""]
    rec = 3600.0
    ev = 1
    for p in result["promos"]:
        lines.append(f"* PROMO {p['rank']} ({p['type']}, {p['duration']:.1f}s, score {p['score']:.2f})")
        for s in p["segments"]:
            dur = s["end"] - s["start"]
            reel = f"EP{s['episode']:03d}"
            lines.append(f"{ev:03d}  {reel:<8} V     C        {_tc_frames(s['start'])} {_tc_frames(s['end'])} "
                         f"{_tc_frames(rec)} {_tc_frames(rec + dur)}")
            lines.append(f"* FROM CLIP NAME: EPISODE {s['episode']}")
            rec += dur
            ev += 1
        rec += gap_sec
        lines.append("")
    return "\n".join(lines)
