"""Turn a picked promo into an actual MP4 with ffmpeg.

Two sources:
- real episode videos uploaded by the editor (one file per episode), cut at the promo's timecodes;
- demo footage, generated on the fly for shows without video: a test pattern with the episode number, the
  running episode timecode and the script's dialogue as subtitles, so anyone can check the cut lands on the
  right line. Demo footage is clearly labelled and never mixed with real footage.

Every segment is re-encoded to one common format (H.264/AAC, 25 fps) so stitched promos concatenate cleanly.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Optional

FPS = 25
VERTICAL = (720, 1280)
LANDSCAPE = (1280, 720)
FONT = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf") if os.path.exists(p)), None)
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v")
_ffmpeg_slot = threading.Semaphore(int(os.getenv("PROMOFINDER_FFMPEG_JOBS", "1")))


class VideoError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str], timeout: int = 240) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise VideoError("ffmpeg took too long") from exc
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-3:]
        raise VideoError("ffmpeg failed: " + " | ".join(tail))
    return p.stdout


def probe(path: str) -> dict:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
                "-of", "json", path], timeout=60)
    data = json.loads(out or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise VideoError("file has no video stream")
    return {"duration": float(data.get("format", {}).get("duration", 0) or 0), "width": v.get("width"),
            "height": v.get("height"), "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def episode_from_filename(name: str) -> Optional[int]:
    base = os.path.splitext(os.path.basename(name))[0].lower()
    m = re.search(r"(?:ep|episode|e)[\s_\-.]*0*(\d{1,4})", base) or re.search(r"(\d{1,4})", base)
    return int(m.group(1)) if m else None


def _fit_filter(w: int, h: int) -> str:
    # fill the frame and crop the overflow (centre crop): the usual way to make 9:16 from 16:9 footage
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,fps={FPS},format=yuv420p")


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def _srt_time(t: float) -> str:
    t = max(t, 0.0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def _write_srt(blocks: list[dict], seg_start: float, path: str) -> bool:
    lines, n = [], 0
    for b in blocks:
        if b.get("kind") != "DIALOGUE":
            continue
        n += 1
        text = f"{b.get('speaker', '')}: {b.get('text', '')}"
        lines += [str(n), f"{_srt_time(b['start'] - seg_start)} --> {_srt_time(b['end'] - seg_start)}", text, ""]
    if not n:
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return True


def _encode_args(out: str) -> list[str]:
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", "-y", out]


def render_segment_real(src: str, start: float, end: float, out: str, size: tuple, has_audio: bool) -> None:
    dur = max(0.1, end - start)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src]
    if not has_audio:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
    cmd += ["-vf", _fit_filter(*size), "-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0", "-shortest"]
    _run(cmd + _encode_args(out))


def render_segment_demo(episode: int, start: float, end: float, blocks: list[dict], out: str, size: tuple,
                        workdir: str) -> None:
    dur = max(0.1, end - start)
    w, h = size
    fs = max(22, w // 22)
    font = f"fontfile={FONT}:" if FONT else ""
    label = _escape_drawtext(f"DEMO FOOTAGE  |  Episode {episode}")
    vf = [f"drawtext={font}text='{label}':fontsize={fs}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=12:"
          f"x=(w-tw)/2:y={h // 12}",
          f"drawtext={font}text='%{{pts\\:hms\\:{start:.3f}}}':fontsize={int(fs * 1.8)}:fontcolor=white:box=1:"
          f"boxcolor=black@0.6:boxborderw=14:x=(w-tw)/2:y={h // 12 + int(fs * 2.4)}"]
    srt = os.path.join(workdir, f"demo_{episode}_{start:.1f}.srt")
    if _write_srt(blocks, start, srt):
        # libass sizes are relative to a 288-px-high canvas, so these give ~40 px text in both formats
        style = f"Fontsize={9 if h > w else 16},Alignment=2,MarginV={20 if h > w else 14},BorderStyle=3,Outline=1,Shadow=0"
        vf.append(f"subtitles={srt}:force_style='{style}'")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={FPS}:duration={dur:.3f}",
           "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
           "-vf", ",".join(vf), "-shortest"]
    _run(cmd + _encode_args(out))


def render_promo(promo: dict, out_path: str, sources: dict[int, dict] | None = None, vertical: bool = True,
                 demo: bool = False, offsets: dict[int, float] | None = None) -> str:
    """promo: a promo dict from an analysis result (needs segments with episode/start/end; blocks for demo).
    sources: {episode: {"path": ..., "duration": ..., "has_audio": ...}} for real footage."""
    if not ffmpeg_available():
        raise VideoError("ffmpeg is not installed on this server")
    size = VERTICAL if vertical else LANDSCAPE
    offsets = offsets or {}
    with _ffmpeg_slot, tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, seg in enumerate(promo["segments"]):
            ep = int(seg["episode"])
            part = os.path.join(tmp, f"part{i}.mp4")
            if demo:
                render_segment_demo(ep, seg["start"], seg["end"], seg.get("blocks", []), part, size, tmp)
            else:
                src = (sources or {}).get(ep)
                if not src:
                    raise VideoError(f"no video uploaded for episode {ep}")
                off = offsets.get(ep, 0.0)
                start, end = seg["start"] + off, seg["end"] + off
                if src.get("duration") and start >= src["duration"]:
                    raise VideoError(f"episode {ep} video is only {src['duration']:.1f}s long; promo starts at {start:.1f}s")
                render_segment_real(src["path"], max(0.0, start), end, part, size, src.get("has_audio", True))
            parts.append(part)
        if len(parts) == 1:
            shutil.move(parts[0], out_path)
        else:
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w") as fh:
                fh.writelines(f"file '{p}'\n" for p in parts)
            _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                  "-c", "copy", "-movflags", "+faststart", "-y", out_path])
    return out_path


def cache_key(job_id: str, promo_id: str, vertical: bool, demo: bool, sources: dict, offsets: dict) -> str:
    sig = json.dumps({"j": job_id, "p": promo_id, "v": vertical, "d": demo,
                      "s": {str(k): [v.get("path"), v.get("mtime")] for k, v in sorted((sources or {}).items())},
                      "o": offsets or {}}, sort_keys=True)
    return hashlib.sha1(sig.encode()).hexdigest()[:16]
