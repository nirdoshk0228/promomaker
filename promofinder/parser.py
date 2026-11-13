"""Parse timecoded micro-drama screenplays (Kuku-style text format) into a Show.

Expected episode layout (tolerant of small variations):

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. HAVELI – NIGHT
    **CAMERA:**
    Start: (00:00.00) | End: (00:03.00)
    Wide shot of ...
    **MEERA (Nervous, Soft):**
    Start: (00:03.30) | End: (00:05.10)
    "Ji, Rani Sa."
    --- SCRIPT END ---
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from typing import Iterable

from .models import Block, Episode, Show, Shot

LABEL_RE = re.compile(r"^\*\*\s*(.+?)\s*:?\s*\*\*\s*:?\s*$")
TIME_RE = re.compile(r"Start:\s*\(?\s*([\d:.]+)\s*\)?\s*\|\s*End:\s*\(?\s*([\d:.]+)\s*\)?", re.I)
HEADER_RE = {
    "number": re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I),
    "logline": re.compile(r"^EPISODE LOGLINE:\s*(.*)$", re.I),
    "notes": re.compile(r"^CONTINUITY NOTES:\s*(.*)$", re.I),
}
SCENE_RE = re.compile(r"^SCENE\s+(\d+)", re.I)
HEADING_RE = re.compile(r"^Scene heading:\s*(.*)$", re.I)
SKIP_RE = re.compile(r"^(Scene start:|Scene end:|Actions and dialogue:|---\s*SCRIPT)", re.I)

NON_SPEAKER_LABELS = {
    "CAMERA": "CAMERA", "SHOT": "CAMERA",
    "ACTION": "ACTION", "TEXT ON SCREEN": "ACTION", "SUPER": "ACTION",
    "SOUND FX": "SFX", "SFX": "SFX", "SOUND": "SFX", "MUSIC": "SFX",
}
MODIFIER_RE = re.compile(r"^(v\.?o\.?|o\.?s\.?|o\.?c\.?|off[- ]?screen|on (phone|tv|video|monitor|speakers?|radio|call)|"
                         r"to (himself|herself|themselves|camera|the .+|[a-z]+)|into phone|reading|cont'?d|continuing|"
                         r"through the door|over phone|on video call)", re.I)

MAX_FILES = 500
MAX_TOTAL_BYTES = 25 * 1024 * 1024


class ParseError(ValueError):
    pass


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ParseError(f"bad timecode '{value}'") from exc
    secs = 0.0
    for n in nums:
        secs = secs * 60 + n
    return secs


def _split_label(label: str):
    """'KAMLA BAI (V.O., Fearful)' -> ('KAMLA BAI', ['Fearful'], ['V.O.'])"""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label)
    name, inner = (m.group(1), m.group(2)) if m else (label, "")
    emotions, modifiers = [], []
    for tok in [t.strip() for t in inner.split(",") if t.strip()]:
        (modifiers if MODIFIER_RE.match(tok) else emotions).append(tok)
    return name.strip(), emotions, modifiers


def parse_episode_text(text: str, source_name: str = "", fallback_number: int | None = None) -> tuple[Episode, list[str]]:
    warnings: list[str] = []
    number, logline, notes = None, "", ""
    scene_no, heading = 0, ""
    blocks: list[Block] = []
    cur = None  # dict for the block being collected

    def flush():
        nonlocal cur
        if cur is None:
            return
        body = " ".join(cur["lines"]).strip()
        if cur["kind"] == "DIALOGUE":
            body = body.strip().strip('"“”').strip()
        if cur["start"] is None:
            prev_end = blocks[-1].end if blocks else 0.0
            cur["start"], cur["end"] = prev_end, prev_end + 2.0
            warnings.append(f"{source_name}: block '{cur['label']}' had no timecode; inferred {prev_end:.1f}s")
        if body or cur["kind"] == "CAMERA":
            blocks.append(Block(
                episode=number or fallback_number or 0, index=len(blocks), scene=max(scene_no, 1),
                scene_heading=heading, kind=cur["kind"], text=body, start=cur["start"], end=max(cur["end"], cur["start"]),
                speaker=cur.get("speaker"), emotions=cur.get("emotions", []), modifiers=cur.get("modifiers", []),
            ))
        cur = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        hit = False
        for key, rx in HEADER_RE.items():
            m = rx.match(line)
            if m:
                if key == "number":
                    number = int(m.group(1))
                elif key == "logline":
                    logline = m.group(1).strip()
                else:
                    notes = m.group(1).strip()
                hit = True
                break
        if hit:
            continue
        m = SCENE_RE.match(line)
        if m:
            flush(); scene_no = int(m.group(1)); continue
        m = HEADING_RE.match(line)
        if m:
            flush(); heading = m.group(1).strip(); continue
        if SKIP_RE.match(line):
            flush(); continue
        m = LABEL_RE.match(line)
        if m:
            flush()
            label = m.group(1).strip()
            base = re.sub(r"\s*\(.*\)\s*$", "", label).upper().strip()
            if base in NON_SPEAKER_LABELS:
                cur = {"kind": NON_SPEAKER_LABELS[base], "label": label, "lines": [], "start": None, "end": None}
            else:
                name, emotions, mods = _split_label(label)
                cur = {"kind": "DIALOGUE", "label": label, "speaker": name.upper(), "emotions": emotions,
                       "modifiers": mods, "lines": [], "start": None, "end": None}
            continue
        m = TIME_RE.search(line)
        if m and cur is not None and cur["start"] is None:
            cur["start"], cur["end"] = parse_time(m.group(1)), parse_time(m.group(2))
            continue
        if cur is not None:
            cur["lines"].append(line)
    flush()

    if number is None:
        number = fallback_number if fallback_number is not None else _number_from_name(source_name)
        if number is None:
            raise ParseError(f"{source_name or 'file'}: could not find an episode number")
        warnings.append(f"{source_name}: no EPISODE NUMBER header; using {number}")
    for b in blocks:
        b.episode = number
    if not blocks:
        raise ParseError(f"{source_name or 'episode'}: no timecoded blocks found — is this a screenplay in the expected format?")
    ep = Episode(number=number, title=f"Episode {number}", logline=logline, continuity_notes=notes,
                 blocks=blocks, source_name=source_name)
    ep.shots = build_shots(ep)
    return ep, warnings


def build_shots(ep: Episode) -> list[Shot]:
    """Group blocks into cuttable shots. A CAMERA block opens a shot; following non-camera blocks belong to it.
    If an episode has no CAMERA blocks, every block becomes its own shot."""
    has_camera = any(b.kind == "CAMERA" for b in ep.blocks)
    shots: list[Shot] = []
    for b in ep.blocks:
        if not has_camera or b.kind == "CAMERA" or not shots:
            shots.append(Shot(episode=ep.number, index=len(shots), start=b.start, end=b.end,
                              camera=b if b.kind == "CAMERA" else None, blocks=[b]))
        else:
            shots[-1].blocks.append(b)
    shots.sort(key=lambda s: s.start)
    for i, s in enumerate(shots):
        s.index = i
        if s.camera is None:
            s.end = max(x.end for x in s.blocks)
    # make shot spans contiguous and non-overlapping
    for a, b in zip(shots, shots[1:]):
        if a.end > b.start:
            a.end = b.start
    return shots


def _number_from_name(name: str):
    m = re.findall(r"(\d+)", os.path.basename(name or ""))
    return int(m[0]) if m else None


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def iter_texts_from_upload(filename: str, data: bytes) -> Iterable[tuple[str, str]]:
    """Yield (name, text) pairs from an uploaded .txt or .zip. Zip members are read in memory (no extraction)."""
    if filename.lower().endswith(".zip") or data[:2] == b"PK":
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ParseError("uploaded zip is corrupted") from exc
        members = [m for m in zf.infolist() if not m.is_dir() and m.filename.lower().endswith(".txt")
                   and not os.path.basename(m.filename).startswith((".", "__"))]
        if len(members) > MAX_FILES:
            raise ParseError(f"zip has {len(members)} text files; limit is {MAX_FILES}")
        total = sum(m.file_size for m in members)
        if total > MAX_TOTAL_BYTES:
            raise ParseError("zip contents too large (limit 25 MB uncompressed)")
        for m in members:
            yield m.filename, _decode(zf.read(m))
    else:
        yield filename, _decode(data)


def _title_from_names(names: list[str]) -> str:
    dirs = {os.path.dirname(n).split("/")[-1] for n in names if os.path.dirname(n)}
    return dirs.pop() if len(dirs) == 1 else "Uploaded show"


def parse_show(files: list[tuple[str, str]], title: str | None = None) -> Show:
    """files: list of (name, text). Files that are not episodes (no timecoded blocks) are skipped with a warning."""
    episodes, warnings = [], []
    for name, text in files:
        if "Start:" not in text:
            warnings.append(f"skipped {os.path.basename(name)}: no timecodes")
            continue
        try:
            ep, w = parse_episode_text(text, source_name=os.path.basename(name))
        except ParseError as exc:
            warnings.append(str(exc))
            continue
        episodes.append(ep)
        warnings.extend(w)
    if not episodes:
        raise ParseError("no episodes could be parsed. " + "; ".join(warnings[:3]))
    seen = {}
    for ep in episodes:
        if ep.number in seen:
            warnings.append(f"duplicate episode number {ep.number} ({seen[ep.number]} and {ep.source_name}); keeping the first")
        else:
            seen[ep.number] = ep.source_name
    episodes = [ep for ep in sorted(episodes, key=lambda e: e.number) if seen.get(ep.number) == ep.source_name]
    return Show(title=title or _title_from_names([n for n, _ in files]), episodes=episodes, warnings=warnings)


def load_show_from_dir(path: str, title: str | None = None) -> Show:
    files = []
    for name in sorted(os.listdir(path)):
        if name.lower().endswith(".txt"):
            with open(os.path.join(path, name), "rb") as fh:
                files.append((name, _decode(fh.read())))
    return parse_show(files, title=title or os.path.basename(os.path.normpath(path)))
