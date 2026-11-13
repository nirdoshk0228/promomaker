"""Per-block and per-shot features. Computed once per episode, reused by every candidate."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from . import lexicon as L
from .models import Block, Episode

CONJUNCTION_START = re.compile(r"^(toh|to|par|aur|lekin|phir|isliye|matlab|so|but|and|then|haan toh)\b", re.I)
CUT_TO_BLACK = re.compile(r"cut to black|fade (to|out)", re.I)


@dataclass
class BlockFeatures:
    physical: int
    reveal: int
    threat: int
    energy: float
    question: bool
    exclaim: int
    trailing: bool
    cliff_cue: int
    resolution: int
    exposition: int
    backref: int
    dream: bool
    flashback: bool
    establishing: bool
    comedy: bool
    words: int
    intensity: float


def block_features(b: Block) -> BlockFeatures:
    text = b.text or ""
    is_dlg = b.kind == "DIALOGUE"
    emo_text = " ".join(b.emotions)
    f = dict(
        physical=min(L.count(L.PHYSICAL, text), 2),
        reveal=min(L.count(L.REVEAL, text), 2),
        threat=min(L.count(L.THREAT, text), 2) if is_dlg else min(L.count(L.THREAT, text), 1),
        energy=L.emotion_energy(b.emotions) if is_dlg else 0.0,
        question=is_dlg and text.rstrip().rstrip('"').endswith(("?", "?!", "!?")),
        exclaim=min(text.count("!"), 2) if is_dlg else 0,
        trailing=is_dlg and text.rstrip().rstrip('"').endswith(("...", "…", "—", "-")),
        cliff_cue=min(L.count(L.CLIFF_CUES, text), 2) if b.kind in ("CAMERA", "SFX", "ACTION") else 0,
        resolution=min(L.count(L.RESOLUTION, text + " " + emo_text), 2),
        exposition=min(L.count(L.EXPOSITION, text + " " + emo_text), 3),
        backref=min(L.count(L.BACK_REFERENCE, text), 2) if is_dlg else 0,
        dream=bool(L.DREAM.search(text)),
        flashback=bool(L.FLASHBACK.search(text)) or ("V.O." in " ".join(b.modifiers)),
        establishing=bool(L.ESTABLISHING.search(text)) and b.kind == "CAMERA",
        comedy=bool(L.COMEDY.search(text + " " + emo_text)),
        words=len(text.split()),
    )
    # visual shock counts fully in stage directions; a violent word inside dialogue is only a mention
    visual = f["physical"] * (0.3 if is_dlg else 0.9)
    # what a line says lands harder when it is delivered with heat
    delivery = (0.6 + 0.6 * max(f["energy"], 0) - 0.2 * max(-f["energy"], 0)) if is_dlg else 1.0
    raw = (visual + delivery * (0.6 * f["reveal"] + 0.6 * f["threat"]) + 0.6 * max(f["energy"], 0)
           + 0.25 * f["question"] + 0.15 * f["exclaim"] + 0.3 * f["cliff_cue"]
           - 0.4 * max(-f["energy"], 0) - 0.3 * min(f["exposition"], 2) - 0.25 * f["establishing"]
           - 0.3 * f["comedy"])
    if is_dlg and f["words"] > 20 and f["energy"] <= 0:
        raw -= 0.3   # long, calm speeches read as exposition
    f["intensity"] = max(0.0, min(1.0, raw / 1.8))
    return BlockFeatures(**f)


class EpisodeIndex:
    """Precomputed features for one episode."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.bf = {b.index: block_features(b) for b in ep.blocks}
        self.shot_ok = [not (s.camera and CUT_TO_BLACK.search(s.camera.text or "")) for s in ep.shots]
        self.dialogues = [b for b in ep.blocks if b.kind == "DIALOGUE"]
        self._sorted = sorted(ep.blocks, key=lambda b: (b.start, b.index))
        self._starts = [b.start for b in self._sorted]
        last_ok = max((i for i, ok in enumerate(self.shot_ok) if ok), default=len(ep.shots) - 1)
        self.last_content_shot = last_ok

    def blocks_between(self, t0: float, t1: float) -> list[Block]:
        lo = bisect.bisect_left(self._starts, t0 - 1e-6)
        hi = bisect.bisect_left(self._starts, t1 - 1e-6)
        return self._sorted[lo:hi]

    def clean_cut_at(self, t: float, tol: float = 0.35) -> bool:
        """True if no line of dialogue is still being spoken across time t."""
        for d in self.dialogues:
            if d.start < t - tol and d.end > t + tol:
                return False
        return True

    def previous_dialogue(self, t: float):
        prev = [d for d in self.dialogues if d.end <= t + 0.05]
        return prev[-1] if prev else None
