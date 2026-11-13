"""Core data structures for PromoFinder."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

KINDS = ("CAMERA", "ACTION", "DIALOGUE", "SFX")


@dataclass
class Block:
    """One timecoded line of the screenplay (a camera shot, an action, a line of dialogue or a sound cue)."""
    episode: int
    index: int                 # position inside the episode (0-based, file order)
    scene: int                 # 1-based scene number
    scene_heading: str
    kind: str                  # CAMERA | ACTION | DIALOGUE | SFX
    text: str
    start: float               # seconds from episode start
    end: float
    speaker: Optional[str] = None
    emotions: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)   # V.O., O.S., On phone ...

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Shot:
    """A cuttable unit: one CAMERA block plus the action/dialogue/sfx blocks that start inside it."""
    episode: int
    index: int
    start: float
    end: float
    camera: Optional[Block]
    blocks: list[Block]        # all blocks in this shot, camera first


@dataclass
class Episode:
    number: int
    title: str
    logline: str
    continuity_notes: str
    blocks: list[Block]
    shots: list[Shot] = field(default_factory=list)
    source_name: str = ""

    @property
    def duration(self) -> float:
        return max((b.end for b in self.blocks), default=0.0)


@dataclass
class Show:
    title: str
    episodes: list[Episode]
    warnings: list[str] = field(default_factory=list)

    def episode(self, number: int) -> Episode:
        for ep in self.episodes:
            if ep.number == number:
                return ep
        raise KeyError(number)

    @property
    def runtime(self) -> float:
        return sum(e.duration for e in self.episodes)


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    shot_from: int             # first shot index (inclusive)
    shot_to: int               # last shot index (inclusive)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Candidate:
    segments: list[Segment]
    scores: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 0.0
    llm: Optional[dict] = None

    @property
    def kind(self) -> str:
        return "single" if len(self.segments) == 1 else "stitched"

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})

    def to_dict(self) -> dict:
        d = {
            "type": self.kind,
            "duration": round(self.duration, 2),
            "score": round(self.score, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "reasons": self.reasons,
            "warnings": self.warnings,
            "segments": [asdict(s) for s in self.segments],
        }
        if self.llm:
            d["llm"] = self.llm
        return d
