"""Learning-to-rank: learn how much each signal matters from labelled promos and editor votes.

The v1 scorer uses hand-set weights (30% hook, 30% cliffhanger ...). This module fits a logistic-regression
ranker on the same signals (component scores + ~24 raw features), so the weights come from data instead.

Training needs scikit-learn; scoring does not — a trained model is a small JSON file (feature means,
scales, coefficients) that the app applies with plain Python.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from .evaluate import iou, load_key
from .models import Candidate

SCORE_KEYS = ["hook", "cliff", "standalone", "intensity", "duration"]


def feature_names(sample: Candidate) -> list[str]:
    return [f"score_{k}" for k in SCORE_KEYS] + sorted(sample.features)


def featurize(c: Candidate, names: list[str]) -> list[float]:
    out = []
    for n in names:
        if n.startswith("score_"):
            out.append(float(c.scores.get(n[6:], 0.0)))
        else:
            out.append(float(c.features.get(n, 0.0)))
    return out


def candidate_segs(c: Candidate):
    return [(s.episode, s.start, s.end) for s in c.segments]


def label(c: Candidate, gold: list, threshold: float = 0.5, start_tol: float = 2.0, end_tol: float = 3.0) -> tuple[int, float]:
    """Positive = the cut an editor would actually make: it overlaps a gold promo (IoU >= threshold) AND starts
    within `start_tol` s of the gold start AND ends within `end_tol` s of the gold end. Overlap alone is not
    enough — a clip that starts 8 s early overlaps well but misses the hook."""
    segs = candidate_segs(c)
    best, pos = 0.0, 0
    for g in gold:
        v = iou(segs, g["segs"])
        best = max(best, v)
        gs, ge = g["segs"][0], g["segs"][-1]
        if (v >= threshold and segs[0][0] == gs[0] and abs(segs[0][1] - gs[1]) <= start_tol
                and segs[-1][0] == ge[0] and abs(segs[-1][2] - ge[2]) <= end_tol):
            pos = 1
    return pos, best


@dataclass
class LearnedRanker:
    names: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    meta: dict

    def prob(self, c: Candidate) -> float:
        x = featurize(c, self.names)
        z = self.intercept + sum(w * (v - m) / s for w, v, m, s in zip(self.coef, x, self.mean, self.scale))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def apply(self, cands: Iterable[Candidate], blend: float = 0.5) -> None:
        """Final score = (1 - blend) * rule score + blend * learned probability.
        Held-out tests favoured the 50/50 blend over either score alone (see scripts/train_ranker.py)."""
        for c in cands:
            p = self.prob(c)
            c.scores["learned"] = p
            c.score = (1 - blend) * c.scores.get("rules", c.score) + blend * p

    def top_weights(self, k: int = 8) -> list[dict]:
        pairs = sorted(zip(self.names, self.coef), key=lambda t: -abs(t[1]))[:k]
        return [{"feature": n, "weight": round(w, 3)} for n, w in pairs]

    def to_dict(self) -> dict:
        return {"names": self.names, "mean": self.mean, "scale": self.scale, "coef": self.coef,
                "intercept": self.intercept, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedRanker":
        return cls(d["names"], d["mean"], d["scale"], d["coef"], d["intercept"], d.get("meta", {}))

    @classmethod
    def load(cls, path: str) -> Optional["LearnedRanker"]:
        if not path or not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)


def fit(X: list[list[float]], y: list[int], names: list[str], sample_weight=None, C: float = 0.3,
        meta: Optional[dict] = None) -> LearnedRanker:
    """Standardise features and fit an L2 logistic regression (balanced classes)."""
    from sklearn.linear_model import LogisticRegression  # training-time dependency only

    n = len(names)
    mean = [sum(r[j] for r in X) / len(X) for j in range(n)]
    var = [sum((r[j] - mean[j]) ** 2 for r in X) / len(X) for j in range(n)]
    scale = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    Z = [[(r[j] - mean[j]) / scale[j] for j in range(n)] for r in X]
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit(Z, y, sample_weight=sample_weight)
    return LearnedRanker(names, mean, scale, [float(w) for w in clf.coef_[0]], float(clf.intercept_[0]),
                         meta or {})


def training_rows(cands: list[Candidate], key: dict, episodes: Optional[set] = None):
    """Label every candidate whose footage lies entirely inside `episodes` by IoU with the answer key."""
    gold, _ = load_key(key)
    names = feature_names(cands[0])
    X, y = [], []
    for c in cands:
        if episodes is not None and not set(c.episodes) <= episodes:
            continue
        X.append(featurize(c, names))
        y.append(label(c, gold)[0])
    return X, y, names


def featurize_payload(p: dict, names: list[str]) -> list[float]:
    """Rebuild a feature vector from a promo as stored in a job result (used to learn from editor votes)."""
    out = []
    for n in names:
        if n.startswith("score_"):
            out.append(float(p.get("scores", {}).get(n[6:], 0.0)))
        else:
            out.append(float(p.get("features", {}).get(n, 0.0)))
    return out
