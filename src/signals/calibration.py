"""
calibration.py — measure whether a probability source can be trusted.

A source is calibrated when its stated probabilities match observed frequencies:
of all the times it says 60%, about 60% happen. Sharpness (confidence) without
calibration loses money. Every brain tool (swarm, pricer, ML) must clear a
calibration check on resolved markets before it sizes a real position.
"""

from __future__ import annotations

import math


def _validate(probs, outcomes) -> None:
    if len(probs) != len(outcomes):
        raise ValueError(
            f"probs and outcomes must be the same length: "
            f"{len(probs)} != {len(outcomes)}"
        )


def brier_score(probs, outcomes) -> float:
    """Mean squared error between probabilities and binary outcomes.

    0 is perfect; a source that always predicts the base rate p scores p(1-p).
    Lower is better.
    """
    _validate(probs, outcomes)
    if not probs:
        return 0.0
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss(probs, outcomes, eps: float = 1e-15) -> float:
    """Negative average log-likelihood; punishes confident wrong answers hard.

    Probabilities are clamped to [eps, 1-eps] so a 0 or 1 prediction stays finite.
    """
    _validate(probs, outcomes)
    if not probs:
        return 0.0
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = min(1.0 - eps, max(eps, p))
        total += o * math.log(p) + (1 - o) * math.log(1.0 - p)
    return -total / len(probs)


def reliability_curve(probs, outcomes, n_bins: int = 10) -> list[dict]:
    """Reliability-diagram data: for each populated probability bin, the mean
    predicted probability vs the observed outcome frequency. Empty bins are
    omitted (not emitted as NaN)."""
    _validate(probs, outcomes)
    bins: list[dict] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        # last bin is closed on the right so prob == 1.0 lands somewhere
        members = [
            (p, o) for p, o in zip(probs, outcomes)
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]
        if not members:
            continue
        ps = [p for p, _ in members]
        os = [o for _, o in members]
        bins.append({
            "bin": i,
            "count": len(members),
            "predicted": sum(ps) / len(ps),
            "observed": sum(os) / len(os),
        })
    return bins


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


class Calibrator:
    """Platt scaling: learn a source's bias on training data and correct it.

    Fits `calibrated = sigmoid(a * logit(raw) + b)` by gradient descent on log
    loss. The slope `a` is clamped to be non-negative so calibration never
    inverts the source's ranking — it rescales confidence, it doesn't flip
    signal. Fit on a train fold, apply to the untouched test fold: no look-ahead.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, probs, outcomes, lr: float = 0.5, iters: int = 2000) -> "Calibrator":
        _validate(probs, outcomes)
        if not probs:
            return self
        xs = [_logit(p) for p in probs]
        ys = list(outcomes)
        n = len(xs)
        a, b = 1.0, 0.0
        for _ in range(iters):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                pred = _sigmoid(a * x + b)
                err = pred - y
                ga += err * x
                gb += err
            a -= lr * ga / n
            b -= lr * gb / n
        self.a = max(0.0, a)   # never invert ranking
        self.b = b
        return self

    def transform(self, probs) -> list[float]:
        return [
            max(0.0, min(1.0, _sigmoid(self.a * _logit(p) + self.b)))
            for p in probs
        ]


def calibration_report(probs, outcomes, n_bins: int = 10) -> dict:
    """Full calibration summary for a probability source."""
    _validate(probs, outcomes)
    n = len(probs)
    base_rate = (sum(outcomes) / n) if n else 0.0
    return {
        "n": n,
        "base_rate": base_rate,
        "brier": brier_score(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "reliability": reliability_curve(probs, outcomes, n_bins=n_bins),
    }
