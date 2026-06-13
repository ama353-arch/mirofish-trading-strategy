#!/usr/bin/env python3
"""
refine_swarm.py — the calibration-driven swarm refinement loop (E5).

Runs the real swarm on resolved markets, fits a Platt calibrator on the TRAIN
fold, applies it to the held-out TEST fold, and reports whether calibration
lowers the swarm's Brier out-of-sample. No look-ahead: the calibrator only ever
sees train outcomes.

A swarm change is "accepted" only if it improves the TEST Brier here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.signals.calibration import Calibrator, brier_score, log_loss
from run_walkforward import make_swarm_prob_fn, SAMPLE_PATH


def main() -> None:
    events = json.loads(Path(SAMPLE_PATH).read_text())
    events.sort(key=lambda e: e["timestamp"])
    split = int(len(events) * 0.6)
    train, test = events[:split], events[split:]

    prob_fn = make_swarm_prob_fn()
    train_p = [prob_fn(e) for e in train]
    train_y = [e["actual_outcome"] for e in train]
    test_p = [prob_fn(e) for e in test]
    test_y = [e["actual_outcome"] for e in test]

    cal = Calibrator().fit(train_p, train_y)
    test_cal = cal.transform(test_p)

    raw_brier = brier_score(test_p, test_y)
    cal_brier = brier_score(test_cal, test_y)
    base = sum(test_y) / len(test_y)

    print("=== SWARM REFINEMENT (calibration layer, out-of-sample) ===")
    print(f"  Train / test markets:   {len(train)} / {len(test)}")
    print(f"  Fitted Platt:           a={cal.a:.3f}, b={cal.b:.3f}")
    print(f"  Base-rate Brier (test): {base * (1 - base):.4f}")
    print(f"  RAW swarm Brier (test): {raw_brier:.4f}  | log loss {log_loss(test_p, test_y):.4f}")
    print(f"  CAL swarm Brier (test): {cal_brier:.4f}  | log loss {log_loss(test_cal, test_y):.4f}")
    verdict = "ACCEPT (calibration helps OOS)" if cal_brier < raw_brier else "REJECT (no OOS gain)"
    print(f"  Verdict:                {verdict}")
    print("\n  Note: calibration fixes systematic over/under-confidence. It cannot")
    print("  manufacture signal — if the uninformed swarm is near-noise, the win is")
    print("  small. Real lift needs the swarm fed real information (next phase).")


if __name__ == "__main__":
    main()
