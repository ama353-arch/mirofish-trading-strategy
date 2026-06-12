"""
engine_report.py — one report that ties the brain to its evidence.

Runs a probability source through the walk-forward harness, measures its
calibration on the resolved markets, and shows the portfolio allocation its
out-of-sample trades would produce. This is the instrument for calibration-driven
refinement: a swarm change ships only if this report improves out-of-sample.
"""

from __future__ import annotations

from src.signals.calibration import calibration_report
from src.signals.portfolio import allocate
from src.signals.walkforward import WalkForwardBacktester


def _prob_brain(prob_fn):
    """Wrap a probability source as a harness signal_fn (edge vs market)."""
    def brain(event: dict, params: dict) -> dict:
        edge = prob_fn(event) - event["market_prob"]
        if abs(edge) < params["edge_min"]:
            return {"side": None}
        return {"side": "buy" if edge > 0 else "sell",
                "size_pct": params.get("size_pct", 0.02)}
    return brain


def build_engine_report(
    events,
    prob_fn,
    train_size: int,
    test_size: int,
    edge_min: float = 0.05,
    max_deployed: float = 0.20,
    max_position: float = 0.05,
) -> dict:
    """Calibration + out-of-sample performance + portfolio allocation, together.

    `prob_fn(event) -> prob` is the probability source under test (swarm, pricer,
    ensemble). The harness scores it out-of-sample; calibration is measured over
    all events; allocations size the OOS trades under the bankroll caps.
    """
    # 1. out-of-sample backtest
    wf = WalkForwardBacktester(train_size=train_size, test_size=test_size)
    result = wf.run(events, signal_fn=_prob_brain(prob_fn),
                    param_grid=[{"edge_min": edge_min}])

    # 2. calibration over every resolved event
    probs = [prob_fn(e) for e in events]
    outcomes = [e["actual_outcome"] for e in events]
    cal = calibration_report(probs, outcomes)

    # 3. portfolio allocation of the OOS trades
    event_by_id = {e["id"]: e for e in events}
    signals = []
    for t in result.oos_trades:
        e = event_by_id[t["event_id"]]
        edge = abs(prob_fn(e) - e["market_prob"])
        signals.append({"id": t["event_id"], "edge": edge, "odds": 1.0})
    allocations = allocate(signals, max_deployed=max_deployed,
                           max_position=max_position, kelly_fraction=0.25)

    return {
        "calibration": cal,
        "oos": result.stats(),
        "allocations": allocations,
    }
