#!/usr/bin/env python3
"""
Run both paper harnesses under one process, and restart either if it dies.

Hosting gives you one process per service, but these are two independent measurements that should not
share a fate: a crash in the MLB harness must not silently stop the crypto one. So each runs as a
child and is restarted with backoff, and every restart is logged — a harness that quietly stopped
looks exactly like a harness finding nothing.

READ-ONLY and fake money throughout. Neither child can place an order.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# MLB is deliberately NOT here. Its backtest edge came apart once the first-inning trades were
# excluded (+0.85c on one half, +5.33c on the other), and the rule had to be changed after seeing
# live losses, which makes it post-hoc. Crypto is the one measurement still worth running.
CHILDREN = [
    ("crypto", [sys.executable, "-u", "run_paper.py", "--contracts", "100"]),
]
MIN_BACKOFF, MAX_BACKOFF = 5, 300


def stamp() -> str:
    return f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}"


def main() -> None:
    data_dir = os.environ.get("PAPER_DATA_DIR", "data/paper")
    os.makedirs(data_dir, exist_ok=True)
    print(f"{stamp()} supervisor up — data dir {data_dir}", flush=True)

    procs: dict[str, subprocess.Popen] = {}
    backoff = {name: MIN_BACKOFF for name, _ in CHILDREN}
    next_start = {name: 0.0 for name, _ in CHILDREN}

    while True:
        now = time.time()
        for name, cmd in CHILDREN:
            p = procs.get(name)
            if p is not None and p.poll() is None:
                continue
            if p is not None:
                print(f"{stamp()} [{name}] exited with {p.returncode}; "
                      f"restarting in {backoff[name]}s", flush=True)
                next_start[name] = now + backoff[name]
                backoff[name] = min(backoff[name] * 2, MAX_BACKOFF)
                procs[name] = None
                continue
            if now < next_start[name]:
                continue
            print(f"{stamp()} [{name}] starting", flush=True)
            procs[name] = subprocess.Popen(cmd)
            # a child that survives a while is healthy; stop punishing it for an old crash
            next_start[name] = now
        for name in list(procs):
            p = procs.get(name)
            if p is not None and p.poll() is None and time.time() - next_start[name] > 600:
                backoff[name] = MIN_BACKOFF
        time.sleep(5)


if __name__ == "__main__":
    main()
