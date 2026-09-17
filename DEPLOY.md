# Running the paper harnesses on Railway

Both harnesses are read-only and trade fake money. There is no code path from either process to a
live order — `ReadOnlyKalshi` raises on any verb but GET, and two tests pin that.

Project: **mirofish-harnesses** (`0afe432f-3fe4-495d-8de4-6408329c839f`), deliberately a separate
Railway project from anything else on the account.

## What runs

| Harness | What it measures |
|---|---|
| `run_paper.py` | 15-minute crypto: the unresolved 99c final-seconds favourite loss rate |
| `run_paper_mlb.py` | MLB in-game: the frozen Monte Carlo model vs the market, threshold 5c |

`run_harnesses.py` supervises both and restarts either on exit with backoff, so a crash in one cannot
silently stop the other.

## Step 1 — set the credentials (you, not Claude)

Claude does not put API keys into hosting dashboards. Run these yourself:

```bash
railway variables --set "KALSHI_API_KEY_ID=<your key id>"
railway variables --set "KALSHI_PRIVATE_KEY=$(cat ~/.kalshi/mirofish-private.pem)"
```

The PEM's newlines matter. If you paste it into the Railway dashboard instead and it arrives with
literal `\n`, the code restores them — but the `$(cat ...)` form above avoids the problem entirely.

## Step 2 — deploy

```bash
railway up --detach
```

## Step 3 — keep the ledgers across redeploys

Without a volume, every redeploy wipes the measurement and the sample restarts at zero.

```bash
railway volume add --mount-path /data
railway up --detach
```

`PAPER_DATA_DIR` is already set to `/data` in the Dockerfile.

## Step 4 — stop the local copies

Two instances writing separate ledgers means two half-samples instead of one whole one.

```bash
pkill -f run_paper.py
pkill -f run_paper_mlb.py
```

## Checking on it

```bash
railway logs                      # live output from both harnesses
railway run cat /data/stats.json  # crypto: settled, losses, break-even, exact p
```

## Reading the result later

Both harnesses log every observation, not just the trades — `mlb_observations.jsonl` and
`quotes.jsonl`. That means the full threshold curve can be checked afterwards without re-running
anything, and without the temptation to pick the threshold that flatters the result.

## Notes

- `railway.json` is Config-as-Code, which Railway deprecates on 2026-12-01. Migrate with
  `railway config migrate` before then.
- The MLB model in `models/mlb_model.json` is frozen on purpose. Refitting it mid-test would make the
  forward test meaningless. `fit_mlb_model.py` regenerates it if you ever deliberately want to.
