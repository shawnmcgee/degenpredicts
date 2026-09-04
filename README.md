# DegenPredicts

Spread and total predictions for every FBS college football game, refreshed daily. Runs
entirely on GitHub — Actions is the scheduler, the repo is the database, Pages is the site.
No server, no manual uploads, no hosting bill.

```
.github/workflows/
  cfb-predict.yml   daily 9am ET   → lines + schedule → models → data/cfb/picks.csv → docs/
  cfb-grade.yml     daily 7am ET   → finals → grade → results.csv, metrics.json
  cfb-train.yml     Tuesdays       → refit models
  test.yml          on push        → offline end-to-end test
cfb/     college football pipeline (live now)
ncaab/   basketball pipeline (built, dormant until November)
```

## Setup — 30 minutes

### 1. Create the repo

```bash
git clone https://github.com/<you>/degenpredicts.git
cd degenpredicts
# copy this bundle in
git add . && git commit -m "initial" && git push
```

### 2. Add secrets

**Settings → Secrets and variables → Actions → New repository secret**

| Secret | Required | Notes |
|---|---|---|
| `CFBD_API_KEY` | yes | you already have one |
| `ODDS_API_KEY` | optional | live book prices for EV/Kelly. Without it the pipeline uses CFBD numbers and assumes −110. **Rotate the old key** (`0a4e0558…`) — it's in your old repo. |

### 3. Turn on Pages

**Settings → Pages → Deploy from a branch → `main` / `/docs`** → Save.
Site appears at `https://<you>.github.io/degenpredicts/`.

### 4. First training run

**Actions → CFB retrain → Run workflow.** This pulls 2015–2026 games, lines, SP+ and
returning production from CFBD (about 40 API calls out of your 1,000/month), then fits four
models. Roughly 5–10 minutes — far faster than a scraper, because CFBD serves whole seasons in
one call.

When it's done, `git pull` and open `data/cfb/models/meta.json`.

### 5. First picks

**Actions → CFB picks → Run workflow.** Then check your Pages URL.

Locally, if you'd rather see it before it's public:

```bash
pip install -r requirements.txt
export CFBD_API_KEY=...
python -m cfb.predict --dry-run
```

### 6. Let it run

The crons are already set. Predict runs every morning, grade every morning, retrain Tuesdays.

---

## About this weekend

Week 1 runs Thursday Sept 3 through Monday Sept 7. If you set this up today you'll have a
board for the Saturday slate.

**Every week-1 pick will be flagged `early season` and staked at zero.** That's deliberate, not
a bug. No team has played a snap this season, so the model is running on last year's ratings,
last year's SP+, and returning production. The numbers are shown so you can see them and track
them; they just aren't staked. The flag clears once teams have `DEGEN_MIN_GAMES` games in the
book (default 2, so it lifts in week 3).

If you want week 1–2 staked anyway, set repo variable `DEGEN_MIN_GAMES` to `0`. I'd let the
first two weeks grade themselves first.

---

## What the model does

Ratings are replayed game by game (`cfb/ratings.py`): an Elo-style margin rating with a
soft-capped margin so 63–0 doesn't overstate the winner, plus scoring offence and defence
ratings. Season-to-season carry-over is 0.72, reflecting real roster churn.

`cfb/features.py` builds features chronologically — a feature on game N only ever saw games
1..N-1. This is verified in CI by a test that rebuilds from a truncated schedule and asserts
identical output. Two preseason-known inputs carry the early weeks:

- **prior-season SP+** (joined only from season−1, so it can't leak)
- **returning production** (share of last year's PPA coming back — published in the offseason)

Four models: totals and margin, each with and without market features. Because CFBD serves
historical lines, the market-aware models train from day one. When a line exists, the model
predicts *against* it, then the published number is the line shrunk toward the model by a
weight fitted on the holdout season — not a guess.

Per game you get: prediction, edge, win probability (from the fitted residual sigma), EV
against the de-vigged price, and a quarter-Kelly stake.

### The number that decides whether to bet any of this

Evaluation is **walk-forward**: train on everything before season S, predict S, repeat for the
last four *complete* seasons, then pool. The in-progress season is never the holdout — early-
season games are unrepresentative, and calibrating on a few dozen of them produces nonsense.

`data/cfb/models/meta.json`:

```json
"margin_market": {
  "test_seasons": [2022, 2023, 2024, 2025],
  "n_test_total": 3400,
  "mae_model": 12.10,
  "mae_market_baseline": 12.05,   ← the line's own error
  "ats_rate": 50.4,
  "ats_stderr": 0.86,             ← 1 s.e. on a coin flip at this sample size
  "beats_market": false,
  "shrink": 0.15,
  "per_season": [ ... ]
}
```

Read it in this order:

1. **`beats_market`** — if false, the model does not beat the closing line. Don't bet it.
2. **`ats_rate` vs `ats_stderr`** — you need roughly `50 + 3×stderr` before a rate is
   distinguishable from luck, and `52.4` to break even at −110. An impressive rate on a few
   hundred games is noise; that is the normal state of affairs, not a bug.
3. **`per_season`** — a model that only works in one season didn't work.

`shrink` is how far off the closing line the published number moves. It is fitted on the pooled
walk-forward residuals, clamped by `DEGEN_SHRINK_CAP` (default 0.6), and forced back to the
default if fewer than 500 decided games back it. Shrink near 0 means "the line is better than
you, publish the line" — which is the correct answer more often than not.

Watch **CLV** (on the site, and in `results.csv`). If the line moves toward your side after you
pick, that's real evidence of an edge and it shows up in weeks rather than the seasons a win
rate would need.

---

## When to cancel PythonAnywhere

Cancel once all four are true:

- [ ] `CFB retrain` finished green and `data/cfb/models/meta.json` is in the repo
- [ ] `CFB picks` produced a board you're happy with
- [ ] Your Pages URL loads
- [ ] You've exported anything you want from PythonAnywhere (old prediction CSVs are the only
      thing with real value — they're historical lines)

That's the same afternoon you set this up. **Nothing here runs on PythonAnywhere.** Actions is
free for public repos and includes 2,000 minutes/month on free private accounts; these
workflows use roughly 40 minutes/month.

---

## Basketball

`ncaab/` is a complete parallel pipeline, already written and tested, using ncaa-api for
scores, The Odds API for lines, and Barttorvik daily snapshots for tempo/efficiency. It has no
scheduled workflows yet. In early November, add workflows mirroring the `cfb-*.yml` files with
`python -m ncaab.train` / `.predict` / `.grade`, and run the training job once to back-fill.

---

## Local development

```bash
pip install -r requirements.txt pytest
pytest -q tests/test_cfb.py     # offline, no network, no API key
python -m cfb.train --no-fetch
python -m cfb.predict --dry-run
python -m cfb.site && open docs/index.html
```

## Knobs (repo variables or env vars)

| Var | Default | Meaning |
|---|---|---|
| `DEGEN_TOTAL_EDGE` | 3.5 | min points of edge to publish a totals play |
| `DEGEN_SPREAD_EDGE` | 2.5 | same for spreads |
| `DEGEN_MIN_GAMES` | 2 | below this, picks are flagged early-season and not staked |
| `DEGEN_KELLY` | 0.25 | Kelly fraction |
| `DEGEN_BOARD_DAYS` | 7 | how far ahead to post games |
| `DEGEN_FIRST_SEASON` | 2015 | earliest season to train on |
| `DEGEN_VENUE` | `kalshi_taker` | cost model: `sportsbook`, `kalshi_taker`, `kalshi_maker`, `exchange_zero` |
| `DEGEN_BREAK_EVEN` | (derived) | override the computed break-even win rate |

### Kalshi markets

Three series are wired in, all confirmed against live payloads:

| Series | Shape | Model input |
|---|---|---|
| `KXNCAAFGAME` | "&lt;team&gt; wins" | `P(margin > 0)` |
| `KXNCAAFSPREAD` | "&lt;team&gt; wins by over X" — **ladder** | `P(margin > X)`, mirrored for the away side |
| `KXNCAAFTOTAL` | "Over X points scored" — **ladder** | `P(total > X)` |

Spreads and totals list many strikes per game, so every rung is priced against the model's
distribution and the best tradeable positive-EV rung is kept. Quarter and half series
(`KXNCAAF1HSPREAD` etc.) exist and are likely softer, but the models predict full-game results,
so those would need their own training target.

**Why the guards are strict.** Picking the best-EV rung from a ~20-rung ladder is a *maximum
over noisy estimates*: it returns a positive number almost every time, even from a model with
no edge. The first live run produced 40 total and 34 spread "picks" out of 106 games, which is
not plausible and was the winner's curse plus a normal approximation overstating tail
probabilities. Four guards now apply, all tunable:

| Guard | Default | Why |
|---|---|---|
| `DEGEN_KALSHI_MIN_EV` | 0.05 | 5c after fees, so marginal noise doesn't clear the bar |
| `DEGEN_KALSHI_PROB_MIN/MAX` | 0.20 / 0.80 | football margins cluster on 3/7/10/14 and have thinner tails than a Gaussian, so tail probabilities are unreliable |
| `DEGEN_KALSHI_MAX_GAP` | 7.0 | ignore rungs far from the sportsbook number — the tails by another name |
| thin-data flag | — | no exchange pick on a game the model itself passes on |

`kt_rungs` / `ks_rungs` record how many rungs survived the guards, so you can see the breadth of
the search that produced a pick. A pick chosen from 15 candidates deserves more scepticism than
one chosen from 2.

Two more guards worth knowing about:

* **Liquidity.** EV is computed against the **ask**, never the mid, and a pick only surfaces if
  the quote passes width/size/volume gates. A live sample quoted 0.07 x 0.90 with zero volume;
  an "edge" against that ask is imaginary.
* **Monotonicity.** P(win by >8.5) can never exceed P(win by >4.5). The live spread sample
  violated this (90c vs 17c). Games whose ladders contradict themselves are flagged on the site,
  because at least one quote is stale.

`ks_book_gap` / `kt_book_gap` show how far Kalshi's strike sits from the sportsbook number. A
stale exchange strike is a far likelier source of profit than the model outsmarting Bovada —
watch that column more than the model's own disagreement.

### Venue matters more than any model tweak

Break-even win rate by venue, for a contract priced near 50c:

| Venue | Break-even |
|---|---|
| Sportsbook at −110 | 52.38% |
| Kalshi taker fee (0.07 formula) | 51.75% |
| Kalshi maker fee (quarter rate, some series) | 50.44% |
| Zero-fee exchange | 50.00% |

Kalshi's published taker fee is `roundup(0.07 × contracts × P × (1−P))`, which peaks at 1.75c
per contract at 50c and falls toward the wings. **Verify the current schedule at
kalshi.com/fee-schedule before sizing anything** — they revise it periodically.

This is not a rounding difference. A 51.9% cover rate loses ~0.9% at −110 and gains ~0.3% at
Kalshi taker fees. `train.py` computes ROI and "standard errors above break-even" at whatever
`DEGEN_VENUE` you set, so the tables tell you about *your* costs, not a sportsbook's.

### Finding soft spots

`market_softness` in `meta.json` splits pooled out-of-sample results by proxies for how much
attention a game gets — number of books posting, P4/G5 tier, week, whether the line moved since
open, favourite size — plus a `soft_and_loud` cross of "quiet game AND big model disagreement".
Each row reports `vs_break_even_se`: standard errors above your venue's break-even. Treat
anything under +2 as unproven, and remember you're looking at ~20 segments, so the best one
being +2 is roughly what chance alone produces.

`DEGEN_TOTAL_EDGE` / `DEGEN_SPREAD_EDGE` apply to the model's **raw disagreement with the
line** (`|model − line|`), not to the shrunk display edge. Set them from the
`ats_by_disagreement` table in `models/meta.json`: pick the smallest bucket whose `cover_pct`
clears 52.4 by more than about two standard errors, on at least a few hundred games. If no
bucket does, no threshold makes this profitable and the right setting is "don't bet".

---

## An honest note

College football spread and total markets are efficient. A public-data model is unlikely to
beat closing numbers by much, and the realistic outcome is a thin edge in soft spots — early
season, Group of Five, games with few books posting — or no edge at all. A dozen weeks of
results cannot reliably distinguish those. That's why CLV is tracked front and centre. Size
small, treat the units column as bookkeeping, and stop if CLV is negative after a few hundred
picks.
