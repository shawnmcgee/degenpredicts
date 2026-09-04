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

`data/cfb/models/meta.json`:

```json
"margin_market": {
  "mae_model": 10.2,
  "mae_market_baseline": 10.1,   ← the line's own error
  "ats_rate": 50.9,              ← how often the model's side covered
  "shrink": 0.25
}
```

If `mae_model` isn't below `mae_market_baseline`, the model has no edge over the book and
`ats_rate` will hover near 50. That is the *expected* result for a first pass, and it's not a
failure of the code — it means don't bet it yet. Break-even at −110 is 52.4%.

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

After four or five weeks, read the **accuracy by edge size** table on the site and set the two
edge minimums to the smallest bucket clearing 52.4% on a real sample.

---

## An honest note

College football spread and total markets are efficient. A public-data model is unlikely to
beat closing numbers by much, and the realistic outcome is a thin edge in soft spots — early
season, Group of Five, games with few books posting — or no edge at all. A dozen weeks of
results cannot reliably distinguish those. That's why CLV is tracked front and centre. Size
small, treat the units column as bookkeeping, and stop if CLV is negative after a few hundred
picks.
