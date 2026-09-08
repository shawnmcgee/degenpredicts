# DegenPredicts

Spread and total predictions for every FBS college football game, refreshed daily. Runs
entirely on GitHub — Actions is the scheduler, the repo is the database, Pages is the site.
No server, no manual uploads, no hosting bill.

```
.github/workflows/
  cfb-predict.yml   daily 9:00am ET → lines + schedule → models → data/cfb/picks.csv → docs/
  cfb-grade.yml     daily 7:00am ET → finals → grade → results.csv, metrics.json
  cfb-train.yml     Tuesdays        → refit models
  nfl-predict.yml   daily 9:30am ET → same, into data/nfl/ → docs/nfl/
  nfl-grade.yml     daily 7:30am ET
  nfl-train.yml     Tuesdays
  test.yml          on push         → offline end-to-end tests, both sports
cfb/     college football pipeline (live now)
nfl/     NFL pipeline (live now)
ncaab/   basketball pipeline (built, dormant until November)
core/    shared HTTP session and settings
```

Two sports, one Pages deployment: college football serves `docs/index.html`, the NFL serves
`docs/nfl/index.html`. Neither overwrites the other.

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

## NFL

`nfl/` is the same architecture pointed at a different sport. Most of it ports over unchanged —
Actions as scheduler, repo as database, Pages as frontend; online ratings replayed
chronologically with a leak-free test in CI; four XGBoost models with shrinkage fitted on a
holdout; EV and Kelly output. Four things genuinely differ.

### 1. Data source: nflverse, no key, no quota

There is no CFBD equivalent, and nflverse is better than one for this design. It is static CSVs
served from GitHub, so there is no secret to rotate and no monthly call budget.

| Feed | What it gives | Size |
|---|---|---|
| `nfldata/games.csv` | schedule and results 1999+, **closing spread, total and moneyline**, rest days, divisional flag, roof, surface, temp, wind, stadium, starting QBs — including for games not yet played | ~2 MB, one request |
| `stats_team_week_<season>.csv` | team-week offensive EPA and play counts → opponent-adjusted season EPA/play | ~220 KB/season |
| `snap_counts` + `rosters` + `players` | snap-weighted roster continuity | ~3.4 MB/season, plus a 7 MB crosswalk fetched once |

Because the schedule file carries historical closing lines, the market-aware models train from
day one rather than after a season of self-logging — the same property that made CFBD work.

**Set up with `Actions → NFL retrain → Run workflow`.** No secrets required. The first run
backfills EPA and continuity (a few minutes); afterwards it refreshes two seasons.

### 2. Feature analogues, and the NFL-specific ones

SP+ becomes **opponent-adjusted prior-season EPA per play**, offence and defence, computed here
from team-week data and joined only from season−1 so it cannot leak. Returning production
becomes **snap-weighted roster continuity** — the share of last season's snaps still on the
roster, ~0.74 offence and ~0.65 defence league-wide in a typical offseason.

Then the ones with no college analogue worth modelling, which is where this actually diverges:

- **Quarterback** — `qb_new` flags a starter who did not start the team's last game, `qb_starts`
  counts prior starts. Both replayed chronologically from the schedule, so they are leak-free
  and available on the board. In the first real training run these were the highest-weighted
  non-market features in the margin model.
- **Rest** — rest differential, byes, and short weeks (Thursday off a Sunday game).
- **Travel** — great-circle distance and time-zone crossings, measured from the stadium each
  team actually played its home games in that season. Relocations, the Rams' Coliseum years and
  London "home" games all come out right with no special cases.
- **Venue and weather** — dome vs outdoors, temperature, wind. Indoor games are stated as 68°F
  and no wind rather than left blank.
- **Divisional** games, which are played twice a year between teams that know each other.

Two things the college model uses are deliberately **absent**: line movement (nflverse publishes
only the closing number, so training on a move feature that is zero historically and non-zero
live would be a train/serve mismatch) and book count (there is one number, not a panel). The
de-vigged **moneyline** is added instead — a second, independent market view of the same game.

### 3. Market efficiency: expect no edge, and check that you are told so

NFL closing lines are the sharpest market in sports. The thresholds, the Kelly fraction and the
Kalshi guards are all strictly tighter than the college pipeline's, and a test asserts they stay
that way. Here is what the first real training run produced (2010–2025, six-season walk-forward,
1,693 out-of-sample games):

```json
"margin_market": {
  "mae_model": 9.88,
  "mae_market_baseline": 9.77,   ← the closing line's own error
  "ats_rate": 52.0,
  "ats_stderr": 1.23,            ← 1 s.e. at this sample size
  "beats_market": false,
  "shrink": 0.2
}
```

**The model does not beat the closing line, and no disagreement bucket clears break-even by two
standard errors.** That is the expected result, not a bug, and the site says so on the page
rather than burying it in JSON. `shrink` fitting to 0.2 is the same finding stated differently:
the published number is the line nudged 20% toward the model.

#### Read `market_softness` with the correction, not the raw z-score

Every segment carries `vs_break_even_se`, and reading that number on its own is the single
easiest way to talk yourself into a bet. Across twenty segments the chance at least one clears
|z| ≥ 2 by pure chance is about 60%. So each model's `significance` block states how many looks
were taken and what |z| a segment actually needs:

```json
"significance": {
  "comparisons": 20,
  "z_required": 3.02,
  "verdict": "no segment survives correction for the number of looks taken"
}
```

Each segment also carries `season_cover_pct` and `seasons_above_break_even`, because per-season
stability is what settles it. A real edge shows up in most seasons; a fluke is two bad years and
four ordinary ones.

**A worked example, because this table nearly produced a bad model change.** The
`away better rested` segment came back at 43.4% cover, −2.2 s.e. — the largest deviation in the
file, and it reads like the model over-crediting away rest. It is not:

- the model's point predictions are the **most accurate** of any rest segment there (mean error
  −0.07, against +0.60 for even rest and +1.03 for home-rested);
- it barely deviates from the line at all (−0.38 vs −0.19 for even rest) and takes the home side
  47.4% of the time against 47.8% — a 0.2-point deviation cannot move a cover rate 11 points;
- it is **not monotonic** — the most extreme away-rest bucket (≥6 days) covers 53.7%, and the
  whole effect sits in one middle bucket;
- it is **not stable** — 53.6 / 57.1 / 50.0 / 34.5 / 33.3 / 38.7 across six seasons of ~30 games.

The market's own error there is −0.45 ± 0.92. The market is slightly off, the model correctly
followed it, and what remains is a coin flip. Fitting the model to that segment would have been
overfitting to noise, so the report was changed instead of the model.

Default thresholds (`DEGEN_SPREAD_EDGE=5.0`, `DEGEN_TOTAL_EDGE=6.0`) sit **above every bucket
that showed anything**, so almost nothing is flagged as a play. Deliberate. Every game is still
predicted, graded and CLV-tracked whether or not it is staked, so evidence accumulates without
money at risk. Lower them only when `ats_by_disagreement` gives you a reason.

### 4. Sample size and calibration

272 games a season against college's ~800. So: the walk-forward pools **six** seasons rather
than four; trees are shallower and more regularised; between-season rating carry-over is **0.55**
against college's 0.72, because NFL rosters churn harder; and home-field is **1.7 points**, not
2.5. Stakes are **eighth-Kelly**, not quarter — Kelly sizing assumes you know your edge, and
against this market you do not. Weeks **1–3** are flagged and unstaked (college unstakes 1–2),
because NFL roster turnover makes preseason ratings untrustworthy for longer.

Two era corrections that would otherwise be silent:

- **2009 is loaded as a warm-up season.** It advances the ratings and the quarterback tracker
  but emits no training rows. Without it, every team enters Week 1 of 2010 rated identically —
  all 16 games shared one `h_margin` value — and the whole first season runs on ratings that
  started from zero.
- **2020 carries a `no_crowd` flag, and home-field is suppressed in the rating replay for it.**
  Mean home margin that season was **+0.14**, against +2.25 across 2010–2019 and +2.06 across
  2021–2025. The season is kept — it is 269 games of real football — but crediting it a normal
  home edge would push every 2020 home team's rating down by an advantage that did not exist.

Everything the model sees is a rate or a per-game figure, so the 2021 move from a 16- to a
17-game season cannot leak in through a season total: EPA is per play, continuity is a ratio,
form is a mean, and week numbers are normalised.

### Isolation over shared code

Each sport is a **self-contained package** — its own ratings engine, config, sources, HTTP
session, tests and workflows. `nfl/` imports nothing from `cfb/`, `ncaab/` or `core/`, and a
test walks the package and fails on any cross-sport import.

This is a deliberate choice against factoring the Elo engine and EV/Kelly math into a shared
package. The duplication is real and measurable — `cfb/ratings.py` and `nfl/ratings.py` differ
by 15 logic lines, 12 of which are tuning constants — but the sports are independently
scheduled jobs committing to `main` on their own crons, and the thing worth optimising is
blast radius, not line count. One bad edit to a shared rating engine takes down college
football, the NFL and basketball at once; the same edit in `nfl/ratings.py` takes down one
sport. `ncaab/` already had its own `http.py`, so this follows the repo's existing grain
rather than cutting against it.

The same rule applies to the tests: `tests/test_nfl.py` asserts absolute thresholds rather
than comparing against `cfb.config`, so retuning the college pipeline cannot fail the NFL
suite. CI runs one job per sport, so a red check names the sport that broke.

### The guardrails that earned their place

Front-loaded because NFL team abbreviations are a minefield, and each of these caught something
real during the build:

- **`nfl/teams.py` is the only place a team code is translated**, and it *raises* on anything
  unmapped rather than passing it through — a silently wrong code splits or merges a franchise's
  rating history and nothing downstream can detect it. It fired on its first run: nflverse
  spells St. Louis `STL` in the schedule and `SL` in the roster files. Relocations
  (OAK→LV, SD→LAC, STL→LA) collapse so a franchise's rating carries across the move. CI asserts
  every code and every stadium in the committed data resolves.
- **The spread sign is flipped exactly once.** nflverse quotes `+3 = home favoured`; this
  pipeline uses CFBD's `−3 = home favoured` everywhere downstream. Getting it backwards raises
  nothing — it just picks the wrong side of every game and lands ATS near 48%.
- **Week numbers are normalised.** The regular season went from 17 weeks to 18 in 2021, so week
  18 is the Wild Card round in 2019 and a regular-season game in 2022. The model sees a fraction
  of that season's regular season plus a separate playoff round.
- **Time-zone shift wraps across the date line.** A Los Angeles team playing in Melbourne shifts
  6 hours, not the 18 that plain offset subtraction reports.
- **Roster continuity joins through the GSIS crosswalk.** Joining snap counts to rosters on
  `pfr_id` looks like it works and drops nearly every offensive lineman, reporting league-wide
  offensive continuity of 0.39 instead of 0.74.
- **Source modules read `config.NAME` at call time.** Importing paths by value froze them at
  import, which let the test suite train on committed production data while believing it was
  sandboxed.

`nfl/sources/kalshi.py` is wired for the exchange the same way the college module is, but its
series tickers could **not** be confirmed against the live API from the machine this was built
on. They are environment-overridable, the rules regexes accept several sport wordings, and every
Kalshi path degrades to "no exchange prices" instead of failing. Confirm them once with:

```bash
python -m nfl.sources.kalshi --discover
```

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
pytest -q tests/test_cfb.py tests/test_nfl.py   # offline, no network, no API key

python -m cfb.train --no-fetch
python -m cfb.predict --dry-run
python -m cfb.site && open docs/index.html

python -m nfl.train --no-fetch                  # nflverse needs no key at all
python -m nfl.predict --dry-run
python -m nfl.site && open docs/nfl/index.html
```

## Knobs (repo variables or env vars)

| Var | Default | Meaning |
|---|---|---|
| `DEGEN_TOTAL_EDGE` | 3.5 | min points of edge to publish a totals play |
| `DEGEN_SPREAD_EDGE` | 2.5 | same for spreads |
| `DEGEN_MIN_GAMES` | 2 (cfb) / 3 (nfl) | below this, picks are flagged early-season and not staked |
| `DEGEN_KELLY` | 0.25 | Kelly fraction |
| `DEGEN_BOARD_DAYS` | 7 | how far ahead to post games |
| `DEGEN_FIRST_SEASON` | 2015 (cfb) / 2010 (nfl) | earliest season to train on |
| `DEGEN_WARMUP_SEASONS` | 1 (nfl) | seasons loaded before the training window to warm the ratings up |
| `DEGEN_WALK_SEASONS` | 6 (nfl) | seasons pooled by the walk-forward evaluation |
| `DEGEN_NFL_HTTP_TIMEOUT` | 60 | NFL-only HTTP timeout; nflverse serves multi-MB files |
| `DEGEN_NFL_DOCS` | `docs/nfl` | where the NFL site is written |
| `DEGEN_KALSHI_ML_SERIES` | `KXNFLGAME` | Kalshi moneyline series (also `..._SPREAD_SERIES`, `..._TOTAL_SERIES`) |
| `DEGEN_SUPPORT_URL` | (unset) | Buy Me a Coffee link shown at the top; omit and the button hides |
| `DEGEN_SUPPORT_LABEL` | Buy me a coffee | button text |
| `DEGEN_THEME` | `ticker` | site look: `ticker`, `scoreboard`, `field` |
| `DEGEN_SUPPORT_URL` | (blank) | e.g. a Buy Me a Coffee link; blank hides the button |
| `DEGEN_SUPPORT_LABEL` | `Buy me a coffee` | button text |

### Reading the board

Each market shows a probability axis with two pins: the **dark pin is the model**, the **grey pin
is Kalshi's ask**. The bar between them is green when our number sits above the ask (the contract
is cheaper than we think it should be) and red when it doesn't.

Prices are multipliers rather than cents:

* **pays** — what a winning contract returns per dollar, taker fee included. A 58c ask with a 1c
  fee costs 59c and returns $1, so it pays 1.69x.
* **fair** — what our probability says it should return (1 / probability).

Buy when **pays** is larger than **fair**. Everything else on the card is context.
| `DEGEN_VENUE` | `kalshi_taker` | cost model: `sportsbook`, `kalshi_taker`, `kalshi_maker`, `exchange_zero` |
| `DEGEN_BREAK_EVEN` | (derived) | override the computed break-even win rate |

### Reading the board

Exchange quotes show as **payout multiples**, not cents:

* **pays** — what Kalshi's ask returns per dollar risked, after the taker fee. A 48c contract
  pays about 2.01x.
* **fair** — what it would have to pay to break even at the model's probability.
* **edge %** — pays against fair. Positive means the market is offering more than our number
  says it should. This is the same figure as ROI.

The rows always show the strike nearest the sportsbook line, tagged `play` (cleared every
guard), `no play` (real quote, edge too small) or `illiquid` (quote too wide or untraded).

Games with no scheduled kickoff carry CFBD's `startTimeTBD` flag and render as "Time TBD"
rather than the placeholder time CFBD stamps on them.

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
