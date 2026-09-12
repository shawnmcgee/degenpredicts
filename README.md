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
  nfl-kalshi-discover.yml  manual  → confirm Kalshi's series tickers, runs from a phone
  epl-predict.yml   daily 8:00am UTC → fixtures + prices → models → data/epl/picks.csv → docs/
  epl-grade.yml     daily 6:30am UTC → results → grade → results.csv, metrics.json
  epl-train.yml     Tuesdays        → refit models
  epl-source-check.yml     manual  → confirm football-data's odds columns, runs from a phone
  test.yml          on push         → offline tests, one job per sport plus the landing page
cfb/     college football pipeline (live now)
nfl/     NFL pipeline (live now)
epl/     Premier League pipeline (live now)
ncaab/   basketball pipeline (built, dormant until November)
core/    sport-neutral bits only: the landing page, shared settings
```

One Pages deployment, one folder per sport. `docs/index.html` is a **chooser** rendered by
`core.landing`; each board lives in its own subfolder (`docs/cfb/`, `docs/nfl/`) so no sport
can overwrite another and adding one needs no coordination.

```
docs/
  index.html      ← the chooser: a card per sport, with week, board size and season record
  cfb/index.html  ← college football board
  nfl/index.html  ← NFL board
  epl/index.html  ← Premier League board
```

`core/landing.py` imports nothing from `cfb`, `nfl` or `ncaab` — it reads the files those
pipelines have already published and renders a card for each. A sport that has published
nothing simply has no card, so `ncaab/` will appear on its own the first time it runs. Every
publishing workflow rebuilds the chooser after its own board, so the front page refreshes
whenever any sport does and self-heals if a run is skipped.

Note the college football board moved from `docs/` to `docs/cfb/` when the chooser was added,
so its `picks.csv`, `results.csv` and `metrics.json` moved with it.

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

"Staked at zero" now means it. Until recently `_strength` was computed *after* the Kelly
numbers were already in the frame and nothing ever zeroed them, so week 1 of 2026 graded 105
games that were all `pass` or `thin` and still carried 13.01 units of totals stake — and the
front page quoted a 25% ROI on bets the system had declined. Stakes are now gated on strength
in all three sports, and the landing card shows units only where something was actually
staked; where nothing cleared the threshold it shows the model's side record, labelled
unstaked. Those 105 rows have had their stakes and units zeroed retroactively, because they
were the output of a bug rather than a decision.

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

CLV is measured against `first_seen_spread` / `first_seen_total` — the number we *first*
published for a game, which survives every later overwrite of that row — and against a
closing line refetched at grade time. Both halves matter: `cfb.grade` used to read the close
out of the same cached snapshot the pick was built from, which is subtraction of a number
from itself, and all 105 graded games came back with CLV of exactly 0.00 and zero variance.
A test now fails if graded CLV has no variance across a synthetic line move. The 105 historic
rows carry CLV of `null` rather than `0.00`: it is genuinely unrecoverable for them, and
unknown is the honest value.

---

## What is in `games.csv`, and what the board publishes

`cfb/sources/cfbd.py` asks CFBD for FBS games. It asks with `classification` *and* `division`,
because the parameter was renamed and unknown parameters are silently ignored rather than
rejected — which is how a request for FBS quietly returned 699 teams in 2025, Kenyon, Sewanee
and Wayland Baptist among them.

So the question is not trusted. `features.fbs_membership` resolves the FBS roster for each
season from `sp_ratings.csv` (SP+ only rates FBS, so its roster *is* the membership list) and
from a `classification` column now carried on each game, and `cfbd.update_games` warns when a
season falls outside 700–1200 games or 110–160 teams.

### Three populations, not two

"Non-FBS rows" is not one thing, and the distinction is the whole answer. Of the 26,381
completed training rows:

| Category | Rows | Share | What it is |
| --- | ---: | ---: | --- |
| FBS vs FBS | 8,354 | 31.7% | what the board publishes |
| **FBS vs FCS** | **1,245** | **4.7%** | the mixed games — ~120 a season |
| non-FBS vs non-FBS | 16,782 | 63.6% | FCS-vs-FCS and below, including D-II and D-III |

The 68% that is not FBS-vs-FBS is overwhelmingly the third row, not the second. Since 2021 the
broken filter has pulled in ~2,900 of those a season against ~120 real mixed games.

### What the flag gates, and why

Deliberately asymmetric, and measured rather than assumed. Walk-forward over 2022–25, five
seeds, scored **only on FBS-vs-FBS holdout games** — the population the board publishes. Seed
noise is ±0.01–0.03 MAE:

| Rating replay | Training rows | Margin MAE | ATS |
| --- | --- | ---: | ---: |
| all | all | **12.186** | **51.84%** |
| all | FBS + mixed | 12.202 | 51.32% |
| all | drop mixed | 12.232 | 51.02% |
| all | FBS only | 12.252 | 51.24% |
| FBS-involved | all | 12.228 | 50.70% |
| FBS only | FBS only | 12.290 | 49.94% |

- **The board is filtered.** 36 of one week's 85 published games were FBS vs FCS — Miami −56.5
  against Florida A&M, priced off a rating Florida A&M does not have, on a site that says it
  covers FBS. Those are gone.
- **Training and the rating replay are not filtered.** Going fully FBS-only costs **0.10 points
  of MAE and 1.9 points of ATS** — far outside seed noise. Splitting the two knobs shows where
  that comes from:
  - **The replay does most of the work.** Dropping FCS-vs-FCS from the replay alone costs 0.042
    MAE and 1.14 ATS. Those games are what *rate* the FCS teams, so a repeat visitor is a known
    quantity by the time it matters (2025: non-FBS mean margin −2.0 against FBS +10.8) rather
    than an average FBS team.
  - **The mixed rows punch far above their weight.** Removing just those 1,245 rows — 4.7% of
    the file — costs 0.046 MAE and 0.82 ATS, because they are the only rows that connect the
    two rating pools.

Dropping the non-FBS rows would have been a plausible-sounding change that made the model
worse, and dropping only the mixed games would have been worse still per row discarded.

### Why the mixed games stay off the board anyway

Not because they are unpredictable. Scored on mixed holdout games the model runs 50.72% ATS on
margin (±2.35, n=453) against a 51.75% break-even — no edge. But bucketed by line size it is
not the FCS that hurts:

| \|spread\| | FBS vs FBS | FBS vs FCS |
| --- | ---: | ---: |
| 7–14 | 55.0% | 68.6% (n=35) |
| 14–21 | 50.4% | 50.9% |
| 21–28 | 53.8% | 54.1% |
| 28+ | 51.2% | 50.2% |

At the same spread the two populations behave the same. The edge lives in the 7–14 and 21–28
buckets, and mixed games have a median spread of **30.5** against 8.5 for FBS games, with 56%
of them past 28 — they land almost entirely where nobody beats the number. Publishing them
would add ~120 games a season of board volume and no expected value.

### Reading the record

The board publishes FBS vs FBS today. It did not always, and 66 of the first 105 graded picks
were FBS vs FCS — so the headline record used to average two populations, one of which the
board will never offer again. It flattered the numbers badly: those 66 ran 63.6% on totals
against 51.3% for the comparable picks, reporting a 59% season.

`grade.tag_fbs` now flags every graded row and `grade.metrics` scores only the comparable ones,
reporting the rest under `off_board` rather than dropping them — they were real published picks,
and hiding a sample is how a record starts flattering itself. Once the pre-filter picks age out
of the current season the split goes away on its own.

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

### Confirming the Kalshi tickers

**Confirmed against the live API on 2026-09-08.** `KXNFLGAME` returned 64 open moneyline
markets, `KXNFLSPREAD` 404 ladder rungs and `KXNFLTOTAL` 304, and all three parsed. The
defaults are correct; they stay environment-overridable because a series can be renamed, and
every Kalshi path still degrades to "no exchange prices" rather than failing.

That run also corrected a real assumption. **Kalshi does not use full club names.** The spread
ladder names the favourite by city ("If Kansas City wins by more than 7.5 points"), the
moneyline subtitle uses a one-letter disambiguator ("New York G"), and the rules matchup uses a
short city form ("NY Giants vs LA Rams") — three styles in one payload. The matcher had been
built for "Kansas City Chiefs" and resolved none of the city-only or truncated forms, so every
spread rung would have failed to join, visible only as `matched 0/16` in a log. All three forms
are now mapped, with bare "New York" and "Los Angeles" refused rather than guessed, and the
tests use the real payloads verbatim.

It needs a network that can reach `api.elections.kalshi.com`. Market data is public — no
account, no key, nothing is written.

**From anywhere, including a phone: Actions → NFL Kalshi discover → Run workflow.** The result
is written to the job summary, which the GitHub mobile app renders as a page rather than as raw
logs, along with a table of what each outcome means and which repo variables to set. That
workflow is manual-only and read-only; it never commits.

Locally, if you prefer:

```bash
pip install -r requirements.txt
python -m nfl.sources.kalshi --discover
python -m nfl.sources.kalshi --discover --keyword FOOTBALL   # if NFL finds nothing
```

It prints every Kalshi series whose ticker or title mentions the NFL, then tries each of the
three configured tickers and shows the raw `rules_primary` text next to what the parser made of
it. Three outcomes:

| What you see | Meaning | What to do |
|---|---|---|
| `N open markets` and `parsed:` showing sensible teams, dates and strikes | Defaults are right | Nothing |
| The listing shows different NFL tickers, and the configured ones return `0 open markets` | Tickers differ | Set the repo variables below |
| `!! markets returned but none parsed` | Ticker is right, Kalshi reworded its rules | Update the regexes at the top of the module |
| Everything `0` plus connection warnings | The host is unreachable, not misconfigured | Retry from a network that can reach it |

The parsed line is the part worth reading carefully. Confirm the **home team is the second name**
in the matchup (Kalshi phrases these away-first) and that a multi-word club comes through whole —
"New York Giants", not "New". That exact truncation is a bug this module already had once.

If the tickers differ, set them under **Settings → Secrets and variables → Actions → Variables**:

| Variable | Default |
|---|---|
| `DEGEN_KALSHI_ML_SERIES` | `KXNFLGAME` |
| `DEGEN_KALSHI_SPREAD_SERIES` | `KXNFLSPREAD` |
| `DEGEN_KALSHI_TOTAL_SERIES` | `KXNFLTOTAL` |

`nfl-predict.yml` forwards all three, and leaving them unset keeps the defaults — an unset repo
variable arrives as an empty string, which `config._env` treats as unset rather than as a blank
ticker.

To confirm it took, run the picks job and look for `kalshi board: N sides` in the log instead of
`no markets returned`.

---

## Premier League

`epl/` is the same architecture pointed at association football — Actions as scheduler, repo as
database, Pages as frontend; ratings replayed chronologically with a leak-free test in CI; four
models with shrinkage fitted on a holdout; EV and Kelly output. But this is the first sport in
this repo that is not gridiron, and the **model at the centre of it is a different kind of
object**. That is the section worth reading.

### 1. Data source: a match archive on GitHub, no key, no quota

`football-data.co.uk` is the canonical archive for this sport and the pipeline was built on it.
**It refuses a GitHub Actions runner.** It is not down and it is not slow — it answers with
HTTP **503**, consistently, from every runner tried. The module is kept working and tested and
can be re-selected with `DEGEN_EPL_SOURCE=footballdata` if that ever changes; from a residential
IP it serves normally, so a manual fetch committed to the repo is also a viable path if you
ever want the true closing columns.

The primary is now a GitHub-hosted aggregate of those same archives, served from
`raw.githubusercontent.com` — the host nflverse is served from, which this repo has been using
reliably for months.

| What | Detail |
|---|---|
| One CSV, all divisions | E0, E1, E2, E3, EC, plus the continental leagues, 2000 → present |
| Prices | `OddHome/Draw/Away`, `Over25/Under25`, **`HandiSize/HandiHome/HandiAway`** |
| Size | ~45 MB, **one request** — replacing ~50 requests to a host that does not answer |
| Naming | football-data's own spelling, so `epl/teams.py` resolves every E0 and E1 club with no new aliases |
| Handicap sign | the same convention — negative means the home side gives goals — verified on the data, not assumed |

**Set up with `Actions → EPL retrain → Run workflow`.** No secrets required. The whole assemble
step — fetch, parse 8,670 matches, opponent-adjust two divisions — takes about 15 seconds.

**The one real downgrade, stated plainly.** football-data.co.uk publishes explicit *closing*
columns (`PSCH`, `AHCh`, `PC>2.5`); this aggregate does not distinguish opening from closing. So
`mae_market_baseline` is measured against a number that may be softer than the true close, and a
model that appears to beat it may only be beating an opening price. Every row is marked
`is_closing: false`, the training metadata records it per season, and any edge this source
appears to show deserves more suspicion than the same edge measured against a close.

#### The board comes from the live price feed

The archive holds played matches only, so it cannot supply fixtures. `epl/sources/odds.py`
builds the board from The Odds API instead — the only feed in the project that knows about a
match before it is played, and it carries the prices too, so one call supplies both halves.

That feed quotes no Asian handicap, so `ah_home` is empty on the board and the market's
supremacy is inverted out of the 1X2 price through the same scoreline model the predictions come
out of. It therefore lands on the model's own scale, and "we differ by 0.4 goals" stays a
meaningful sentence rather than a comparison of two different quantities.

**Without `ODDS_API_KEY` there is no board.** Training, grading and the ratings all work; there
is simply nothing to publish. The job says so and exits cleanly rather than failing.

### 2. The model is a scoreline distribution, not a margin

The other two sports predict a margin and read probabilities off a normal CDF. That is the
wrong instrument here, for three reasons that all point the same way:

- **The scale is tiny and discrete.** A match produces about 2.8 goals, so the whole
  distribution lives on the integers 0–5. A continuous density is not approximating anything.
- **The draw is a real outcome** — about 24% of matches, and its own market. `P(margin == 0)`
  is exactly zero under a Gaussian, so you would have to bolt on a fudge, and the fudge would
  be doing the most important work in the model.
- **Low scores are correlated.** Independent Poisson marginals understate 0-0 and 1-1 and
  overstate 1-0 and 0-1 — the four scorelines that decide whether a match is drawn.

So the two models predict **supremacy** (home goals minus away) and **total goals**, and
`epl/poisson.py` turns that pair into a Dixon–Coles bivariate Poisson grid over every scoreline.
Every market is then read off the same grid:

```
lambda_home = (total + supremacy) / 2      P(i,j) = Poisson(i;lh) * Poisson(j;la) * tau(i,j)
lambda_away = (total - supremacy) / 2      tau = the Dixon-Coles low-score correction
```

| Market | Read off the grid as |
|---|---|
| 1X2 | `sum(i>j)`, `sum(i==j)`, `sum(i<j)` |
| Asian handicap | `sum(i-j+h > 0)`, with quarter lines split across the two adjacent lines |
| Over/under | `sum(i+j > line)`, with a real push leg on whole-number lines |
| Both teams to score | `sum(i>0 and j>0)` |

The payoff is consistency: the 1X2 price, the handicap and the total **cannot contradict each
other**, so when our number disagrees with the book in the same direction across all three,
that is one piece of evidence rather than three.

At the default settings the grid reproduces the league from first principles. Averaged over
3,040 real fixtures, the replayed ratings imply **42.9% / 23.7% / 33.4%** home/draw/away against
those matches' actual **44.3% / 22.9% / 32.8%**.

Note that this has to be averaged over real fixtures, not read off one "average match": a single
0.35-goal supremacy gives 44.9 / 26.0 / 29.0, which overstates draws by two points because one
representative fixture ignores the spread of team strengths across the league. The draw rate is
a property of the distribution of mismatches, not of the average mismatch.

### 3. What actually differs about football

**Promotion and relegation.** Three of twenty clubs are replaced every season and have no
top-flight history at all. This has no NFL analogue whatsoever — the same 32 franchises come
back every year. Two things follow:

- A club entering the league is **seeded at a promoted-club prior** (−0.22 attack, +0.20
  defence in log-goals, about 0.4 goals a match below average), not at league average. Starting
  them at average hands them roughly a third of a goal a match they have not earned, which is
  larger than any feature in the model.
- The **division below is fetched too**, and a promoted club's Championship season is carried
  up scaled by a division gap, flagged `promoted` so the model can learn how much to trust it
  rather than being told.

**European commitments.** A club playing Thursday in the Europa League and Sunday in the league
is carrying a real load, but the league schedule cannot see those midweek matches. `in_europe`
is derived from prior-season finishing position — legitimately known before a ball is kicked,
and the honest version of a fixture-congestion feature given what this data contains.

**The behind-closed-doors window is a date range, not a season.** 2019-20 was played in front of
full grounds until March and empty from June, so flagging whole seasons would mislabel 288
matches that had crowds. Measured across 2015-16 to 2024-25, home supremacy ran **+0.302 goals**
in normal seasons and **+0.161** in the empty ones — a natural experiment large enough that
crediting those matches a normal home edge would push every home club's rating down by an
advantage that was not there.

**Matchweek is deliberately not a feature.** football-data publishes no round number, and the
one you would reconstruct from dates is a lie: postponements and European fixtures put two clubs
three matches apart in the same calendar week. The model sees `h_games` / `a_games` — matches
actually played by that club — which is the honest version of the same question.

**Derbies and travel.** England has no time zones, so the NFL's body-clock term does not exist.
What remains is a genuine north–south haul (Newcastle to Bournemouth is 470 km) and, at the
other end, the local derby, which is detected geometrically: two grounds within 20 km.

Prior-season strength is the SP+ / prior-EPA analogue, opponent-adjusted by multiplicative
fixed point, in **two flavours: goals and shots on target**. The shot version carries the
preseason weight, for the reason every football analytics department rediscovered a decade ago —
over 38 matches, shot volume predicts next season's goals better than this season's goals do.

### 4. Expect no edge, and check that you are told so

The closing Asian handicap on a Premier League match is, by most measures, the most efficient
price in world sport: enormous limits, sharp money, and twenty clubs that thousands of people
model full-time. The realistic outcome is `beats_market: false` and a fitted `shrink` near zero,
and the site says so on the page rather than burying it in JSON.

Here is what the first real training run produced — 8,000 matches from 2005 to 2026, evaluated
walk-forward across five seasons and 1,900 out-of-sample matches:

```json
"sup_market": {
  "mae_model": 1.309,
  "mae_market_baseline": 1.295,   ← the market's own error, in goals
  "cover_rate": 50.3,
  "cover_stderr": 1.19,
  "break_even_pct": 51.28,
  "beats_market": false,
  "shrink": 0.00                  ← publish the market's number; the model adds nothing
}
```

**The model does not beat the market, no disagreement bucket clears break-even, and no segment
survives correction for the twenty looks taken** (a segment needs |z| ≥ 3.02, and the best one
manages −0.54). The fitted shrink of exactly **0.00** is the same finding stated as bluntly as
the fitter can state it: every published number is the market's number.

Note the largest disagreements are the *worst* bucket, at 47.3% — when this model departs
furthest from the price, it is most often simply wrong. That is the ordinary result for a
public-data model against a mature football market, and it is why the default thresholds sit
above every bucket in the table.

**But point error is not the whole test here**, and that is genuinely new in this repo. Two
models can have identical mean absolute error on supremacy while disagreeing completely about
how often matches are drawn. So `models/meta.json` carries a `probability` block that scores the
implied 1X2 probabilities directly. From the same run:

```json
"probability": {
  "log_loss_model": 0.9605,
  "log_loss_market": 0.9598,     ← the de-vigged market's own log loss
  "log_loss_edge": -0.0007,      ← negative means the market's probabilities were better
  "beats_market_log_loss": false,
  "draw": { "model_pct": 23.2, "market_pct": 23.3, "actual_pct": 23.9 }
}
```

A log-loss gap of 0.0007 is a dead heat — the model's probabilities are about as good as the
market's, and not better. **The draw line is the one genuinely encouraging number in this
report**: 23.2% predicted against 23.9% actual, within a point of both reality and the market's
own view. That is the outcome a Gaussian margin model cannot express at all, and getting it
right is the whole reason this pipeline models scorelines. It does not amount to an edge. It
does mean the probabilities are honest, which is the precondition for ever finding one.

Read it in this order:

1. **`log_loss_edge`** — the proper scoring rule, and the honest answer to "does this model know
   anything the price does not". Expect it negative.
2. **`draw`** — predicted against actual. This is the specific failure a Gaussian margin model
   cannot even express, so it is the first thing to check that this one gets right.
3. **`calibration`** — do the probabilities mean what they say? A different and more basic
   question than whether they beat the price, and worth passing before anything else matters.

A model that beat the market on log loss while losing on supremacy MAE would not be a
contradiction — it would mean the edge is in the *shape* of the distribution rather than its
centre, which for football is the more plausible of the two.

#### Three-way de-vigging is not proportional de-vigging

With two outcomes at 1.95/1.95, dividing each implied probability by their sum is exactly right.
With three it is not, because bookmakers load proportionally more margin onto the longshot —
the favourite-longshot bias, one of the most replicated findings in the literature. On a
1.25 / 6.00 / 12.00 book, proportional de-vigging says the away side is 7.9%; Shin's method says
7.0%.

That 0.9 of a point is not academic: it is subtracted from exactly the leg where a model most
easily talks itself into a "value" bet. So `epl/odds_math.py` uses **Shin by default**, with the
proportional and power methods available for comparison.

#### Where the edge would actually be, if anywhere

Not here. `DEGEN_EPL_LEAGUE` points the whole pipeline at a different division — football-data
publishes the Championship (`E1`), League One (`E2`) and League Two (`E3`) in an identical
column layout, and the same code produces a board for any of them with no other change. Those
markets take smaller limits, attract less modelling attention, and have far more roster churn.
If a public-data edge in English football exists, it is far likelier a division or two down than
in the most-modelled league on earth.

The `market_softness` segments are cut with that in mind: promoted clubs, European load, closed
doors, derbies, favourite size, stage of season — each reported with `vs_break_even_se` **and**
per-season stability, and each stamped with whether it survives correction for the ~20 looks
taken. Treat anything under +2 as unproven, and remember that the best of twenty segments
landing at +2 is roughly what chance alone produces.

### The guardrails that earned their place

Front-loaded, because football naming and football dates are both worse than they look:

- **Club names raise rather than guess, and ambiguous short forms raise too.** "Sheffield" is
  two clubs. So is "Bristol", and so is "Manchester". Resolving one of those either way silently
  merges two clubs' rating histories and nothing downstream can detect it, so `epl/teams.py`
  refuses them by name and says which clubs it could have meant. CI asserts every club in the
  committed data resolves, so a newly promoted side fails the build the week it appears rather
  than entering the league as an unrated stranger.
- **Dates are parsed day-first, and the format is detected rather than inferred.**
  football-data writes `01/02/2024` for 1 February. Parsed month-first that is 2 January — which
  raises nothing, reorders a third of every season, and breaks the one guarantee this whole
  project rests on. The GitHub mirror writes ISO dates instead, so the convention is detected
  once per file and then forced; pandas will otherwise infer a different one for different
  chunks of the same column.
- **The odds columns changed shape in 2019-20 and every quantity is resolved through an ordered
  candidate list.** Betbrain's aggregates (`BbAvH`, `BbAHh`, `BbAv>2.5`) were dropped and
  replaced with `AvgH` / `AHh` / `Avg>2.5`, plus a whole parallel set of *closing* columns
  (`PSCH`, `AHCh`, `PC>2.5`). Reading one era's spelling does not raise — it yields NaN for
  every row in the other era, and the market models then train on whichever half of history
  happened to match. The winning columns are recorded per row in `odds_source` and summarised
  per season in `meta.json`, so this is visible rather than quiet.
- **`game_id` is season-plus-clubs, deliberately not date-based.** English football rearranges
  fixtures constantly. A date-keyed id would mint a *new* id when a postponed match is finally
  played, so the pick published for it would never grade and would sit in `picks.csv` forever.
- **Quarter handicaps are split, not rounded.** A −0.75 line is half at −0.5 and half at −1.0,
  so half the stake can win while the other half pushes. Both the pricing and the grading
  express that, including "half win" and "half loss" as settlement outcomes. Rounding a quarter
  line away would misprice the market that is quoted on most matches in this league.
- **Push legs are returned, not lost.** A whole-number handicap or goal line pushes on an exact
  hit — on a level handicap that is the ~24% of matches that end drawn. Folding those into the
  loss column would understate EV by more than any edge being measured.
- **The season boundary is August, not July.** 2019-20 was suspended in March 2020 and its last
  rounds were played behind closed doors from 17 June to 26 July. A July boundary files those as
  2020-21 — 66 Premier League matches joined to the wrong prior-season strength ratings and
  crossing the rating engine's season rollover in the wrong place. The match archive shows it
  exactly: 314 matches in "2019" and 446 in "2020", against a normal 380. No English league
  season has ever kicked off in July.
- **The ratings are anchored to the league mean.** Nothing otherwise forces mean attack and mean
  defence to zero, so in a high-scoring season every club's ratings drift up together. That
  breaks the season rollover, which regresses toward zero — no longer the mean — and quietly
  deflates predicted scoring every August. A league-level term absorbs the era instead, and the
  club ratings are re-centred around it at each boundary. After the fix, an eight-season replay
  predicts a mean supremacy of +0.249 against an actual +0.257, and a mean total of 2.851
  against an actual 2.850.

### Checking the source

The one thing here that cannot be verified without reaching the live host is the shape of
football-data's odds columns — and that layout has already changed once. It is
environment-overridable, every path degrades to "no market prices" rather than failing, and
there is a results-only GitHub mirror behind it that logs loudly when it is used, so the
pipeline runs either way.

**From anywhere, including a phone: Actions → EPL source check → Run workflow.** The result goes
to the job summary, which the GitHub mobile app renders as a page rather than as raw logs: a row
per season showing how many matches and prices resolved, how long each took, and from which
columns, plus a banner saying whether the host answered at all.

#### If the host does not answer

This pipeline fetches far more files per run than the other two — a first backfill is around
fifty — so the per-request retry budget is multiplied by fifty. That is not hypothetical: the
first version shipped a 45-second timeout with four retries, which is **4.1 minutes of dead time
per unreachable file**, and it turned the six-file schema check into a 25-minute job and would
have made a first retrain a three-hour one.

Three things now bound it, and it is worth knowing which does what:

- **`Retry-After` is not honoured.** This is the one that actually bit. The host answers 503,
  which is in the retry list, and a 503 may carry a `Retry-After` header. urllib3 honours that
  header by default, and a `Retry-After` sleep is **not** capped by `backoff_max` — that cap
  applies only to computed exponential backoff. So a single file fetch was parked for **288
  seconds**, which no amount of tuning `backoff_factor`, `backoff_max` or the retry count could
  have prevented. Declining to honour the header is the only thing that bounds it, and against a
  static archive it costs nothing: if the host wants us gone it keeps saying 503, and the
  breaker notices.
- **A wall-clock budget on the primary host.** Thirty seconds of *wasted* time and the host is
  abandoned for the rest of the run. Note what this could and could not do: it correctly tripped
  on that 288-second fetch and let the job finish — but it is checked *after* a request returns,
  so it cannot interrupt one that is already asleep. It bounds the run, not the request. Only
  time from *failed* requests counts, so a merely slow-but-working host is never abandoned.
- **Short timeouts**, as a first line rather than the defence: 8s to connect, 15s to read, two
  attempts. Both failure paths are bounded under 45 seconds, and a test asserts both, because
  which one fires is not ours to choose — the diagnosis here was wrong twice before the logs
  settled it, first as a packet drop and then as a stall, when it was neither.
- **A circuit breaker.** Once either bound is hit the primary host is treated as down for the
  rest of the run and every later fetch goes straight to the mirror. Discovering a dead host
  once beats discovering it fifty times. The state is per-process and never persisted, so an
  outage heals by itself on the next scheduled run, and a single success part-way through clears
  the counter so intermittent failures never trip it.
- **`timeout-minutes` on every EPL workflow**, so a hung upstream can never burn Actions minutes
  for hours even if the first two fail.

When the breaker trips, the run still **completes** — served entirely from the results-only
mirror, with market-aware models skipped and a loud line in the log saying so. A run that
finishes and tells you it has no odds is worth far more than one that hangs.

Locally:

```bash
python -m epl.sources.footballdata --check
python -m epl.sources.footballdata --check --league E1 --seasons 2018,2019,2024
```

A season showing matches but zero priced means the layout moved: add the new spellings to
`CLOSING_1X2` / `OPENING_AH` and friends at the top of `epl/sources/footballdata.py`.

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
pytest -q tests/                                # offline, no network, no API key

python -m cfb.train --no-fetch
python -m cfb.predict --dry-run
python -m cfb.site && open docs/cfb/index.html

python -m nfl.train --no-fetch                  # nflverse needs no key at all
python -m nfl.predict --dry-run
python -m nfl.site && open docs/nfl/index.html

python -m epl.train --no-fetch                  # football-data needs no key either
python -m epl.predict --dry-run
python -m epl.site && open docs/epl/index.html
python -m epl.sources.footballdata --check      # what the odds resolver actually found

python -m core.landing && open docs/index.html  # the chooser, built from what is published
```

## Knobs (repo variables or env vars)

| Var | Default | Meaning |
|---|---|---|
| `DEGEN_TOTAL_EDGE` | 3.5 | min points of edge to publish a totals play |
| `DEGEN_SPREAD_EDGE` | 2.5 | same for spreads |
| `DEGEN_MIN_GAMES` | 2 (cfb) / 3 (nfl) | below this, picks are flagged early-season and not staked |
| `DEGEN_KELLY` | 0.25 | Kelly fraction |
| `DEGEN_BOARD_DAYS` | 7 | how far ahead to post games |
| `DEGEN_CFB_SLATES` | `15,18,22.5` | ET hours cutting the four kickoff slates (noon / afternoon / night / late). A bad value falls back to the defaults |
| `DEGEN_FIRST_SEASON` | 2015 (cfb) / 2010 (nfl) | earliest season to train on |
| `DEGEN_WARMUP_SEASONS` | 1 (nfl) | seasons loaded before the training window to warm the ratings up |
| `DEGEN_WALK_SEASONS` | 6 (nfl) | seasons pooled by the walk-forward evaluation |
| `DEGEN_NFL_HTTP_TIMEOUT` | 60 | NFL-only HTTP timeout; nflverse serves multi-MB files |
| `DEGEN_NFL_DOCS` | `docs/nfl` | where the NFL board is written |
| `DEGEN_EPL_SOURCE` | `matchdata` | history backend: `matchdata` (GitHub archive, the default) or `footballdata` (football-data.co.uk, when it is reachable) |
| `DEGEN_MATCHDATA_URL` | (the archive) | override the archive location |
| `DEGEN_EPL_PRIMARY_BUDGET` | 30 | seconds of wasted wall-clock before a source host is abandoned for the run |
| `DEGEN_EPL_LEAGUE` | `E0` | which division the EPL pipeline predicts: `E0` Premier League, `E1` Championship, `E2` League One, `E3` League Two |
| `DEGEN_SUP_EDGE` | 0.60 | min goals of supremacy disagreement to publish a handicap play |
| `DEGEN_GOALS_EDGE` | 0.70 | same for total goals |
| `DEGEN_DC_RHO` | −0.04 | Dixon-Coles low-score correction; more negative lifts 0-0 and 1-1 |
| `DEGEN_EPL_DOCS` | `docs/epl` | where the Premier League board is written |
| `DEGEN_CFB_DOCS` | `docs/cfb` | where the college football board is written |
| `DEGEN_KALSHI_ML_SERIES` | `KXNFLGAME` | Kalshi moneyline series (also `..._SPREAD_SERIES`, `..._TOTAL_SERIES`) |
| `DEGEN_SUPPORT_URL` | (unset) | Buy Me a Coffee link shown at the top; omit and the button hides |
| `DEGEN_SUPPORT_LABEL` | Buy me a coffee | button text |
| `DEGEN_THEME` | `ticker` | site look: `ticker`, `scoreboard`, `field` |
| `DEGEN_SUPPORT_URL` | (blank) | e.g. a Buy Me a Coffee link; blank hides the button |
| `DEGEN_SUPPORT_LABEL` | `Buy me a coffee` | button text |

### Filtering a Saturday

A CFB Saturday is around 50 board games, which is too many to read as one list. Above the board
there are two independent chip rows — **day** and **kickoff slate** — and they combine, so
"Saturday + Night" is two clicks.

The slates are the windows the schedule actually clusters into, not round numbers. A typical
week:

| Slate | ET window | Games |
| --- | --- | ---: |
| Noon | before 3:00 | 10 |
| Afternoon | 3:00 – 5:59 | 14 |
| Night | 6:00 – 10:29 | 21 |
| Late | 10:30 onwards | 3 |

Two deliberate choices. **6pm counts as night**, because it is an evening kickoff by any normal
reading and the afternoon window is the 3:30–4:00 block. And a game with **no announced kickoff
gets its own `Time TBD` chip** rather than being dropped — games a week out often have no time
yet, and a filter that silently shrank the board would be worse than one extra chip. A slate
chip only appears when it has games, so an empty chip can never blank the board.

Cuts are set by `DEGEN_CFB_SLATES` if you disagree with them.

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
