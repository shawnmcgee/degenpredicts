# NHL totals: proof of concept

A from-scratch NHL over/under model, built to answer one question: **can a model hit 55% or better
on over/under against the market total?** It is self-contained in this folder and changes nothing
in `nhl/`, `data/`, the workflows or the live board.

## The answer: no, and a 55% that looks real is a trap

**No model built here beats the market's own price on totals.** Five model families were tried,
plus two market-anchored designs and five market-structure strategies. On the 3,405 games where
the over/under price is known (2023-24 to Jan 2026), every one of them lost money once the juice
was counted. None of them predicted results better than the devigged price.

A 55% *does* show up: the ensemble went **54.97% on 2,958 games** against 2012-22 closing lines.
It is not an edge, and it is worth understanding why, because it is the easiest number in sports
betting to fool yourself with:

- The 2012-22 archive has the closing line (5.5, 6, 6.5) but **not its juice**. At a 5.5 the
  juice does the real work: an over at -140 is a 58% proposition.
- A model can therefore beat 50% "against the line" just by learning **which side the book had
  juiced**, and team ratings are a good proxy for that.
- **Proof 1.** On the priced games, backing whichever side the market already favoured wins
  **52.4% of 3,405 games** and still loses 3.1% per bet (the juice costs more than the extra hit
  rate earns).
- **Proof 2.** A "juice model" that only imitates the price a book would post (fitted to 2023-26
  prices, never shown a result) picks **52.2% winners on 7,470 games in 2016-22, and 53.2% on its
  most confident quarter**. That is the same range as the team-feature models, with no hockey
  knowledge beyond what a price contains.
- Given the real price, the same team-feature models fall to ~50% and lose 3-6% per bet.

**A hit rate is only meaningful next to the break-even of the prices actually taken.** At -110 that
is 52.4%. At a juiced -130 it is 56.5%.

## What was built

| Piece | File | What it does |
|---|---|---|
| Data | `fetch.py` | Downloads 2010-11 to 2026-27 from [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) (scraped from the NHL's own API, so game ids match `data/nhl/games.csv`): play-by-play with xG per shot and on-ice skaters, skater box scores (who dressed, TOI), goalie box scores (who started), referees. ~350 MB, git-ignored. |
| Aggregation | `build.py` | Per game, per team: xG and shot attempts by strength, power-play xG, penalties, empty-net goals. Per goalie: xG faced vs goals allowed. Per skater: TOI, individual xG, points, 5v5 on-ice xG for/against. Goals reconcile with `games.csv` on 99.99% of games. |
| Features | `features.py` | One chronological replay: every game is described from the state the night before, then the state is updated. About 150 features (below). |
| Models | `models.py` | `base`, `logit`, `gbm`, `pois`, `ens`, plus the market-anchored `Juice` and `Resid` models. |
| Backtest | `backtest.py` | Walk-forward by season. Each test season is scored blind by models fitted on earlier seasons only. |
| Market tests | `market_checks.py` | Tests of the market itself: juice, calibration, shrinkage, puck-line vs total, feature scan. |

**Features, by family:**

- **Team.** Fast (10-game) and slow (40-game) weighted rates of xG, goals, shot attempts, shots,
  power-play xG, penalties drawn/taken and empty-net goals, for and against. Each is a ratio to a
  league rate that drifts with the scoring environment, with a summer regression toward the mean.
- **Goalies.** Tonight's starter's goals saved against xG (decayed, shrunk). Also how established
  he is as the #1, games of experience and rest. A *projected* starter is computed too: the usual
  #1, except the previous night's goalie on a back-to-back. It is right 69% of the time.
- **Lineups.** Every skater carries his own decayed TOI, individual xG, points and 5v5 on-ice xG
  for/against. Tonight's dressed lineup is valued against the team's usual lineup at the same
  ratings, so injuries, scratches, call-ups and trades show up as goal deltas. Also counted:
  regulars missing and new faces.
- **Referees.** Each referee's penalties, goals and over/under margin against the closing line,
  relative to expectation, shrunk toward neutral.
- **Schedule.** Rest, back-to-backs, 3-in-4, travel km, time-zone shift.
- **Market and context.** Line, moneyline favourite, the price where known, league scoring vs the
  line, each team's recent over/under margin, early season, playoffs.

The play-by-play files were scored by different xG model versions: a game summed to about 7.4 xG
before 2023-24 and about 6.1 after, against about 6 real goals throughout. So every xG figure is
converted to goals with a running within-season goals/xG ratio before it reaches a rating.

## How it was tested

Two tracks, because the two archives of lines know different things:

- **Close track (2012-13 to 2022-23, 11,634 decided games).** SBR closing totals, no prices. A
  closing line has seen the confirmed starters and lineups, so the models may too. Only a hit rate
  can be measured.
- **Noon track (2023-24 to Jan 2026, plus this season, 3,405 decided games).** Odds API noon
  prices. At noon most starters and lineups aren't confirmed, so the honest models see only the
  projected starter and the usual lineup. A *noon + news* variant gives them tonight's real
  starters and lineups as an upper bound. Here the price exists, so models are scored against it.

## Results

### Priced games (2023-26): the test that matters

| Model | Log loss (market 0.69253) | Hit rate, all games | Bets at EV ≥ 0 | Hit rate | Break-even | ROI ± 1 s.e. |
|---|---:|---:|---:|---:|---:|---:|
| `logit` (team features) | 0.69814 | 50.0% | 2,109 | 50.6% | 53.2% | −4.6% ± 2.0 |
| `gbm` | 0.69550 | 49.7% | 1,629 | 50.4% | 52.8% | −4.3% ± 2.3 |
| `pois` (NB count model) | 0.69633 | 50.5% | 2,289 | 50.9% | 53.0% | −3.7% ± 1.9 |
| `ens` | 0.69574 | 50.0% | 1,813 | 50.6% | 53.2% | −4.6% ± 2.2 |
| `resid_logit` (price + features) | 0.69275 | 52.2%* | 499 | 48.9% | 52.6% | −6.2% ± 4.2 |
| `resid_gbm` | 0.69335 | 51.5%* | 728 | 49.3% | 52.7% | −6.1% ± 3.5 |
| `pois` **with tonight's real starters and lineups** | 0.69607 | 51.5% | 2,386 | 52.1% | 53.0% | −1.5% ± 1.9 |
| *Back the side the market favours* | – | *52.4%* | 3,405 | 52.4% | – | *−3.1% ± 1.6* |

\* Mostly agreeing with the juiced side; see the last row.

Raising the EV bar does not help. The only positive buckets anywhere are 48 and 116 bets, both well
inside one standard error.

### Closing lines (2012-22): where the "55%" lives

| Model | Hit rate, all | Most confident picks |
|---|---:|---:|
| `base` (line + favourite only) | 51.4% | 47.5% on 301 |
| `logit` | 52.4% | 53.8% on 4,669 |
| `gbm` | 51.9% | 54.2% on 2,570 |
| `ens` | 52.1% | **55.0% on 2,958** |
| *Juice model alone (no results, imitates a book's price), 2016-22* | *52.2%* | *53.2% on 1,773* |

When the price is put back in (2016-22, with the modelled price as the anchor), the juice model
alone has the **best** log loss, 0.69218. Every model that adds team features on top of it is
worse (0.6932-0.6934): the features carry nothing the price didn't.

### This season (2026-27, 85 decided games)

Every model is between 46% and 53%. The market's favoured side is 54%. Eighty-five games carry a
standard error of 5.4 points, so this says nothing either way. The live board's 47.1% (40-45-2)
is the same: within one standard error of 50%.

## Why 55% isn't there

**Total goals are close to unpredictable game to game.** On the priced games, the correlation
between actual total goals and the best pre-game forecast is:

- market-implied mean: 0.070
- this model with real starters: 0.067
- this model with projected starters: 0.060

That is an R² of about 0.5%. Given the market's mean, the model's disagreement with it is worth
0.135 ± 0.123 goals per goal: indistinguishable from nothing.

**The arithmetic of 55%.** Near a total of 6, each 0.1 goals of expected total moves P(over) by
about 1.5 points. For the most confident tenth of games to win 55% against *fair* prices, a model
needs predictable goal variation the market has not priced, with a spread of about **0.19 goals**.
The market's entire predictable spread across games is about **0.4 goals**. Finding unpriced
structure half that size, from public data, in a mature market, is not a realistic target.
For context, long-run winning bettors are generally described as living in the low-to-mid 50s at
standard juice, not above 55%.

## What else was tried

| Idea | Result |
|---|---|
| Market spreads its totals too wide (bet toward the league mean) | Actual totals move 0.41 goals per goal of market mean (partly measurement attenuation). Unders on the top fifth: +0.6% ± 3.7 overall; by season +0.1, +8.1, −15.2, −18.0. Not tradeable. |
| Fade heavily juiced sides | Market-favoured side at ≥ 3 points of juice: 49.8% vs 54.1% fair on 452 games (ROI for the other side +4.6% ± 4.8). One standard error, found by looking. |
| Line-specific bias | Under 6.5: +0.9% ± 3.0. Every other line negative both ways. |
| Puck-line price as a second opinion on goals | The puck-line price is 98.8% explained by the moneyline and total. Its residual vs the totals result: z = −0.54. |
| Goalie news the noon market didn't have | Real-vs-projected starter quality vs result minus price: z = 0.9. |
| Referee crews (penalties, goals, O/U record) | Close track z ≤ 2.3 for penalties; nothing against priced lines. |
| Team over/under streaks vs the line | Nothing (close z 0.4). |

### Weak signals, for what they are worth

With about 40 features scanned, the largest |z| expected from chance alone is about 2.5. A few
point the same way in most seasons on both tracks:

- **Goaltending luck regresses** (`sv_s`). A team whose goals-against has run above its xG-against
  goes under the market more often: priced z = −2.2, close z = −1.9, negative in 14 of 15 seasons.
- **Lineup changes** (`lu_dpts`). The dressed lineup's scoring vs the usual: close z = +3.6, but
  against real prices only z = +0.9. Closing books move juice on the same news.
- **Back-to-backs and 3-in-4** go under the market slightly. Back-to-backs: priced z = −2.0,
  close z = −1.0. 3-in-4: priced z = −1.2, close z = −2.5.

Even taken at face value, these are correlations of 0.02-0.04: a point or two of hit rate on
selected games. They are not 55%.

## If you want an edge on totals, it isn't in the model

Every result above says the totals price already contains what a public-data model knows. What
moves the needle is **price and timing**, not a better rating:

- **Line shopping.** At a two-sided -110 market the break-even is 52.4%. Taking the best of
  several books (or an exchange price) can bring it near 51%, and that alone is worth more than
  any feature tested here.
- **Speed on news.** The real-starter/lineup ("news") variant was the least bad model: −1.5% ±
  1.9 at noon prices. Information that arrives after the line is set is the plausible edge, but it
  is only capturable by betting *before* the book moves. Backtesting it needs the timestamped
  evening/closing prices that the live board has been logging since this season (`snapshots.csv`).
  In a season, CLV from those snapshots can answer it. No historical archive can.
- **Prediction markets.** On an exchange such as Kalshi you can post limit orders instead of paying
  a spread. The model still has to beat the price, but there is no 4.5% vig to clear.

**Recommendation: keep the current totals model as it is.** Its own backtest found the same thing
(log loss 0.6925 vs the market's 0.6926). Chasing 55% on totals is likely to produce an overfit
model that looks good on the archive and loses on the board. The puck line, where the existing
README reports its one structural signal, is the better place to spend effort.

## Caveats

- **The priced archive is noon prices from one or two books (BetMGM, Caesars), not closing
  lines.** An edge against them would be an edge against a softer market. None was found anyway.
- **The close track uses tonight's real starters and dressed lineups.** They are confirmed before
  puck drop, which is what a closing line also knows, but a few lineup changes happen at warmups.
- **The juice model assumes 2023-26 books shade prices the way 2012-22 closing books did.** It is
  only used to show where the apparent skill came from. No result in the priced table depends on it.
- **The feature scan and market tests looked at the priced games**, so any pattern found there
  (the "weak signals") needs out-of-sample confirmation. The 2026-27 season is the only clean test
  left.
- **The xG values come from sportsdataverse's models.** They are re-scaled to goals within each
  season, so their level doesn't matter, but their quality does.

## Running it

```bash
pip install -r poc/nhl_totals/requirements.txt
python -m poc.nhl_totals.fetch          # ~350 MB into poc/nhl_totals/cache/ (set POC_CACHE to move it)
python -m poc.nhl_totals.build          # ~30 s: per-game team / goalie / skater / referee tables
python -m poc.nhl_totals.features       # ~70 s: the chronological feature replay
python -m poc.nhl_totals.backtest       # ~45 s: every model, both tracks -> results/backtest.json
python -m poc.nhl_totals.market_checks  # market tests and the feature scan -> results/market_checks.json
```

`results/` holds the JSON summaries and every out-of-sample prediction (`preds_*.csv.gz`), so the
tables above can be checked without re-running anything.
