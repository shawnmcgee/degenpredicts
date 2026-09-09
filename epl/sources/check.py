"""Report what the ACTIVE source actually resolves, whichever backend is configured.

    python -m epl.sources.check
    python -m epl.sources.check --league E1

This replaces a check that was hardcoded to ``footballdata``. That was correct while
football-data.co.uk was the primary and actively misleading afterwards: it went on reporting
503s from a host the pipeline no longer reads, which looks like a broken pipeline when nothing
is wrong. A check that does not follow the configuration is worse than no check, because it
answers a question nobody asked with an alarming number.

Output goes to the job summary, which the GitHub mobile app renders as a page rather than as
raw logs. Read-only; it never writes to the data cache.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

from .. import config
from . import active, name

log = logging.getLogger("epl.sources.check")


def run(league: str) -> str:
    src = active()
    t0 = time.time()
    lines = [f"# EPL source check — {config.LEAGUE_NAMES.get(league, league)} (`{league}`)", "",
             f"Backend: **`{name()}`** (`{src.__name__}`)", ""]

    games = lines_ = None
    try:
        if hasattr(src, "fetch_all"):
            raw = src.fetch_all()
            if raw.empty:
                lines += ["> **The archive did not return anything.** This is a reachability "
                          "problem. The pipeline keeps whatever is already cached; re-run "
                          "later.", ""]
                return "\n".join(lines)
            games, lines_ = src.parse(raw, league)
        else:
            from .footballdata import _check, _summary
            return _summary(_check(_default_seasons(), league), league)
    except Exception as e:                     # a check must report, never raise
        lines += [f"> **The check failed:** `{e}`", ""]
        return "\n".join(lines)

    elapsed = time.time() - t0
    if games.empty:
        lines += [f"> **No `{league}` matches found in the archive.** Either the division code "
                  "is wrong or the upstream layout moved.", ""]
        return "\n".join(lines)

    priced = len(lines_)
    closing = int(lines_["is_closing"].astype(bool).sum()) if priced else 0
    ah = int(lines_["ah_home"].notna().sum()) if priced else 0
    ou = int(lines_["price_over"].notna().sum()) if priced else 0
    if closing:
        closing_note = f"{closing:,} rows carry explicit closing prices."
    else:
        closing_note = ("All prices are marked opening/unknown rather than closing - this "
                        "backend does not distinguish them, so the market baseline may be "
                        "softer than a true close.")
    lines += [
        f"> **{len(games):,} matches, {priced:,} with prices** "
        f"({ah:,} with an Asian handicap, {ou:,} with an over/under), fetched and parsed in "
        f"{elapsed:.0f}s.", "",
        f"Seasons {int(games.season.min())}-{int(games.season.max())}. {closing_note}", "",
        "| Season | Matches | Priced | AH | O/U | Columns resolved |",
        "|---|---|---|---|---|---|",
    ]
    by = lines_.groupby("season") if priced else None
    for s, grp in list(games.groupby("season"))[-8:]:
        p = by.get_group(s) if by is not None and s in by.groups else None
        lines.append(
            f"| {int(s)}-{(int(s)+1) % 100:02d} | {len(grp)} | {len(p) if p is not None else 0} "
            f"| {int(p['ah_home'].notna().sum()) if p is not None else 0} "
            f"| {int(p['price_over'].notna().sum()) if p is not None else 0} "
            f"| `{p['odds_source'].iloc[0] if p is not None and len(p) else '—'}` |")

    lines += ["", "## What to do", "",
              "| What you see | Meaning | What to do |", "|---|---|---|",
              "| Matches and priced both healthy | The source is good | Run **EPL retrain** |",
              "| Matches healthy, priced 0 | Results only, no market | The market-aware models "
              "will have no rows; check the upstream column names |",
              "| Nothing returned | Reachability | Re-run later; the cache is untouched |", "",
              "The board does **not** come from this source — the archive holds played matches "
              "only. It comes from the live price feed, so **EPL picks** needs `ODDS_API_KEY`.",
              ]
    return "\n".join(lines)


def _default_seasons():
    cur = config.season_of(config.today_uk())
    return sorted({2015, 2018, 2019, 2022, cur - 1, cur})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the active EPL source.")
    ap.add_argument("--league", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    text = run(a.league or config.LEAGUE)
    print(text)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
