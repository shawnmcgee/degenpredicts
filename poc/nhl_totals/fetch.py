"""Download the public NHL datasets this proof of concept is built on.

    python -m poc.nhl_totals.fetch                  # every season 2010-11 -> current
    python -m poc.nhl_totals.fetch --seasons 2026 2027

Source: the season files sportsdataverse publishes as GitHub releases on
`sportsdataverse/sportsdataverse-data` (the same files `fastRhockey::load_nhl_*` reads). They are
scraped from the NHL's own API, so game ids match data/nhl/games.csv. File seasons are named by
the year the season ENDS (2025 = 2024-25); the repo's own files use the year it starts.

Nothing here is committed: the files land in the cache directory (POC_CACHE, default
poc/nhl_totals/cache/, git-ignored) and are ~350 MB in all, almost all of it play-by-play.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
from pathlib import Path

import requests

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

# (release tag, file prefix) - see fastRhockey/R/nhl_loaders.R
DATASETS = {
    "pbp": ("nhl_pbp_lite", "play_by_play_lite"),       # every event, on-ice skaters, xG per shot
    "skater_box": ("nhl_skater_boxscores", "skater_box"),  # who dressed, TOI, points
    "goalie_box": ("nhl_goalie_boxscores", "goalie_box"),  # who started in net
    "officials": ("nhl_officials", "officials"),         # the two referees and two linesmen
}
FIRST_SEASON = 2011   # 2010-11, the first season with on-ice skaters and xG
LAST_SEASON = 2027    # 2026-27


def cache_dir() -> Path:
    d = Path(os.environ.get("POC_CACHE", Path(__file__).resolve().parent / "cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_for(key: str, season: int) -> Path:
    tag, prefix = DATASETS[key]
    return cache_dir() / tag / f"{prefix}_{season}.parquet"


def _one(key: str, season: int, force: bool) -> str:
    tag, prefix = DATASETS[key]
    p = path_for(key, season)
    if p.exists() and not force:
        return f"{p.name}: cached"
    p.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{BASE}/{tag}/{prefix}_{season}.parquet", timeout=600)
    if r.status_code != 200:
        return f"{p.name}: HTTP {r.status_code}"
    tmp = p.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(p)
    return f"{p.name}: {len(r.content) / 1e6:.1f} MB"


def fetch(seasons=None, keys=None, force=False):
    seasons = seasons or range(FIRST_SEASON, LAST_SEASON + 1)
    keys = keys or list(DATASETS)
    jobs = [(k, s) for k in keys for s in seasons]
    with cf.ThreadPoolExecutor(6) as ex:
        for msg in ex.map(lambda j: _one(*j, force), jobs):
            print(msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*")
    ap.add_argument("--only", nargs="*", choices=list(DATASETS))
    ap.add_argument("--force", action="store_true", help="re-download the current season")
    a = ap.parse_args()
    fetch(a.seasons, a.only, a.force)
