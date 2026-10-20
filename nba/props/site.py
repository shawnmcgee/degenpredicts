"""Render the props tab into docs/nba/props/.

    python -m nba.props.site
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import config
from ..sources import kalshi
from .board import MARKET_LABEL

log = logging.getLogger("nba.props.site")
TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
UNIT = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "3PM"}
ORDER = ["points", "rebounds", "assists", "threes"]
POS_ORDER = {"G": 0, "F": 1, "C": 2}
INJURY_LABEL = {"out": "Out", "suspension": "Out", "doubtful": "Doubtful",
                "questionable": "Questionable", "day-to-day": "Day-to-day", "probable": "Probable"}


def _pays(price):
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    c = kalshi.cost(price)
    return None if c != c else round(1 / c, 2)


def _pct(p):
    return None if p is None or p != p else round(100 * float(p))


def _num(v, nd=1):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, nd)


def _rung(r) -> dict:
    no_px = 1 - r["yes_bid"] if r["yes_bid"] == r["yes_bid"] else np.nan
    mid = r["mid"] if r["mid"] == r["mid"] else np.nan
    p = r["p_yes"]
    return {"k": int(r["k"]), "p": _pct(r["p_model"]), "pub": _pct(p),
            "mid_c": None if mid != mid else round(100 * mid),
            "yes_c": None if r["yes_ask"] != r["yes_ask"] else round(100 * r["yes_ask"]),
            "no_c": None if no_px != no_px else round(100 * no_px),
            "yes_pays": _pays(r["yes_ask"]), "no_pays": _pays(no_px),
            "fair_yes": round(1 / p, 2) if p == p and p > 0 else None,
            "fair_no": round(1 / (1 - p), 2) if p == p and p < 1 else None,
            "flag": r["flag"] if isinstance(r["flag"], str) else "",
            "edge": max(_num(r.get("edge_yes"), 3) or -9, _num(r.get("edge_no"), 3) or -9),
            "thin": not bool(r["tradeable"])}


def _value(rungs: list[dict]) -> dict | None:
    """The flagged rung with the biggest edge, phrased as the side to take."""
    flagged = [x for x in rungs if x["flag"]]
    if not flagged:
        return None
    b = max(flagged, key=lambda x: x["edge"])
    yes = b["flag"] == "yes"
    return {"side": "Yes" if yes else "No", "k": b["k"],
            "p": b["pub"] if yes else (100 - b["pub"] if b["pub"] is not None else None),
            "cents": b["yes_c"] if yes else b["no_c"],
            "pays": b["yes_pays"] if yes else b["no_pays"],
            "fair": b["fair_yes"] if yes else b["fair_no"], "edge": round(100 * b["edge"])}


def _market_block(g: pd.DataFrame) -> dict:
    g = g.sort_values("k")
    m = g["market"].iloc[0]
    mid = (g["yes_bid"] + g["yes_ask"]) / 2
    main = g.iloc[int((mid - 0.5).abs().fillna(9).values.argmin())]
    rungs = [_rung(r) for _, r in g.iterrows()]
    proj = g["proj_avg"].iloc[0] if g["proj_avg"].notna().any() else g["proj_mean"].iloc[0]
    return {"market": m, "label": MARKET_LABEL[m], "unit": UNIT[m], "proj": _num(proj),
            "median": _num(g["proj_median"].iloc[0], 0), "main": _rung(main), "rungs": rungs,
            "value": _value(rungs), "flagged": any(x["flag"] for x in rungs)}


def board_context() -> tuple[list[dict], dict]:
    if not config.PROPS_BOARD.exists():
        return [], {}
    b = pd.read_csv(config.PROPS_BOARD, dtype={"game_id": str})
    # A pull that failed leaves the last board in place; its games that have since tipped are
    # not a board. (The NBA plays nightly - without this, a dead Kalshi read at 6pm would leave
    # last night's ladders up.)
    tip = pd.to_datetime(b["tip_utc"], utc=True, errors="coerce")
    b = b[tip.isna() | (tip > pd.Timestamp.now(tz="UTC"))]
    if b.empty:
        return [], {}
    b["tradeable"] = b["tradeable"].astype(str).str.lower().isin(["true", "1"])
    games = []
    for gid, gg in b.groupby("game_id", sort=False):
        first = gg.iloc[0]
        players = []
        for _, pg in gg.groupby("athlete_id"):
            p0 = pg.iloc[0]
            blocks = [_market_block(mg) for _, mg in pg.groupby("market")]
            blocks.sort(key=lambda x: ORDER.index(x["market"]))
            inj = p0["injury"] if isinstance(p0["injury"], str) else ""
            players.append({"name": p0["player_name"] if isinstance(p0["player_name"], str)
                            else str(p0["athlete_id"]),
                            "pos": p0["position"], "team": p0["team"],
                            "injury": INJURY_LABEL.get(inj, inj.title()),
                            "markets": blocks, "flagged": any(x["flagged"] for x in blocks),
                            "kinds": " ".join(x["market"] for x in blocks)})
        players.sort(key=lambda x: (x["team"] != first["away_team"], POS_ORDER.get(x["pos"], 9),
                                    x["name"]))
        games.append({"game_id": gid, "away": first["away_team"], "home": first["home_team"],
                      "tip": first["tip_et"] if isinstance(first["tip_et"], str) else "",
                      "sort": first["tip_utc"] if isinstance(first["tip_utc"], str) else "",
                      "players": players})
    games.sort(key=lambda x: (x["sort"] or "9", x["game_id"]))
    summary = {"rungs": int(len(b)), "players": int(b["athlete_id"].nunique()),
               "games": int(b["game_id"].nunique()),
               "flagged": int(b["flag"].isin(["yes", "no"]).sum()),
               "pulled": str(b["pulled_at"].max())[:16].replace("T", " ") + " UTC"}
    return games, summary


def backtest_note() -> dict:
    p = config.PROPS_MODEL_DIR / "meta.json"
    if not p.exists():
        return {}
    meta = json.loads(p.read_text())
    out = {"trained_at": meta.get("trained_at", "")[:10], "markets": []}
    for m in ORDER:
        r = meta["markets"].get(m)
        if not r or r.get("skipped"):
            continue
        pl = r.get("proxy_line") or {}
        out["markets"].append({
            "label": MARKET_LABEL[m], "market": m, "mae_model": r.get("mae_model"),
            "mae_base": r.get("mae_baseline"), "brier_model": pl.get("brier_model"),
            "brier_base": pl.get("brier_trailing_avg"), "hit_conf": pl.get("side_hit_rate_confident"),
            "n_conf": pl.get("n_confident"), "under": pl.get("always_under_hit_rate"),
            "seasons": f"{config.season_label(r['seasons'][0])} to {config.season_label(r['seasons'][-1])}"})
    return out


def graded() -> dict:
    p = config.PROPS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def build() -> None:
    config.PROPS_DOCS.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    games, summary = board_context()
    html = env.get_template("props.html").render(
        title=config.SITE_TITLE, games=games, summary=summary, bt=backtest_note(), g=graded(),
        updated=str(config.today_et()), support_url=config.SUPPORT_URL,
        support_label=config.SUPPORT_LABEL, min_edge=round(100 * config.PROPS_MIN_EDGE),
        shrink=round(100 * config.PROPS_SHRINK), p_min=round(100 * config.PROPS_PROB_MIN),
        p_max=round(100 * config.PROPS_PROB_MAX), max_gap=round(100 * config.PROPS_MAX_GAP),
        markets=[(m, MARKET_LABEL[m]) for m in ORDER])
    (config.PROPS_DOCS / "index.html").write_text(html)
    (config.PROPS_DOCS / ".nojekyll").touch()
    for src in (config.PROPS_BOARD, config.PROPS / "metrics.json"):
        if src.exists():
            shutil.copy(src, config.PROPS_DOCS / src.name)
    log.info("built %s: %d games, %d players", config.PROPS_DOCS / "index.html",
             summary.get("games", 0), summary.get("players", 0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
