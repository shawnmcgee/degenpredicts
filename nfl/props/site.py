"""Render the props tab into docs/nfl/props/.

    python -m nfl.props.site
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

log = logging.getLogger("nfl.props.site")
TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
UNIT = {"pass_yds": "pass yds", "rush_yds": "rush yds", "rec_yds": "rec yds",
        "receptions": "rec", "anytime_td": "TD"}
ORDER = ["pass_yds", "rush_yds", "rec_yds", "receptions", "anytime_td"]
POS_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def _pays(price):
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if not (0 < price < 1):
        return None
    return round(1 / (price + kalshi.fee(price)), 2)


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
    return {"k": int(r["k"]), "p": _pct(r["p_yes"]),
            "yes_c": None if r["yes_ask"] != r["yes_ask"] else round(100 * r["yes_ask"]),
            "no_c": None if no_px != no_px else round(100 * no_px),
            "yes_pays": _pays(r["yes_ask"]), "no_pays": _pays(no_px),
            "fair_yes": round(1 / r["p_yes"], 2) if r["p_yes"] and r["p_yes"] == r["p_yes"] and r["p_yes"] > 0 else None,
            "fair_no": round(1 / (1 - r["p_yes"]), 2) if r["p_yes"] == r["p_yes"] and r["p_yes"] < 1 else None,
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
            "p": b["p"] if yes else (100 - b["p"] if b["p"] is not None else None),
            "cents": b["yes_c"] if yes else b["no_c"],
            "pays": b["yes_pays"] if yes else b["no_pays"],
            "fair": b["fair_yes"] if yes else b["fair_no"],
            "edge": round(100 * b["edge"])}


def _market_block(g: pd.DataFrame) -> dict:
    g = g.sort_values("k")
    m = g["market"].iloc[0]
    mid = (g["yes_bid"] + g["yes_ask"]) / 2
    main = g.iloc[int((mid - 0.5).abs().fillna(9).values.argmin())]
    rungs = [_rung(r) for _, r in g.iterrows()]
    proj = g["proj_avg"].iloc[0] if "proj_avg" in g and g["proj_avg"].notna().any() \
        else g["proj_mean"].iloc[0]
    if m == "anytime_td":
        proj = g["proj_mean"].iloc[0]
    return {"market": m, "label": MARKET_LABEL[m], "unit": UNIT[m],
            "proj": _pct(proj) if m == "anytime_td" else _num(proj),
            "median": None if m == "anytime_td" else _num(g["proj_median"].iloc[0], 0),
            "main": _rung(main), "rungs": rungs, "value": _value(rungs),
            "flagged": any(x["flag"] for x in rungs)}


def board_context() -> tuple[list[dict], dict]:
    if not config.PROPS_BOARD.exists():
        return [], {}
    b = pd.read_csv(config.PROPS_BOARD, dtype={"game_id": str, "player_id": str})
    if b.empty:
        return [], {}
    b["tradeable"] = b["tradeable"].astype(str).str.lower().isin(["true", "1"])
    games = []
    for gid, gg in b.groupby("game_id", sort=False):
        first = gg.iloc[0]
        players = []
        for pid, pg in gg.groupby("player_id"):
            p0 = pg.iloc[0]
            blocks = [_market_block(mg) for _, mg in pg.groupby("market")]
            blocks.sort(key=lambda x: ORDER.index(x["market"]))
            players.append({"name": p0["player_name"], "pos": p0["position"], "team": p0["team"],
                            "injury": (p0["injury"] if isinstance(p0["injury"], str) else "").title(),
                            "markets": blocks, "flagged": any(x["flagged"] for x in blocks),
                            "kinds": " ".join(x["market"] for x in blocks)})
        players.sort(key=lambda x: (x["team"] != first["away_team"], POS_ORDER.get(x["pos"], 9), x["name"]))
        games.append({"game_id": gid, "away": first["away_team"], "home": first["home_team"],
                      "tip": first.get("tip_et") if isinstance(first.get("tip_et"), str) else "",
                      "kick": first.get("kickoff_utc") if isinstance(first.get("kickoff_utc"), str) else "",
                      "players": players})
    games.sort(key=lambda x: (x["kick"] or "9", x["game_id"]))
    summary = {"rungs": int(len(b)), "players": int(b["player_id"].nunique()),
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
        if not r:
            continue
        row = {"label": MARKET_LABEL[m], "market": m}
        if m == "anytime_td":
            row.update({"ll_model": r.get("log_loss_model"), "ll_base": r.get("log_loss_baseline"),
                        "n": r.get("n"), "seasons": f"{r['seasons'][0]}-{r['seasons'][-1]}"})
        else:
            pl = r.get("proxy_line") or {}
            row.update({"mae_model": r.get("mae_model"), "mae_base": r.get("mae_baseline"),
                        "n": pl.get("n"), "brier_model": pl.get("brier_model"),
                        "brier_base": pl.get("brier_trailing_avg"),
                        "hit_conf": pl.get("side_hit_rate_confident"),
                        "n_conf": pl.get("n_confident"),
                        "under": pl.get("always_under_hit_rate"),
                        "base_hit": pl.get("trailing_avg_side_hit_rate"),
                        "seasons": f"{r['seasons'][0]}-{r['seasons'][-1]}"})
        out["markets"].append(row)
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
        p_min=round(100 * config.PROPS_PROB_MIN), p_max=round(100 * config.PROPS_PROB_MAX),
        max_gap=round(100 * config.PROPS_MAX_GAP),
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
