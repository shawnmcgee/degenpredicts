"""Score model variants against each other on identical folds.

    python -m nfl.compare                       # walk-forward + this season's replay
    python -m nfl.compare --out compare.json    # also write the numbers to a file

Two tests, both strictly out of sample:

* **Walk-forward** over the last complete seasons (2020-2025 today): for each season S, train
  on everything before S, predict every game of S against its closing line.
* **This season's replay**: for each completed week W, train on every game kicked off before
  week W - which is what the Tuesday retrain sees - and predict week W against the line the
  board actually graded that game on (results.csv), so the replay is scored on the same
  numbers as the published record.

Every variant runs through the same folds with the same learner, so differences between them
are differences between the models. The side-by-side that matters is the paired one: on the
games where two variants pick different sides, which one was right more often. Games where
they agree cannot separate them.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import config, train
from .features import CORE_FEATURES, INJURY_FEATURES, LEAGUE_FEATURES, MARKET_FEATURES

log = logging.getLogger("nfl.compare")

# Settled before any result was looked at. `halflife` 0 is the unweighted fit used today.
VARIANTS = {
    "current": dict(feats=CORE_FEATURES, halflife=0.0),
    "home_field": dict(feats=CORE_FEATURES + LEAGUE_FEATURES, halflife=4.0),
    "injuries": dict(feats=CORE_FEATURES + INJURY_FEATURES, halflife=0.0),
    "both": dict(feats=CORE_FEATURES + LEAGUE_FEATURES + INJURY_FEATURES, halflife=4.0),
}
KINDS = {"margin": ("home_margin", "spread_home"), "total": ("total_points", "total_line")}


def _line(df: pd.DataFrame, kind: str) -> pd.Series:
    """The market's number on the target's own scale: expected home margin, or the total."""
    return -df["spread_home"] if kind == "margin" else df["total_line"]


def _grade(pred, line, actual) -> np.ndarray:
    """+1 win, -1 loss, 0 push, for the side the prediction takes against the line."""
    pred, line, actual = map(np.asarray, (pred, line, actual))
    side = np.sign(pred - line)
    side[side == 0] = -1          # a dead-even prediction takes the under / the away side
    return np.sign((actual - line) * side).astype(int)


def walk_forward(df: pd.DataFrame, kind: str, variant: dict,
                 n_seasons: int = train.WALK_FORWARD_SEASONS) -> pd.DataFrame:
    target, line_col = KINDS[kind]
    d = df.dropna(subset=[target, line_col])
    comp = train.complete_seasons(d)
    tests = [s for s in comp[-n_seasons:] if s > comp[0]]
    feats = variant["feats"] + MARKET_FEATURES
    out = []
    for s in tests:
        tr, te = d[d["season"] < s], d[d["season"] == s]
        model, _ = train._fit(tr, feats, target, halflife=variant["halflife"])
        pred = model.predict(te[feats])
        line = _line(te, kind).values
        out.append(pd.DataFrame({"game_id": te["game_id"].values, "season": s,
                                 "pred": pred, "line": line, "actual": te[target].values,
                                 "result": _grade(pred, line, te[target].values)}))
    return pd.concat(out, ignore_index=True)


def board_lines() -> pd.DataFrame:
    """The spread and total each graded game was published and graded against."""
    if not config.RESULTS.exists():
        return pd.DataFrame(columns=["game_id", "board_spread", "board_total"])
    r = pd.read_csv(config.RESULTS, dtype={"game_id": str})
    return r[["game_id", "spread_home", "total_line"]].rename(
        columns={"spread_home": "board_spread", "total_line": "board_total"})


def replay(df: pd.DataFrame, kind: str, variant: dict, season: int,
           lines: pd.DataFrame) -> pd.DataFrame:
    """Week-by-week rerun of `season`: retrain before each week, predict it on board lines."""
    target, line_col = KINDS[kind]
    feats = variant["feats"] + MARKET_FEATURES
    cur = df[(df["season"] == season)].merge(lines, on="game_id", how="inner")
    cur = cur.dropna(subset=[target]).copy()
    if cur.empty:
        return pd.DataFrame()
    # what the board saw: its own line, and the two features computed from the line
    cur["spread_home"] = cur["board_spread"].combine_first(cur["spread_home"])
    cur["total_line"] = cur["board_total"].combine_first(cur["total_line"])
    cur["spread_vs_model"] = cur["exp_margin"] + cur["spread_home"]
    cur["total_vs_model"] = cur["exp_total"] - cur["total_line"]
    cur = cur.dropna(subset=[line_col])
    hist = df.dropna(subset=[target, line_col])
    out = []
    for week, te in cur.groupby("week"):
        start = pd.to_datetime(te["date"]).min()
        tr = hist[pd.to_datetime(hist["date"]) < start]
        model, _ = train._fit(tr, feats, target, halflife=variant["halflife"])
        pred = model.predict(te[feats])
        line = _line(te, kind).values
        out.append(pd.DataFrame({"game_id": te["game_id"].values, "season": season,
                                 "week": int(week), "pred": pred, "line": line,
                                 "actual": te[target].values,
                                 "result": _grade(pred, line, te[target].values)}))
    return pd.concat(out, ignore_index=True)


def record(res: pd.Series) -> dict:
    w, l, p = int((res == 1).sum()), int((res == -1).sum()), int((res == 0).sum())
    return {"wins": w, "losses": l, "pushes": p,
            "pct": round(100 * w / (w + l), 1) if w + l else None}


def summarise(pool: pd.DataFrame) -> dict:
    out = record(pool["result"])
    out["mae"] = round(float((pool["pred"] - pool["actual"]).abs().mean()), 2)
    out["mae_line"] = round(float((pool["line"] - pool["actual"]).abs().mean()), 2)
    if "week" in pool:
        out["by_week"] = {int(k): record(g["result"]) for k, g in pool.groupby("week")}
    else:
        out["by_season"] = {int(k): record(g["result"])["pct"] for k, g in pool.groupby("season")}
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Head to head on the games where the two variants picked different sides."""
    m = a.merge(b, on="game_id", suffixes=("_a", "_b"))
    side_a = np.sign(m["pred_a"] - m["line_a"])
    side_b = np.sign(m["pred_b"] - m["line_b"])
    diff = m[(side_a != side_b) & (m["result_a"] != 0)]
    k, n = int((diff["result_a"] == 1).sum()), int(len(diff))
    p = binomtest(k, n, 0.5).pvalue if n else None
    return {"games_that_differ": n, "variant_right": k, "current_right": n - k,
            "p_value": round(float(p), 3) if p is not None else None}


def run(variants: dict | None = None, season: int | None = None) -> dict:
    variants = variants or VARIANTS
    df = train.assemble(fetch=False)
    season = season or config.season_of(config.today_et())
    lines = board_lines()
    out = {"learner": train.make_model()[1], "season": season, "walk_forward": {},
           "replay": {}}
    for kind in KINDS:
        wf = {name: walk_forward(df, kind, v) for name, v in variants.items()}
        rp = {name: replay(df, kind, v, season, lines) for name, v in variants.items()}
        out["walk_forward"][kind] = {}
        out["replay"][kind] = {}
        for name in variants:
            s = summarise(wf[name])
            if name != "current":
                s["vs_current"] = paired(wf[name], wf["current"])
            out["walk_forward"][kind][name] = s
            if len(rp[name]):
                r = summarise(rp[name])
                if name != "current":
                    r["vs_current"] = paired(rp[name], rp["current"])
                out["replay"][kind][name] = r
    out["live"] = _live_record(season)
    return out


def _live_record(season: int) -> dict:
    """What the board actually published this season, for comparison with the replay."""
    if not config.RESULTS.exists():
        return {}
    r = pd.read_csv(config.RESULTS, dtype={"game_id": str})
    r = r[r["season"] == season]
    conv = {"win": 1, "loss": -1, "push": 0}
    return {"margin": record(r["spread_result"].map(conv)),
            "total": record(r["total_result"].map(conv))}


def markdown(res: dict) -> str:
    def rec(r):
        return f"{r['wins']}-{r['losses']}" + (f"-{r['pushes']}" if r["pushes"] else "") \
            + (f" ({r['pct']}%)" if r["pct"] is not None else "")

    lines = [f"Learner: `{res['learner']}`", ""]
    for kind, label in (("margin", "Spreads"), ("total", "Totals")):
        wf = res["walk_forward"][kind]
        lines += [f"### {label} - walk-forward, closing lines", "",
                  "| Variant | Record | MAE (line) | Per season | Where it differs from current |",
                  "|---|---|---|---|---|"]
        for name, s in wf.items():
            seasons = " ".join(f"{k}:{v}" for k, v in s["by_season"].items())
            vs = s.get("vs_current")
            vs_txt = (f"{vs['variant_right']}-{vs['current_right']} over {vs['games_that_differ']}"
                      f" games, p={vs['p_value']}") if vs else ""
            lines.append(f"| {name} | {rec(s)} | {s['mae']} ({s['mae_line']}) | {seasons} | {vs_txt} |")
        rp = res["replay"].get(kind) or {}
        if rp:
            live = (res.get("live") or {}).get(kind)
            lines += ["", f"### {label} - {res['season']} replay, board lines"
                      + (f" (published record: {rec(live)})" if live else ""), "",
                      "| Variant | Record | By week | Where it differs from current |",
                      "|---|---|---|---|"]
            for name, s in rp.items():
                weeks = " ".join(f"W{k}:{v['wins']}-{v['losses']}" for k, v in s["by_week"].items())
                vs = s.get("vs_current")
                vs_txt = (f"{vs['variant_right']}-{vs['current_right']} over "
                          f"{vs['games_that_differ']} games") if vs else ""
                lines.append(f"| {name} | {rec(s)} | {weeks} | {vs_txt} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--season", type=int)
    args = ap.parse_args(argv)
    res = run(season=args.season)
    md = markdown(res)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## NFL model comparison\n\n" + md + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
