"""Render the static NHL board into docs/nhl/ (GitHub Pages serves the docs folder).

    python -m nhl.site

Laid out like the other boards - same palette, same tabs, same controls - with the parts that
are hockey's own:

* a **probability strip** per game: home win in regulation, overtime, away win in regulation.
  Overtime is a fifth of all games and the reason a one-goal favourite and a -1.5 favourite are
  such different bets, so it gets its own segment rather than being folded into a win %;
* **who is in net**, labelled the way the day's news has it - confirmed, likely or projected -
  or as the model's own guess from recent starts, with its confidence, before there is any
  news. A backup is flagged, and a pick that clears the bar waits until both starters are
  confirmed or likely;
* the puck line and total as **prices**: our probability, the market's, and the EV at the posted
  price, because in hockey the line barely moves and the price is the whole bet.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config

log = logging.getLogger("nhl.site")
TEMPLATES = Path(__file__).resolve().parent / "templates"

# A bucket needs this many graded picks before its hit rate is coloured against expectation.
HIT_RATE_MIN_N = 20


def _pct(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(100 * f, 1)


def _american(dec) -> str:
    from .odds_math import decimal_to_american
    try:
        a = decimal_to_american(float(dec))
    except (TypeError, ValueError):
        return ""
    return "" if a != a else f"{a:+.0f}"


def _recent_results(limit: int = 12) -> list[dict]:
    if not config.RESULTS.exists():
        return []
    df = pd.read_csv(config.RESULTS, dtype={"game_id": str}, low_memory=False)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date", ascending=False).head(limit)
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def _context(g) -> str:
    """The one-line 'why this game is different' strip. Only facts the model actually uses."""
    bits = []
    for side, team in (("h", g.get("home_team")), ("a", g.get("away_team"))):
        if g.get(f"{side}_b2b") == 1:
            bits.append(f"{team} on a back-to-back")
        elif g.get(f"{side}_3in4") == 1:
            bits.append(f"{team} 3rd game in 4 nights")
    hr, ar = g.get("h_rest"), g.get("a_rest")
    if hr is not None and ar is not None and hr == hr and ar == ar and abs(hr - ar) >= 2:
        side = g.get("home_team") if hr > ar else g.get("away_team")
        bits.append(f"{side} +{int(abs(hr - ar))} days rest")
    km = g.get("a_km")
    if km is not None and km == km and km >= 2500:
        bits.append(f"{g.get('away_team')} {int(km):,} km trip")
    tz = g.get("a_tz")
    if tz is not None and tz == tz and abs(tz) >= 2:
        bits.append(f"{g.get('away_team')} {abs(int(tz))} time zones")
    return " · ".join(bits)


def _goalies(g) -> list[dict]:
    """Each side's goalie and how sure the board is: the day's news where there is any, the
    model's own guess with its confidence where there is not."""
    out = []
    for side, team in (("a", g.get("away_team")), ("h", g.get("home_team"))):
        name = g.get(f"g_{side}_top")
        if not name:
            continue
        status = g.get(f"g_{side}_status") or ""
        if status not in ("confirmed", "likely", "projected"):
            conf = g.get(f"g_{side}_conf")
            status, label = "guess", ("our guess" if conf is None or conf != conf
                                      else f"our guess {int(round(100 * conf))}%")
        else:
            label = status
        out.append({"team": team, "name": str(name).split(" ")[-1], "full": str(name),
                    "status": status, "label": label,
                    "backup": bool(g.get(f"g_{side}_backup"))})
    return out


def _board() -> list[dict]:
    """Upcoming games, freshest prediction per game, ready for the template."""
    if not config.PICKS.exists():
        return []
    df = pd.read_csv(config.PICKS, dtype={"game_id": str}, low_memory=False)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] >= config.today_et()]
    if df.empty:
        return []
    df = df.sort_values("prediction_date").drop_duplicates("game_id", keep="last").copy()
    evs = df[["spread_ev", "total_ev"]].astype(float)
    df["best_ev"] = evs.max(axis=1)
    df["has_play"] = df["spread_strength"].isin(["play", "bold"]) | \
        df["total_strength"].isin(["play", "bold"])
    df["_rank"] = df["best_ev"].fillna(-1) + df["has_play"].astype(float)
    df = df.sort_values("_rank", ascending=False)
    for c in ("p_home_win", "p_away_win", "p_reg_home", "p_ot", "p_reg_away", "spread_p_win",
              "spread_mkt_p", "total_p_win", "total_mkt_p", "mkt_p_home"):
        if c in df:
            df[c + "_pct"] = [_pct(v) for v in df[c]]
    for c in ("spread_ev", "total_ev"):
        df[c + "_pct"] = [_pct(v) for v in df[c]]
    # Prices are formatted from the decimal price at render time. The American string written
    # by predict comes back from the CSV as a float and would print as "-182.0".
    for c in ("spread", "total"):
        df[f"{c}_price"] = [_american(v) for v in df.get(f"{c}_dec", pd.Series(index=df.index, dtype=float))]
    # the total's probabilities are shown net of the push, like the price they are compared with
    live = 1 - df["total_p_push"].astype(float).fillna(0)
    df["total_p_live_pct"] = [_pct(p / l) if l else None
                              for p, l in zip(df["total_p_win"].astype(float), live)]
    df["day"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %-d")
    df["day_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["time_label"] = [(t if isinstance(t, str) and t.strip() else "Time TBD")
                        for t in df.get("puck_drop", pd.Series([""] * len(df), index=df.index))]
    df["kick_sort"] = [(t if isinstance(t, str) else "")
                       for t in df.get("puck_drop_utc", pd.Series([""] * len(df), index=df.index))]
    df["search"] = (df["home_team"].fillna("") + " " + df["away_team"].fillna("") + " "
                    + df.get("g_h_top", pd.Series([""] * len(df), index=df.index)).fillna("") + " "
                    + df.get("g_a_top", pd.Series([""] * len(df), index=df.index)).fillna("")).str.lower()
    def flag(col):
        # a picks.csv written before a column existed has no such column: read that as 0
        return pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1) if col in df \
            else pd.Series(False, index=df.index)

    df["b2b"] = (flag("h_b2b") | flag("a_b2b")).astype(int)
    df["backup"] = (flag("g_h_backup") | flag("g_a_backup")).astype(int)
    df["waiting"] = df["spread_strength"].eq("wait") | df["total_strength"].eq("wait")
    # pandas cannot hold None inside a float64 column, so `where(notna, None)` leaves NaN in place
    # - and bool(nan) is True, which makes every Jinja `{% if %}` pass and renders "nan" on the
    # page. Casting to object first is what actually clears them.
    df = df.astype(object).where(pd.notna(df), None)
    records = df.to_dict("records")
    for r in records:
        r["context"] = _context(r)
        r["goalies"] = _goalies(r)
    return records


def _meta() -> dict:
    path = config.MODEL_DIR / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def _metrics() -> dict:
    if config.METRICS.exists():
        try:
            return json.loads(config.METRICS.read_text())
        except ValueError:
            return {}
    return {}


def _rule(block: dict) -> dict | None:
    for r in block.get("rules") or []:
        if "staking rule" in str(r.get("segment", "")):
            return r
    return None


def _model_note(meta: dict) -> dict:
    """What the last training run concluded, surfaced where a reader will see it."""
    ev = meta.get("eval") or {}
    ml, sp, to = ev.get("moneyline") or {}, ev.get("spread") or {}, ev.get("total") or {}
    if not ml:
        return {}
    pos = next((r for r in sp.get("rules") or [] if "positive" in str(r.get("segment", ""))), None)
    def verdict(edge):
        """How a log-loss difference reads to someone who is not going to look it up."""
        if edge is None:
            return None
        if abs(edge) < 0.0002:
            return "tied"
        return "edged" if edge > 0 else "trailed"

    close = (ml.get("vs_closing") or {}).get("log_loss_edge")
    strict = meta.get("eval_strict") or {}
    ssp = strict.get("spread") or {}
    s_pos = next((r for r in ssp.get("rules") or [] if "positive" in str(r.get("segment", ""))), None)
    return {"strict_seasons": strict.get("test_seasons") or [], "strict_pos": s_pos,
            "strict_rule": _rule(ssp), "strict_z": (ssp.get("significance") or {}).get("z_required"),
            "seasons": ev.get("test_seasons") or [],
            "ml_edge": ml.get("log_loss_edge"), "ml_close": close, "ml_close_verdict": verdict(close),
            "ml_pregame": (ml.get("vs_pregame") or {}).get("log_loss_edge"),
            "spread_rule": _rule(sp), "spread_pos": pos,
            "spread_verdict": (sp.get("significance") or {}).get("verdict"),
            "spread_z": (sp.get("significance") or {}).get("z_required"),
            "total_beats": to.get("beats_market"), "total_ll": to.get("log_loss_model"),
            "total_ll_mkt": to.get("log_loss_market"),
            "goalies": ev.get("goalies") or {},
            "shrink": meta.get("shrink") or {}}


def _backtest(meta: dict) -> dict:
    """ROI by expected value from the backtest, shown until live results can replace it.

    The STRICT holdout when there is one - nothing in it was fitted on the seasons it scores -
    because the page's headline says that is the number to trust, and a table beside it quoting
    the rosier walk-forward would undo the headline.
    """
    strict = meta.get("eval_strict") or {}
    ev = strict if (strict.get("spread") or {}).get("roi_by_ev") else (meta.get("eval") or {})
    sp, to = ev.get("spread") or {}, ev.get("total") or {}
    if not sp.get("roi_by_ev"):
        return {}
    by_total = {b["ev"]: b for b in to.get("roi_by_ev") or []}
    rows = [{"bucket": b["ev"], "spread": b, "total": by_total.get(b["ev"])}
            for b in sp["roi_by_ev"]]
    for b in to.get("roi_by_ev") or []:
        if b["ev"] not in {r["bucket"] for r in rows}:
            rows.append({"bucket": b["ev"], "spread": None, "total": b})
    sig = sp.get("significance") or {}
    return {"rows": rows, "seasons": ev.get("test_seasons") or [], "strict": ev is strict,
            "z_required": sig.get("z_required"), "verdict": sig.get("verdict"),
            "rules": (sp.get("rules") or [])}


def _hit_rows(metrics: dict) -> list[dict]:
    """This season's record by EV at the posted price, puck line and totals side by side."""
    rows: dict[str, dict] = {}
    for key, side in (("spreads", "spread"), ("totals", "total")):
        for r in (metrics.get(key) or {}).get("by_edge") or []:
            row = rows.setdefault(r["bucket"], {"lo": r["lo"], "hi": r["hi"], "spread": None,
                                                "total": None, "label": _ev_label(r["lo"], r["hi"])})
            tone = ""
            if (r.get("n") or 0) >= HIT_RATE_MIN_N and r.get("expected_pct") is not None:
                tone = "up" if (r.get("win_pct") or 0) > r["expected_pct"] else "down"
            row[side] = {**r, "tone": tone}
    return sorted(rows.values(), key=lambda r: r["lo"])


def _ev_label(lo, hi) -> str:
    if lo < 0:
        return "below 0%"
    return f"{100 * lo:g}%+" if hi >= 1 else f"{100 * lo:g}–{100 * hi:g}%"


def _hit_for(rows: list[dict], side: str, ev) -> dict | None:
    try:
        v = float(ev)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    for r in rows:
        if r["lo"] <= v < r["hi"]:
            return {**r[side], "label": r["label"]} if r[side] else None
    return None


def _days(picks: list[dict]) -> list[dict]:
    seen: dict[str, str] = {}
    for p in picks:
        k = p.get("day_key")
        if k and k not in seen:
            seen[k] = p.get("day") or k
    return [{"key": k, "label": v} for k, v in sorted(seen.items())]


def build() -> None:
    config.ensure_dirs()
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["money"] = lambda v: ("+" if (v or 0) >= 0 else "") + f"{v or 0:.2f}"
    picks = _board()
    metrics = _metrics()
    meta = _meta()
    hit_rows = _hit_rows(metrics)
    for g in picks:
        g["spread_hit"] = _hit_for(hit_rows, "spread", g.get("spread_ev"))
        g["total_hit"] = _hit_for(hit_rows, "total", g.get("total_ev"))
    html = env.get_template("index.html").render(
        title=config.SITE_TITLE, picks=picks, m=metrics, model=_model_note(meta),
        backtest=_backtest(meta), calibration=meta.get("scoreline_calibration") or [],
        hit_rows=hit_rows, hit_min=HIT_RATE_MIN_N, results=_recent_results(),
        support_url=config.SUPPORT_URL, support_label=config.SUPPORT_LABEL,
        days=_days(picks), updated=metrics.get("updated", "") or str(config.today_et()),
        spread_min=config.SPREAD_EV_MIN, total_min=config.TOTAL_EV_MIN,
        trained=(meta.get("trained_at") or "")[:10])
    (config.DOCS / "index.html").write_text(html)
    (config.DOCS / ".nojekyll").touch()
    for src in (config.PICKS, config.METRICS, config.RESULTS):
        if src.exists():
            shutil.copy(src, config.DOCS / src.name)
    log.info("built %s: %d games", config.DOCS / "index.html", len(picks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
