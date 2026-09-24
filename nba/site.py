"""Render the static NBA board into docs/nba/ (GitHub Pages serves the docs folder).

    python -m nba.site

Laid out like the other boards - same palette, same tabs, same controls - with the parts that
are basketball's own:

* **who is out, and what it is worth** - each club's notable absences and question marks from
  the day's injury report, with the points the player is worth over his replacement, because
  that is the single largest thing that moves an NBA line;
* **three bets a game** - the spread and total pointed like the NFL's (our number against the
  line), and the moneyline priced like the NHL's (our probability against the price);
* **our number beside the market's**, and the ratings-alone number beside both, so a reader can
  see how much of the published view is the market and how much is the model.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config

log = logging.getLogger("nba.site")
TEMPLATES = Path(__file__).resolve().parent / "templates"

# A bucket needs this many graded picks before its hit rate is coloured against break-even.
HIT_RATE_MIN_N = 20
# An absence or question mark is shown on the card once it is worth this many points.
NOTE_MIN_POINTS = 1.0
STATUS_LABEL = {"out": "Out", "suspension": "Out", "doubtful": "Doubtful",
                "questionable": "Questionable", "day-to-day": "Day-to-day",
                "probable": "Probable", "absent": "Missed recent games"}


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
    tz = g.get("a_tz")
    if tz is not None and tz == tz and abs(tz) >= 2:
        bits.append(f"{g.get('away_team')} {abs(int(tz))} time zones")
    alt = g.get("altitude")
    if alt is not None and alt == alt and alt >= 1000:
        bits.append("at altitude")
    if g.get("neutral"):
        bits.append(f"neutral site{' - ' + g['venue_city'] if g.get('venue_city') else ''}")
    return " · ".join(bits)


def _injuries(g) -> list[dict]:
    """Each side's notable absences and question marks, largest first."""
    out = []
    for side, team in (("a", g.get("away_team")), ("h", g.get("home_team"))):
        try:
            notes = json.loads(g.get(f"{side}_notes") or "[]")
        except (TypeError, ValueError):
            notes = []
        items = []
        for n in notes:
            if (n.get("impact") or 0) < NOTE_MIN_POINTS:
                continue
            items.append({"name": str(n.get("name") or "").split(" ")[-1] or str(n.get("id")),
                          "full": n.get("name"), "status": n.get("status"),
                          "label": STATUS_LABEL.get(n.get("status"), n.get("status")),
                          "impact": n.get("impact"),
                          "sure": (n.get("p_out") or 0) >= 1.0})
        if items:
            out.append({"team": team, "players": items[:4]})
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
    strengths = ("spread_strength", "total_strength", "ml_strength")
    df["has_play"] = df[list(strengths)].isin(["play", "bold"]).any(axis=1)
    df["waiting"] = df[list(strengths)].eq("wait").any(axis=1)
    gap = pd.concat([df["spread_disagree"].abs(), df["total_disagree"].abs()], axis=1).max(axis=1)
    df["_rank"] = gap.fillna(0) + 100 * df["has_play"].astype(float)
    df = df.sort_values("_rank", ascending=False)
    for c in ("p_home_win", "p_away_win", "spread_p_win", "spread_mkt_p", "total_p_win",
              "total_mkt_p", "ml_p_win", "ml_mkt_p", "mkt_p_home"):
        if c in df:
            df[c + "_pct"] = [_pct(v) for v in df[c]]
    for c in ("spread_ev", "total_ev", "ml_ev"):
        df[c + "_pct"] = [_pct(v) for v in df[c]]
    # A whole-number line can push, and the market's de-vigged probability is the chance of
    # winning GIVEN no push - so ours is shown the same way, or a 7-point spread would read two
    # points worse than the price when it is not.
    for c in ("spread", "total"):
        live = 1 - pd.to_numeric(df[f"{c}_p_push"], errors="coerce").fillna(0)
        df[f"{c}_p_live_pct"] = [_pct(pw / lv) if lv else None
                                 for pw, lv in zip(pd.to_numeric(df[f"{c}_p_win"], errors="coerce"), live)]
    # Prices are formatted from the decimal price at render time. The American string written
    # by predict comes back from the CSV as a float and would print as "-110.0".
    for c in ("spread", "total", "ml"):
        df[f"{c}_price"] = [_american(v) for v in df.get(f"{c}_dec", pd.Series(index=df.index, dtype=float))]
    df["day"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %-d")
    df["day_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["time_label"] = [(t if isinstance(t, str) and t.strip() else "Time TBD")
                        for t in df.get("tip_label", pd.Series([""] * len(df), index=df.index))]
    df["kick_sort"] = [(t if isinstance(t, str) else "")
                       for t in df.get("tip_sort", pd.Series([""] * len(df), index=df.index))]
    df["search"] = (df["home_team"].fillna("") + " " + df["away_team"].fillna("") + " "
                    + df.get("h_notes", pd.Series([""] * len(df), index=df.index)).fillna("") + " "
                    + df.get("a_notes", pd.Series([""] * len(df), index=df.index)).fillna("")).str.lower()

    def flag(col):
        # a picks.csv written before a column existed has no such column: read that as 0
        return pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1) if col in df \
            else pd.Series(False, index=df.index)

    df["b2b"] = (flag("h_b2b") | flag("a_b2b")).astype(int)
    df["out_star"] = [int(any(i["impact"] >= 3 for s in _injuries(r) for i in s["players"]))
                      for r in df.to_dict("records")]
    # pandas cannot hold None inside a float64 column, so `where(notna, None)` leaves NaN in place
    # - and bool(nan) is True, which makes every Jinja `{% if %}` pass and renders "nan" on the
    # page. Casting to object first is what actually clears them.
    df = df.astype(object).where(pd.notna(df), None)
    records = df.to_dict("records")
    for r in records:
        r["context"] = _context(r)
        r["injuries"] = _injuries(r)
        m, nm = r.get("pub_margin"), r.get("nm_margin")
        if m is not None:
            r["model_side"] = (r["home_team"] if m >= 0 else r["away_team"], abs(m))
        if nm is not None:
            r["ratings_side"] = (r["home_team"] if nm >= 0 else r["away_team"], abs(nm))
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


def _model_note(meta: dict) -> dict:
    """What the last training run concluded, surfaced where a reader will see it. If the model
    does not beat the closing line, the page says so - publishing the number while hiding that
    would be the dishonest version of this project."""
    ev = meta.get("eval") or {}
    mm, tm = ev.get("margin_market") or {}, ev.get("total_market") or {}
    mn, tn = ev.get("margin_nomarket") or {}, ev.get("total_nomarket") or {}
    if not mm.get("mae_model"):
        return {}
    ml = ev.get("moneyline") or {}
    # the moneyline bucket at the staking bar, if the backtest had one - the page states it with
    # its standard error and per-season record, never as a headline ROI on its own
    bar = next((r for r in ml.get("roi_by_ev") or [] if r.get("candidate")
                and r.get("lo", -1) >= config.ML_EV_MIN - 1e-9), None)
    seasons = mm.get("test_seasons") or []
    return {"seasons": seasons, "n": mm.get("n_test_total"),
            "span": (f"{config.season_label(seasons[0])} to {config.season_label(seasons[-1])}"
                     if seasons else ""),
            "ml_bar": bar, "ml_z": (ml.get("significance") or {}).get("z_required"),
            "m_mae": mm.get("mae_model"), "m_line": mm.get("mae_market_baseline"),
            "m_ats": mm.get("ats_rate"), "m_ats_n": mm.get("ats_n"), "m_se": mm.get("ats_stderr"),
            "m_beats": mm.get("beats_market"), "t_mae": tm.get("mae_model"),
            "t_line": tm.get("mae_market_baseline"), "t_ats": tm.get("ats_rate"),
            "t_beats": tm.get("beats_market"), "nm_mae": mn.get("mae_model_on_closing_games"),
            "nt_mae": tn.get("mae_model_on_closing_games"),
            "verdict": (mm.get("significance") or {}).get("verdict"),
            "t_verdict": (tm.get("significance") or {}).get("verdict"),
            "ml": ev.get("moneyline") or {}, "shrink": meta.get("shrink") or {}}


def _backtest(meta: dict) -> dict:
    """ATS by disagreement from the walk-forward, shown until live results can replace it."""
    ev = meta.get("eval") or {}
    sp, to = ev.get("margin_market") or {}, ev.get("total_market") or {}
    if not sp.get("ats_by_disagreement"):
        return {}
    sig = sp.get("significance") or {}
    seasons = sp.get("test_seasons") or []
    return {"spread": sp["ats_by_disagreement"], "total": to.get("ats_by_disagreement") or [],
            "seasons": seasons, "n": sp.get("n_test_total"),
            "span": (f"{config.season_label(seasons[0])} to {config.season_label(seasons[-1])}"
                     if seasons else ""),
            "z_required": sig.get("z_required"), "verdict": sig.get("verdict"),
            "t_verdict": (to.get("significance") or {}).get("verdict"),
            "break_even": sp.get("break_even_pct")}


def _bucket_label(key: str, lo: float, hi: float) -> str:
    """A grader bucket the way a reader says it: "0.5–1" and "3+" points off the line, or
    "negative", "2–5%" and "10%+" EV. Built from the row's own bounds, never re-stated edges, so
    a card and the table beside it always name the same bucket."""
    if key == "moneyline":
        if hi <= 0:
            return "negative"
        lo_pct, hi_pct = round(100 * lo, 1), round(100 * hi, 1)
        return f"{lo_pct:g}%+" if hi >= 1 else f"{lo_pct:g}–{hi_pct:g}%"
    return f"{lo:g}+" if hi >= 99 else f"{lo:g}–{hi:g}"


def _hit_row(key: str, r: dict, be: float) -> dict:
    """One bucket's record, labelled, and coloured against its bar once it has the games to be.

    Spreads and totals are coloured against break-even; the moneyline against the rate its own
    prices implied, because a favourite winning 70% of the time is what the price already said.
    """
    ref = r.get("expected_pct") if key == "moneyline" else be
    tone = ""
    if ref is not None and (r.get("n") or 0) >= HIT_RATE_MIN_N:
        tone = "up" if (r.get("win_pct") or 0) > ref else "down"
    return {**r, "tone": tone, "label": _bucket_label(key, float(r["lo"]), float(r["hi"]))}


def _hit_rows(metrics: dict) -> dict:
    """This season's record by disagreement (spreads, totals) and by EV (moneyline)."""
    be = metrics.get("break_even") or 52.38
    out = {}
    for key in ("spreads", "totals", "moneyline"):
        rows = [_hit_row(key, r, be) for r in (metrics.get(key) or {}).get("by_edge") or []
                if (r.get("wins") or 0) + (r.get("losses") or 0)]
        if rows:
            out[key] = rows
    return out


def _hit_for(metrics: dict, key: str, value) -> dict | None:
    """The season record at a pick's own disagreement (or EV), for one market, or None.

    A disagreement is bucketed by size, either side of the line; EV keeps its sign, because a
    pick priced at -4% is not one priced at +4%.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    v = v if key == "moneyline" else abs(v)
    for r in (metrics.get(key) or {}).get("by_edge") or []:
        if r["lo"] <= v < r["hi"] and (r.get("wins") or 0) + (r.get("losses") or 0):
            return _hit_row(key, r, metrics.get("break_even") or 52.38)
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
    for g in picks:
        g["spread_hit"] = _hit_for(metrics, "spreads", g.get("spread_disagree"))
        g["total_hit"] = _hit_for(metrics, "totals", g.get("total_disagree"))
        g["ml_hit"] = _hit_for(metrics, "moneyline", g.get("ml_ev"))
    html = env.get_template("index.html").render(
        title=config.SITE_TITLE, picks=picks, m=metrics, model=_model_note(meta),
        backtest=_backtest(meta), hit_rows=_hit_rows(metrics),
        hit_min=HIT_RATE_MIN_N, results=_recent_results(),
        support_url=config.SUPPORT_URL, support_label=config.SUPPORT_LABEL,
        days=_days(picks), updated=metrics.get("updated", "") or str(config.today_et()),
        spread_min=config.SPREAD_EDGE_MIN, total_min=config.TOTAL_EDGE_MIN,
        ml_min=config.ML_EV_MIN, wait_points=config.WAIT_POINTS,
        season=config.season_label(config.season_of(config.today_et())),
        trained=(meta.get("trained_at") or "")[:10], odds_source=config.ODDS_SOURCE)
    (config.DOCS / "index.html").write_text(html)
    (config.DOCS / ".nojekyll").touch()
    for src in (config.PICKS, config.METRICS, config.RESULTS):
        if src.exists():
            shutil.copy(src, config.DOCS / src.name)
    log.info("built %s: %d games", config.DOCS / "index.html", len(picks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
