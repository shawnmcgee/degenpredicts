"""The NBA props job, end to end - what ``nba-props.yml`` runs.

    python -m nba.props.run              # refresh, (re)train if due, price, grade, render
    python -m nba.props.run --retrain    # force a refit
    python -m nba.props.run --no-fetch   # use the committed player files

Retraining is due when there are no models, when they are a week old, or when they were fitted
under a different scikit-learn than the one installed - a pickled model is only promised to load
under the version that wrote it, so a version change refits rather than gambling on a load. A
fit is about eight minutes on a runner: the replay, five walk-forward seasons and the final fit
for each of four markets.

Each step is independent: a failed Kalshi pull still grades and renders, and a failed grade still
renders the board.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone

from .. import config

log = logging.getLogger("nba.props.run")
MAX_MODEL_AGE = timedelta(days=7)


def retrain_due() -> str | None:
    meta = config.PROPS_MODEL_DIR / "meta.json"
    if not meta.exists():
        return "no models yet"
    m = json.loads(meta.read_text())
    missing = [k for k in m.get("markets", {})
               if not (config.PROPS_MODEL_DIR / f"{k}.joblib").exists()]
    if missing:
        return f"model files missing: {missing}"
    try:
        import sklearn
        if m.get("sklearn") != sklearn.__version__:
            return f"scikit-learn {m.get('sklearn')} -> {sklearn.__version__}"
    except ImportError:
        pass
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(m["trained_at"])
        if age > MAX_MODEL_AGE:
            return f"models are {age.days} days old"
    except (KeyError, ValueError):
        return "no training date"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args(argv)

    from ..sources import hoopr
    from . import board, grade, model, site
    if not a.no_fetch:
        hoopr.update()
    why = "asked to" if a.retrain else retrain_due()
    if why:
        log.info("retraining props models (%s)", why)
        model.train(model.assemble())
    for step, fn in (("board", board.run), ("grade", grade.grade), ("site", site.build)):
        try:
            fn()
        except Exception:
            log.exception("props %s failed - carrying on", step)
            if step == "site":
                raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
