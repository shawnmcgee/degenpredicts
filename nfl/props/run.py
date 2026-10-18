"""The props job, end to end - what ``nfl-props.yml`` runs.

    python -m nfl.props.run              # refresh, (re)train if due, price, grade, render
    python -m nfl.props.run --retrain    # force a refit

Retraining is due when there are no models, when they are a week old, or when they were
fitted under a different scikit-learn than the one installed - a pickled model is only
promised to load under the version that wrote it, so a version change refits rather than
gambling on a load. A fit takes a few minutes on a runner.

Each step is independent: a failed Kalshi pull still grades and renders, and a failed grade
still renders the board.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone

from .. import config
from ..sources import nflverse

log = logging.getLogger("nfl.props.run")
MAX_MODEL_AGE = timedelta(days=7)


def retrain_due() -> str | None:
    meta = config.PROPS_MODEL_DIR / "meta.json"
    if not meta.exists():
        return "no models yet"
    m = json.loads(meta.read_text())
    missing = [k for k in m.get("markets", {}) if not (config.PROPS_MODEL_DIR / f"{k}.joblib").exists()]
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

    from . import board, data, grade, model, site
    if not a.no_fetch:
        data.update()
        nflverse.update_games()
    why = "asked to" if a.retrain else retrain_due()
    if why:
        log.info("retraining props models (%s)", why)
        model.train(model.assemble(fetch=False))
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
