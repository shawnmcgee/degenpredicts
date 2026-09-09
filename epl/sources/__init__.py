"""Selects the backend that supplies match history and prices.

Two are implemented and both produce the identical games/lines schema, so everything downstream
- features, training, prediction, grading - is written against the schema and never against a
particular source. That is what made swapping the primary a configuration change rather than a
rewrite when the original host stopped answering.

    matchdata     (default) a GitHub-hosted aggregate of the football-data archives
    footballdata  football-data.co.uk itself, kept working for when it is reachable again
"""
from __future__ import annotations

import logging

from .. import config

log = logging.getLogger(__name__)


def active():
    """The configured source module."""
    if config.SOURCE == "footballdata":
        from . import footballdata
        return footballdata
    from . import matchdata
    return matchdata


def name() -> str:
    return config.SOURCE
