"""One HTTP session for the NFL pipeline, with retries, timeouts and a browser-ish UA.

Sport-local on purpose, matching ``ncaab/http.py``. The college pipeline runs against CFBD
and Kalshi - small JSON responses on a metered API key. This one runs against nflverse, which
serves multi-megabyte static CSVs from GitHub releases: different sizes, different failure
modes, different sensible timeouts. Sharing one session would mean a retry or timeout tuned
for a 7 MB roster crosswalk silently changing the behaviour of a live college job that commits
to main every morning.

Keeping them separate costs about forty lines and buys a blast radius of one sport.
"""
from __future__ import annotations

import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

log = logging.getLogger(__name__)
_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(total=config.HTTP_RETRIES, backoff_factor=1.5,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=8))
        s.headers.update({"User-Agent": "degenpredicts-nfl/1.0 (+https://github.com)"})
        _session = s
    return _session


def get(url: str, **kw) -> requests.Response | None:
    kw.setdefault("timeout", config.HTTP_TIMEOUT)
    try:
        return session().get(url, **kw)
    except requests.RequestException as e:
        log.warning("GET %s failed: %s", url, e)
        return None


def get_json(url: str, **kw):
    r = get(url, **kw)
    if r is None or r.status_code != 200:
        if r is not None:
            log.debug("GET %s -> HTTP %s", url, r.status_code)
        return None
    try:
        return r.json()
    except ValueError:
        log.warning("GET %s returned non-JSON (upstream layout change?)", url)
        return None
