"""One HTTP session for the NBA pipeline, with retries, timeouts and a browser-ish UA.

Sport-local on purpose, matching ``nhl/http.py`` and ``epl/http.py``. This one mostly talks to
GitHub release storage (a season of box scores is one file) and ESPN's public JSON; sharing a
session would mean a timeout tuned for one of them silently changing the behaviour of a live
job in another sport that commits to main every morning.
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
        # Retry-After is not honoured, for the reason the EPL pipeline learned the hard way: a
        # hostile Retry-After sleep is not capped by backoff_max, and one parked a single fetch
        # for 288 seconds. Against a host that wants us gone, a bounded failure is better.
        retry = Retry(total=config.HTTP_RETRIES, backoff_factor=1.5, backoff_max=20,
                      respect_retry_after_header=False,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=8))
        s.headers.update({"User-Agent": "degenpredicts-nba/1.0 (+https://github.com)",
                          "Accept": "application/json,text/csv,*/*"})
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
            log.warning("GET %s -> HTTP %s", url, r.status_code)
        return None
    try:
        return r.json()
    except ValueError:
        log.warning("GET %s returned non-JSON (upstream layout change?)", url)
        return None
