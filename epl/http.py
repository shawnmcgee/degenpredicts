"""One HTTP session for the EPL pipeline, with retries, timeouts and a browser-ish UA.

Sport-local on purpose, matching ``nfl/http.py`` and ``ncaab/http.py``. This one talks to
football-data.co.uk, a small static-file host that is neither a CDN nor an API: it is slower
than GitHub's raw endpoint, it occasionally serves a stale byte range mid-refresh, and it is
the sort of site that goes down for an afternoon. Sharing a session with the other sports
would mean a retry policy tuned for that host silently changing the behaviour of a live job
that commits to main every morning.

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
        # backoff_factor is deliberately small and the total is capped: this pipeline fetches
        # ~50 files on a first backfill, so any per-request slack is paid fifty times over.
        #
        # respect_retry_after_header=False is the important one, and it was learned the hard
        # way. football-data.co.uk answers a GitHub runner with HTTP 503 - it is neither
        # stalling nor dropping packets, it is actively refusing - and a 503 may carry a
        # Retry-After header. urllib3 honours that header by DEFAULT, and a Retry-After sleep
        # is NOT capped by backoff_max: the cap applies only to the computed exponential
        # backoff. So a hostile Retry-After parked a single file fetch for 288 seconds, which
        # no amount of tuning backoff_factor or backoff_max could have prevented.
        #
        # Nor could the wall-clock budget, which is checked after a request returns and so
        # cannot interrupt one that is already asleep. Declining to honour the header is the
        # only thing that bounds this, and for a static file archive it costs nothing: if the
        # host wants us gone it will keep saying 503, and the breaker will notice.
        retry = Retry(total=config.HTTP_RETRIES, connect=config.HTTP_RETRIES,
                      read=config.HTTP_RETRIES, status=config.HTTP_RETRIES,
                      backoff_factor=0.5, backoff_max=4,
                      respect_retry_after_header=False,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=8))
        s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=8))
        # football-data.co.uk serves a plain static file but 403s a bare python-requests UA.
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; degenpredicts-epl/1.0; +https://github.com)",
            "Accept": "text/csv,text/plain,*/*",
        })
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
