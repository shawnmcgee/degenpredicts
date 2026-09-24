"""Tests for the Odds API quota alert. Offline: no network, no API key, no GitHub.

The alert's whole value is saying the right thing at the right time, so what is pinned here is
when it speaks - out of credits, a refused key, or running low with the reset still too far
away - and, just as much, when it keeps quiet: a low balance the reset will refill, the morning
of the 1st, a daily refresh, and an alert its owner already closed.
"""
from __future__ import annotations

import pathlib
from datetime import datetime, timezone

import pytest
import requests

from core import odds_quota as oq

ROOT = pathlib.Path(__file__).resolve().parent.parent


def at(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def quota(used, remaining, status=200, message=""):
    return oq.Quota(status, used, remaining, message)


# ---------------------------------------------------------------------------------
# Reading the headers
# ---------------------------------------------------------------------------------

def test_the_quota_is_read_from_the_headers():
    q = oq.parse(200, {"x-requests-used": "412", "x-requests-remaining": "88",
                       "x-requests-last": "0"}, [])
    assert (q.used, q.remaining, q.message) == (412, 88, "")
    refused = oq.parse(401, {}, {"message": "API key is not valid", "error_code": "INVALID_KEY"})
    assert refused.remaining is None and refused.message == "API key is not valid"
    assert oq.parse(503, {}, None).message == "HTTP 503"


# ---------------------------------------------------------------------------------
# When it speaks, and when it keeps quiet
# ---------------------------------------------------------------------------------

def test_running_low_before_the_reset_is_an_alert():
    v = oq.assess(quota(420, 80), at(2026, 10, 20, 23), alert_below=100)
    assert v.level == "low" and v.title == "Odds API credits running low: 80 left"
    text = " ".join(v.lines)
    assert "80 of 500 credits left" in text and "about 21 a day" in text
    assert "run out around Oct 24, before they reset on Nov 1" in text


def test_a_low_balance_the_reset_will_refill_is_not_an_alert():
    # 20 left on the evening of the 30th at ~16 a day: the reset comes first
    v = oq.assess(quota(480, 20), at(2026, 9, 30, 23), alert_below=100)
    assert v.level == "ok" and "last until they reset on Oct 1" in " ".join(v.lines)


def test_plenty_left_is_not_an_alert_even_at_a_fast_pace():
    assert oq.assess(quota(300, 200), at(2026, 10, 12, 23), alert_below=100).level == "ok"


def test_the_morning_of_the_first_does_not_cry_wolf():
    # 30 credits by 3pm on the 1st is 48 a day if extrapolated from a single morning - which
    # would "run out" by the 11th. The pace is taken over at least three days until three days
    # have passed, and at 10 a day 470 credits last the month.
    v = oq.assess(quota(30, 470), at(2026, 11, 1, 15), alert_below=500)
    assert "about 10 a day" in " ".join(v.lines)
    assert v.level == "ok"
    later = oq.assess(quota(300, 200), at(2026, 11, 10, 15), alert_below=500)
    assert later.level == "low", "once the pace is real, the same rule does speak up"


@pytest.mark.parametrize("q, level", [
    (quota(500, 0), "out"),
    (quota(None, None, 401, "Usage quota has been reached"), "out"),
    (quota(None, None, 401, "API key is not valid"), "refused"),
    (quota(None, None, 503, "HTTP 503"), "unknown"),
    (quota(None, None, 429, "Requests are being made too frequently"), "unknown"),
    (quota(None, None), "unknown"),
])
def test_out_refused_and_unknown(q, level):
    v = oq.assess(q, at(2026, 10, 20, 23), alert_below=100)
    assert v.level == level
    if level in oq.ALERTS:
        assert v.title and v.lines


def test_the_reset_rolls_over_the_year():
    assert oq.next_reset(at(2026, 12, 15, 23)) == at(2027, 1, 1)
    assert oq.next_reset(at(2027, 1, 31, 23, 59)) == at(2027, 2, 1)


def test_an_empty_or_garbled_threshold_falls_back_to_the_default(monkeypatch):
    for raw in ("", "  ", "lots"):
        monkeypatch.setenv("DEGEN_ODDS_ALERT_BELOW", raw)
        assert oq._env_int("DEGEN_ODDS_ALERT_BELOW", 100) == 100
    monkeypatch.setenv("DEGEN_ODDS_ALERT_BELOW", "150")
    assert oq._env_int("DEGEN_ODDS_ALERT_BELOW", 100) == 150


# ---------------------------------------------------------------------------------
# The issue: one at a time, and quiet unless something changed for the worse
# ---------------------------------------------------------------------------------

class FakeGitHub:
    def __init__(self, issue=None):
        self.issue = issue
        self.calls = []

    def latest_alert(self):
        return self.issue

    def create(self, title, body):
        self.calls.append(("create", title, body))
        return {"number": 7}

    def edit(self, number, **fields):
        self.calls.append(("edit", number, fields))
        return {}

    def comment(self, number, body):
        self.calls.append(("comment", number, body))
        return {}

    def did(self, kind):
        return [c for c in self.calls if c[0] == kind]


def alert_issue(level, state="open", closed_at=None):
    return {"number": 7, "state": state, "closed_at": closed_at,
            "body": oq.MARKER.format(level=level) + "\n@owner something"}


NOW = at(2026, 10, 20, 23)
LOW = oq.assess(quota(420, 80), NOW, alert_below=100)
OUT = oq.assess(quota(500, 0), NOW, alert_below=100)
OK = oq.assess(quota(200, 300), NOW, alert_below=100)


def test_the_first_alert_opens_an_issue_that_mentions_the_owner():
    gh = FakeGitHub()
    assert oq.sync(LOW, gh, "@shawnmcgee", NOW) == "opened #7"
    (_, title, body), = gh.did("create")
    assert title == LOW.title
    assert body.startswith(oq.MARKER.format(level="low"))
    assert "@shawnmcgee" in body and "80 of 500 credits left" in body


def test_a_daily_refresh_edits_quietly():
    gh = FakeGitHub(alert_issue("low"))
    assert oq.sync(LOW, gh, "@shawnmcgee", NOW) == "refreshed #7"
    assert gh.did("edit") and not gh.did("comment") and not gh.did("create"), \
        "an edit sends no notification; a daily comment would be spam"


def test_getting_worse_is_a_new_notification():
    gh = FakeGitHub(alert_issue("low"))
    assert oq.sync(OUT, gh, "@shawnmcgee", NOW) == "escalated #7 from low to out"
    (_, _, body), = gh.did("comment")
    assert body.startswith("@shawnmcgee Odds API credits used up")


def test_the_reset_closes_the_alert():
    gh = FakeGitHub(alert_issue("out"))
    assert oq.sync(OK, gh, "@shawnmcgee", NOW) == "closed #7"
    assert gh.did("comment")
    closing = gh.did("edit")[-1][2]
    assert (closing["state"], closing["state_reason"]) == ("closed", "completed")
    assert oq.MARKER.format(level="ok") in closing["body"]
    quiet = FakeGitHub(alert_issue("low", state="closed", closed_at="2026-10-02T23:00:05Z"))
    assert oq.sync(OK, quiet, "@shawnmcgee", NOW).startswith("nothing") and not quiet.calls


def test_an_alert_you_closed_stays_closed_this_month_unless_it_gets_worse():
    closed = alert_issue("low", state="closed", closed_at="2026-10-18T09:00:00Z")
    gh = FakeGitHub(closed)
    assert oq.sync(LOW, gh, "@shawnmcgee", NOW) == "nothing: #7 was closed this month"
    assert not gh.calls
    worse = FakeGitHub(closed)
    assert oq.sync(OUT, worse, "@shawnmcgee", NOW) == "opened #7"
    last_month = FakeGitHub(alert_issue("low", state="closed", closed_at="2026-09-29T23:00:00Z"))
    assert oq.sync(LOW, last_month, "@shawnmcgee", NOW) == "opened #7", "a new month, a new alert"


def test_the_resets_own_close_does_not_silence_the_new_month():
    """September's alert, closed by the check itself when the credits reset on October 1, must
    not read as an acknowledgement of October's."""
    gh = FakeGitHub(alert_issue("out"))
    oq.sync(OK, gh, "@shawnmcgee", at(2026, 10, 1, 23))
    closed_by_reset = {**alert_issue("out", state="closed", closed_at="2026-10-01T23:00:04Z"),
                       "body": gh.did("edit")[-1][2]["body"]}
    later = FakeGitHub(closed_by_reset)
    assert oq.sync(LOW, later, "@shawnmcgee", NOW) == "opened #7"


def test_a_failed_check_touches_nothing():
    gh = FakeGitHub(alert_issue("low"))
    v = oq.assess(quota(None, None, 503, "HTTP 503"), NOW, alert_below=100)
    assert oq.sync(v, gh, "@shawnmcgee", NOW).startswith("nothing") and not gh.calls


# ---------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------

def test_no_key_means_nothing_to_check(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(oq.requests, "get", lambda *a, **k: pytest.fail("no key, no call"))
    assert oq.main([]) == 0


def test_the_key_never_reaches_the_log(monkeypatch, caplog):
    secret = "abc123secretkey"
    monkeypatch.setenv("ODDS_API_KEY", secret)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def boom(url, params=None, **kw):
        raise requests.ConnectionError(
            f"Max retries exceeded with url: {url}?apiKey={params['apiKey']}")
    monkeypatch.setattr(oq.requests, "get", boom)
    with caplog.at_level("INFO"):
        assert oq.main([]) == 0, "an unreachable API is not an alert: the next check retries"
    assert secret not in caplog.text and "ConnectionError" in caplog.text


def test_an_alert_that_cannot_be_delivered_fails_the_run(monkeypatch):
    class Response:
        status_code, headers = 200, {"x-requests-used": "495", "x-requests-remaining": "5"}

        def json(self):
            return []

    class Unreachable:
        def __init__(self, *a, **k):
            pass

        def latest_alert(self):
            raise requests.HTTPError("403 Resource not accessible by integration")

    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setattr(oq.requests, "get", lambda *a, **k: Response())
    monkeypatch.setattr(oq, "GitHub", Unreachable)
    monkeypatch.setattr(oq, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: NOW)}))
    assert oq.main([]) == 1, "a failed run is GitHub's own email - the alert still gets through"
    assert oq.main(["--dry-run"]) == 0


def test_the_workflow_runs_daily_costs_nothing_and_never_commits():
    wf = (ROOT / ".github" / "workflows" / "odds-quota.yml").read_text()
    assert "schedule:" in wf and "workflow_dispatch:" in wf
    assert "issues: write" in wf and "contents: read" in wf and "contents: write" not in wf
    assert "secrets.ODDS_API_KEY" in wf and "secrets.GITHUB_TOKEN" in wf
    assert "vars.DEGEN_ODDS_ALERT_BELOW" in wf
    assert "git push" not in wf
    assert oq.SPORTS_URL.endswith("/v4/sports"), "the free endpoint - the check must cost nothing"


def test_a_full_run_sends_the_right_github_calls(monkeypatch, tmp_path):
    """End to end through main() and the real GitHub class, with only the network faked."""
    class Resp:
        def __init__(self, status=200, payload=None, headers=None):
            self.status_code, self._payload, self.headers = status, payload, headers or {}
            self.content = b"x" if payload is not None else b""

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    sent = []

    def github(self, method, url, **kw):
        sent.append((method, url, kw.get("params"), kw.get("json"),
                     self.headers.get("Authorization")))
        if method == "GET":      # one unrelated issue and one pull request: neither is the alert
            return Resp(payload=[{"number": 3, "state": "open", "body": "a bug"},
                                 {"number": 4, "state": "open", "pull_request": {},
                                  "body": oq.MARKER.format(level="low")}])
        return Resp(payload={"number": 12})

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    monkeypatch.setenv("GITHUB_REPOSITORY", "shawnmcgee/degenpredicts")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "shawnmcgee")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("DEGEN_ODDS_ALERT_BELOW", raising=False)
    monkeypatch.setattr(oq.requests, "get", lambda url, params=None, **k: Resp(
        payload=[], headers={"x-requests-used": "420", "x-requests-remaining": "80"}))
    monkeypatch.setattr(oq.requests.Session, "request", github)
    monkeypatch.setattr(oq, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: NOW),
                                                       "fromisoformat": datetime.fromisoformat}))
    assert oq.main([]) == 0
    (m1, u1, p1, _, auth), (m2, u2, _, j2, _) = sent
    assert (m1, u1) == ("GET", "https://api.github.com/repos/shawnmcgee/degenpredicts/issues")
    assert p1 == {"state": "all", "per_page": 100} and auth == "Bearer t0ken"
    assert (m2, u2) == ("POST", "https://api.github.com/repos/shawnmcgee/degenpredicts/issues")
    assert j2["title"] == "Odds API credits running low: 80 left"
    assert "@shawnmcgee" in j2["body"] and "run out around Oct 24" in j2["body"]
    assert "Alert issue: opened #12" in summary.read_text()
