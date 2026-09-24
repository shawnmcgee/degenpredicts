"""Warn before the shared Odds API key runs out of credits.

    python -m core.odds_quota              # check, write the job summary, raise or clear the alert
    python -m core.odds_quota --dry-run    # check and print; never touch an issue

Every board that prices off The Odds API shares one key, and the free plan is 500 credits a
month. When they run out, every one of those boards loses its prices at once and nothing goes
red: the pipelines fall back to assumed prices or stop staking, quietly. This makes it loud.

**The check is free.** It reads the quota from the headers of ``GET /v4/sports``, an endpoint
The Odds API does not charge for, so it can run every day at no cost.

**The alert is a GitHub issue that @-mentions the repo's owner** - an email, and a push from the
GitHub mobile app. It is raised when:

* the key is out of credits, or the API refuses it; or
* fewer than ``DEGEN_ODDS_ALERT_BELOW`` credits remain (default 100, about six days at the
  busiest point of the year) and, at this month's pace, they run out before the credits reset
  on the 1st. Near the end of a month with credits to spare, a low balance is not a problem, so
  it is not an alert.

There is only ever one issue. It is opened once, its numbers are refreshed quietly on each check,
it gets a new comment - a new notification - only when things get worse, and the first check
after the credits reset closes it. Closing it yourself means "I know": nothing more is said that
month unless it gets worse. If an alert is needed and the issue cannot be written, the run fails
instead, so GitHub's own failed-run email still gets through.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger("core.odds_quota")

SPORTS_URL = "https://api.the-odds-api.com/v4/sports"
MARKER = "<!-- odds-quota-alert level={level} -->"
MARKER_RE = re.compile(r"<!-- odds-quota-alert level=(\w+) -->")
SEVERITY = {"ok": 0, "low": 1, "out": 2, "refused": 2}
ALERTS = ("low", "out", "refused")
# The pace is read as if at least this many days of the month had passed. On the morning of the
# 1st the month holds a single run's usage, and extrapolating that to 30 days cries wolf.
MIN_PACE_DAYS = 3.0
WHAT_TO_DO = ("**What to do:** upgrade the plan at the-odds-api.com, or pause a board you can "
              "spare until the reset (**Actions → its picks workflow → ⋯ → Disable workflow**). "
              "This issue refreshes itself on each daily check and closes once the credits reset.")


def _env_int(name: str, default: int) -> int:
    """Empty counts as unset: Actions passes an unset repo variable as ''."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not a whole number; using %d", name, raw, default)
        return default


@dataclass
class Quota:
    status: int                  # HTTP status of the check
    used: int | None             # credits used since the last reset
    remaining: int | None        # credits left until the next one
    message: str = ""            # the API's own words, when it refused the key


@dataclass
class Verdict:
    level: str                   # ok | low | out | refused | unknown
    title: str = ""
    lines: list[str] = field(default_factory=list)


def parse(status: int, headers, body) -> Quota:
    def num(name):
        try:
            return int(float(headers.get(name)))
        except (TypeError, ValueError):
            return None
    message = ""
    if status != 200:
        if isinstance(body, dict):
            message = str(body.get("message") or body.get("error_code") or "")
        message = message or f"HTTP {status}"
    return Quota(status, num("x-requests-used"), num("x-requests-remaining"), message)


def read(key: str) -> Quota | None:
    """One free call. None if the API could not be reached at all: the next check retries."""
    try:
        r = requests.get(SPORTS_URL, params={"apiKey": key}, timeout=30)
    except requests.RequestException as e:
        # the exception's text holds the URL, and the URL holds the key: log the kind only
        log.warning("Odds API unreachable (%s)", type(e).__name__)
        return None
    try:
        body = r.json()
    except ValueError:
        body = None
    return parse(r.status_code, r.headers, body)


def month_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_reset(now: datetime) -> datetime:
    """Credits reset on the 1st of each month."""
    return (month_start(now) + timedelta(days=32)).replace(day=1)


def assess(q: Quota, now: datetime, alert_below: int) -> Verdict:
    reset = next_reset(now)
    on_reset = f"on {reset:%b} 1"
    if q.remaining is not None and q.remaining <= 0:
        return Verdict("out", "Odds API credits used up", [
            f"All {q.used or 0:,} credits are used.",
            f"Every board that prices off the key is without prices until they reset {on_reset}."])
    if q.status != 200:
        if q.status in (401, 403):
            if re.search(r"quota|credit|usage", q.message, re.I):
                return Verdict("out", "Odds API credits used up", [
                    f"The API says: “{q.message}”",
                    f"Every board that prices off the key is without prices until they reset "
                    f"{on_reset}."])
            return Verdict("refused", "Odds API key refused", [
                f"The API says: “{q.message}”",
                "Check the `ODDS_API_KEY` secret: the key may have been rotated, mistyped or "
                "cancelled. Every board that prices off it has no prices until it is fixed."])
        return Verdict("unknown",
                       lines=[f"The quota check failed ({q.message}); the next one retries."])
    if q.used is None or q.remaining is None:
        return Verdict("unknown", lines=["The API answered without its quota headers."])

    days = max((now - month_start(now)).total_seconds() / 86400, MIN_PACE_DAYS)
    pace = q.used / days
    lines = [f"{q.remaining:,} of {q.used + q.remaining:,} credits left "
             f"({q.used:,} used this month, about {pace:.0f} a day)."]
    runs_out = now + timedelta(days=q.remaining / pace) if pace > 0 else None
    if runs_out is not None and runs_out < reset:
        lines.append(f"At that pace they run out around {runs_out:%b} {runs_out.day}, "
                     f"before they reset {on_reset}.")
    else:
        lines.append(f"At that pace they last until they reset {on_reset}.")
    if q.remaining <= alert_below and runs_out is not None and runs_out < reset:
        return Verdict("low", f"Odds API credits running low: {q.remaining:,} left", lines)
    return Verdict("ok", lines=lines)


class GitHub:
    """The handful of issue calls the alert needs, on the workflow's own token."""

    def __init__(self, repo: str, token: str, api: str = "https://api.github.com"):
        self.base = f"{api}/repos/{repo}"
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})

    def _call(self, method: str, path: str, **kw):
        r = self.s.request(method, self.base + path, timeout=30, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    def latest_alert(self) -> dict | None:
        """The newest alert issue, open or closed. The API lists newest first."""
        for issue in self._call("GET", "/issues", params={"state": "all", "per_page": 100}) or []:
            if "pull_request" not in issue and MARKER_RE.search(issue.get("body") or ""):
                return issue
        return None

    def create(self, title: str, body: str) -> dict:
        return self._call("POST", "/issues", json={"title": title, "body": body})

    def edit(self, number: int, **fields) -> dict:
        return self._call("PATCH", f"/issues/{number}", json=fields)

    def comment(self, number: int, body: str) -> dict:
        return self._call("POST", f"/issues/{number}/comments", json={"body": body})


def _bullets(lines) -> str:
    return "\n".join(f"- {line}" for line in lines)


def body_for(v: Verdict, mention: str) -> str:
    return (f"{MARKER.format(level=v.level)}\n{mention} {v.title}.\n\n{_bullets(v.lines)}\n\n"
            f"{WHAT_TO_DO}\n")


def _level_of(issue: dict) -> str:
    found = MARKER_RE.search(issue.get("body") or "")
    return found.group(1) if found else "low"


def _closed_this_month(issue: dict, now: datetime) -> bool:
    try:
        closed = datetime.fromisoformat(str(issue.get("closed_at")).replace("Z", "+00:00"))
    except ValueError:
        return False
    return closed >= month_start(now)


def sync(v: Verdict, gh, mention: str, now: datetime) -> str:
    """Raise, refresh, escalate or clear the one alert issue. Returns what it did."""
    if v.level not in SEVERITY:
        return "nothing: the check itself failed"
    issue = gh.latest_alert()
    is_open = issue is not None and issue.get("state") == "open"
    if v.level == "ok":
        if not is_open:
            return "nothing: the credits are fine"
        gh.comment(issue["number"], f"The credits are fine again.\n\n{_bullets(v.lines)}")
        # recorded as ok: closed on the 1st by the reset, it must not read as "I know" about the
        # new month's alerts - only a close by you is that
        gh.edit(issue["number"], state="closed", state_reason="completed",
                body=MARKER_RE.sub(MARKER.format(level="ok"), issue.get("body") or ""))
        return f"closed #{issue['number']}"
    body = body_for(v, mention)
    if not is_open:
        # closed this month at this level or worse - by you, as "I know", or by a check that saw
        # the pace ease - so saying it again would be noise; only something worse speaks up
        if (issue is not None and _closed_this_month(issue, now)
                and SEVERITY.get(_level_of(issue), 0) >= SEVERITY[v.level]):
            return f"nothing: #{issue['number']} was closed this month"
        return f"opened #{gh.create(v.title, body)['number']}"
    was = _level_of(issue)
    gh.edit(issue["number"], title=v.title, body=body)
    # a new comment is a new notification: only when it is worse, never for a daily refresh
    if v.level != was and SEVERITY[v.level] >= SEVERITY.get(was, 0):
        gh.comment(issue["number"], f"{mention} {v.title}.\n\n{_bullets(v.lines)}")
        return f"escalated #{issue['number']} from {was} to {v.level}"
    return f"refreshed #{issue['number']}"


def _summary(text: str) -> None:
    """The job summary: what the GitHub mobile app shows for a run, rendered as a page."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="check and print; never touch an issue")
    args = ap.parse_args(argv)

    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        log.info("no ODDS_API_KEY: nothing to check")
        _summary("### Odds API quota\n\nNo `ODDS_API_KEY` secret is set, so there is nothing "
                 "to check.")
        return 0
    q = read(key)
    now = datetime.now(timezone.utc)
    v = (assess(q, now, _env_int("DEGEN_ODDS_ALERT_BELOW", 100)) if q else
         Verdict("unknown", lines=["The Odds API could not be reached; the next check retries."]))
    for line in v.lines:
        log.info("%s", line)

    code = 0
    repo, token = os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_TOKEN", "")
    if args.dry_run:
        did = "dry run: no issue touched"
    elif not (repo and token):
        did = "no GITHUB_REPOSITORY / GITHUB_TOKEN: printed only"
    else:
        owner = os.environ.get("GITHUB_REPOSITORY_OWNER") or repo.split("/")[0]
        try:
            did = sync(v, GitHub(repo, token), f"@{owner}", now)
        except requests.RequestException as e:
            did = f"could not write the alert issue ({type(e).__name__})"
            # an alert nobody receives is no alert: fail, and GitHub emails the failed run
            code = 1 if v.level in ALERTS else 0
    log.info("%s: %s", v.level, did)
    _summary(f"### Odds API quota: {v.level}\n\n{_bullets(v.lines)}\n\nAlert issue: {did}.")
    return code


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
