"""Tests for the site's front door.

The landing page belongs to no sport, so it gets its own suite rather than living in either
sport's. Its whole job is to read what the pipelines have published and turn that into a
chooser, so the things worth pinning are: it never imports a sport, it skips a sport that has
published nothing, and it degrades rather than raising when a file is missing or malformed.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

from core import landing

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Every sport with a live board. ncaab is deliberately absent: it has no workflows yet, so it
# publishes nothing and correctly has no card and no tab.
LIVE_SPORTS = {
    "cfb": ("cfb/templates/index.html", "College football"),
    "nfl": ("nfl/templates/index.html", "NFL"),
    "epl": ("epl/templates/index.html", "Premier League"),
}


def _publish(docs, slug, *, index=True, metrics=None, picks=None):
    folder = docs / slug
    folder.mkdir(parents=True, exist_ok=True)
    if index:
        (folder / "index.html").write_text("<html><body>board</body></html>")
    if metrics is not None:
        (folder / "metrics.json").write_text(json.dumps(metrics))
    if picks is not None:
        (folder / "picks.csv").write_text(picks)
    return folder


HEADER = ("game_id,date,week,total_strength,spread_strength\n")


def test_never_imports_a_sport():
    """The chooser reads published files. Importing a pipeline would couple every sport to
    the front page and undo the isolation the rest of the repo is built on."""
    from pathlib import Path
    src = (Path(landing.__file__)).read_text()
    for i, line in enumerate(src.splitlines(), 1):
        assert not re.match(r"\s*(from|import)\s+(cfb|nfl|ncaab)\b", line), \
            f"core/landing.py:{i} imports a sport: {line.strip()}"


def test_only_shows_sports_that_published_something(tmp_path):
    docs = tmp_path / "docs"
    _publish(docs, "cfb", metrics={"updated": "2026-09-08"}, picks=HEADER)
    # nfl folder exists but has no index.html - the pipeline has not published yet
    (docs / "nfl").mkdir(parents=True, exist_ok=True)
    cards = landing.collect(docs, today="2026-09-08")
    assert [c["slug"] for c in cards] == ["cfb"]

    _publish(docs, "nfl", metrics={"updated": "2026-09-08"}, picks=HEADER)
    cards = landing.collect(docs, today="2026-09-08")
    # order follows the registry, not the filesystem
    assert [c["slug"] for c in cards] == ["cfb", "nfl"]


def test_counts_only_upcoming_games(tmp_path):
    """picks.csv accumulates history and can hold several predictions per game."""
    docs = tmp_path / "docs"
    picks = HEADER + "\n".join([
        "g1,2026-09-01,1,pass,pass",       # already played
        "g2,2026-09-10,2,play,pass",       # upcoming, a play
        "g3,2026-09-11,2,pass,bold",       # upcoming, a play
        "g4,2026-09-12,2,pass,pass",       # upcoming, no play
        "g4,2026-09-12,2,pass,pass",       # duplicate prediction for the same game
    ]) + "\n"
    _publish(docs, "nfl", metrics={"updated": "2026-09-08"}, picks=picks)
    card = landing.collect(docs, today="2026-09-08")[0]
    assert card["board"]["games"] == 3, "played games and duplicates must not be counted"
    assert card["board"]["week"] == 2
    assert card["board"]["plays"] == 2
    assert card["has_data"] is True


def test_survives_missing_and_malformed_files(tmp_path):
    """One sport publishing something broken must not take the front page down with it."""
    docs = tmp_path / "docs"
    _publish(docs, "cfb")                                   # no metrics, no picks at all
    _publish(docs, "nfl", metrics={"updated": "2026-09-08"},
             picks="this is not,a valid picks file\n")
    (docs / "cfb" / "metrics.json").write_text("{not json")

    cards = landing.collect(docs, today="2026-09-08")
    assert len(cards) == 2
    cfb = next(c for c in cards if c["slug"] == "cfb")
    assert cfb["updated"] is None and cfb["has_data"] is False
    landing.build(docs)                                       # must render regardless
    assert (docs / "index.html").exists()


def test_renders_a_chooser_with_a_link_per_sport(tmp_path):
    docs = tmp_path / "docs"
    picks = HEADER + "g1,2026-09-10,2,play,pass\n"
    _publish(docs, "cfb", metrics={
        "updated": "2026-09-08",
        "spreads": {"all_games": {"n": 40, "units": 2.5, "win_pct": 55.0, "clv": 0.31}},
        "totals": {"all_games": {"n": 40, "units": -1.25, "win_pct": 47.5, "clv": -0.10}},
    }, picks=picks)
    _publish(docs, "nfl", metrics={"updated": "2026-09-08"}, picks=HEADER)

    html = landing.build(docs).read_text()
    assert 'href="cfb/"' in html and 'href="nfl/"' in html
    assert "College Football" in html and "NFL" in html
    assert "+2.50u" in html and "-1.25u" in html          # signed units, both directions
    assert "+0.31" in html                                 # CLV carried through
    assert "No games on the board right now." in html      # the NFL card, with an empty board
    assert "1 play" in html and "1 plays" not in html      # singular, not "1 plays"
    # a pandas-style NaN must never reach the page
    body = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I).lower()
    assert "nan" not in body


def test_renders_with_nothing_published(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    html = landing.build(docs).read_text()
    assert "Nothing published yet" in html
    assert (docs / ".nojekyll").exists()


def test_every_publishing_workflow_rebuilds_the_front_page():
    """A sport that publishes without refreshing the chooser leaves the front page stale."""
    from pathlib import Path
    wf = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    for name in ("cfb-predict", "cfb-grade", "nfl-predict", "nfl-grade"):
        text = (wf / f"{name}.yml").read_text()
        assert "python -m core.landing" in text, f"{name}.yml does not rebuild the chooser"


# ---------------------------------------------------------------------------------
# Cross-sport navigation
# ---------------------------------------------------------------------------------
def test_every_board_links_back_to_the_chooser():
    """Each board is its own page under docs/<slug>/, so without a link up there is no way back
    to the other sports except the browser's back button."""
    for slug, (tpl, _label) in LIVE_SPORTS.items():
        html = (ROOT / tpl).read_text()
        assert 'href="../"' in html, f"{slug} has no link back to the chooser"
        assert 'nav class="sports"' in html, f"{slug} has no sport tabs"


def test_sport_tabs_are_reciprocal():
    """Every live board must link to every other one.

    The tabs were added to cfb and nfl before the Premier League existed, so those two pages
    linked only to each other. The chooser could reach the EPL board but neither sport page
    could, which makes it a dead end from anywhere except the front door. A new sport has to
    update the others, and this is what says so.
    """
    missing = []
    for slug, (tpl, _label) in LIVE_SPORTS.items():
        html = (ROOT / tpl).read_text()
        for other in LIVE_SPORTS:
            if other == slug:
                continue
            if f'href="../{other}/"' not in html:
                missing.append(f"{slug} does not link to {other}")
    assert not missing, "sport tabs are not reciprocal: " + "; ".join(missing)


def test_each_board_marks_itself_as_the_current_tab():
    """Without aria-current the tab row gives no indication of which page you are on, and it is
    also what the stylesheet keys the highlight off."""
    for slug, (tpl, label) in LIVE_SPORTS.items():
        html = (ROOT / tpl).read_text()
        assert 'aria-current="page" href="./"' in html, f"{slug} marks no current tab"
        nav = re.search(r'<nav class="sports".*?</nav>', html, re.S).group(0)
        current = re.search(r'aria-current="page" href="\./">([^<]+)<', nav).group(1)
        assert current == label, f"{slug} labels its own tab {current!r}, expected {label!r}"


def test_every_live_sport_has_a_card_on_the_chooser():
    """A board that publishes but has no entry in core.landing.SPORTS is unreachable from the
    front page."""
    slugs = {s["slug"] for s in landing.SPORTS}
    for slug in LIVE_SPORTS:
        assert slug in slugs, f"{slug} publishes a board but has no card on the chooser"


def test_every_committing_workflow_commits_before_pulling():
    """The ordering bug that lost the first EPL board, checked across all three sports.

    `python -m <sport>.site` writes docs/<sport>/index.html as an UNTRACKED file. If the remote
    has gained a commit that also creates it, git refuses to clobber it and aborts the pull -
    and a trailing `|| true` swallows that abort, so the commit lands on a stale base and the
    push is rejected non-fast-forward.

    This lives in the sport-neutral suite on purpose: it is a property of every publishing
    workflow, and pinning it per sport is how it came to be fixed in one place and left broken
    in six others.
    """
    wf = ROOT / ".github" / "workflows"
    jobs = sorted(p.name for p in wf.glob("*.yml")
                  if any(k in p.name for k in ("-predict", "-grade", "-train")))
    assert len(jobs) >= 9, f"expected every sport's three jobs, found {jobs}"
    for name in jobs:
        text = (wf / name).read_text()
        if "git push" not in text:
            continue
        add, commit = text.index("git add "), text.index("git commit -m")
        pull, push = text.index("git pull"), text.index("git push")
        assert add < pull, f"{name}: pulls before staging"
        assert commit < pull, f"{name}: commits after pulling"
        assert pull < push, f"{name}: pushes before rebasing"
        # Strip comments first: the explanation of this very bug quotes `|| true`.
        commands = [ln for ln in text.split("git config user.name")[1].splitlines()
                    if not ln.strip().startswith("#")]
        assert not any("|| true" in ln for ln in commands), \
            f"{name}: swallows a failed git command, which hides a doomed push"
