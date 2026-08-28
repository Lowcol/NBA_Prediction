"""Offline tests for the live injury-report fetch/parse used by
Team{1,2}_PlayersOut. No real network or real PDF files: the row-clustering
and column-classification logic is tested directly against small
extract_words()-shaped fixtures (mirroring the real report's coordinates),
and the fetch path is tested by mocking at the network/PDF-parsing boundary,
the same altitude test_api.py uses for its own external-data mocks.
"""

import httpx
import pytest

import injury_report as ir


def word(text, x0, top):
    """Minimal stand-in for one entry of pdfplumber's page.extract_words()."""
    return {"text": text, "x0": x0, "top": top}


# One page's worth of words, shaped like a real downloaded report (see
# injury_report.py's module docstring / this feature's design doc): header row,
# then two player rows under one team block, then a stray Reason-continuation
# line with no PlayerName/CurrentStatus of its own (must be ignored).
SAMPLE_PAGE_WORDS = [
    word("GameDate", 23.1, 107.7),
    word("GameTime", 119.6, 107.7),
    word("Matchup", 200.0, 107.7),
    word("Team", 264.2, 107.7),
    word("PlayerName", 425.0, 107.7),
    word("CurrentStatus", 585.7, 107.7),
    word("Reason", 666.1, 107.7),
    word("01/06/2026", 24.1, 128.9),
    word("07:00(ET)", 120.6, 128.9),
    word("CLE@IND", 201.0, 128.9),
    word("ClevelandCavaliers", 265.2, 128.9),
    word("Allen,Jarrett", 426.0, 128.9),
    word("Questionable", 586.7, 128.9),
    word("Injury/Illness-Illness;Illness", 667.1, 128.9),
    word("Livingston,Chris", 426.0, 151.0),
    word("Out", 586.7, 151.0),
    word("GLeague-Two-Way", 667.1, 151.0),
    # Stray wrapped-Reason line: no PlayerName/CurrentStatus, must not be counted.
    word("continued-reason-text", 667.1, 217.1),
]


def test_cluster_rows_groups_same_line_words_and_splits_different_lines():
    rows = ir._cluster_rows(SAMPLE_PAGE_WORDS)
    tops = [round(row[0]["top"], 1) for row in rows]
    assert tops == [107.7, 128.9, 151.0, 217.1]


def test_find_header_row_locates_the_header():
    rows = ir._cluster_rows(SAMPLE_PAGE_WORDS)
    header = ir._find_header_row(rows)
    assert {w["text"] for w in header} == {
        "GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason",
    }


def test_classify_column_assigns_words_to_nearest_header():
    rows = ir._cluster_rows(SAMPLE_PAGE_WORDS)
    boundaries = ir._column_boundaries(ir._find_header_row(rows))
    assert ir._classify_column(426.0, boundaries) == "PlayerName"
    assert ir._classify_column(586.7, boundaries) == "CurrentStatus"
    assert ir._classify_column(265.2, boundaries) == "Team"


def test_parse_pages_forward_fills_team_and_counts_unavailable_statuses():
    counts = ir._parse_pages([SAMPLE_PAGE_WORDS])
    # Both players belong to Cleveland (Team only appears on the first row of
    # its block); only "Out" is unavailable here ("Questionable" is not).
    assert counts == {"cleveland cavaliers": 1}


def test_parse_pages_ignores_reason_only_rows():
    # The stray continued-reason-text row (no PlayerName/CurrentStatus) must not
    # produce a spurious team entry or crash the classifier.
    counts = ir._parse_pages([SAMPLE_PAGE_WORDS])
    assert sum(counts.values()) == 1


def test_parse_pages_carries_team_and_boundaries_across_pages_without_their_own_header():
    # Real downloaded reports don't repeat the header on every page (confirmed
    # against a live report) -- a later page can start mid-block with no Team
    # or header words of its own, so both must carry over from the prior page.
    page_two_words = [
        word("Doncic,Luka", 426.0, 80.0),
        word("Doubtful", 586.7, 80.0),
        word("Reason-text-here", 667.1, 80.0),
    ]
    counts = ir._parse_pages([SAMPLE_PAGE_WORDS, page_two_words])
    assert counts["cleveland cavaliers"] == 2  # Livingston(Out) + Doncic(Doubtful)


def test_parse_pages_includes_zero_count_teams_mentioned_in_the_report():
    # A team with only "Available"/"Probable" players should still show up
    # (at 0), not be omitted -- callers rely on .get(team, 0) either way, but
    # the report explicitly mentions this team so it belongs in the dict.
    page_words = [
        word("GameDate", 23.1, 107.7),
        word("Team", 264.2, 107.7),
        word("PlayerName", 425.0, 107.7),
        word("CurrentStatus", 585.7, 107.7),
        word("MiamiHeat", 265.2, 128.9),
        word("White,Coby", 426.0, 128.9),
        word("Available", 586.7, 128.9),
    ]
    counts = ir._parse_pages([page_words])
    assert counts == {"miami heat": 0}


def test_parse_pages_ignores_the_repeated_page_title_line():
    # Every real page repeats an "Injury Report: <date> <time> AM/PM" title line
    # above the table (confirmed against a live downloaded report). Its "Injury"
    # token sits close enough to the Team column's x0 to be nearest-neighbor-
    # misclassified as a team, which would otherwise steal counts from the real
    # team via forward-fill. This caught a real bug during live testing.
    title_row = [
        word("Injury", 289.1, 45.5),
        word("Report:", 355.6, 45.5),
        word("04/10/26", 438.3, 45.5),
        word("05:30", 539.8, 45.5),
        word("PM", 602.5, 45.5),
    ]
    page_words = title_row + SAMPLE_PAGE_WORDS
    counts = ir._parse_pages([page_words])
    assert "injury" not in counts
    assert counts == {"cleveland cavaliers": 1}


def test_normalize_team_name_handles_concatenated_report_names():
    assert ir._normalize_team_name("ClevelandCavaliers") == "cleveland cavaliers"
    assert ir._normalize_team_name("LAClippers") == "la clippers"
    assert ir._normalize_team_name("LosAngelesLakers") == "los angeles lakers"
    assert ir._normalize_team_name("Philadelphia76ers") == "philadelphia 76ers"


def test_normalize_team_name_falls_back_to_lowercase_for_unknown_input():
    assert ir._normalize_team_name("Some Unknown Team") == "some unknown team"


def test_fetch_latest_injury_counts_returns_empty_dict_on_network_failure(monkeypatch):
    ir._cache["counts"] = None
    ir._cache["fetched_at"] = 0.0

    def raise_network_error(url, timeout):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(ir.httpx, "get", raise_network_error)

    result = ir.fetch_latest_injury_counts()

    assert result == {}


def test_fetch_latest_injury_counts_returns_empty_dict_when_nothing_found(monkeypatch):
    ir._cache["counts"] = None
    ir._cache["fetched_at"] = 0.0

    class NotFoundResponse:
        status_code = 404

    monkeypatch.setattr(ir.httpx, "get", lambda url, timeout: NotFoundResponse())

    result = ir.fetch_latest_injury_counts()

    assert result == {}


def test_fetch_latest_injury_counts_parses_a_successful_response(monkeypatch):
    ir._cache["counts"] = None
    ir._cache["fetched_at"] = 0.0

    class OkResponse:
        status_code = 200
        content = b"fake-pdf-bytes"

    monkeypatch.setattr(ir.httpx, "get", lambda url, timeout: OkResponse())
    monkeypatch.setattr(ir, "_parse_pdf_bytes", lambda content: {"denver nuggets": 3})

    result = ir.fetch_latest_injury_counts()

    assert result == {"denver nuggets": 3}


def test_fetch_latest_injury_counts_uses_cache_within_ttl(monkeypatch):
    ir._cache["counts"] = {"denver nuggets": 1}
    ir._cache["fetched_at"] = ir.time.monotonic()

    def fail_if_called(url, timeout):
        raise AssertionError("should not re-fetch while cache is fresh")

    monkeypatch.setattr(ir.httpx, "get", fail_if_called)

    assert ir.fetch_latest_injury_counts() == {"denver nuggets": 1}
