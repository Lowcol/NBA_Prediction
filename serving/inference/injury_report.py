"""Live NBA injury report fetch + parse, for the Team{1,2}_PlayersOut feature.

Unlike every other model feature (precomputed ahead of time into a rolling-stats
snapshot file), a team's current injury report is a fact about *today's* specific
game -- it can't be baked into a static snapshot. Serving therefore fetches and
parses the report at prediction time, the same reasoning that makes B2B computed
dynamically rather than snapshotted (see is_back_to_back() in predictor.py).

Parsing uses pdfplumber's raw per-word coordinates (extract_words()) rather than
its table-extraction helpers: the injury report PDF has no reliable ruling lines
for extract_table()/extract_tables() to key off (confirmed against a real
downloaded report). This also avoids the `nbainjuries` package used for the
historical pull (scripts/data_pull/injury_report_pull.py) -- that needs
tabula-py + a JVM, and the serving Docker images deliberately don't carry Java.

Like the rest of this codebase's date handling (season_label_for_date(),
date.today() in predictor.py/main.py), this doesn't do real timezone conversion
-- it just uses local wall-clock time. The 48-hour lookback when walking
backward through report timestamps comfortably absorbs the resulting slop.
"""

import io
import logging
import time
from datetime import datetime, timedelta

import httpx
import pdfplumber

logger = logging.getLogger(__name__)

REPORT_URL_TEMPLATE = (
    "https://ak-static.cms.nba.com/referee/injury/"
    "Injury-Report_{ts:%Y-%m-%d}_{hour12:02d}_{ts.minute:02d}{ampm}.pdf"
)
STEP_MINUTES = 30
LOOKBACK_HOURS = 48
REQUEST_TIMEOUT_SECONDS = 10

# Must match the training-side historical-injury-feature script's threshold
# exactly (same pair, not more/fewer) so both halves of this feature agree on
# what "unavailable" means.
UNAVAILABLE_STATUSES = {"Out", "Doubtful"}

ROW_TOLERANCE = 2.0  # points; safety margin for clustering words onto one visual row
REQUIRED_HEADERS = {"Team", "PlayerName", "CurrentStatus"}
HEADER_TEXTS = {"GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"}

# The report's Team column has no space between city and nickname (e.g.
# "ClevelandCavaliers", "LAClippers", "Philadelphia76ers" -- confirmed against a
# real downloaded report). Map the concatenated form back to this repo's
# "city name" lowercase convention (matches TEAM_NAME in predictor.py's
# load_team_stats()) via a static lookup rather than guessing word boundaries,
# which is fragile for cases like the digit in "76ers".
CANONICAL_TEAM_NAMES = [
    "atlanta hawks", "boston celtics", "brooklyn nets", "charlotte hornets",
    "chicago bulls", "cleveland cavaliers", "dallas mavericks", "denver nuggets",
    "detroit pistons", "golden state warriors", "houston rockets", "indiana pacers",
    "la clippers", "los angeles lakers", "memphis grizzlies", "miami heat",
    "milwaukee bucks", "minnesota timberwolves", "new orleans pelicans",
    "new york knicks", "oklahoma city thunder", "orlando magic",
    "philadelphia 76ers", "phoenix suns", "portland trail blazers",
    "sacramento kings", "san antonio spurs", "toronto raptors", "utah jazz",
    "washington wizards",
]
_TEAM_NAME_BY_CONCAT = {name.replace(" ", ""): name for name in CANONICAL_TEAM_NAMES}

# In-process cache: fine for the nightly batch job (which only calls this once
# per run anyway) and keeps the API from re-downloading + re-parsing a
# multi-page PDF on every single prediction request. 10 minutes balances
# freshness (reports are published roughly hourly-to-half-hourly) against load.
CACHE_TTL_SECONDS = 600
_cache: dict = {"counts": None, "fetched_at": 0.0}


def _normalize_team_name(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "")
    return _TEAM_NAME_BY_CONCAT.get(key, raw.strip().lower())


def _report_url(ts: datetime) -> str:
    hour12 = ts.hour % 12
    if hour12 == 0:
        hour12 = 12
    ampm = "AM" if ts.hour < 12 else "PM"
    return REPORT_URL_TEMPLATE.format(ts=ts, hour12=hour12, ampm=ampm)


def _candidate_timestamps(start: datetime):
    """Walk backward in 30-min steps from `start` (rounded down to :00/:30),
    capped at LOOKBACK_HOURS."""
    minute = 0 if start.minute < 30 else 30
    ts = start.replace(minute=minute, second=0, microsecond=0)
    steps = (LOOKBACK_HOURS * 60) // STEP_MINUTES
    for _ in range(steps + 1):
        yield ts
        ts -= timedelta(minutes=STEP_MINUTES)


def _cluster_rows(words: list[dict], tolerance: float = ROW_TOLERANCE) -> list[list[dict]]:
    """Group words into visual rows by clustering on `top` (y-position)."""
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda w: w["top"]):
        if rows and abs(word["top"] - rows[-1][0]["top"]) <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return rows


def _find_header_row(rows: list[list[dict]]) -> list[dict] | None:
    for row in rows:
        texts = {w["text"] for w in row}
        if REQUIRED_HEADERS.issubset(texts):
            return row
    return None


def _column_boundaries(header_row: list[dict]) -> list[tuple[float, str]]:
    """(x0, column_name) pairs, sorted by x0, for the header's own columns."""
    return sorted((w["x0"], w["text"]) for w in header_row if w["text"] in HEADER_TEXTS)


def _classify_column(x0: float, boundaries: list[tuple[float, str]]) -> str:
    return min(boundaries, key=lambda b: abs(b[0] - x0))[1]


def _parse_pages(pages: list[list[dict]]) -> dict[str, int]:
    """Core parsing algorithm, operating on each page's extract_words() output.

    Team and Matchup/GameDate/GameTime are only present on the first row of
    their block (not repeated per player), so Team is forward-filled from the
    most recent row that had one -- across pages too, since a real report's
    header row (confirmed against a live download) does NOT necessarily repeat
    on every page; later pages can start mid-block with no Team of their own.
    Column boundaries are likewise carried forward from the most recent page
    that had a header, rather than assumed fixed or required on every page.

    The Reason column is never read: it's the only column that wraps across
    multiple lines, and ignoring it means "does this row have both a
    PlayerName and a CurrentStatus" is a safe row-validity check.
    """
    counts: dict[str, int] = {}
    last_team: str | None = None
    boundaries: list[tuple[float, str]] | None = None

    for words in pages:
        if not words:
            continue
        rows = _cluster_rows(words)
        header_row = _find_header_row(rows)
        if header_row is not None:
            boundaries = _column_boundaries(header_row)
        if boundaries is None:
            # No header seen yet on any page so far -- nothing to classify against.
            continue

        for row in rows:
            if row is header_row:
                continue
            # Every page repeats a title line ("Injury Report: <date> <time> AM/PM")
            # above the table -- confirmed against a real downloaded report. Its
            # "Injury" token sits close enough to the Team column's x0 to be
            # nearest-neighbor-misclassified as a Team value, which would corrupt
            # forward-fill for real rows above the page's first genuine team
            # mention. "Report:" (with the colon) never appears in real report
            # data, so it's a safe, distinctive marker to filter this line on.
            if any(w["text"] == "Report:" for w in row):
                continue
            by_col: dict[str, list[str]] = {}
            for word in row:
                col = _classify_column(word["x0"], boundaries)
                by_col.setdefault(col, []).append(word["text"])

            team = " ".join(by_col["Team"]) if "Team" in by_col else None
            player = " ".join(by_col["PlayerName"]) if "PlayerName" in by_col else None
            status = " ".join(by_col["CurrentStatus"]) if "CurrentStatus" in by_col else None

            if team:
                last_team = _normalize_team_name(team)
            if player and status and last_team is not None:
                counts.setdefault(last_team, 0)
                if status in UNAVAILABLE_STATUSES:
                    counts[last_team] += 1

    return counts


def _parse_pdf_bytes(content: bytes) -> dict[str, int]:
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        return _parse_pages([page.extract_words() for page in pdf.pages])


def _fetch_latest_injury_counts_uncached(now: datetime | None = None) -> dict[str, int]:
    start = now or datetime.now()
    for ts in _candidate_timestamps(start):
        url = _report_url(ts)
        try:
            resp = httpx.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            logger.debug("Injury report request failed for %s: %s", url, exc)
            continue
        if resp.status_code != 200:
            continue
        try:
            counts = _parse_pdf_bytes(resp.content)
        except Exception as exc:
            logger.warning("Failed to parse injury report at %s: %s", url, exc)
            continue
        if counts:
            logger.info("Loaded injury report %s (%d teams)", url, len(counts))
            return counts
    logger.warning(
        "No usable NBA injury report found in the last %d hours; "
        "Team{1,2}_PlayersOut will default to 0 for this run.", LOOKBACK_HOURS,
    )
    return {}


def fetch_latest_injury_counts(now: datetime | None = None) -> dict[str, int]:
    """{team_name_lowercase: count of players Out/Doubtful} for every team
    mentioned in the most recently published NBA injury report.

    Cached in-process for CACHE_TTL_SECONDS (see module docstring). Never
    raises: any failure (network error, no report found within the lookback
    window, unexpected PDF layout) is logged and results in an empty dict, so
    a missing injury signal degrades prediction quality slightly rather than
    breaking serving -- the same posture as load_predictor()'s registry
    fallback. Callers should default any team missing from the dict to 0.
    """
    cached = _cache["counts"]
    if cached is not None and (time.monotonic() - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached

    try:
        counts = _fetch_latest_injury_counts_uncached(now)
    except Exception as exc:  # belt-and-suspenders: this must never crash the caller
        logger.warning("Injury report fetch failed unexpectedly: %s", exc)
        counts = {}

    _cache["counts"] = counts
    _cache["fetched_at"] = time.monotonic()
    return counts
