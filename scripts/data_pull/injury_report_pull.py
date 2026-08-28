"""Pulls raw player-level injury report rows for each unique game date across
5 NBA seasons (2021-22 through 2025-26) -- the coverage window supported by
the `nbainjuries` package's PDF-parsing logic (see NOTE below).

For each game date D, this resolves and fetches the injury report published
the evening before (nominally 5:30pm ET the day prior, matching the NBA's own
"report by 5pm local the day before" rule), searching a small window of
nearby half-hour timestamps if the exact 5:30pm report isn't available. This
is a deliberately crude, pregame-safe report selection: it never uses a
report timestamped on or after the game date itself (that would only be
correct for back-to-back second games, and using it uniformly would be
inconsistent/leak same-day info).

Output: NBAdata/injury_reports/injury_reports_<season>.csv, one row per
player per resolved report, with the resolved report timestamp and target
game date attached as extra columns. This is raw, unaggregated data --
turning it into an "unavailable player count" feature is a separate
follow-up task.

NOTE: nbainjuries only knows how to parse report PDFs from 2021-10-01 onward
(older reports use a different column layout it can't parse) -- do not
extend SEASONS below that without first fixing the parsing gap.
"""
import asyncio
import os
import time
from datetime import datetime, timedelta

import aiohttp
import pandas as pd
from nbainjuries import injury_asy

SEASONS = ["2021_22", "2022_23", "2023_24", "2024_25", "2025_26"]
MATCHUPS_FILE = "NBAdata/matchups/NBA_{season}_Matchups.csv"
OUTPUT_DIR = "NBAdata/injury_reports"

REPORT_HOUR, REPORT_MINUTE = 17, 30  # 5:30pm ET, the standard "evening before" report
# Offsets (minutes) from the primary 5:30pm-the-day-before target, tried closest-first.
# Capped at 10 candidates spanning a ~4.5hr window (15:00-19:30 ET).
SEARCH_OFFSETS_MIN = [0, -30, 30, -60, 60, -90, 90, -120, 120, -150]

VALIDATE_CONCURRENCY = 15
FETCH_CONCURRENCY = 15


def load_target_dates(season: str) -> list:
    path = MATCHUPS_FILE.format(season=season)
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["DATE"]).dt.date.unique()
    return sorted(dates)


def candidate_timestamps(game_date) -> list:
    prev_day = game_date - timedelta(days=1)
    base = datetime(prev_day.year, prev_day.month, prev_day.day, REPORT_HOUR, REPORT_MINUTE)
    return [base + timedelta(minutes=m) for m in SEARCH_OFFSETS_MIN]


async def resolve_timestamp(game_date, session, validate_sem):
    for ts in candidate_timestamps(game_date):
        async with validate_sem:
            try:
                if await injury_asy.check_reportvalid(ts, session=session):
                    return ts
            except Exception:
                continue
    return None


async def _do_fetch(ts, session, fetch_sem):
    async with fetch_sem:
        return await injury_asy.get_reportdata(ts, session=session, return_df=True)


async def fetch_report_cached(ts, session, fetch_sem, cache, cache_lock):
    """Fetch a report, reusing an in-flight/completed fetch if two dates
    resolve to the same underlying report timestamp."""
    async with cache_lock:
        if ts not in cache:
            cache[ts] = asyncio.create_task(_do_fetch(ts, session, fetch_sem))
        task = cache[ts]
    return await task


async def process_date(game_date, session, validate_sem, fetch_sem, cache, cache_lock):
    ts = await resolve_timestamp(game_date, session, validate_sem)
    if ts is None:
        return game_date, None, None, "no_valid_report_found"
    try:
        df = await fetch_report_cached(ts, session, fetch_sem, cache, cache_lock)
    except Exception as e:
        return game_date, ts, None, f"fetch_error: {type(e).__name__}: {e}"
    return game_date, ts, df, None


async def process_season(season: str):
    target_dates = load_target_dates(season)
    validate_sem = asyncio.Semaphore(VALIDATE_CONCURRENCY)
    fetch_sem = asyncio.Semaphore(FETCH_CONCURRENCY)
    cache = {}
    cache_lock = asyncio.Lock()

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_date(d, session, validate_sem, fetch_sem, cache, cache_lock)
            for d in target_dates
        ]
        results = await asyncio.gather(*tasks)

    rows = []
    failures = []
    for game_date, ts, df, err in results:
        if df is None:
            failures.append((game_date, ts, err))
            continue
        d = df.copy()
        d["target_game_date"] = game_date
        d["report_timestamp"] = ts
        cols = ["target_game_date", "report_timestamp"] + [
            c for c in d.columns if c not in ("target_game_date", "report_timestamp")
        ]
        rows.append(d[cols])

    if rows:
        season_df = pd.concat(rows, ignore_index=True)
        season_df["Team"] = season_df["Team"].astype(str).str.strip().str.lower()
    else:
        season_df = pd.DataFrame()

    return season_df, target_dates, failures


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_targeted = 0
    overall_resolved = 0
    t_start = time.time()

    for season in SEASONS:
        print(f"\n=== {season} ===")
        t0 = time.time()
        season_df, target_dates, failures = asyncio.run(process_season(season))
        elapsed = time.time() - t0

        n_target = len(target_dates)
        n_resolved = n_target - len(failures)
        overall_targeted += n_target
        overall_resolved += n_resolved

        out_path = os.path.join(OUTPUT_DIR, f"injury_reports_{season}.csv")
        season_df.to_csv(out_path, index=False)

        print(f"Saved: {out_path} ({len(season_df)} rows)")
        print(f"Coverage: {n_resolved}/{n_target} dates resolved ({n_resolved / n_target:.1%})")
        print(f"Elapsed: {elapsed:.1f}s")
        if failures:
            print(f"Failed dates ({len(failures)}):")
            for game_date, ts, err in failures[:20]:
                print(f"   {game_date}: {err}")
            if len(failures) > 20:
                print(f"   ... and {len(failures) - 20} more")

    total_elapsed = time.time() - t_start
    print("\n=== Overall ===")
    print(f"Coverage: {overall_resolved}/{overall_targeted} dates resolved ({overall_resolved / overall_targeted:.1%})")
    print(f"Total wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
