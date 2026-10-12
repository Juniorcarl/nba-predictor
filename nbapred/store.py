"""Incremental storage of games and box scores as plain CSV files.

Files are per season and append-mostly so git stores daily changes cheaply.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd

from . import espn
from .config import BOX_DIR, ET, GAMES_DIR, SEASON_TYPES

log = logging.getLogger(__name__)

GAME_COLS = ["game_id", "date", "start_utc", "season", "season_type", "completed", "state", "neutral",
             "home_id", "home_abbr", "home_name", "home_score",
             "away_id", "away_abbr", "away_name", "away_score"]
BOX_COLS = ["game_id", "team_id", "player_id", "player", "pos", "starter", "dnp", "min",
            "pts", "reb", "ast", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "oreb", "tov", "stl", "blk"]


def today_et() -> date:
    return datetime.now(ET).date()


def season_for(day: date) -> int:
    """ESPN season label (ending year) for a calendar date."""
    return day.year + 1 if day.month >= 8 else day.year


def season_window(season: int) -> tuple[date, date]:
    return date(season - 1, 10, 1), date(season, 6, 30)


# ------------------------------------------------------------------ games
def games_path(season: int):
    return GAMES_DIR / f"{season}.csv"


def load_games(seasons=None) -> pd.DataFrame:
    files = sorted(GAMES_DIR.glob("*.csv"))
    if seasons is not None:
        files = [f for f in files if int(f.stem) in set(seasons)]
    if not files:
        return pd.DataFrame(columns=GAME_COLS)
    df = pd.concat([pd.read_csv(f, dtype={"game_id": str, "home_id": str, "away_id": str}) for f in files],
                   ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _save_games(season: int, df: pd.DataFrame) -> None:
    df = df.drop_duplicates("game_id", keep="last").sort_values(["date", "game_id"])
    df[GAME_COLS].to_csv(games_path(season), index=False)


def update_schedule(start: date, end: date, sleep: float = 0.2) -> int:
    """Fetch scoreboards for each ET day in [start, end] and upsert games."""
    by_season: dict[int, list[dict]] = {}
    d = start
    fails = 0
    while d <= end:
        try:
            rows = espn.parse_scoreboard(espn.fetch_scoreboard(d))
            fails = 0
        except RuntimeError as e:
            log.error("%s", e)
            rows = []
            fails += 1
            if fails >= 8:  # ESPN unreachable: stop instead of retrying for hours
                raise RuntimeError("ESPN scoreboard failed 8 days in a row - giving up this run") from e
        for r in rows:
            r["date"] = d.isoformat()
            if r["season"] is None:
                r["season"] = season_for(d)
            by_season.setdefault(int(r["season"]), []).append(r)
        d += timedelta(days=1)
        time.sleep(sleep)

    n = 0
    for season, rows in by_season.items():
        new = pd.DataFrame(rows)
        p = games_path(season)
        old = pd.read_csv(p, dtype={"game_id": str, "home_id": str, "away_id": str}) if p.exists() else None
        df = pd.concat([old, new], ignore_index=True) if old is not None else new
        for c in GAME_COLS:
            if c not in df:
                df[c] = None
        _save_games(season, df)
        n += len(rows)
    return n


# ------------------------------------------------------------------ box scores
def box_path(season: int):
    return BOX_DIR / f"{season}.csv"


def load_box(seasons=None) -> pd.DataFrame:
    files = sorted(BOX_DIR.glob("*.csv"))
    if seasons is not None:
        files = [f for f in files if int(f.stem) in set(seasons)]
    if not files:
        return pd.DataFrame(columns=BOX_COLS)
    return pd.concat([pd.read_csv(f, dtype={"game_id": str, "team_id": str, "player_id": str}) for f in files],
                     ignore_index=True)


def update_boxscores(max_games: int | None = None, sleep: float = 0.25) -> int:
    """Fetch box scores for completed games that we don't have yet."""
    games = load_games()
    if games.empty:
        return 0
    games = games[games["completed"].astype(bool) & games["season_type"].isin(list(SEASON_TYPES))]
    have = set(load_box()["game_id"].astype(str))
    todo = games[~games["game_id"].isin(have)].sort_values("date")
    if max_games:
        todo = todo.head(max_games)
    log.info("box scores to fetch: %d", len(todo))

    fetched = 0
    pending: dict[int, list[dict]] = {}
    for i, g in enumerate(todo.itertuples(index=False), 1):
        try:
            rows = espn.parse_boxscore(espn.fetch_summary(g.game_id), g.game_id)
        except RuntimeError as e:
            log.error("%s", e)
            continue
        if not rows:
            log.warning("empty box score for %s", g.game_id)
            continue
        pending.setdefault(int(g.season), []).extend(rows)
        fetched += 1
        if i % 100 == 0:
            _flush_box(pending)
            pending = {}
            log.info("  %d / %d", i, len(todo))
        time.sleep(sleep)
    _flush_box(pending)
    return fetched


def _flush_box(pending: dict[int, list[dict]]) -> None:
    for season, rows in pending.items():
        p = box_path(season)
        df = pd.DataFrame(rows)[BOX_COLS]
        df.to_csv(p, mode="a", header=not p.exists(), index=False)
