"""ESPN public (unofficial) NBA data feed.

Endpoints are undocumented, so every parser is defensive: missing fields become
None rather than crashing the daily run. Network code is kept separate from
parsing so the parsers can be tested against saved JSON.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date

import requests

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0 (nba-predictor; personal project)"}

_session = requests.Session()
_session.headers.update(HEADERS)


def _get(url: str, params: dict | None = None, tries: int = 3) -> dict:
    for attempt in range(tries):
        try:
            r = _session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            log.warning("ESPN %s -> HTTP %s", url, r.status_code)
        except requests.RequestException as e:
            log.warning("ESPN %s failed: %s", url, e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ESPN request failed after {tries} tries: {url} {params}")


# ---------------------------------------------------------------- fetchers
def fetch_scoreboard(day: date) -> dict:
    return _get(f"{BASE}/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 100})


def fetch_summary(event_id: str) -> dict:
    return _get(f"{BASE}/summary", {"event": event_id})


def fetch_injuries() -> dict:
    return _get(f"{BASE}/injuries")


def fetch_roster(team_id: str) -> dict:
    return _get(f"{BASE}/teams/{team_id}/roster")


# ---------------------------------------------------------------- parsers
def _int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def parse_scoreboard(js: dict) -> list[dict]:
    """One row per game: ids, teams, scores, status, season."""
    games = []
    for ev in js.get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        season = ev.get("season") or {}
        status = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        row = {
            "game_id": str(ev.get("id")),
            "start_utc": ev.get("date") or comp.get("date"),
            "season": _int(season.get("year")),
            "season_type": _int(season.get("type")),
            "completed": bool(status.get("completed")),
            "state": status.get("state"),  # pre / in / post
            "neutral": bool(comp.get("neutralSite")),
        }
        for c in comp.get("competitors", []) or []:
            side = "home" if c.get("homeAway") == "home" else "away"
            t = c.get("team") or {}
            row[f"{side}_id"] = str(t.get("id"))
            row[f"{side}_abbr"] = t.get("abbreviation")
            row[f"{side}_name"] = t.get("displayName")
            row[f"{side}_score"] = _int(c.get("score")) if row["completed"] else None
        if "home_id" in row and "away_id" in row:
            games.append(row)
    return games


def _minutes(s) -> float | None:
    if s in (None, "", "--"):
        return None
    s = str(s)
    if ":" in s:  # "34:12"
        m, sec = s.split(":", 1)
        return _int(m) + (_int(sec) or 0) / 60 if _int(m) is not None else None
    try:
        return float(s)
    except ValueError:
        return None


def _made_att(s) -> tuple[int | None, int | None]:
    if not s or "-" not in str(s):
        return None, None
    a, b = str(s).split("-", 1)
    return _int(a), _int(b)


# ESPN has used both label styles ("3PT") and key styles over the years.
_LABEL_ALIASES = {
    "MIN": "min", "minutes": "min",
    "FG": "fg", "fieldGoalsMade-fieldGoalsAttempted": "fg",
    "3PT": "fg3", "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "fg3",
    "FT": "ft", "freeThrowsMade-freeThrowsAttempted": "ft",
    "OREB": "oreb", "offensiveRebounds": "oreb",
    "DREB": "dreb", "defensiveRebounds": "dreb",
    "REB": "reb", "rebounds": "reb",
    "AST": "ast", "assists": "ast",
    "STL": "stl", "steals": "stl",
    "BLK": "blk", "blocks": "blk",
    "TO": "tov", "turnovers": "tov",
    "PF": "pf", "fouls": "pf",
    "+/-": "pm", "plusMinus": "pm",
    "PTS": "pts", "points": "pts",
}


def parse_boxscore(js: dict, game_id: str) -> list[dict]:
    """One row per player who appeared (or was listed DNP) in the game."""
    rows = []
    box = js.get("boxscore") or {}
    for team_block in box.get("players", []) or []:
        team = team_block.get("team") or {}
        team_id = str(team.get("id"))
        for stat_block in team_block.get("statistics", []) or []:
            labels = stat_block.get("labels") or stat_block.get("names") or stat_block.get("keys") or []
            cols = [_LABEL_ALIASES.get(l, l) for l in labels]
            for a in stat_block.get("athletes", []) or []:
                ath = a.get("athlete") or {}
                stats = a.get("stats") or []
                vals = dict(zip(cols, stats))
                fgm, fga = _made_att(vals.get("fg"))
                fg3m, fg3a = _made_att(vals.get("fg3"))
                ftm, fta = _made_att(vals.get("ft"))
                minutes = _minutes(vals.get("min"))
                dnp = bool(a.get("didNotPlay")) or not stats or not minutes
                rows.append({
                    "game_id": str(game_id),
                    "team_id": team_id,
                    "player_id": str(ath.get("id")),
                    "player": ath.get("displayName"),
                    "pos": (ath.get("position") or {}).get("abbreviation"),
                    "starter": bool(a.get("starter")),
                    "dnp": dnp,
                    "min": 0.0 if dnp else minutes,
                    "pts": _int(vals.get("pts")) or 0,
                    "reb": _int(vals.get("reb")) or 0,
                    "ast": _int(vals.get("ast")) or 0,
                    "fg3m": fg3m or 0, "fg3a": fg3a or 0,
                    "fgm": fgm or 0, "fga": fga or 0,
                    "ftm": ftm or 0, "fta": fta or 0,
                    "oreb": _int(vals.get("oreb")) or 0,
                    "tov": _int(vals.get("tov")) or 0,
                    "stl": _int(vals.get("stl")) or 0,
                    "blk": _int(vals.get("blk")) or 0,
                })
    return rows


_ID_RE = re.compile(r"/id/(\d+)")


def parse_injuries(js: dict) -> list[dict]:
    """Flatten the league injury report."""
    out = []
    for team in js.get("injuries", []) or []:
        team_name = team.get("displayName")
        for inj in team.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            pid = ath.get("id")
            if not pid:
                for link in ath.get("links", []) or []:
                    m = _ID_RE.search(link.get("href", ""))
                    if m:
                        pid = m.group(1)
                        break
            team_id = (ath.get("team") or {}).get("id") or team.get("id")
            out.append({
                "player_id": str(pid) if pid else None,
                "player": ath.get("displayName"),
                "team_id": str(team_id) if team_id else None,
                "team": team_name,
                "status": inj.get("status") or (inj.get("type") or {}).get("description"),
                "comment": inj.get("shortComment") or "",
            })
    return out


def parse_roster(js: dict) -> list[dict]:
    """Current roster for one team. ESPN returns either a flat list or position groups."""
    out = []
    for a in js.get("athletes", []) or []:
        items = a.get("items") if isinstance(a, dict) and "items" in a else [a]
        for p in items or []:
            if p.get("id"):
                out.append({"player_id": str(p["id"]), "player": p.get("displayName") or p.get("fullName"),
                            "pos": (p.get("position") or {}).get("abbreviation"),
                            "height": _num(p.get("height")), "weight": _num(p.get("weight")),
                            "birth": (p.get("dateOfBirth") or "")[:10] or None,
                            "exp": _num((p.get("experience") or {}).get("years"))})
    return out


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_rosters(team_ids, bio_out: list | None = None) -> dict:
    """{team_id: [player_id, ...]} for every team that answered. Teams that fail are left out,
    and the model falls back to recent history for them. Player bio rows (height etc.) go to bio_out."""
    out = {}
    for tid in sorted(set(map(str, team_ids))):
        try:
            players = parse_roster(fetch_roster(tid))
            if players:
                out[tid] = [p["player_id"] for p in players]
                if bio_out is not None:
                    bio_out.extend(players)
        except RuntimeError as e:
            log.warning("roster %s failed: %s", tid, e)
        time.sleep(0.2)
    return out
