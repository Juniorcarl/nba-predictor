"""Generate a synthetic league in the same CSV format the ESPN pipeline writes.

Used for offline testing and local UI work only. Never used by the daily job.
Usage: NBAPRED_DATA_DIR=demo/data python scripts/make_demo_data.py
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.config import BOX_DIR, GAMES_DIR  # noqa: E402
from nbapred.store import BOX_COLS, GAME_COLS  # noqa: E402

TEAMS = [("1", "ATL", "Atlanta Hawks"), ("2", "BOS", "Boston Celtics"), ("17", "BKN", "Brooklyn Nets"),
         ("30", "CHA", "Charlotte Hornets"), ("4", "CHI", "Chicago Bulls"), ("5", "CLE", "Cleveland Cavaliers"),
         ("6", "DAL", "Dallas Mavericks"), ("7", "DEN", "Denver Nuggets"), ("8", "DET", "Detroit Pistons"),
         ("9", "GS", "Golden State Warriors"), ("10", "HOU", "Houston Rockets"), ("11", "IND", "Indiana Pacers"),
         ("12", "LAC", "LA Clippers"), ("13", "LAL", "Los Angeles Lakers"), ("29", "MEM", "Memphis Grizzlies"),
         ("14", "MIA", "Miami Heat"), ("15", "MIL", "Milwaukee Bucks"), ("16", "MIN", "Minnesota Timberwolves"),
         ("3", "NO", "New Orleans Pelicans"), ("18", "NY", "New York Knicks"), ("25", "OKC", "Oklahoma City Thunder"),
         ("19", "ORL", "Orlando Magic"), ("20", "PHI", "Philadelphia 76ers"), ("21", "PHX", "Phoenix Suns"),
         ("22", "POR", "Portland Trail Blazers"), ("23", "SAC", "Sacramento Kings"), ("24", "SA", "San Antonio Spurs"),
         ("28", "TOR", "Toronto Raptors"), ("26", "UTAH", "Utah Jazz"), ("27", "WSH", "Washington Wizards")]
FIRST = ["Alex", "Jordan", "Chris", "Malik", "Tyrese", "Luka", "Devin", "Jalen", "Marcus", "Andre", "Kobe", "Darius",
         "Isaiah", "Cam", "Trey", "Zion", "Evan", "Miles", "Noah", "Caleb"]
LAST = ["Walker", "Johnson", "Okafor", "Silva", "Brooks", "Mensah", "Kovac", "Diallo", "Hayes", "Moreno", "Bennett",
        "Adeyemi", "Price", "Novak", "Grant", "Tshabalala", "Reed", "Fofana", "Lindqvist", "Carter"]
POS = ["PG", "SG", "SF", "PF", "C"]

rng = np.random.default_rng(7)


def make_players():
    players, pid = [], 1000
    names = [f"{a} {b}" for a in FIRST for b in LAST]
    rng.shuffle(names)
    for tid, _, _ in TEAMS:
        mins = sorted(rng.uniform(8, 37, 13), reverse=True)
        for i, m in enumerate(mins):
            pos = POS[i % 5]
            big = pos in ("PF", "C")
            players.append({
                "player_id": str(pid), "player": names[pid - 1000], "pos": pos, "team_id": tid,
                "min": m, "pts_pm": rng.uniform(0.3, 0.85), "reb_pm": rng.uniform(0.25, 0.45) if big else rng.uniform(0.1, 0.22),
                "ast_pm": rng.uniform(0.04, 0.1) if big else rng.uniform(0.08, 0.28), "fg3_rate": rng.uniform(0.0, 0.15) if big else rng.uniform(0.1, 0.45),
                "starter": i < 5,
            })
            pid += 1
    return pd.DataFrame(players)


def simulate(start: date, end: date, season: int, players: pd.DataFrame, completed_until: date):
    games, box = [], []
    strength = {t[0]: rng.normal(0, 0.06) for t in TEAMS}
    gid = season * 10000
    d = start
    while d <= end:
        if rng.random() < 0.9:
            ids = [t[0] for t in TEAMS]
            rng.shuffle(ids)
            n = int(rng.integers(3, 8)) * 2
            for k in range(0, n, 2):
                h, a = ids[k], ids[k + 1]
                gid += 1
                info = {t[0]: t for t in TEAMS}
                row = {"game_id": str(gid), "date": d.isoformat(), "start_utc": f"{d.isoformat()}T23:30Z", "season": season,
                       "season_type": 2, "completed": d < completed_until, "state": "post" if d < completed_until else "pre",
                       "neutral": False, "home_id": h, "home_abbr": info[h][1], "home_name": info[h][2],
                       "away_id": a, "away_abbr": info[a][1], "away_name": info[a][2], "home_score": None, "away_score": None}
                if d < completed_until:
                    scores = {}
                    for tid in (h, a):
                        roster = players[players.team_id == tid]
                        out = rng.random(len(roster)) < 0.08
                        avail_min = roster["min"].where(~out, 0)
                        scale = 240 / avail_min.sum()
                        tot = 0
                        mult = 1 + strength[tid] + (0.03 if tid == h else 0)
                        for (p, is_out, m_scaled) in zip(roster.itertuples(), out, avail_min * scale):
                            if is_out:
                                box.append({"game_id": str(gid), "team_id": tid, "player_id": p.player_id, "player": p.player,
                                            "pos": p.pos, "starter": False, "dnp": True, "min": 0.0, **{c: 0 for c in BOX_COLS[8:]}})
                                continue
                            mins = max(1.0, rng.normal(m_scaled, 4))
                            pts = int(rng.poisson(p.pts_pm * mins * mult))
                            fga = int(max(pts / 2.2 + rng.normal(0, 2), 0))
                            fg3a = int(fga * p.fg3_rate * 2.2)
                            fg3m = int(rng.binomial(fg3a, 0.36)) if fg3a else 0
                            stat = {"pts": pts, "reb": int(rng.poisson(p.reb_pm * mins)), "ast": int(rng.poisson(p.ast_pm * mins * mult)),
                                    "fg3m": fg3m, "fg3a": fg3a, "fgm": int(fga * 0.47), "fga": fga, "ftm": int(pts * 0.15), "fta": int(pts * 0.19),
                                    "oreb": int(rng.poisson(0.05 * mins)), "tov": int(rng.poisson(0.05 * mins)),
                                    "stl": int(rng.poisson(0.03 * mins)), "blk": int(rng.poisson(0.02 * mins))}
                            tot += pts
                            box.append({"game_id": str(gid), "team_id": tid, "player_id": p.player_id, "player": p.player, "pos": p.pos,
                                        "starter": bool(p.starter), "dnp": False, "min": round(mins, 1), **stat})
                        scores[tid] = tot
                    if scores[h] == scores[a]:
                        scores[h] += 3
                    row["home_score"], row["away_score"] = scores[h], scores[a]
                games.append(row)
        d += timedelta(days=1)
    return pd.DataFrame(games)[GAME_COLS], pd.DataFrame(box)[BOX_COLS]


def _american(p):
    p = min(max(p, 0.03), 0.97)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def fake_odds(gp, pp, prop_games=3):
    """Odds-API-shaped events built from noisy versions of our own predictions (demo only)."""
    from nbapred.odds import PROP_MARKETS
    r = np.random.default_rng(1)
    games, props = [], {}
    for i, g in enumerate(gp.sort_values("start_utc").itertuples()):
        books = []
        hname = g.home_name.replace("LA ", "Los Angeles ")  # Odds API uses full city names
        for bk in ("bookA", "bookB", "bookC"):
            ph = min(max(g.p_home_win + r.normal(0, 0.07), 0.05), 0.95)
            spread = round((-(g.p_margin) + r.normal(0, 2.5)) * 2) / 2
            total = round((g.p_total + r.normal(0, 4)) * 2) / 2
            books.append({"key": bk, "markets": [
                {"key": "h2h", "outcomes": [{"name": hname, "price": _american(ph + 0.02)},
                                            {"name": g.away_name, "price": _american(1 - ph + 0.02)}]},
                {"key": "spreads", "outcomes": [{"name": hname, "price": -110, "point": spread},
                                                {"name": g.away_name, "price": -110, "point": -spread}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": total},
                                               {"name": "Under", "price": -110, "point": total}]}]})
        games.append({"id": f"ev{i}", "home_team": hname, "away_team": g.away_name,
                      "bookmakers": books})
        if i < prop_games:
            mk = []
            for key, stat in PROP_MARKETS.items():
                outs = []
                for pl in pp[(pp.game_id == g.game_id) & (pp.p_min > 18)].itertuples():
                    line = np.floor(getattr(pl, f"p_{stat}") + r.normal(0, 1.5 if stat in ("pts", "pra") else 0.7)) + 0.5
                    if line < 0.5:
                        continue
                    outs += [{"name": "Over", "description": pl.player, "price": int(r.choice([-125, -115, -110, 100])), "point": line},
                             {"name": "Under", "description": pl.player, "price": int(r.choice([-115, -110, -105])), "point": line}]
                mk.append({"key": key, "outcomes": outs})
            props[g.game_id] = {"bookmakers": [{"key": "bookA", "markets": mk}]}
    return games, props


if __name__ == "__main__":
    players = make_players()
    today = date.today()
    for season, start, end in [(2025, date(2024, 10, 22), date(2025, 4, 13)),
                               (2026, date(2025, 10, 21), date(2026, 4, 12)),
                               (2027, date(2026, 10, 1), today + timedelta(days=3))]:
        g, b = simulate(start, end, season, players, completed_until=today)
        g.to_csv(GAMES_DIR / f"{season}.csv", index=False)
        if len(b):
            b.to_csv(BOX_DIR / f"{season}.csv", index=False)
        print(season, len(g), "games", len(b), "box rows")
