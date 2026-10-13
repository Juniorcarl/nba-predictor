"""Extra factor groups. Each group can be switched on or off in data/model_config.json.

All features use only information available before tip-off.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import CANDIDATE_MIN, RECENT_TEAM_GAMES, REGULAR_MIN

_ALPHA = 1 - math.exp(math.log(0.5) / 6)

# ---------------------------------------------------------------- the groups
# name -> (plain-English description, player-model columns, game-model columns)
GROUPS = {
    "position_defense": (
        "How much the opponent allows to the player's position (guards, forwards, centers), not just overall",
        ["pd_pts_rel", "pd_reb_rel", "pd_ast_rel", "pd_fg3m_rel"], []),
    "size": (
        "Player height and weight, and the height gap to the opponent's tallest and average available players",
        ["height", "weight", "h_vs_opp_max", "h_vs_opp_avg"], []),
    "opposing_defenders": (
        "Strength of the opponent's available defenders: best shot-blocker and ball thieves, accounting for injuries",
        ["o_pres_blk_max", "o_pres_stl_sum", "o_pres_height_max"], ["h_o_pres_blk_max", "a_o_pres_blk_max"]),
    "blowout_risk": (
        "Expected size of the win or loss: starters sit late in blowouts",
        ["exp_margin_abs", "exp_margin_signed"], []),
    "fatigue_travel": (
        "Games in the last 4 and 7 days, time zones crossed, road-trip length, altitude (Denver, Utah), recent minutes load",
        ["t_g4", "t_g7", "t_tz_shift", "t_road_streak", "t_altitude", "min_7d"],
        ["h_t_g4", "a_t_g4", "h_t_g7", "a_t_g7", "a_t_tz_shift", "a_t_road_streak", "a_t_altitude"]),
    "injury_return": (
        "Games missed right before this one: players coming back are often on a minutes limit",
        ["missed_games", "games_since_return"], []),
    "revenge_game": (
        "Playing against a former team",
        ["revenge"], []),
    "vs_opponent_history": (
        "How the player has done against this opponent before, compared with his usual level",
        ["vs_pts_diff", "vs_reb_diff", "vs_ast_diff", "vs_n"], []),
    "motivation": (
        "Team and opponent record, games left, playoffs: tanking teams and rested stars late in the season",
        ["t_winpct", "o_t_winpct", "t_games_left", "playoffs"],
        ["h_t_winpct", "a_t_winpct", "h_t_games_left", "playoffs"]),
    "shooting_luck": (
        "Recent shooting compared with the player's normal: hot and cold streaks mostly even out",
        ["fg_pct_e3", "fg_pct_szn", "fg_hot", "fg3_pct_e3", "fg3_pct_szn", "fg3_hot"], []),
    "age_experience": (
        "Age and years in the league: older players fade more on back-to-backs",
        ["age", "exp"], []),
}

# Team locations: hours relative to US Eastern, and high-altitude arenas.
TZ = {"ATL": 0, "BOS": 0, "BKN": 0, "CHA": 0, "CLE": 0, "DET": 0, "IND": 0, "MIA": 0, "NY": 0, "NYK": 0,
      "ORL": 0, "PHI": 0, "TOR": 0, "WSH": 0, "WAS": 0, "CHI": -1, "DAL": -1, "HOU": -1, "MEM": -1, "MIL": -1,
      "MIN": -1, "NO": -1, "NOP": -1, "OKC": -1, "SA": -1, "SAS": -1, "DEN": -2, "PHX": -2, "UTAH": -2, "UTA": -2,
      "GS": -3, "GSW": -3, "LAC": -3, "LAL": -3, "POR": -3, "SAC": -3}
ALTITUDE = {"DEN", "UTAH", "UTA"}


def pos_group(pos) -> str:
    p = str(pos or "").upper()
    if p.startswith("C"):
        return "C"
    if "F" in p:
        return "F"
    return "G"


# ---------------------------------------------------------------- lineup context loop
def lineup_context(base: pd.DataFrame, bio: pd.DataFrame):
    """Per (game, team): profile of the *available* players. Per (game, player): missed games, revenge.

    Returns (team_ctx, player_ctx, state) where state is used for upcoming games.
    """
    h = dict(zip(bio["player_id"], bio["height"])) if len(bio) else {}
    df = base[["game_id", "date", "team_id", "opp_id", "player_id", "min", "blk", "stl"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_id"])
    team_idx, players = {}, {}
    team_rows, player_rows = [], []
    for (gid, tid), grp in df.groupby(["game_id", "team_id"], sort=False):
        idx = team_idx.get(tid, 0) + 1
        team_idx[tid] = idx
        played = grp[grp["min"] > 0]
        opp = grp["opp_id"].iloc[0]
        # profile of who actually suited up, using their form before this game
        prof = [players[p] for p in played["player_id"] if p in players and players[p]["min"] >= REGULAR_MIN]
        team_rows.append({"game_id": gid, "team_id": tid, **_profile(prof, h)})
        for r in played.itertuples(index=False):
            s = players.get(r.player_id)
            if s is None:
                player_rows.append({"game_id": gid, "player_id": r.player_id, "missed_games": 0,
                                    "games_since_return": 99, "revenge": 0})
                players[r.player_id] = {"id": r.player_id, "min": r.min, "blk": r.blk, "stl": r.stl, "team": tid,
                                        "last_idx": idx, "prev": set(), "since": 99}
                continue
            missed = idx - s["last_idx"] - 1 if s["team"] == tid else 0
            since = 0 if missed >= 3 else s["since"] + 1
            player_rows.append({"game_id": gid, "player_id": r.player_id, "missed_games": min(max(missed, 0), 82),
                                "games_since_return": min(since, 99), "revenge": int(opp in s["prev"])})
            if s["team"] != tid:
                s["prev"].add(s["team"])
            for k in ("min", "blk", "stl"):
                s[k] = _ALPHA * getattr(r, k) + (1 - _ALPHA) * s[k]
            s.update(team=tid, last_idx=idx, since=since)
    state = {"team_idx": team_idx, "players": players, "height": h}
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows), state


def _profile(prof: list[dict], heights: dict) -> dict:
    if not prof:
        return {"pres_blk_max": np.nan, "pres_stl_sum": np.nan, "pres_height_max": np.nan, "pres_height_avg": np.nan}
    hs = [(heights.get(p["id"]), p["min"]) for p in prof]
    hs = [(x, w) for x, w in hs if x is not None and not pd.isna(x)]
    return {"pres_blk_max": max(p["blk"] for p in prof),
            "pres_stl_sum": sum(sorted((p["stl"] for p in prof), reverse=True)[:5]),
            "pres_height_max": max(x for x, _ in hs) if hs else np.nan,
            "pres_height_avg": (sum(x * w for x, w in hs) / sum(w for _, w in hs)) if hs else np.nan}


def upcoming_lineup_context(state: dict, cand: pd.DataFrame, upcoming: pd.DataFrame):
    """Same features for upcoming games, from the players expected to be available (not ruled out)."""
    if cand is None or cand.empty:
        return pd.DataFrame(), pd.DataFrame()
    players, h = state["players"], state["height"]
    team_rows, player_rows = [], []
    opp_of = {}
    for g in upcoming.itertuples(index=False):
        opp_of[(g.game_id, g.home_id)] = g.away_id
        opp_of[(g.game_id, g.away_id)] = g.home_id
    for (gid, tid), grp in cand.groupby(["game_id", "team_id"]):
        prof = [players[p] for p in grp["player_id"] if p in players and players[p]["min"] >= REGULAR_MIN]
        team_rows.append({"game_id": gid, "team_id": tid, **_profile(prof, h)})
        nxt = state["team_idx"].get(tid, 0) + 1
        opp = opp_of.get((gid, tid))
        for pid in grp["player_id"]:
            s = players.get(pid)
            if s is None:
                player_rows.append({"game_id": gid, "player_id": pid, "missed_games": 0, "games_since_return": 99, "revenge": 0})
                continue
            missed = min(max(nxt - s["last_idx"] - 1, 0), 82) if s["team"] == tid else 0
            prev = s["prev"] | ({s["team"]} if s["team"] != tid else set())
            player_rows.append({"game_id": gid, "player_id": pid, "missed_games": missed,
                                "games_since_return": 0 if missed >= 3 else min(s["since"] + 1, 99),
                                "revenge": int(opp in prev)})
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)


# ---------------------------------------------------------------- position defense
def position_defense(base: pd.DataFrame, upcoming_rows: pd.DataFrame | None) -> pd.DataFrame:
    """Per (game, defending team, position group): what that defense allows per 48 minutes, relative to league.

    upcoming_rows: frame with game_id, date, season, opp_id (the defending team) for upcoming games.
    """
    d = base[base["min"] > 0].copy()
    d["pgrp"] = d["pos"].map(pos_group)
    agg = d.groupby(["game_id", "date", "season", "opp_id", "pgrp"], as_index=False)[["min", "pts", "reb", "ast", "fg3m"]].sum()
    for c in ["pts", "reb", "ast", "fg3m"]:
        agg[f"r_{c}"] = agg[c] / agg["min"].replace(0, np.nan) * 48
    if upcoming_rows is not None and len(upcoming_rows):
        up = upcoming_rows[["game_id", "date", "season", "opp_id"]].drop_duplicates()
        up = pd.concat([up.assign(pgrp=p) for p in ("G", "F", "C")], ignore_index=True)
        agg = pd.concat([agg, up], ignore_index=True)
    agg["date"] = pd.to_datetime(agg["date"])
    agg = agg.sort_values(["opp_id", "pgrp", "date", "game_id"]).reset_index(drop=True)
    g = agg.groupby(["opp_id", "pgrp"], group_keys=False)
    for c in ["pts", "reb", "ast", "fg3m"]:
        agg[f"e_{c}"] = g[f"r_{c}"].transform(lambda s: s.shift(1).ewm(halflife=10, ignore_na=True, min_periods=3).mean())
        lg = agg.groupby(["season", "pgrp"])[f"r_{c}"].transform(lambda s: s.expanding(min_periods=20).mean().shift(1))
        agg[f"pd_{c}_rel"] = (agg[f"e_{c}"] / lg.where(lg > 0)).replace([np.inf, -np.inf], np.nan)
    return agg[["game_id", "opp_id", "pgrp"] + [f"pd_{c}_rel" for c in ["pts", "reb", "ast", "fg3m"]]]


# ---------------------------------------------------------------- team schedule context
def team_schedule(T: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Fatigue, travel and motivation per (game, team). T is sorted per team by date."""
    abbr = dict(zip(games["home_id"], games["home_abbr"])) | dict(zip(games["away_id"], games["away_abbr"]))
    home_of = dict(zip(games["game_id"], games["home_id"]))
    X = T[["game_id", "team_id", "date", "season", "season_type", "home", "margin", "t_ngames"]].copy()
    X["date"] = pd.to_datetime(X["date"])
    X = X.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)
    X["loc"] = X["game_id"].map(home_of).map(abbr)
    X["tz"] = X["loc"].map(TZ).fillna(0)
    g = X.groupby("team_id", group_keys=False)
    X["t_tz_shift"] = (X["tz"] - g["tz"].shift(1)).abs().fillna(0)
    X["t_altitude"] = (X["loc"].isin(ALTITUDE) & ~X["team_id"].map(abbr).isin(ALTITUDE)).astype(int)
    # games in the previous 4 / 7 days (this game included)
    for days, col in ((3, "t_g4"), (6, "t_g7")):
        X[col] = _count_window(X, days)
    # consecutive away games including this one
    X["t_road_streak"] = _road_streak(X)
    won = (X["margin"] > 0).astype(float).where(X["margin"].notna())
    X["t_winpct"] = won.groupby([X["team_id"], X["season"]]).transform(lambda s: s.shift(1).expanding().mean())
    X["t_games_left"] = (82 - X["t_ngames"]).clip(lower=0)
    X.loc[X["season_type"] == 3, "t_games_left"] = 0
    X["playoffs"] = (X["season_type"] == 3).astype(int)
    return X[["game_id", "team_id", "t_tz_shift", "t_altitude", "t_g4", "t_g7", "t_road_streak",
              "t_winpct", "t_games_left", "playoffs"]]


def _count_window(X, days):
    out = np.zeros(len(X))
    for _, idx in X.groupby("team_id").indices.items():
        dates = X["date"].values[idx]
        j = 0
        for k in range(len(idx)):
            while dates[j] < dates[k] - np.timedelta64(days, "D"):
                j += 1
            out[idx[k]] = k - j + 1
    return out


def _road_streak(X):
    out = np.zeros(len(X))
    for _, idx in X.groupby("team_id").indices.items():
        run = 0
        for k in idx:
            run = run + 1 if X["home"].values[k] == 0 else 0
            out[k] = run
    return out


# ---------------------------------------------------------------- player-level extras
def player_extras(P: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """Shooting luck, minutes load, history vs this opponent, size and age. P sorted by player/date."""
    g = P.groupby("player_id", group_keys=False)

    def lag_ewm(col, hl):
        return g[col].transform(lambda s: s.shift(1).ewm(halflife=hl, ignore_na=True, min_periods=1).mean())

    gs = P.groupby(["player_id", "season"], group_keys=False)

    def lag_szn(col):
        return gs[col].transform(lambda s: s.shift(1).expanding().sum())

    fgm3, fga3 = lag_ewm("fgm", 3), lag_ewm("fga", 3)
    P["fg_pct_e3"] = fgm3 / fga3.replace(0, np.nan)
    P["fg_pct_szn"] = lag_szn("fgm") / lag_szn("fga").replace(0, np.nan)
    P["fg_hot"] = P["fg_pct_e3"] - P["fg_pct_szn"]
    m3, a3 = lag_ewm("fg3m", 3), lag_ewm("fg3a", 3)
    P["fg3_pct_e3"] = m3 / a3.replace(0, np.nan)
    P["fg3_pct_szn"] = lag_szn("fg3m") / lag_szn("fg3a").replace(0, np.nan)
    P["fg3_hot"] = P["fg3_pct_e3"] - P["fg3_pct_szn"]

    # minutes over the previous 7 days
    out = np.zeros(len(P))
    for _, idx in P.groupby("player_id").indices.items():
        dates, mins = P["date"].values[idx], P["min"].fillna(0).values[idx]
        j, run = 0, 0.0
        for k in range(len(idx)):
            while dates[j] < dates[k] - np.timedelta64(7, "D"):
                run -= mins[j]
                j += 1
            out[idx[k]] = run
            run += mins[k]
    P["min_7d"] = out

    # vs this opponent, compared with usual level
    gv = P.groupby(["player_id", "opp_id"], group_keys=False)
    for c in ["pts", "reb", "ast"]:
        P[f"vs_{c}_diff"] = gv[c].transform(lambda s: s.shift(1).expanding().mean()) - P[f"{c}_e10"]
    P["vs_n"] = gv.cumcount()

    if len(bio):
        b = bio.drop_duplicates("player_id").set_index("player_id")
        P["height"] = P["player_id"].map(b["height"]).astype(float)
        P["weight"] = P["player_id"].map(b["weight"]).astype(float)
        birth = pd.to_datetime(P["player_id"].map(b["birth"]), errors="coerce")
        P["age"] = (P["date"] - birth).dt.days / 365.25
        # experience at game time: current years minus seasons since then
        P["exp"] = P["player_id"].map(b["exp"]).astype(float) - (P["season"].max() - P["season"])
    else:
        for c in ("height", "weight", "age", "exp"):
            P[c] = np.nan
    return P


def attach(df: pd.DataFrame, team_ctx: pd.DataFrame, player_ctx: pd.DataFrame, pdef: pd.DataFrame,
           sched: pd.DataFrame) -> pd.DataFrame:
    """Merge every extra feature onto the player model frame."""
    df = df.copy()
    df["pgrp"] = df["pos"].map(pos_group)
    df = df.merge(pdef, on=["game_id", "opp_id", "pgrp"], how="left")
    oc = team_ctx.rename(columns={"team_id": "opp_id", **{c: f"o_{c}" for c in team_ctx.columns if c.startswith("pres_")}})
    df = df.merge(oc, on=["game_id", "opp_id"], how="left")
    df = df.merge(player_ctx.drop_duplicates(["game_id", "player_id"]), on=["game_id", "player_id"], how="left")
    df["h_vs_opp_max"] = df["height"] - df["o_pres_height_max"]
    df["h_vs_opp_avg"] = df["height"] - df["o_pres_height_avg"]
    df = df.merge(sched, on=["game_id", "team_id"], how="left")
    os_ = sched[["game_id", "team_id", "t_winpct"]].rename(columns={"team_id": "opp_id", "t_winpct": "o_t_winpct"})
    df = df.merge(os_, on=["game_id", "opp_id"], how="left")
    hadj = np.where(df["home"] == 1, 2.5, -2.5)
    df["exp_margin_signed"] = df["t_margin_e"] - df["o_t_margin_e"] + hadj
    df["exp_margin_abs"] = df["exp_margin_signed"].abs()
    return df


def attach_game(G: pd.DataFrame, team_ctx: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    G = G.copy()
    for side, other in (("h", "home"), ("a", "away")):
        s = sched.rename(columns={"team_id": f"{other}_id", **{c: f"{side}_{c}" for c in sched.columns if c.startswith("t_")}})
        G = G.merge(s.drop(columns=["playoffs"]), on=["game_id", f"{other}_id"], how="left")
        # opponent's available defenders, seen from this side
        opp = "away" if other == "home" else "home"
        tc = team_ctx[["game_id", "team_id", "pres_blk_max"]].rename(columns={"team_id": f"{opp}_id", "pres_blk_max": f"{side}_o_pres_blk_max"})
        G = G.merge(tc, on=["game_id", f"{opp}_id"], how="left")
    pl = sched[["game_id", "playoffs"]].drop_duplicates("game_id")
    G = G.merge(pl, on="game_id", how="left")
    return G
