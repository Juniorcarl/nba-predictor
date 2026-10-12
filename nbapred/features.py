"""Feature engineering.

Every feature for a game uses only information from *before* that game
(values are shifted by one game), so the same code builds training rows from
history and prediction rows for upcoming games.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OUT_STATUSES = {"out", "doubtful", "suspension", "suspended", "inactive"}
ROLL_STATS = ["min", "pts", "reb", "ast", "fg3m", "fga", "fg3a", "fta", "tov"]
SEASON_STATS = ["min", "pts", "reb", "ast", "fg3m"]
STD_STATS = ["min", "pts", "reb", "ast", "fg3m"]
TEAM_STATS = ["pts", "reb", "ast", "fg3m", "fga", "fta", "oreb", "tov"]

REGULAR_MIN = 12.0     # ewm minutes to count as a rotation regular
CANDIDATE_MIN = 8.0    # ewm minutes to get a projection at all
RECENT_TEAM_GAMES = 10
_HL = 6
_ALPHA = 1 - math.exp(math.log(0.5) / _HL)


# ---------------------------------------------------------------- helpers
def _lag_ewm(g, col, hl):
    return g[col].transform(lambda s: s.shift(1).ewm(halflife=hl, ignore_na=True, min_periods=1).mean())


def _lag_roll(g, col, n, fn="mean"):
    return g[col].transform(lambda s: getattr(s.shift(1).rolling(n, min_periods=2), fn)())


def _lag_expanding_mean(g, col):
    return g[col].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())


def attach_game_info(box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "date", "season", "season_type", "home_id", "away_id"]]
    df = box.merge(g, on="game_id", how="inner")
    df["home"] = (df["team_id"] == df["home_id"]).astype(int)
    df["opp_id"] = np.where(df["home"] == 1, df["away_id"], df["home_id"])
    return df.drop(columns=["home_id", "away_id"])


# ---------------------------------------------------------------- team level
def team_games(box: pd.DataFrame, games: pd.DataFrame, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (game, team) with lagged team/opponent form features."""
    played = box[box["min"] > 0]
    agg = played.groupby(["game_id", "team_id"], as_index=False)[TEAM_STATS].sum()
    done = games[games["completed"].astype(bool)]
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = done[["game_id", "date", "season", "season_type", f"{side}_id", f"{other}_id",
                     f"{side}_score", f"{other}_score"]].copy()
        part.columns = ["game_id", "date", "season", "season_type", "team_id", "opp_id", "score", "opp_score"]
        part["home"] = int(side == "home")
        rows.append(part)
    T = pd.concat(rows, ignore_index=True).merge(agg, on=["game_id", "team_id"], how="left")
    T["pts"] = T["score"].fillna(T["pts"])
    T["poss"] = T["fga"] - T["oreb"] + T["tov"] + 0.44 * T["fta"]
    opp = T[["game_id", "team_id", "pts", "reb", "ast", "fg3m", "poss"]].rename(
        columns={"team_id": "opp_id", "pts": "a_pts", "reb": "a_reb", "ast": "a_ast", "fg3m": "a_fg3m", "poss": "opp_poss"})
    T = T.merge(opp, on=["game_id", "opp_id"], how="left")
    T["pace"] = (T["poss"] + T["opp_poss"]) / 2
    T["margin"] = T["pts"] - T["a_pts"]

    if upcoming is not None and len(upcoming):
        up = []
        for side, other in (("home", "away"), ("away", "home")):
            part = upcoming[["game_id", "date", "season", "season_type", f"{side}_id", f"{other}_id"]].copy()
            part.columns = ["game_id", "date", "season", "season_type", "team_id", "opp_id"]
            part["home"] = int(side == "home")
            up.append(part)
        T = pd.concat([T] + up, ignore_index=True)

    T["date"] = pd.to_datetime(T["date"])
    T = T.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)
    g = T.groupby("team_id", group_keys=False)
    for c in ["pts", "a_pts", "a_reb", "a_ast", "a_fg3m", "pace", "margin", "fga", "fg3m"]:
        T[f"t_{c}_e"] = _lag_ewm(g, c, 8)
    T["t_margin_season"] = T.groupby(["team_id", "season"], group_keys=False)["margin"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    T["t_rest"] = g["date"].diff().dt.days.clip(upper=10).fillna(10)
    T["t_b2b"] = (T["t_rest"] <= 1).astype(int)
    T["t_ngames"] = T.groupby(["team_id", "season"]).cumcount()

    # League scoring environment (season to date, before this date) for relative features.
    L = T.dropna(subset=["pts"]).groupby(["season", "date"]).agg(
        p=("pts", "sum"), r=("reb", "sum"), a=("ast", "sum"), f=("fg3m", "sum"), n=("pts", "size")).reset_index()
    L = L.sort_values("date")
    for k in "praf":
        L[f"cum_{k}"] = L.groupby("season")[k].cumsum() - L[k]
    L["cum_n"] = L.groupby("season")["n"].cumsum() - L["n"]
    for k, name in zip("praf", ["pts", "reb", "ast", "fg3m"]):
        L[f"lg_{name}"] = L[f"cum_{k}"] / L["cum_n"].replace(0, np.nan)
    L = L[["season", "date", "lg_pts", "lg_reb", "lg_ast", "lg_fg3m"]]
    T = pd.merge_asof(T.sort_values("date"), L.sort_values("date"), on="date", by="season",
                      allow_exact_matches=False, direction="backward")
    # Fall back to long-run NBA averages early in a season.
    for name, default in (("lg_pts", 114.0), ("lg_reb", 43.5), ("lg_ast", 26.5), ("lg_fg3m", 13.0)):
        T[name] = T[name].fillna(default)
    T["t_def_rel"] = T["t_a_pts_e"] / T["lg_pts"]
    T["t_off_rel"] = T["t_pts_e"] / T["lg_pts"]
    for s in ["reb", "ast", "fg3m"]:
        T[f"t_a_{s}_rel"] = T[f"t_a_{s}_e"] / T[f"lg_{s}"]
    return T


# ---------------------------------------------------------------- absences
@dataclass
class RosterState:
    team_idx: dict = field(default_factory=dict)        # team -> games played so far
    players: dict = field(default_factory=dict)         # player -> state dict

    def regulars(self, team, next_idx, min_ewm=REGULAR_MIN):
        return [p for p, s in self.players.items()
                if s["team"] == team and next_idx - s["last_idx"] <= RECENT_TEAM_GAMES and s["min"] >= min_ewm]


def absences(box: pd.DataFrame) -> tuple[pd.DataFrame, RosterState]:
    """For each (game, team): minutes/shots of rotation regulars who did not play.

    Returns the per-game frame and the final roster state (used for upcoming games).
    """
    df = box[["game_id", "date", "team_id", "player_id", "player", "pos", "min", "fga"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_id"])
    st = RosterState()
    out = []
    for (gid, tid), grp in df.groupby(["game_id", "team_id"], sort=False):
        idx = st.team_idx.get(tid, 0) + 1
        st.team_idx[tid] = idx
        played = grp[grp["min"] > 0]
        played_ids = set(played["player_id"])
        absent = [p for p in st.regulars(tid, idx) if p not in played_ids]
        out.append({"game_id": gid, "team_id": tid,
                    "miss_min": sum(st.players[p]["min"] for p in absent),
                    "miss_fga": sum(st.players[p]["fga"] for p in absent),
                    "miss_n": len(absent)})
        for r in played.itertuples(index=False):
            s = st.players.get(r.player_id)
            if s is None:
                s = {"min": r.min, "fga": r.fga, "team": tid, "last_idx": idx, "player": r.player, "pos": r.pos,
                     "last_date": r.date}
                st.players[r.player_id] = s
            else:
                s["min"] = _ALPHA * r.min + (1 - _ALPHA) * s["min"]
                s["fga"] = _ALPHA * r.fga + (1 - _ALPHA) * s["fga"]
                s.update(team=tid, last_idx=idx, player=r.player, pos=r.pos or s.get("pos"), last_date=r.date)
    return pd.DataFrame(out), st


def _norm(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for suf in (" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return "".join(ch for ch in s if ch.isalnum())


def injury_status_map(injuries: list[dict]) -> tuple[dict, dict]:
    by_id, by_name = {}, {}
    for i in injuries or []:
        st = (i.get("status") or "").strip()
        if i.get("player_id"):
            by_id[str(i["player_id"])] = (st, i.get("comment", ""))
        if i.get("player"):
            by_name[_norm(i["player"])] = (st, i.get("comment", ""))
    return by_id, by_name


ROSTER_RECENT_DAYS = 400  # covers an offseason plus most of a season lost to injury


def team_pool(st: RosterState, tid: str, rosters: dict | None, as_of) -> list[str]:
    """Players likely to play for team `tid` next game.

    With a current ESPN roster: everyone on it with enough recent history, wherever they played before
    (so offseason trades and signings land on the right team). Without one: the team's recent regulars.
    """
    if rosters and rosters.get(tid):
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=ROSTER_RECENT_DAYS)
        out = []
        for pid in rosters[tid]:
            s = st.players.get(pid)
            if s and s["min"] >= CANDIDATE_MIN and pd.Timestamp(s.get("last_date")) >= cutoff:
                out.append(pid)
        return out
    return st.regulars(tid, st.team_idx.get(tid, 0) + 1, min_ewm=CANDIDATE_MIN)


def upcoming_candidates(st: RosterState, upcoming: pd.DataFrame, injuries: list[dict], rosters: dict | None = None):
    """Players to project for upcoming games, plus team absence features."""
    by_id, by_name = injury_status_map(injuries)

    def status(pid, name):
        return by_id.get(pid) or by_name.get(_norm(name)) or ("", "")

    cand, team_abs = [], []
    for g in upcoming.itertuples(index=False):
        for side in ("home", "away"):
            tid = getattr(g, f"{side}_id")
            miss_min = miss_fga = miss_n = 0.0
            for pid in team_pool(st, tid, rosters, g.date):
                s = st.players[pid]
                stat, comment = status(pid, s["player"])
                is_out = stat.lower() in OUT_STATUSES
                if is_out and s["min"] >= REGULAR_MIN:
                    miss_min += s["min"]
                    miss_fga += s["fga"]
                    miss_n += 1
                if not is_out:
                    cand.append({"game_id": g.game_id, "team_id": tid, "player_id": pid, "player": s["player"],
                                 "pos": s.get("pos"), "inj_status": stat, "inj_comment": comment})
            team_abs.append({"game_id": g.game_id, "team_id": tid,
                             "miss_min": miss_min, "miss_fga": miss_fga, "miss_n": miss_n})
    return pd.DataFrame(cand), pd.DataFrame(team_abs)


# ---------------------------------------------------------------- player level
def player_features(P: pd.DataFrame) -> pd.DataFrame:
    """P: one row per player-game (played games + upcoming rows with NaN stats)."""
    P = P.copy()
    P["date"] = pd.to_datetime(P["date"])
    P = P.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    P["starter"] = P["starter"].astype(float)
    g = P.groupby("player_id", group_keys=False)
    for c in ROLL_STATS:
        P[f"{c}_e3"] = _lag_ewm(g, c, 3)
        P[f"{c}_e10"] = _lag_ewm(g, c, 10)
        P[f"{c}_r10"] = _lag_roll(g, c, 10)
    for c in STD_STATS:
        P[f"{c}_sd10"] = _lag_roll(g, c, 10, "std")
    gs = P.groupby(["player_id", "season"], group_keys=False)
    for c in SEASON_STATS:
        P[f"{c}_szn"] = _lag_expanding_mean(gs, c)
    P["starter_e"] = _lag_ewm(g, "starter", 3)
    P["starter_last"] = g["starter"].shift(1)
    P["n_games"] = g.cumcount()
    P["n_szn"] = gs.cumcount()
    P["rest"] = g["date"].diff().dt.days.clip(upper=15).fillna(15)
    # Per-minute production
    for c in ["pts", "reb", "ast", "fg3m", "fga"]:
        P[f"{c}_pm"] = P[f"{c}_e10"] / P["min_e10"].replace(0, np.nan)
    return P


TEAM_FEATS = ["t_pts_e", "t_pace_e", "t_off_rel", "t_rest", "t_b2b", "t_margin_e"]
OPP_FEATS = ["t_def_rel", "t_a_reb_rel", "t_a_ast_rel", "t_a_fg3m_rel", "t_pace_e", "t_a_pts_e", "t_margin_e"]


def player_model_frame(P: pd.DataFrame, T: pd.DataFrame, absn: pd.DataFrame) -> pd.DataFrame:
    tcols = T[["game_id", "team_id"] + TEAM_FEATS]
    ocols = T[["game_id", "team_id"] + OPP_FEATS].rename(columns={"team_id": "opp_id", **{c: f"o_{c}" for c in OPP_FEATS}})
    df = P.merge(tcols, on=["game_id", "team_id"], how="left")
    df = df.merge(ocols, on=["game_id", "opp_id"], how="left")
    df = df.merge(absn, on=["game_id", "team_id"], how="left")
    oab = absn.rename(columns={"team_id": "opp_id", "miss_min": "o_miss_min", "miss_fga": "o_miss_fga", "miss_n": "o_miss_n"})
    df = df.merge(oab, on=["game_id", "opp_id"], how="left")
    # How much of the missing workload this player could absorb
    df["miss_min_share"] = df["miss_min"] * df["min_e10"] / 240.0
    df["miss_fga_share"] = df["miss_fga"] * df["fga_pm"]
    return df


PLAYER_FEATURES = (
    [f"{c}_{k}" for c in ROLL_STATS for k in ("e3", "e10", "r10")]
    + [f"{c}_sd10" for c in STD_STATS]
    + [f"{c}_szn" for c in SEASON_STATS]
    + ["starter_e", "starter_last", "n_games", "n_szn", "rest", "home"]
    + [f"{c}_pm" for c in ["pts", "reb", "ast", "fg3m", "fga"]]
    + TEAM_FEATS + [f"o_{c}" for c in OPP_FEATS]
    + ["miss_min", "miss_fga", "miss_n", "o_miss_min", "o_miss_n", "miss_min_share", "miss_fga_share"]
)


# ---------------------------------------------------------------- game level
GAME_SIDE_FEATS = ["t_pts_e", "t_a_pts_e", "t_pace_e", "t_margin_e", "t_margin_season", "t_rest", "t_b2b",
                   "t_ngames", "t_off_rel", "t_def_rel", "miss_min", "miss_n"]


def game_model_frame(T: pd.DataFrame, absn: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    TT = T.merge(absn, on=["game_id", "team_id"], how="left")
    h = TT[TT["home"] == 1][["game_id"] + GAME_SIDE_FEATS].rename(columns={c: f"h_{c}" for c in GAME_SIDE_FEATS})
    a = TT[TT["home"] == 0][["game_id"] + GAME_SIDE_FEATS].rename(columns={c: f"a_{c}" for c in GAME_SIDE_FEATS})
    G = games.merge(h, on="game_id").merge(a, on="game_id")
    G["margin"] = G["home_score"] - G["away_score"]
    G["total"] = G["home_score"] + G["away_score"]
    G["d_margin_e"] = G["h_t_margin_e"] - G["a_t_margin_e"]
    G["exp_total"] = (G["h_t_pts_e"] + G["a_t_a_pts_e"]) / 2 + (G["a_t_pts_e"] + G["h_t_a_pts_e"]) / 2
    G["d_miss"] = G["h_miss_min"].fillna(0) - G["a_miss_min"].fillna(0)
    G["neutral"] = G["neutral"].astype(float) if "neutral" in G else 0.0
    return G


GAME_FEATURES = [f"{s}_{c}" for s in ("h", "a") for c in GAME_SIDE_FEATS] + ["d_margin_e", "exp_total", "d_miss", "neutral"]
