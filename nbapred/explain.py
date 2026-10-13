"""Plain-English reasons for each prediction.

XGBoost can split any prediction exactly into how much each input pushed it up or down
(SHAP values, `pred_contribs=True`). Here those pushes are grouped into things a fan would
recognise ("two starters out", "tough defense", "back-to-back") and given a short explanation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xgboost as xgb

from .extra import GROUPS

UNIT = {"pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "threes", "min": "min", "pra": "PRA"}
LONG_RUN = ("_e10", "_r10", "_szn", "_pm", "_sd10")


# ---------------------------------------------------------------- player feature -> reason category
def player_category(f: str) -> str:
    for g, (_, cols, _) in GROUPS.items():
        if f in cols:
            return g
    if f in ("miss_min", "miss_fga", "miss_n", "miss_min_share", "miss_fga_share"):
        return "teammates"
    if f in ("o_miss_min", "o_miss_n"):
        return "opp_injuries"
    if f.startswith("o_t_def") or f.startswith("o_t_a_"):
        return "opp_defense"
    if f in ("t_pace_e", "o_t_pace_e"):
        return "pace"
    if f in ("t_pts_e", "t_off_rel", "t_margin_e", "o_t_margin_e"):
        return "strength"
    if f in ("rest", "t_rest", "t_b2b"):
        return "rest"
    if f == "home":
        return "home"
    if f in ("starter_e", "starter_last"):
        return "role"
    if f.endswith("_e3"):
        return "form"
    return "level"  # long-run averages, per-minute rates, games played


def _ft(inches):
    if inches is None or pd.isna(inches):
        return "?"
    return f"{int(inches // 12)}'{int(round(inches % 12))}\""


def player_detail(cat: str, r: pd.Series, stat: str, opp: str, amt: float) -> tuple[str, str]:
    """(label, detail) for a reason category, using the actual numbers behind it."""
    s = "pts" if stat == "pra" else stat
    u = UNIT[stat]
    v = lambda k, d=np.nan: r.get(k, d)  # noqa: E731
    if cat == "form":
        return ("Recent form", f"Last few games {v(s + '_e3'):.1f} {UNIT[s]} in {v('min_e3'):.0f} min, "
                               f"usual {v(s + '_e10'):.1f} in {v('min_e10'):.0f} min")
    if cat == "role":
        return ("Role", "Started last game" if v("starter_last") == 1 else "Came off the bench last game")
    if cat == "teammates":
        n = int(v("miss_n", 0) or 0)
        if n:
            return ("Teammates out", f"{n} rotation teammate{'s' if n > 1 else ''} out, about {v('miss_min'):.0f} minutes to share")
        return ("Full squad", "No rotation teammates out, so no extra minutes or shots to pick up")
    if cat == "opp_injuries":
        n = int(v("o_miss_n", 0) or 0)
        return ("Opponent injuries", f"{opp} missing {n} rotation player{'s' if n != 1 else ''}" if n else f"{opp} at full strength")
    if cat == "opp_defense":
        key = {"reb": "o_t_a_reb_rel", "ast": "o_t_a_ast_rel", "fg3m": "o_t_a_fg3m_rel"}.get(s, "o_t_def_rel")
        rel = v(key, 1.0)
        what = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "min": "points"}[s]
        if pd.isna(rel):
            return ("Opponent defense", f"{opp}'s defense")
        word = "more" if rel > 1 else "fewer"
        return ("Opponent defense", f"{opp} allow {abs(rel - 1) * 100:.0f}% {word} {what} than the league average lately")
    if cat == "pace":
        p = (v("t_pace_e") + v("o_t_pace_e")) / 2
        return ("Game pace", f"Both teams together average about {p:.0f} possessions: "
                             + ("a fast game means more chances" if amt > 0 else "a slower game means fewer chances"))
    if cat == "strength":
        m = v("t_margin_e") - v("o_t_margin_e")
        return ("Team strength", f"His team is {abs(m):.1f} points per game {'better' if m >= 0 else 'worse'} than {opp} lately")
    if cat == "rest":
        if v("t_b2b") == 1:
            return ("Rest", "Second game of a back-to-back")
        return ("Rest", f"{int(v('t_rest', 2))} days since the team's last game")
    if cat == "home":
        return ("Home or away", "Home game" if v("home") == 1 else "Road game")
    # optional factor groups
    if cat == "position_defense":
        from .extra import pos_group
        grp = {"G": "guards", "F": "forwards", "C": "centers"}[pos_group(r.get("pos"))]
        rel = v({"reb": "pd_reb_rel", "ast": "pd_ast_rel", "fg3m": "pd_fg3m_rel"}.get(s, "pd_pts_rel"), 1.0)
        return ("Defense vs his position", f"{opp} allow {abs(rel - 1) * 100:.0f}% {'more' if rel > 1 else 'less'} than average to {grp}"
                if not pd.isna(rel) else f"{opp} against {grp}")
    if cat == "size":
        return ("Size matchup", f"He is {_ft(v('height'))}; {opp}'s tallest available player is {_ft(v('o_pres_height_max'))}")
    if cat == "opposing_defenders":
        return ("Opposing defenders", f"{opp}'s best available shot-blocker averages {v('o_pres_blk_max'):.1f} blocks lately")
    if cat == "blowout_risk":
        return ("Blowout risk", f"Expected margin about {v('exp_margin_abs'):.0f} points"
                + (": starters may sit late" if v("exp_margin_abs", 0) >= 10 else ""))
    if cat == "fatigue_travel":
        bits = [f"{int(v('t_g4', 1))} games in 4 nights"]
        if v("t_tz_shift", 0):
            bits.append(f"{int(v('t_tz_shift'))} time zone{'s' if v('t_tz_shift') > 1 else ''} crossed")
        if v("t_altitude", 0):
            bits.append("playing at altitude")
        if v("t_road_streak", 0) >= 3:
            bits.append(f"game {int(v('t_road_streak'))} of a road trip")
        return ("Fatigue and travel", ", ".join(bits))
    if cat == "injury_return":
        if v("missed_games", 0) >= 3:
            return ("Back from injury", f"First game back after missing {int(v('missed_games'))} games")
        if v("games_since_return", 99) <= 3:
            return ("Back from injury", f"Game {int(v('games_since_return')) + 1} since returning")
        return ("Availability", "Has been playing regularly")
    if cat == "revenge_game":
        return ("Former team", f"Facing his former team, {opp}" if v("revenge", 0) else "Not a former team")
    if cat == "vs_opponent_history":
        n = int(v("vs_n", 0) or 0)
        if n:
            return ("History vs opponent", f"{n} past games vs {opp}: {v('vs_pts_diff'):+.1f} pts vs his usual")
        return ("History vs opponent", f"First meeting with {opp} in our data")
    if cat == "motivation":
        return ("Standings", f"Team has won {v('t_winpct', .5) * 100:.0f}% this season, {int(v('t_games_left', 0))} games left")
    if cat == "shooting_luck":
        return ("Shooting streak", f"Shooting {v('fg_pct_e3', 0) * 100:.0f}% lately vs {v('fg_pct_szn', 0) * 100:.0f}% this season")
    if cat == "age_experience":
        return ("Age", f"Age {v('age'):.0f}" if not pd.isna(v("age")) else "Age unknown")
    return (cat.replace("_", " ").title(), "")


def _contribs(model, X: pd.DataFrame) -> np.ndarray:
    return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)


def explain_players(models: dict, info: dict, up: pd.DataFrame, opp_abbr: dict, top: int = 5) -> list[dict]:
    """One dict per row of `up`: {stat: {"level", "pred", "items": [{label, detail, amt}], "other"}}."""
    feats = info["features"]
    if up.empty:
        return []
    X = up[feats]
    cats = [player_category(f) for f in feats]
    per_stat = {}
    for stat, m in models.items():
        C = _contribs(m, X)
        poisson = m.get_params()["objective"].startswith("count:")
        per_stat[stat] = (C, poisson)
    out = []
    for i, (_, r) in enumerate(up.iterrows()):
        opp = opp_abbr.get(r["opp_id"], "the opponent")
        row = {}
        for stat, (C, poisson) in per_stat.items():
            row[stat] = _split(C[i], cats, poisson, float(r[f"p_{stat}"]))
        row["pra"] = _combine([row["pts"], row["reb"], row["ast"]], float(r["p_pra"]))
        expl = {}
        for stat, parts in row.items():
            items = sorted(((c, a) for c, a in parts["cats"].items() if c != "level"), key=lambda x: -abs(x[1]))
            min_amt = {"pts": .3, "pra": .4, "min": .4}.get(stat, .12)
            shown = [(c, a) for c, a in items if abs(a) >= min_amt][:top]
            rest = parts["pred"] - parts["level"] - sum(a for _, a in shown)
            expl[stat] = {"level": round(parts["level"], 1), "pred": round(parts["pred"], 1), "other": round(rest, 1),
                          "items": [dict(zip(("label", "detail"), player_detail(c, r, stat, opp, a)), amt=round(a, 1), cat=c)
                                    for c, a in shown]}
        out.append(expl)
    return out


def _split(c: np.ndarray, cats: list[str], poisson: bool, pred: float) -> dict:
    """Sum contributions per category; express them in the stat's own units."""
    bias = c[-1]
    sums: dict[str, float] = {}
    for cat, v in zip(cats, c[:-1]):
        sums[cat] = sums.get(cat, 0.0) + float(v)
    if not poisson:
        level = bias + sums.pop("level", 0.0)
        return {"level": max(level, 0.0), "pred": pred, "cats": sums}
    level_m = bias + sums.pop("level", 0.0)
    level = math.exp(level_m)
    units = {k: level * (math.exp(v) - 1) for k, v in sums.items()}
    return {"level": level, "pred": pred, "cats": units}


def _combine(parts: list[dict], pred: float) -> dict:
    cats: dict[str, float] = {}
    for p in parts:
        for k, v in p["cats"].items():
            cats[k] = cats.get(k, 0.0) + v
    return {"level": sum(p["level"] for p in parts), "pred": pred, "cats": cats}


# ---------------------------------------------------------------- games
def game_category(f: str) -> str:
    base = f[2:] if f[:2] in ("h_", "a_") else f
    for g, (_, _, gcols) in GROUPS.items():
        if f in gcols:
            return g
    if base in ("miss_min", "miss_n") or f == "d_miss":
        return "injuries"
    if base in ("t_rest", "t_b2b"):
        return "rest"
    if base in ("t_pace_e",) or f == "exp_total":
        return "pace"
    if base == "t_ngames":
        return "season"
    if f == "neutral":
        return "venue"
    if base in ("t_pts_e", "t_off_rel"):
        return "offense"
    if base in ("t_a_pts_e", "t_def_rel"):
        return "defense"
    return "form"  # margins


def game_detail(cat: str, g, target: str) -> tuple[str, str]:
    H, A = g.home_abbr, g.away_abbr
    v = lambda k, d=np.nan: getattr(g, k, d)  # noqa: E731
    if cat == "form":
        return ("Recent form", f"{H} {v('h_t_margin_e'):+.1f} per game lately, {A} {v('a_t_margin_e'):+.1f}")
    if cat == "offense":
        return ("Offense", f"{H} score {v('h_t_pts_e'):.0f} a game lately, {A} {v('a_t_pts_e'):.0f}")
    if cat == "defense":
        return ("Defense", f"{H} allow {v('h_t_a_pts_e'):.0f} a game lately, {A} {v('a_t_a_pts_e'):.0f}")
    if cat == "injuries":
        hn, an = int(v("h_miss_n", 0) or 0), int(v("a_miss_n", 0) or 0)
        if not hn and not an:
            return ("Injuries", "Both teams close to full strength")
        return ("Injuries", f"{H} missing {hn} rotation player{'s' if hn != 1 else ''}, {A} missing {an}")
    if cat == "rest":
        bits = []
        for side, ab in (("h", H), ("a", A)):
            if v(f"{side}_t_b2b") == 1:
                bits.append(f"{ab} on a back-to-back")
        return ("Rest", ", ".join(bits) if bits else f"{H} {int(v('h_t_rest', 2))} days rest, {A} {int(v('a_t_rest', 2))}")
    if cat == "pace":
        return ("Pace", f"{H} average {v('h_t_pace_e'):.0f} possessions, {A} {v('a_t_pace_e'):.0f}")
    if cat == "season":
        return ("Early-season uncertainty", f"{H} have played {int(v('h_t_ngames', 0))} games, {A} {int(v('a_t_ngames', 0))}")
    if cat == "fatigue_travel":
        return ("Fatigue and travel", f"{A}: {int(v('a_t_g4', 1))} games in 4 nights, {int(v('a_t_tz_shift', 0))} time zones crossed"
                + (", at altitude" if v("a_t_altitude", 0) else ""))
    if cat == "motivation":
        return ("Standings", f"{H} have won {v('h_t_winpct', .5) * 100:.0f}%, {A} {v('a_t_winpct', .5) * 100:.0f}%")
    if cat == "opposing_defenders":
        return ("Rim protection", f"Best available shot-blockers: {v('a_o_pres_blk_max', 0):.1f} blocks ({H}), "
                                  f"{v('h_o_pres_blk_max', 0):.1f} ({A})")
    if cat == "venue":
        return ("Venue", "Neutral court" if v("neutral", 0) else f"At {H}")
    return (cat.replace("_", " ").title(), "")


def explain_games(models: dict, info: dict, up: pd.DataFrame, top: int = 5) -> list[dict]:
    feats = info["features"]
    if up.empty:
        return []
    X = up[feats]
    cats = [game_category(f) for f in feats]
    res = {t: _contribs(models[t], X) for t in ("margin", "total")}
    out = []
    for i, g in enumerate(up.itertuples(index=False)):
        e = {}
        for t, C in res.items():
            sums: dict[str, float] = {}
            for cat, val in zip(cats, C[i][:-1]):
                sums[cat] = sums.get(cat, 0.0) + float(val)
            items = sorted(sums.items(), key=lambda x: -abs(x[1]))
            shown = [(c, a) for c, a in items if abs(a) >= 0.4][:top]
            pred = float(C[i].sum())
            base = float(C[i][-1])
            e[t] = {"level": round(base, 1), "pred": round(pred, 1), "other": round(pred - base - sum(a for _, a in shown), 1),
                    "items": [dict(zip(("label", "detail"), game_detail(c, g, t)), amt=round(a, 1), cat=c) for c, a in shown]}
        out.append(e)
    return out
