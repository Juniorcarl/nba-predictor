"""Bold calls, upset picks and same-game multis.

Different job from the main projections: instead of the most likely outcome, look for nights where a
big outcome (a ceiling night, an upset) is clearly more likely than usual, then price it honestly.

Every probability here is calibrated on held-out games: when this page says 25%, it should happen
about 1 time in 4. Most bold calls will lose; the point is that the winners pay enough.
"""
from __future__ import annotations

import itertools
import json
import logging
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import models
from .config import LOG_DIR, PROP_STATS

log = logging.getLogger(__name__)
BOLD_LOG = LOG_DIR / "bold.csv"

MILESTONES = {
    "pts": [10, 15, 20, 25, 30, 35, 40, 45, 50],
    "reb": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    "ast": [4, 6, 8, 10, 12, 14, 16],
    "fg3m": [2, 3, 4, 5, 6, 7, 8],
    "stl": [2, 3, 4, 5],
    "blk": [2, 3, 4, 5, 6],
    "pra": [20, 25, 30, 35, 40, 45, 50, 55, 60],
}
NAME = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "stl": "steals", "blk": "blocks",
        "pra": "points + rebounds + assists"}
BANDS = {"high": (0.10, 0.30), "medium": (0.30, 0.48)}   # chance of hitting
MIN_BOOST = 1.20       # tonight's chance must be at least 20% higher than on his normal night
MIN_UPLIFT = 0.04      # ... and at least 4 percentage points higher
MIN_MINUTES = 24       # only players with a real role tonight
SKIP_STATUS = {"out", "doubtful", "questionable", "game-time decision", "day-to-day", "suspension"}
CORR_STATS = PROP_STATS + ["pra"]


# ================================================================== correlations (for multis)
def fit_corr(valid: pd.DataFrame, var_params: dict, games: pd.DataFrame) -> dict:
    """How outcomes move together, measured on held-out games."""
    v = valid.copy()
    v["pra"] = v["pts"] + v["reb"] + v["ast"]
    v["p_pra"] = v["p_pts"] + v["p_reb"] + v["p_ast"]
    for s in CORR_STATS:
        v[f"z_{s}"] = (v[s] - v[f"p_{s}"]) / np.sqrt(models.variance(v[f"p_{s}"], var_params[s]))
    zc = [f"z_{s}" for s in CORR_STATS]
    same = np.corrcoef(v[zc].to_numpy().T)

    base = v[["game_id", "team_id", "player_id"] + zc]
    pairs = base.merge(base, on="game_id", suffixes=("_a", "_b"))
    pairs = pairs[pairs["player_id_a"] != pairs["player_id_b"]]
    if len(pairs) > 400_000:
        pairs = pairs.sample(400_000, random_state=0)
    mates = pairs[pairs["team_id_a"] == pairs["team_id_b"]]
    opps = pairs[pairs["team_id_a"] != pairs["team_id_b"]]

    def cross(df):
        m = np.zeros((len(CORR_STATS), len(CORR_STATS)))
        for i, a in enumerate(CORR_STATS):
            for j, b in enumerate(CORR_STATS):
                m[i, j] = np.corrcoef(df[f"z_{a}_a"], df[f"z_{b}_b"])[0, 1]
        return (m + m.T) / 2

    # player outcome vs his team beating expectations
    g = games.set_index("game_id")
    hs, as_ = v["game_id"].map(g["home_score"]), v["game_id"].map(g["away_score"])
    team_margin = np.where(v["home"] == 1, hs - as_, as_ - hs)
    exp_margin = v["t_margin_e"] - v["o_t_margin_e"] + np.where(v["home"] == 1, 2.5, -2.5)
    surprise = pd.Series(team_margin - exp_margin)
    ok = surprise.notna().to_numpy() & ~np.isnan(exp_margin.to_numpy())
    win = {s: float(np.corrcoef(v.loc[ok, f"z_{s}"], surprise[ok])[0, 1]) for s in CORR_STATS}
    rnd = lambda m: np.round(np.nan_to_num(m), 3).tolist()  # noqa: E731
    return {"stats": CORR_STATS, "same": rnd(same), "mate": rnd(cross(mates)), "opp": rnd(cross(opps)),
            "win": {k: round(x, 3) for k, x in win.items()}}


# ================================================================== helpers
def kelly(p: float, dec_odds: float, fraction: float = 0.25) -> float:
    """Quarter-Kelly stake as a share of bankroll (0 if no value)."""
    b = dec_odds - 1
    if b <= 0:
        return 0.0
    f = (p * b - (1 - p)) / b
    return max(0.0, f * fraction)


def _cal(stat, p_raw, tails):
    return float(models.calibrated(stat, p_raw, tails))


def _recent_hits(logs: dict, pid: str, stat: str, T: int, n=10):
    games = logs.get(pid, [])[:n]
    if not games:
        return None
    vals = [(g["pts"] + g["reb"] + g["ast"]) if stat == "pra" else g.get(stat, 0) for g in games]
    return {"hits": int(sum(x >= T for x in vals)), "n": len(vals), "best": int(max(vals))}


# ================================================================== bold calls
def bold_calls(pp: pd.DataFrame, tails: dict, logs: dict, abbr: dict, limit: int = 20) -> list[dict]:
    cal = tails.get("tails", {})
    vol = tails.get("vol", {})
    out = []
    for r in pp.itertuples(index=False):
        if r.low_sample or str(r.inj_status or "").lower() in SKIP_STATUS or r.p_min < MIN_MINUTES:
            continue
        why = getattr(r, "why", None) or {}
        best_for_player = []
        for stat, ladder in MILESTONES.items():
            mu = float(getattr(r, f"p_{stat}"))
            vv = float(getattr(r, f"vv_{stat}"))
            # his normal night: recent average with the same volatility
            if stat == "pra":
                mu_n = float(np.nansum([getattr(r, "pts_e10"), getattr(r, "reb_e10"), getattr(r, "ast_e10")]))
            else:
                mu_n = float(getattr(r, f"{stat}_e10") or 0)
            if not mu_n or math.isnan(mu_n) or mu <= 0:
                continue
            vr = vol.get(stat, {}).get(r.player_id, 1.0)
            vv_n = float(models.variance(mu_n, _var_params[stat])) * vr
            for T in ladder:
                if T <= mu:  # bold calls are above the projection
                    continue
                p = _cal(stat, models.tail_prob(mu, vv, T), cal)
                band = next((b for b, (lo, hi) in BANDS.items() if lo <= p <= hi), None)
                if not band:
                    continue
                p_n = _cal(stat, models.tail_prob(mu_n, vv_n, T), cal)
                boost = p / max(p_n, 1e-3)
                if boost < MIN_BOOST or p - p_n < MIN_UPLIFT:
                    continue
                h = _recent_hits(logs, r.player_id, stat, T, n=15)
                if not h or h["hits"] < 1:  # he must have actually done it recently
                    continue
                best_for_player.append({
                    "player_id": r.player_id, "player": r.player, "team": abbr.get(r.team_id, ""), "opp": abbr.get(r.opp_id, ""),
                    "game_id": r.game_id, "home": int(r.home), "stat": stat, "line": T, "p": round(p, 3), "p_normal": round(p_n, 3),
                    "boost": round(boost, 2), "band": band, "fair_odds": round(1 / p, 2), "proj": round(mu, 1),
                    "normal": round(mu_n, 1), "vol": round(vr, 2), "uplift": p - p_n,
                    "reasons": _reasons(r, stat, T, why.get(stat), logs, vr, abbr),
                })
        # best 2 per player: biggest uplift, different stats
        best_for_player.sort(key=lambda c: -c["uplift"])
        seen = set()
        for c in best_for_player:
            if c["stat"] in seen:
                continue
            seen.add(c["stat"])
            out.append(c)
            if len(seen) == 2:
                break
    # headline stats first: a points or PRA spike matters more than an extra steal
    weight = {"pts": 1.4, "pra": 1.3, "reb": 1.15, "ast": 1.15, "fg3m": 1.0, "stl": 0.8, "blk": 0.8}
    out.sort(key=lambda c: -c["uplift"] * weight[c["stat"]])
    for c in out:
        c.pop("uplift")
    return out[:limit]


def _reasons(r, stat, T, w, logs, vr, abbr):
    """Why tonight's ceiling is higher than usual, and what works against it."""
    fors, against = [], []
    if w:
        for it in w["items"]:
            (fors if it["amt"] > 0 else against).append({"label": it["label"], "detail": it["detail"], "amt": it["amt"]})
    names = getattr(r, "out_names", "") or ""
    if isinstance(names, str) and names.strip():
        fors.insert(0, {"label": "Teammates out", "detail": f"Out tonight: {names}", "amt": None})
    h = _recent_hits(logs, r.player_id, stat, T)
    facts = []
    if r.p_min >= r.min_e10 + 2:
        facts.append(f"Projected {r.p_min:.0f} minutes, up from his usual {r.min_e10:.0f}")
    if h:
        facts.append(f"Reached {T}+ {NAME[stat]} in {h['hits']} of his last {h['n']} games (best: {h['best']})")
    if vr >= 1.12:
        facts.append(f"Swings more than most players: his big nights are bigger ({(vr - 1) * 100:.0f}% more variance than typical)")
    elif vr <= 0.88:
        facts.append("A steady player: big spikes are rarer for him, which is already priced in")
    # dedupe teammates line from SHAP list if names given
    if names:
        fors = [f for i, f in enumerate(fors) if not (i > 0 and f["label"] == "Teammates out")]
    return {"for": fors[:4], "against": against[:2], "facts": facts}


_var_params: dict = {}


def build(slate_date, pp, gp, glines, pinfo, tails, corr, players, abbr) -> dict:
    """Everything the Bold page shows for one slate (and log it for grading)."""
    _var_params.clear()
    _var_params.update(pinfo["var"])
    logs = {p["id"]: p.get("logs", []) for p in players}
    calls = bold_calls(pp, tails, logs, abbr) if len(pp) else []
    ups = upsets(gp, glines) if len(gp) else []
    sgms = multis(pp, gp, tails, corr) if len(gp) else []
    if slate_date is not None:
        log_bold(slate_date, calls, ups, sgms)
    return {"date": str(slate_date) if slate_date else None, "calls": calls, "upsets": ups, "multis": sgms,
            "bands": BANDS, "tail_check": tails.get("check", {})}


# ================================================================== upsets
def upsets(gp: pd.DataFrame, glines: dict, limit: int = 6) -> list[dict]:
    out = []
    for g in gp.itertuples(index=False):
        ph = float(g.p_home_win)
        dog_home = ph < 0.5
        p = ph if dog_home else 1 - ph
        if not 0.25 <= p <= 0.49:
            continue
        team = g.home_abbr if dog_home else g.away_abbr
        fav = g.away_abbr if dog_home else g.home_abbr
        side = "home" if dog_home else "away"
        lines = (glines or {}).get(g.game_id) or {}
        ml = lines.get("ml")
        book = ml["sides"][side] if ml else None
        bw = (ml or {}).get("betway", {}).get(side) if ml else None
        reasons = []
        w = (getattr(g, "why", None) or {}).get("margin")
        if w:
            for it in w["items"]:
                helps = it["amt"] < 0 if dog_home is False else it["amt"] > 0
                reasons.append({**it, "helps": bool(helps)})
        for x in reasons:
            x["amt_dog"] = x["amt"] if dog_home else -x["amt"]
        out.append({"game_id": g.game_id, "team": team, "fav": fav, "side": side, "p": round(p, 3),
                    "fair_odds": round(1 / p, 2), "start": g.start_utc,
                    "book_p": round(book["fair"], 3) if book else None, "best_price": book["price"] if book else None,
                    "betway_price": bw, "margin": round(float(g.p_margin), 1),
                    "reasons": [x for x in reasons if x["helps"]][:3] + [x for x in reasons if not x["helps"]][:1]})
    # with lines: biggest gap to the bookmaker first; without: most likely upset first
    out.sort(key=lambda u: (u["book_p"] is not None, (u["p"] - u["book_p"]) if u["book_p"] is not None else u["p"]), reverse=True)
    return out[:limit]


# ================================================================== same-game multis
def _leg_candidates(pp_game: pd.DataFrame, tails: dict, gp_row) -> list[dict]:
    cal = tails.get("tails", {})
    legs = []
    for r in pp_game.itertuples(index=False):
        if r.low_sample or str(r.inj_status or "").lower() in SKIP_STATUS or r.p_min < 22:
            continue
        best = None
        for stat, ladder in MILESTONES.items():
            mu = float(getattr(r, f"p_{stat}"))
            vv = float(getattr(r, f"vv_{stat}"))
            mu_n = float(np.nansum([r.pts_e10, r.reb_e10, r.ast_e10])) if stat == "pra" else float(getattr(r, f"{stat}_e10") or 0)
            for T in ladder:
                p = _cal(stat, models.tail_prob(mu, vv, T), cal)
                if not 0.40 <= p <= 0.80:
                    continue
                vr = tails.get("vol", {}).get(stat, {}).get(r.player_id, 1.0)
                p_n = _cal(stat, models.tail_prob(mu_n, float(models.variance(max(mu_n, .1), _var_params[stat])) * vr, T), cal) if mu_n else p
                score = p - p_n  # better than his normal night
                cand = {"kind": "player", "player_id": r.player_id, "player": r.player, "team_id": r.team_id, "stat": stat,
                        "line": T, "p": round(p, 3), "score": score}
                if best is None or score > best["score"] or (score == best["score"] and p < best["p"]):
                    best = cand
        if best and best["score"] > 0.02:
            legs.append(best)
    legs.sort(key=lambda x: -x["score"])
    legs = legs[:6]
    ph = float(gp_row.p_home_win)
    fav_home = ph >= 0.5
    legs.append({"kind": "win", "team_id": gp_row.home_id if fav_home else gp_row.away_id,
                 "team": gp_row.home_abbr if fav_home else gp_row.away_abbr, "p": round(max(ph, 1 - ph), 3), "score": 0})
    return legs


def _corr_between(a, b, corr) -> float:
    S = corr["stats"]
    if a["kind"] == "win" and b["kind"] == "win":
        return -1.0
    if a["kind"] == "win" or b["kind"] == "win":
        pl, w = (b, a) if a["kind"] == "win" else (a, b)
        c = corr["win"].get(pl["stat"], 0.0)
        return c if pl["team_id"] == w["team_id"] else -c
    i, j = S.index(a["stat"]), S.index(b["stat"])
    if a["player_id"] == b["player_id"]:
        return corr["same"][i][j]
    return corr["mate"][i][j] if a["team_id"] == b["team_id"] else corr["opp"][i][j]


def joint_prob(legs: list[dict], corr: dict, n: int = 40_000, seed: int = 7) -> float:
    """Chance that every leg hits, allowing for how they move together (Gaussian copula)."""
    k = len(legs)
    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            R[i, j] = R[j, i] = _corr_between(legs[i], legs[j], corr)
    w, V = np.linalg.eigh(R)  # make sure the matrix is valid
    R = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(k), R, size=n)
    cut = sps.norm.ppf(1 - np.array([leg["p"] for leg in legs]))
    return float(np.mean(np.all(z >= cut, axis=1)))


def multis(pp: pd.DataFrame, gp: pd.DataFrame, tails: dict, corr: dict | None, limit: int = 8) -> list[dict]:
    if not corr or pp.empty:
        return []
    out = []
    for g in gp.itertuples(index=False):
        legs = _leg_candidates(pp[pp["game_id"] == g.game_id], tails, g)
        cands = []
        for size in (2, 3):
            for combo in itertools.combinations(legs, size):
                if sum(leg["kind"] == "win" for leg in combo) > 1:
                    continue
                if len({(leg.get("player_id"), leg.get("stat")) for leg in combo}) < size:
                    continue
                naive = float(np.prod([leg["p"] for leg in combo]))
                if naive < 0.05:
                    continue
                joint = joint_prob(list(combo), corr, n=20_000)
                if not 0.10 <= joint <= 0.40:
                    continue
                cands.append({"game_id": g.game_id, "matchup": f"{g.away_abbr} @ {g.home_abbr}", "start": g.start_utc,
                              "legs": [_leg_text(leg) for leg in combo], "p": round(joint, 3), "naive": round(naive, 3),
                              "lift": round(joint / naive, 2), "fair_odds": round(1 / joint, 2),
                              "score": joint * sum(leg["score"] for leg in combo) + (joint / naive - 1) * 0.1})
        cands.sort(key=lambda c: -c["score"])
        for c in cands[:2]:
            c.pop("score")
            out.append(c)
    out.sort(key=lambda c: -c["lift"])
    return out[:limit]


def _leg_text(leg):
    if leg["kind"] == "win":
        return {"text": f"{leg['team']} to win", "p": leg["p"], "kind": "win", "team_id": leg["team_id"]}
    return {"text": f"{leg['player']} {leg['line']}+ {NAME[leg['stat']]}", "p": leg["p"], "kind": "player",
            "player_id": leg["player_id"], "stat": leg["stat"], "line": leg["line"]}


# ================================================================== logging and grading
LOG_COLS = ["date", "kind", "game_id", "desc", "detail", "p", "fair_odds", "result"]


def log_bold(slate_date, calls, ups, sgms):
    rows = []
    for c in calls:
        rows.append({"kind": "call", "game_id": c["game_id"], "desc": f"{c['player']} {c['line']}+ {NAME[c['stat']]}",
                     "detail": json.dumps({"player_id": c["player_id"], "stat": c["stat"], "line": c["line"]}),
                     "p": c["p"], "fair_odds": c["fair_odds"]})
    for u in ups:
        rows.append({"kind": "upset", "game_id": u["game_id"], "desc": f"{u['team']} to beat {u['fav']}",
                     "detail": json.dumps({"side": u["side"]}), "p": u["p"], "fair_odds": u["fair_odds"]})
    for m in sgms:
        rows.append({"kind": "multi", "game_id": m["game_id"], "desc": " + ".join(x["text"] for x in m["legs"]),
                     "detail": json.dumps(m["legs"]), "p": m["p"], "fair_odds": m["fair_odds"]})
    if not rows:
        return
    new = pd.DataFrame(rows)
    new["date"], new["result"] = str(slate_date), None
    old = pd.read_csv(BOLD_LOG, dtype={"game_id": str}) if BOLD_LOG.exists() else pd.DataFrame(columns=LOG_COLS)
    old = old[~((old["date"] == str(slate_date)) & old["game_id"].isin(set(new["game_id"])))]
    pd.concat([old, new[LOG_COLS]], ignore_index=True).to_csv(BOLD_LOG, index=False)


def _leg_hit(leg, gid, box_idx, games_idx):
    if leg["kind"] == "win":
        g = games_idx.loc[gid]
        home_won = g["home_score"] > g["away_score"]
        return home_won == (leg["team_id"] == g["home_id"])
    key = (gid, str(leg["player_id"]))
    if key not in box_idx.index:
        return None  # did not play: void
    row = box_idx.loc[key]
    if bool(row["dnp"]):
        return None
    val = row["pts"] + row["reb"] + row["ast"] if leg["stat"] == "pra" else row[leg["stat"]]
    return bool(val >= leg["line"])


def grade_bold(games: pd.DataFrame, box: pd.DataFrame) -> int:
    if not BOLD_LOG.exists():
        return 0
    df = pd.read_csv(BOLD_LOG, dtype={"game_id": str, "result": object})
    df["result"] = df["result"].astype(object)
    done = games[games["completed"].astype(bool)].set_index("game_id")
    bx = box.set_index(["game_id", "player_id"])
    n = 0
    for i in df.index[df["result"].isna()]:
        r = df.loc[i]
        if r["game_id"] not in done.index:
            continue
        d = json.loads(r["detail"])
        if r["kind"] == "call":
            hit = _leg_hit({"kind": "player", **d}, r["game_id"], bx, done)
        elif r["kind"] == "upset":
            g = done.loc[r["game_id"]]
            hit = (g["home_score"] > g["away_score"]) == (d["side"] == "home")
        else:
            res = [_leg_hit(leg, r["game_id"], bx, done) for leg in d]
            hit = None if any(x is None for x in res) else all(res)
        df.at[i, "result"] = "void" if hit is None else ("hit" if hit else "miss")
        n += 1
    df.to_csv(BOLD_LOG, index=False)
    return n


def bold_record() -> dict:
    if not BOLD_LOG.exists():
        return {}
    df = pd.read_csv(BOLD_LOG)
    g = df[df["result"].isin(["hit", "miss"])].copy()
    if g.empty:
        return {"graded": 0}
    g["hit"] = (g["result"] == "hit").astype(float)
    # profit if every call had been bet at exactly fair odds (0 expected) and at 10% better than fair
    g["at_fair"] = np.where(g["hit"] == 1, g["fair_odds"] - 1, -1)
    out = {"graded": int(len(g))}
    for k, d in g.groupby("kind"):
        out[k] = {"n": int(len(d)), "hits": int(d["hit"].sum()), "expected": round(float(d["p"].sum()), 1),
                  "hit_rate": round(float(d["hit"].mean()), 3), "avg_p": round(float(d["p"].mean()), 3),
                  "units_at_fair": round(float(d["at_fair"].sum()), 2)}
    g["bucket"] = pd.cut(g["p"], [0, .15, .2, .25, .3, .4, .5])
    out["calibration"] = [{"bucket": str(b), "said": round(float(d["p"].mean()), 3), "happened": round(float(d["hit"].mean()), 3),
                           "n": int(len(d))} for b, d in g.groupby("bucket", observed=True)]
    out["recent"] = df.sort_values("date", ascending=False).head(40).replace({np.nan: None})[
        ["date", "kind", "desc", "p", "fair_odds", "result"]].to_dict("records")
    return out
