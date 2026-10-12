"""Turn predictions + lines into picks, log them, and grade them after games finish."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import models
from .config import EDGE_THRESHOLD, LOG_DIR, MAX_EDGE
from .odds import norm_player, payout

log = logging.getLogger(__name__)
PICKS = LOG_DIR / "picks.csv"
PROJ = LOG_DIR / "projections.csv"
PICK_COLS = ["date", "game_id", "matchup", "market", "player_id", "player", "side", "line", "price", "book",
             "model_prob", "fair_prob", "edge", "ev", "projection", "flagged", "result", "profit", "actual"]


def _option(market, side, line, s, model_p, **extra):
    price = s["price"]
    ev = model_p * payout(price) - (1 - model_p) if price is not None else None
    return {"market": market, "side": side, "line": line, "price": price, "book": s["book"],
            "model_prob": round(float(model_p), 4), "fair_prob": round(float(s["fair"]), 4),
            "edge": round(float(model_p - s["fair"]), 4), "ev": None if ev is None else round(float(ev), 4), **extra}


def game_options(g, lines: dict, sd_margin: float, sd_total: float) -> list[dict]:
    out = []
    if not lines:
        return out
    mu, tot = g.p_margin, g.p_total
    if lines.get("ml"):
        ph = float(sps.norm.cdf(mu / sd_margin))
        for side, p in (("home", ph), ("away", 1 - ph)):
            out.append(_option("moneyline", side, None, lines["ml"]["sides"][side], p))
    if lines.get("spread"):
        hp = lines["spread"]["point"]
        ph = float(sps.norm.cdf((mu + hp) / sd_margin))
        for side, p in (("home", ph), ("away", 1 - ph)):
            out.append(_option("spread", side, hp if side == "home" else -hp, lines["spread"]["sides"][side], p))
    if lines.get("total"):
        pt = lines["total"]["point"]
        po = float(sps.norm.cdf((tot - pt) / sd_total))
        for side, p in (("over", po), ("under", 1 - po)):
            out.append(_option("total", side, pt, lines["total"]["sides"][side], p))
    return out


def prop_options(pl, plines: dict, var_params: dict) -> list[dict]:
    out = []
    key_name = norm_player(pl.player)
    for stat in ("pts", "reb", "ast", "fg3m", "pra"):
        c = plines.get((key_name, stat))
        if not c:
            continue
        mu = getattr(pl, f"p_{stat}")
        pr = models.count_probs(mu, getattr(pl, f"v_{stat}"), c["point"])
        for side in ("over", "under"):
            if side in c["sides"]:
                out.append(_option(stat, side, c["point"], c["sides"][side], pr[side],
                                   player_id=pl.player_id, player=pl.player, projection=round(float(mu), 2)))
    return out


def flag(opts: list[dict], low_sample=False, inj="") -> None:
    risky = low_sample or inj.lower() in ("questionable", "doubtful", "game-time decision", "day-to-day")
    for o in opts:
        strong = o["ev"] is not None and o["edge"] >= EDGE_THRESHOLD and o["ev"] > 0
        o["check"] = bool(strong and (o["edge"] > MAX_EDGE or risky))
        o["flagged"] = bool(strong and not o["check"])


# ---------------------------------------------------------------- logging
def log_picks(slate_date, rows: list[dict]) -> None:
    """Replace today's not-yet-started entries with the latest run's picks."""
    new = pd.DataFrame(rows)
    if new.empty:
        return
    new["date"] = str(slate_date)
    for c in PICK_COLS:
        if c not in new:
            new[c] = None
    old = pd.read_csv(PICKS, dtype={"game_id": str, "player_id": str}) if PICKS.exists() else pd.DataFrame(columns=PICK_COLS)
    old = old[~((old["date"] == str(slate_date)) & old["game_id"].isin(set(new["game_id"])))]
    pd.concat([old, new[PICK_COLS]], ignore_index=True).to_csv(PICKS, index=False)


def log_projections(slate_date, pp: pd.DataFrame) -> None:
    if pp.empty:
        return
    cols = ["game_id", "player_id", "player", "team_id", "p_min", "p_pts", "p_reb", "p_ast", "p_fg3m", "p_pra"]
    new = pp[cols].copy()
    new.insert(0, "date", str(slate_date))
    new[cols[4:]] = new[cols[4:]].round(2)
    old = pd.read_csv(PROJ, dtype={"game_id": str, "player_id": str}) if PROJ.exists() else None
    if old is not None:
        old = old[~((old["date"] == str(slate_date)) & old["game_id"].isin(set(new["game_id"])))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(PROJ, index=False)


# ---------------------------------------------------------------- grading
def _settle(actual, side, line):
    if actual is None or pd.isna(actual):
        return None
    if side in ("over", "home"):
        diff = actual - line
    else:
        diff = line - actual
    return "push" if diff == 0 else ("win" if diff > 0 else "loss")


def grade(games: pd.DataFrame, box: pd.DataFrame) -> int:
    if not PICKS.exists():
        return 0
    df = pd.read_csv(PICKS, dtype={"game_id": str, "player_id": str, "result": object})
    df["result"] = df["result"].astype(object)
    df["actual"] = df["actual"].astype(float)
    df["profit"] = df["profit"].astype(float)
    open_ = df["result"].isna()
    done = games[games["completed"].astype(bool)].set_index("game_id")
    played = box.copy()
    played["pra"] = played["pts"] + played["reb"] + played["ast"]
    bx = played.set_index(["game_id", "player_id"])
    n = 0
    for i in df.index[open_]:
        r = df.loc[i]
        if r["game_id"] not in done.index:
            continue
        g = done.loc[r["game_id"]]
        m = r["market"]
        if m == "moneyline":
            actual = g["home_score"] - g["away_score"]
            res = _settle(actual, r["side"], 0)
        elif m == "spread":
            actual = g["home_score"] - g["away_score"]
            # line stored from the picked side's perspective
            res = _settle(actual + r["line"], "over", 0) if r["side"] == "home" else _settle(-actual + r["line"], "over", 0)
        elif m == "total":
            actual = g["home_score"] + g["away_score"]
            res = _settle(actual, r["side"], r["line"])
        else:
            key = (r["game_id"], str(r["player_id"]))
            if key not in bx.index:
                res, actual = "void", None  # did not play -> books void the bet
            else:
                row = bx.loc[key]
                if bool(row["dnp"]):
                    res, actual = "void", None
                else:
                    actual = row[m]
                    res = _settle(actual, r["side"], r["line"])
        if res is None:
            continue
        df.at[i, "result"] = res
        df.at[i, "actual"] = actual
        df.at[i, "profit"] = payout(r["price"]) if res == "win" else (-1.0 if res == "loss" else 0.0)
        n += 1
    df.to_csv(PICKS, index=False)
    return n


def track_record(box: pd.DataFrame) -> dict:
    out = {"picks": None, "projections": None}
    if PICKS.exists():
        df = pd.read_csv(PICKS, dtype={"game_id": str, "player_id": str})
        f = df[(df["flagged"].astype(str).str.lower() == "true") & df["result"].isin(["win", "loss", "push"])].copy()
        f = f.sort_values("date")

        def summ(d):
            w, l, p = (d["result"] == "win").sum(), (d["result"] == "loss").sum(), (d["result"] == "push").sum()
            return {"n": int(len(d)), "wins": int(w), "losses": int(l), "pushes": int(p),
                    "win_rate": round(w / (w + l), 3) if w + l else None,
                    "units": round(float(d["profit"].sum()), 2), "roi": round(float(d["profit"].mean()), 3) if len(d) else None,
                    "avg_edge": round(float(d["edge"].mean()), 3) if len(d) else None}
        daily = f.groupby("date")["profit"].sum().cumsum().round(2)
        recent = df[df["flagged"].astype(str).str.lower() == "true"].sort_values("date", ascending=False).head(60)
        out["picks"] = {"all": summ(f), "by_market": {m: summ(d) for m, d in f.groupby("market")},
                        "curve": [{"date": d, "units": float(u)} for d, u in daily.items()],
                        "recent": recent.replace({np.nan: None}).to_dict("records")}
    if PROJ.exists():
        pr = pd.read_csv(PROJ, dtype={"game_id": str, "player_id": str})
        b = box[box["min"] > 0][["game_id", "player_id", "min", "pts", "reb", "ast", "fg3m"]]
        j = pr.merge(b, on=["game_id", "player_id"])
        if len(j):
            j["pra"] = j["pts"] + j["reb"] + j["ast"]
            out["projections"] = {"n": int(len(j)), **{s: round(float((j[f"p_{s}"] - j[s]).abs().mean()), 2)
                                                         for s in ("min", "pts", "reb", "ast", "fg3m", "pra")}}
    return out
