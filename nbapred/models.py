"""Training, validation and prediction."""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats as sps
from xgboost import XGBRegressor

from . import features as F
from .config import PROP_STATS, SEASON_TYPES

log = logging.getLogger(__name__)

OBJECTIVES = {"pts": "reg:squarederror", "min": "reg:squarederror",
              "reb": "count:poisson", "ast": "count:poisson", "fg3m": "count:poisson"}
VALID_FRAC = 0.2
MIN_HISTORY = 3


# ---------------------------------------------------------------- config (what the learning loop changes)
DEFAULT_PARAMS = {"max_depth": 5, "min_child_weight": 20, "learning_rate": 0.03}
DEFAULT_CONFIG = {"groups": [], "params": DEFAULT_PARAMS, "version": 1}


def config_path():
    from .config import DATA_DIR
    return DATA_DIR / "model_config.json"


def load_config() -> dict:
    import json
    p = config_path()
    if not p.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    cfg = json.loads(p.read_text())
    cfg.setdefault("groups", [])
    cfg["params"] = {**DEFAULT_PARAMS, **cfg.get("params", {})}
    return cfg


def save_config(cfg: dict) -> None:
    import json
    config_path().write_text(json.dumps(cfg, indent=2))


def player_feats(cfg: dict) -> list[str]:
    from .extra import GROUPS
    cols = list(F.PLAYER_FEATURES)
    for g in cfg.get("groups", []):
        cols += [c for c in GROUPS[g][1] if c not in cols]
    return cols


def game_feats(cfg: dict) -> list[str]:
    from .extra import GROUPS
    cols = list(F.GAME_FEATURES)
    for g in cfg.get("groups", []):
        cols += [c for c in GROUPS[g][2] if c not in cols]
    return cols


def _xgb(objective, params=None, n_estimators=1500):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return XGBRegressor(n_estimators=n_estimators, learning_rate=p["learning_rate"], max_depth=int(p["max_depth"]),
                        min_child_weight=p["min_child_weight"], subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                        objective=objective, tree_method="hist", n_jobs=-1, early_stopping_rounds=60, verbosity=0)


def _final(objective, n, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return XGBRegressor(n_estimators=max(n, 50), learning_rate=p["learning_rate"], max_depth=int(p["max_depth"]),
                        min_child_weight=p["min_child_weight"], subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                        objective=objective, tree_method="hist", n_jobs=-1, verbosity=0)


def _time_split(df: pd.DataFrame):
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * (1 - VALID_FRAC))]
    return df[df["date"] < cut], df[df["date"] >= cut]


def _fit_var(pred: np.ndarray, resid: np.ndarray) -> dict:
    """Variance as a power of the mean: var = a * mu^b (fit on binned residuals)."""
    q = pd.qcut(pred, 12, duplicates="drop")
    b = pd.DataFrame({"mu": pred, "r2": resid ** 2, "q": q}).groupby("q", observed=True).agg(
        mu=("mu", "mean"), v=("r2", "mean"), n=("mu", "size"))
    b = b[(b["mu"] > 0.05) & (b["v"] > 0)]
    if len(b) < 3:
        return {"a": float(np.mean(resid ** 2)), "b": 0.0}
    slope, icpt = np.polyfit(np.log(b["mu"]), np.log(b["v"]), 1, w=np.sqrt(b["n"]))
    return {"a": float(math.exp(icpt)), "b": float(slope)}


def variance(mu, vp: dict):
    mu = np.maximum(np.asarray(mu, dtype=float), 0.05)
    return vp["a"] * mu ** vp["b"]


# ---------------------------------------------------------------- probabilities
def count_probs(mu: float, var: float, line: float) -> dict:
    """P(over / under / push) for an integer-valued stat vs a betting line."""
    mu = max(float(mu), 0.05)
    var = max(float(var), mu * 1.02)
    r = mu * mu / (var - mu)
    p = r / (r + mu)
    dist = sps.nbinom(r, p)
    if float(line).is_integer():
        k = int(line)
        push = float(dist.pmf(k))
        under = float(dist.cdf(k - 1))
        return {"over": 1 - under - push, "under": under, "push": push}
    k = math.floor(line)
    under = float(dist.cdf(k))
    return {"over": 1 - under, "under": under, "push": 0.0}


def normal_probs(mu: float, sd: float, line: float) -> dict:
    z = (line - mu) / sd
    under = float(sps.norm.cdf(z))
    return {"over": 1 - under, "under": under, "push": 0.0}


# ---------------------------------------------------------------- data assembly
def assemble(box, games, upcoming=None, injuries=None, rosters=None):
    """Return (player frame, game frame). Upcoming rows have is_upcoming=True."""
    games = games.copy()
    games = games[games["season_type"].isin(list(SEASON_TYPES))]
    base = F.attach_game_info(box, games)
    absn_hist, st = F.absences(base)

    upcoming = upcoming if upcoming is not None else pd.DataFrame(columns=games.columns)
    cand, absn_up = (F.upcoming_candidates(st, upcoming, injuries or [], rosters) if len(upcoming)
                     else (pd.DataFrame(), pd.DataFrame(columns=["game_id", "team_id", "miss_min", "miss_fga", "miss_n"])))
    absn = pd.concat([absn_hist, absn_up], ignore_index=True)
    absn[["miss_min", "miss_fga", "miss_n"]] = absn[["miss_min", "miss_fga", "miss_n"]].astype(float)

    T = F.team_games(base, games, upcoming if len(upcoming) else None)

    hist = base[base["min"] > 0].copy()
    hist["is_upcoming"] = False
    parts = [hist]
    if len(cand):
        up = cand.merge(upcoming[["game_id", "date", "season", "season_type", "home_id", "away_id"]], on="game_id")
        up["home"] = (up["team_id"] == up["home_id"]).astype(int)
        up["opp_id"] = np.where(up["home"] == 1, up["away_id"], up["home_id"])
        up = up.drop(columns=["home_id", "away_id"])
        up["is_upcoming"] = True
        up["starter"] = np.nan
        parts.append(up)
    P = pd.concat(parts, ignore_index=True)
    P = F.player_features(P)

    # ---- extra factor groups (always computed; the config decides which ones the models use)
    from . import extra as X
    from .store import load_bio
    bio = load_bio()
    tctx, pctx, lstate = X.lineup_context(base, bio)
    tctx_up, pctx_up = X.upcoming_lineup_context(lstate, cand, upcoming) if len(cand) else (pd.DataFrame(), pd.DataFrame())
    team_ctx = pd.concat([tctx, tctx_up], ignore_index=True)
    player_ctx = pd.concat([pctx, pctx_up], ignore_index=True)
    up_rows = T[T["game_id"].isin(set(upcoming["game_id"]))] if len(upcoming) else None
    pdef = X.position_defense(base, up_rows)
    all_games = pd.concat([games, upcoming], ignore_index=True)
    sched = X.team_schedule(T, all_games)
    P = X.player_extras(P, bio)

    PF = F.player_model_frame(P, T, absn)
    PF = X.attach(PF, team_ctx, player_ctx, pdef, sched)

    G = F.game_model_frame(T, absn, pd.concat([games[games["completed"].astype(bool)], upcoming], ignore_index=True))
    G = X.attach_game(G, team_ctx, sched)
    G["is_upcoming"] = G["game_id"].isin(set(upcoming["game_id"])) if len(upcoming) else False
    G["date"] = pd.to_datetime(G["date"])
    return PF, G


# ---------------------------------------------------------------- training
def train_player_models(PF: pd.DataFrame, cfg: dict | None = None):
    cfg = cfg or load_config()
    feats, params = player_feats(cfg), cfg["params"]
    data = PF[(~PF["is_upcoming"]) & (PF["n_games"] >= MIN_HISTORY)].dropna(subset=["pts"])
    tr, va = _time_split(data)
    X_tr, X_va = tr[feats], va[feats]
    models, var_params, metrics, va_preds = {}, {}, {}, {}
    for t in PROP_STATS + ["min"]:
        m = _xgb(OBJECTIVES[t], params)
        m.fit(X_tr, tr[t], eval_set=[(X_va, va[t])], verbose=False)
        best = int(m.best_iteration or m.n_estimators) + 1
        pred = m.predict(X_va)
        va_preds[t] = pred
        resid = va[t].to_numpy() - pred
        var_params[t] = _fit_var(pred, resid)
        base = va[f"{t}_e10"].fillna(va[f"{t}_szn"]).fillna(tr[t].mean()).to_numpy()
        metrics[t] = {"mae": float(np.mean(np.abs(resid))), "baseline_mae": float(np.mean(np.abs(va[t] - base))),
                      "n_valid": int(len(va)), "trees": best}
        final = _final(OBJECTIVES[t], best, params)
        final.fit(data[feats], data[t], verbose=False)
        models[t] = final
        log.info("%-4s MAE %.2f (baseline %.2f), trees %d", t, metrics[t]["mae"], metrics[t]["baseline_mae"], best)

    pra_pred = va_preds["pts"] + va_preds["reb"] + va_preds["ast"]
    pra_true = (va["pts"] + va["reb"] + va["ast"]).to_numpy()
    var_params["pra"] = _fit_var(pra_pred, pra_true - pra_pred)
    pra_base = (va["pts_e10"] + va["reb_e10"] + va["ast_e10"]).fillna(pra_true.mean()).to_numpy()
    metrics["pra"] = {"mae": float(np.mean(np.abs(pra_true - pra_pred))),
                      "baseline_mae": float(np.mean(np.abs(pra_true - pra_base))), "n_valid": int(len(va))}

    # Calibration check: how often did "over the model's median" lines hit, by predicted probability bucket.
    calib = _calibration(va, va_preds, var_params)
    info = {"metrics": metrics, "var": var_params, "calibration": calib,
            "valid_from": str(pd.Timestamp(va["date"].min()).date()), "valid_to": str(pd.Timestamp(va["date"].max()).date()),
            "train_rows": int(len(data)), "features": feats, "groups": list(cfg.get("groups", []))}
    valid = va.copy()
    for t, pr in va_preds.items():
        valid[f"p_{t}"] = pr
    info["_valid"] = valid  # held-out predictions, used by the pattern finder (not published)
    return models, info


def _calibration(va, va_preds, var_params):
    """Simulate lines at the player's recent average and compare predicted vs actual over rates."""
    rows = []
    for t in PROP_STATS:
        line = np.floor(va[f"{t}_e10"].fillna(0).to_numpy()) + 0.5
        mu = va_preds[t]
        v = variance(mu, var_params[t])
        p_over = np.array([count_probs(m, vv, l)["over"] for m, vv, l in zip(mu, v, line)])
        hit = (va[t].to_numpy() > line).astype(float)
        rows.append(pd.DataFrame({"p": p_over, "hit": hit}))
    d = pd.concat(rows)
    d["bucket"] = pd.cut(d["p"], [0, .3, .4, .45, .5, .55, .6, .7, 1.0])
    out = d.groupby("bucket", observed=True).agg(pred=("p", "mean"), actual=("hit", "mean"), n=("p", "size")).reset_index()
    return [{"bucket": str(r.bucket), "pred": round(r.pred, 3), "actual": round(r.actual, 3), "n": int(r.n)}
            for r in out.itertuples()]


def train_game_models(G: pd.DataFrame, cfg: dict | None = None):
    cfg = cfg or load_config()
    feats, params = game_feats(cfg), cfg["params"]
    data = G[(~G["is_upcoming"]) & (G["h_t_ngames"] >= 3) & (G["a_t_ngames"] >= 3)].dropna(subset=["margin"])
    tr, va = _time_split(data)
    models, info = {}, {"features": feats}
    for t in ("margin", "total"):
        m = _xgb("reg:squarederror", params)
        m.fit(tr[feats], tr[t], eval_set=[(va[feats], va[t])], verbose=False)
        best = int(m.best_iteration or m.n_estimators) + 1
        pred = m.predict(va[feats])
        resid = va[t].to_numpy() - pred
        info[t] = {"mae": float(np.mean(np.abs(resid))), "sd": float(np.std(resid)), "n_valid": int(len(va)), "trees": best}
        if t == "margin":
            info[t]["winner_acc"] = float(np.mean((pred > 0) == (va[t].to_numpy() > 0)))
            info[t]["home_baseline_acc"] = float(np.mean(va[t].to_numpy() > 0))
        final = _final("reg:squarederror", best, params)
        final.fit(data[feats], data[t], verbose=False)
        models[t] = final
        log.info("game %-6s MAE %.2f sd %.2f", t, info[t]["mae"], info[t]["sd"])
    return models, info


# ---------------------------------------------------------------- prediction
def predict_players(models, info, PF: pd.DataFrame) -> pd.DataFrame:
    up = PF[PF["is_upcoming"]].copy()
    if up.empty:
        return up
    X = up[info["features"]]
    for t, m in models.items():
        up[f"p_{t}"] = np.clip(m.predict(X), 0, None)
    up["p_pra"] = up["p_pts"] + up["p_reb"] + up["p_ast"]
    for t in PROP_STATS + ["pra"]:
        up[f"v_{t}"] = variance(up[f"p_{t}"], info["var"][t])
    # Low-sample players get a confidence flag
    up["low_sample"] = up["n_games"] < 10
    return up


def predict_games(models, info, G: pd.DataFrame) -> pd.DataFrame:
    up = G[G["is_upcoming"]].copy()
    if up.empty:
        return up
    X = up[info["features"]]
    up["p_margin"] = models["margin"].predict(X)
    up["p_total"] = models["total"].predict(X)
    sd = info["margin"]["sd"]
    up["p_home_win"] = sps.norm.cdf(up["p_margin"] / sd)
    up["p_home_score"] = (up["p_total"] + up["p_margin"]) / 2
    up["p_away_score"] = (up["p_total"] - up["p_margin"]) / 2
    return up
