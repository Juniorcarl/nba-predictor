"""Write the JSON files the static site reads (docs/data/*.json)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import picks as P
from .config import SITE_DATA


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (o is None or math.isnan(o) or math.isinf(o)) else round(float(o), 3)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o


def _write(name, obj):
    (SITE_DATA / name).write_text(json.dumps(_clean(obj), separators=(",", ":")))


def build_slate(slate_date, gp: pd.DataFrame, pp: pd.DataFrame, glines: dict, plines: dict, ginfo: dict, pinfo: dict):
    """Returns (slate dict, list of pick rows for logging)."""
    games_out, log_rows = [], []
    sd_m, sd_t = ginfo["margin"]["sd"], ginfo["total"]["sd"]
    for g in gp.sort_values("start_utc").itertuples():
        lines = glines.get(g.game_id) or {}
        gopts = P.game_options(g, lines, sd_m, sd_t)
        P.flag(gopts)
        matchup = f"{g.away_abbr} @ {g.home_abbr}"
        for o in gopts:
            log_rows.append({**o, "game_id": g.game_id, "matchup": matchup, "player_id": None, "player": None,
                             "projection": round(float(g.p_total if o["market"] == "total" else g.p_margin), 2)})
        players = []
        gpp = pp[pp["game_id"] == g.game_id].sort_values("p_min", ascending=False)
        for pl in gpp.itertuples():
            popts = P.prop_options(pl, plines.get(g.game_id, {}), pinfo["var"])
            P.flag(popts, low_sample=bool(pl.low_sample), inj=pl.inj_status or "")
            for o in popts:
                log_rows.append({**o, "game_id": g.game_id, "matchup": matchup})
            players.append({
                "id": pl.player_id, "name": pl.player, "pos": pl.pos, "side": "home" if pl.home == 1 else "away",
                "inj": pl.inj_status or "", "inj_note": pl.inj_comment or "", "low_sample": bool(pl.low_sample),
                "proj": {s: getattr(pl, f"p_{s}") for s in ("min", "pts", "reb", "ast", "fg3m", "pra")},
                "var": {s: getattr(pl, f"v_{s}") for s in ("pts", "reb", "ast", "fg3m", "pra")},
                "recent": {s: getattr(pl, f"{s}_r10") for s in ("min", "pts", "reb", "ast", "fg3m")},
                "props": popts,
            })
        games_out.append({
            "id": g.game_id, "start": g.start_utc, "home": {"id": g.home_id, "abbr": g.home_abbr, "name": g.home_name},
            "away": {"id": g.away_id, "abbr": g.away_abbr, "name": g.away_name},
            "pred": {"home_score": g.p_home_score, "away_score": g.p_away_score, "margin": g.p_margin,
                     "total": g.p_total, "home_win": g.p_home_win},
            "missing": {"home": g.h_miss_min, "away": g.a_miss_min},
            "lines": {k: v for k, v in lines.items() if v}, "options": gopts, "players": players,
        })
    slate = {"date": str(slate_date), "games": games_out}
    return slate, log_rows


def write_all(slate: dict, pinfo: dict, ginfo: dict, track: dict, players: list, extra_meta: dict):
    _write("slate.json", slate)
    _write("track.json", track)
    _write("players.json", players)
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "player_model": {k: v for k, v in pinfo.items() if k != "var"}, "var": pinfo.get("var"),
            "game_model": ginfo, **extra_meta}
    _write("meta.json", meta)


def players_index(box: pd.DataFrame, games: pd.DataFrame, pp: pd.DataFrame, n_logs=12) -> list:
    """Recent game logs for everyone who played in the last two seasons."""
    gi = games[["game_id", "date", "season", "home_id", "away_id", "home_abbr", "away_abbr"]]
    df = box[box["min"] > 0].merge(gi, on="game_id")
    if df.empty:
        return []
    recent_seasons = sorted(df["season"].unique())[-2:]
    df = df[df["season"].isin(recent_seasons)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home"] = df["team_id"] == df["home_id"]
    df["team"] = np.where(df["home"], df["home_abbr"], df["away_abbr"])
    df["opp"] = np.where(df["home"], df["away_abbr"], df["home_abbr"])
    df = df.sort_values("date")
    proj = {r.player_id: {s: getattr(r, f"p_{s}") for s in ("min", "pts", "reb", "ast", "fg3m", "pra")} | {"game_id": r.game_id}
            for r in pp.itertuples()} if len(pp) else {}
    out = []
    for pid, d in df.groupby("player_id"):
        last = d.iloc[-1]
        cur = d[d["season"] == d["season"].max()]
        avg = cur[["min", "pts", "reb", "ast", "fg3m"]].mean()
        logs = d.tail(n_logs).iloc[::-1]
        out.append({
            "id": pid, "name": last["player"], "team": last["team"], "pos": last["pos"],
            "season": int(last["season"]), "gp": int(len(cur)), "avg": avg.to_dict(),
            "logs": [{"d": r.date.strftime("%Y-%m-%d"), "opp": ("vs " if r.home else "@ ") + str(r.opp),
                      "min": r.min, "pts": r.pts, "reb": r.reb, "ast": r.ast, "fg3m": r.fg3m} for r in logs.itertuples()],
            "proj": proj.get(pid),
        })
    out.sort(key=lambda p: -(p["avg"].get("min") or 0))
    return out
