"""The learning loop: post-mortems, pattern finding, experiments and proposals.

Nothing here changes the model by itself. Experiments produce *proposals*; the owner approves them
(see scripts/apply_proposal.py), and every approved change is logged and can be undone.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import models
from .config import LOG_DIR, PROP_STATS
from .extra import GROUPS, pos_group

log = logging.getLogger(__name__)
LEARN = LOG_DIR / "learning.json"
TARGETS = PROP_STATS + ["min"]
MIN_GAIN = 0.001          # a change must cut the average miss by at least 0.1% (in both halves of the test period) ...
REJECT_COOLDOWN_DAYS = 45  # ... and rejected ideas are not re-proposed for a while

PARAM_IDEAS = [
    ("max_depth", 4, "Simpler trees (depth 4): less likely to memorise noise"),
    ("max_depth", 6, "Deeper trees (depth 6): can learn more complex interactions"),
    ("min_child_weight", 40, "Require more games per pattern before trusting it"),
    ("min_child_weight", 10, "Allow patterns backed by fewer games"),
    ("recency_halflife", 365, "Weight recent games more: a game a year old counts half as much"),
    ("recency_halflife", 180, "Lean hard on recent games: a game 6 months old counts half as much"),
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="minutes")


def load() -> dict:
    if LEARN.exists():
        d = json.loads(LEARN.read_text())
    else:
        d = {}
    for k in ("proposals", "changes", "postmortems", "patterns", "experiments", "events"):
        d.setdefault(k, [] if k != "patterns" else {})
    return d


def save(d: dict) -> None:
    d["postmortems"] = d["postmortems"][-60:]
    d["experiments"] = d["experiments"][-20:]
    d["events"] = d["events"][-300:]
    LEARN.write_text(json.dumps(d, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def event(d: dict, kind: str, text: str, **extra) -> None:
    d["events"].append({"at": now(), "kind": kind, "text": text, **extra})


# ================================================================== post-mortem
def postmortem(d: dict, projections: pd.DataFrame, box: pd.DataFrame, games: pd.DataFrame) -> list[str]:
    """Explain the misses for every slate whose games have all finished and that wasn't reviewed yet."""
    done_dates = {p["date"] for p in d["postmortems"]}
    if projections.empty:
        return []
    g = games.set_index("game_id")
    completed = set(games.loc[games["completed"].astype(bool), "game_id"])
    new_dates = []
    for date, pr in projections.groupby("date"):
        if date in done_dates or not set(pr["game_id"]).issubset(completed):
            continue
        rep = _review_slate(date, pr, box, g)
        if rep:
            d["postmortems"].append(rep)
            event(d, "postmortem", f"Reviewed {rep['n']} player projections from {date}: "
                                   f"average miss {rep['mae']['pts']} pts, {rep['mae']['reb']} reb, {rep['mae']['ast']} ast.")
            new_dates.append(date)
    return new_dates


def _review_slate(date, pr, box, g):
    b = box.copy()
    team_min = b.groupby(["game_id", "team_id"])["min"].sum()
    j = pr.merge(b, on=["game_id", "player_id"], how="left", suffixes=("", "_box"))
    if j.empty:
        return None
    j["played"] = j["min"].fillna(0) > 0
    rows = []
    for r in j.itertuples(index=False):
        gm = g.loc[r.game_id]
        margin = abs((gm["home_score"] or 0) - (gm["away_score"] or 0))
        ot = team_min.get((r.game_id, r.team_id), 240) > 250
        reason, detail = _explain(r, margin, ot)
        act = {s: (getattr(r, s) if r.played else 0) for s in ("min", "pts", "reb", "ast", "fg3m")}
        rows.append({"player": r.player, "game_id": r.game_id, "played": bool(r.played),
                     "p_min": r.p_min, "p_pts": r.p_pts, "p_reb": r.p_reb, "p_ast": r.p_ast,
                     **{f"a_{k}": v for k, v in act.items()}, "reason": reason, "detail": detail,
                     "err_pts": act["pts"] - r.p_pts})
    R = pd.DataFrame(rows)
    P = R[R["played"]]
    mae = {s: round(float((P[f"a_{s}"] - P[f"p_{s}"]).abs().mean()), 2) for s in ("min", "pts", "reb", "ast")}
    # how much of the points miss came from minutes rather than scoring rate
    rate = P["p_pts"] / P["p_min"].replace(0, np.nan)
    min_part = ((P["a_min"] - P["p_min"]) * rate).abs().sum()
    share_minutes = float(min_part / max(P["err_pts"].abs().sum(), 1e-9))
    big = R[(R["err_pts"].abs() >= 8) | (~R["played"])].copy()
    big["abs"] = big["err_pts"].abs()
    reasons = R[R["err_pts"].abs() >= 6]["reason"].value_counts().to_dict()
    return {
        "date": str(date), "n": int(len(P)), "dnp": int((~R["played"]).sum()), "mae": mae,
        "share_minutes": round(min(share_minutes, 1.0), 3), "reasons": reasons,
        "biggest": big.sort_values("abs", ascending=False).head(12)[
            ["player", "played", "p_pts", "a_pts", "p_min", "a_min", "reason", "detail"]].round(1).to_dict("records"),
    }


def _explain(r, margin, ot):
    """Best guess at why a projection missed, from what the box score can show."""
    if not r.played:
        return "Did not play", "Projected to play but sat out: late scratch, rest or coach's decision"
    pm, am = float(r.p_min or 0), float(r.min or 0)
    if pm > 0 and am < 0.7 * pm:
        if margin >= 20:
            return "Blowout", f"Played {am:.0f} min vs {pm:.0f} expected in a {margin:.0f}-point game"
        if am < 0.5 * pm:
            return "Left early or benched", f"Only {am:.0f} min vs {pm:.0f} expected: possible injury, foul trouble or benching"
        return "Fewer minutes", f"{am:.0f} min vs {pm:.0f} expected"
    if pm > 0 and am > 1.25 * pm:
        if ot:
            return "Overtime", f"Played {am:.0f} min including overtime"
        return "Bigger role", f"{am:.0f} min vs {pm:.0f} expected: rotation change or teammate missing"
    fga = float(getattr(r, "fga", 0) or 0)
    if fga >= 8:
        pct = float(getattr(r, "fgm", 0) or 0) / fga
        if pct >= 0.62:
            return "Hot shooting", f"Shot {int(r.fgm)}/{int(fga)}: well above normal"
        if pct <= 0.32:
            return "Cold shooting", f"Shot {int(r.fgm)}/{int(fga)}: well below normal"
    return "Normal variation", "Minutes and shooting close to expected"


# ================================================================== patterns
SEGMENTS = [
    # (label, function(frame) -> boolean mask, related factor group or None)
    ("Back-to-back games", lambda v: v["t_b2b"] == 1, "fatigue_travel"),
    ("3 games in 4 nights", lambda v: v.get("t_g4", pd.Series(0, index=v.index)) >= 3, "fatigue_travel"),
    ("Games at altitude (Denver, Utah)", lambda v: v.get("t_altitude", pd.Series(0, index=v.index)) == 1, "fatigue_travel"),
    ("Long road trips (4+ away games)", lambda v: v.get("t_road_streak", pd.Series(0, index=v.index)) >= 4, "fatigue_travel"),
    ("Expected blowouts (10+ points)", lambda v: v.get("exp_margin_abs", pd.Series(0, index=v.index)) >= 10, "blowout_risk"),
    ("First 3 games back from injury", lambda v: v.get("games_since_return", pd.Series(99, index=v.index)) <= 2, "injury_return"),
    ("Against a former team", lambda v: v.get("revenge", pd.Series(0, index=v.index)) == 1, "revenge_game"),
    ("Opponent has an elite shot-blocker", lambda v: v.get("o_pres_blk_max", pd.Series(0, index=v.index)) >= 2.2, "opposing_defenders"),
    ("2+ teammates out", lambda v: v["miss_n"] >= 2, None),
    ("Starters", lambda v: v["starter_last"] == 1, None),
    ("Bench players", lambda v: v["starter_last"] == 0, None),
    ("Guards", lambda v: v["pos"].map(pos_group) == "G", "position_defense"),
    ("Forwards", lambda v: v["pos"].map(pos_group) == "F", "position_defense"),
    ("Centers", lambda v: v["pos"].map(pos_group) == "C", "position_defense"),
    ("Home games", lambda v: v["home"] == 1, None),
    ("Road games", lambda v: v["home"] == 0, None),
    ("Hot shooters (last 3 games)", lambda v: v.get("fg_hot", pd.Series(0, index=v.index)) >= 0.08, "shooting_luck"),
    ("Cold shooters (last 3 games)", lambda v: v.get("fg_hot", pd.Series(0, index=v.index)) <= -0.08, "shooting_luck"),
    ("Late season, team out of contention", lambda v: (v.get("t_games_left", pd.Series(50, index=v.index)) <= 15)
     & (v.get("t_winpct", pd.Series(.5, index=v.index)) <= 0.4), "motivation"),
    ("Playoffs", lambda v: v.get("playoffs", pd.Series(0, index=v.index)) == 1, "motivation"),
]
UNITS = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "min": "minutes"}


def find_patterns(valid: pd.DataFrame, enabled: list[str], source: str) -> list[dict]:
    """Groups of games where the model is consistently too high or too low (held-out predictions only)."""
    out = []
    for label, fn, group in SEGMENTS:
        try:
            mask = fn(valid).fillna(False).astype(bool)
        except KeyError:
            continue
        seg = valid[mask]
        if len(seg) < 60:
            continue
        for t in ("pts", "reb", "ast", "min"):
            err = seg[t] - seg[f"p_{t}"]
            bias, sd = float(err.mean()), float(err.std())
            tstat = bias / (sd / np.sqrt(len(seg))) if sd > 0 else 0
            big_enough = abs(bias) >= {"pts": 0.4, "reb": 0.2, "ast": 0.15, "min": 0.6}[t]
            if abs(tstat) >= 3 and big_enough:
                direction = "under-predicts" if bias > 0 else "over-predicts"
                fix = None
                if group and group not in enabled:
                    fix = f"The '{group.replace('_', ' ')}' factor targets this; experiments test whether adding it helps."
                elif group:
                    fix = f"The '{group.replace('_', ' ')}' factor is already on; the gap may shrink as it learns."
                out.append({"segment": label, "stat": t, "bias": round(bias, 2), "n": int(len(seg)), "t": round(tstat, 1),
                            "text": f"{label}: the model {direction} {UNITS[t]} by {abs(bias):.1f} on average ({len(seg):,} cases).",
                            "group": group, "fix": fix, "source": source})
    out.sort(key=lambda p: -abs(p["t"]))
    return out[:15]


def live_frame(projections: pd.DataFrame, PF: pd.DataFrame) -> pd.DataFrame:
    """Live-season predictions joined with outcomes and the features they were made with."""
    if projections.empty:
        return pd.DataFrame()
    pr = projections.rename(columns={c: c for c in projections.columns})
    feats = PF[~PF["is_upcoming"]].drop(columns=[c for c in PF.columns if c.startswith("p_")], errors="ignore")
    j = pr.drop(columns=["player", "team_id"], errors="ignore").merge(feats, on=["game_id", "player_id"], how="inner")
    return j.dropna(subset=["pts"])


# ================================================================== experiments
TARGET_SEGMENTS = {
    # the games each factor group is meant to fix, so a niche factor isn't judged only by the overall average
    "fatigue_travel": ("back-to-backs, 3-in-4s, altitude and long road trips",
                       lambda v: (v["t_b2b"] == 1) | (v["t_g4"] >= 3) | (v["t_altitude"] == 1) | (v["t_road_streak"] >= 4)),
    "blowout_risk": ("expected blowouts (10+ points)", lambda v: v["exp_margin_abs"] >= 10),
    "injury_return": ("first 3 games back from injury", lambda v: v["games_since_return"] <= 2),
    "revenge_game": ("games against a former team", lambda v: v["revenge"] == 1),
    "opposing_defenders": ("games against an elite shot-blocker", lambda v: v["o_pres_blk_max"] >= 2.2),
    "motivation": ("playoffs and the last 15 games of the season", lambda v: (v["playoffs"] == 1) | (v["t_games_left"] <= 15)),
    "shooting_luck": ("players on hot or cold shooting streaks", lambda v: v["fg_hot"].abs() >= 0.08),
    "vs_opponent_history": ("opponents the player has faced 3+ times", lambda v: v["vs_n"] >= 3),
    "age_experience": ("players aged 32+", lambda v: v["age"] >= 32),
    "size": ("centers and players with a big height gap", lambda v: (v["pos"].map(pos_group) == "C") | (v["h_vs_opp_max"].abs() >= 4)),
}
SEG_GAIN = 0.02      # a niche factor must cut the miss in its own games by 2% ...
SEG_MIN_N = 300      # ... over at least this many games ...
SEG_MAX_HARM = 0.0003  # ... without making the overall average worse by more than 0.03%


def _scores(PF, G, cfg, lr=0.06):
    """Held-out absolute errors for each target under a config (fast settings, same for every config)."""
    cfg = {**cfg, "params": {**cfg["params"], "learning_rate": lr}}
    feats, params = models.player_feats(cfg), cfg["params"]
    data = PF[(~PF["is_upcoming"]) & (PF["n_games"] >= models.MIN_HISTORY)].dropna(subset=["pts"])
    tr, va = models._time_split(data)
    out = {"_va": va, "_half": (va["date"] < va["date"].quantile(0.5)).to_numpy()}
    for t in TARGETS:
        m = models._xgb(models.OBJECTIVES[t], params, n_estimators=800)
        m.fit(tr[feats], tr[t], eval_set=[(va[feats], va[t])], sample_weight=models.recency_weights(tr, params), verbose=False)
        out[t] = np.abs(va[t].to_numpy() - m.predict(va[feats]))
    gfeats = models.game_feats(cfg)
    gd = G[(~G["is_upcoming"]) & (G["h_t_ngames"] >= 3) & (G["a_t_ngames"] >= 3)].dropna(subset=["margin"])
    gtr, gva = models._time_split(gd)
    out["_ghalf"] = (gva["date"] < gva["date"].quantile(0.5)).to_numpy()
    for t in ("margin", "total"):
        m = models._xgb("reg:squarederror", params, n_estimators=800)
        m.fit(gtr[gfeats], gtr[t], eval_set=[(gva[gfeats], gva[t])], sample_weight=models.recency_weights(gtr, params), verbose=False)
        out[t] = np.abs(gva[t].to_numpy() - m.predict(gva[gfeats]))
    return out


def _mae(sc, k, mask=None):
    e = sc[k] if mask is None else sc[k][mask]
    return float(e.mean()) if len(e) else float("nan")


def _compare(base, chal, keys):
    gain, halves = {}, [[], []]
    for k in keys:
        h = base["_ghalf"] if k in ("margin", "total") else base["_half"]
        gain[k] = (_mae(base, k) - _mae(chal, k)) / _mae(base, k)
        for i, m in enumerate((h, ~h)):
            halves[i].append((_mae(base, k, m) - _mae(chal, k, m)) / _mae(base, k, m))
    return float(np.mean(list(gain.values()))), gain, [float(np.mean(x)) for x in halves]


def _segment_gain(base, chal, group):
    """Gain on the games a factor targets (points and minutes), split in two halves."""
    if group not in TARGET_SEGMENTS:
        return None
    label, fn = TARGET_SEGMENTS[group]
    va = base["_va"]
    try:
        mask = fn(va).fillna(False).to_numpy(dtype=bool)
    except KeyError:
        return None
    n = int(mask.sum())
    if n < SEG_MIN_N:
        return {"label": label, "n": n, "gain": None, "halves": None}
    h = base["_half"]
    res = []
    for m in (mask, mask & h, mask & ~h):
        g = [(_mae(base, k, m) - _mae(chal, k, m)) / _mae(base, k, m) for k in ("pts", "min") if m.sum()]
        res.append(float(np.mean(g)) if g else 0.0)
    per = {k: {"before": round(_mae(base, k, mask), 3), "after": round(_mae(chal, k, mask), 3)} for k in TARGETS}
    return {"label": label, "n": n, "gain": round(res[0], 4), "halves": [round(res[1], 4), round(res[2], 4)], "per": per}


def run_experiments(d: dict, PF: pd.DataFrame, G: pd.DataFrame) -> list[dict]:
    cfg = models.load_config()
    enabled = list(cfg["groups"])
    log.info("experiments: champion has groups %s", enabled or "none")
    champ = _scores(PF, G, cfg)
    tested, new_props = [], []

    ideas = [("add_group" if g not in enabled else "remove_group", g, None) for g in GROUPS]
    ideas += [("set_param", k, v) for k, v, _ in PARAM_IDEAS if cfg["params"].get(k) != v]

    for kind, target, value in ideas:
        if _cooling_down(d, kind, target, value):
            continue
        chal_cfg = json.loads(json.dumps(cfg))
        if kind == "add_group":
            chal_cfg["groups"] = enabled + [target]
        elif kind == "remove_group":
            chal_cfg["groups"] = [x for x in enabled if x != target]
        else:
            chal_cfg["params"][target] = value
        chal = _scores(PF, G, chal_cfg)
        uses_game = kind == "set_param" or bool(GROUPS.get(target, ("", [], []))[2])
        keys = TARGETS + (["margin", "total"] if uses_game else [])
        avg, per, halves = _compare(champ, chal, keys)
        seg = _segment_gain(champ, chal, target) if kind == "add_group" else None
        if kind == "remove_group":
            passed, basis = avg >= 0 and min(halves) >= -0.0005, "overall"
        elif avg >= MIN_GAIN and min(halves) > 0:
            passed, basis = True, "overall"
        elif seg and seg["gain"] is not None and seg["gain"] >= SEG_GAIN and min(seg["halves"]) > 0 and avg >= -SEG_MAX_HARM:
            passed, basis = True, "targeted"
        else:
            passed, basis = False, None
        rec = {"kind": kind, "target": target, "value": value, "gain": round(avg, 4), "basis": basis,
               "per": {k: {"before": round(_mae(champ, k), 3), "after": round(_mae(chal, k), 3), "change": round(-per[k], 4)} for k in keys},
               "halves": [round(h, 4) for h in halves], "segment": seg, "passed": bool(passed)}
        tested.append(rec)
        log.info("  %-12s %-20s overall %+.2f%% halves %s | targeted %s -> %s", kind, target, 100 * avg,
                 [f"{100*h:+.2f}%" for h in halves],
                 f"{100*seg['gain']:+.1f}% over {seg['n']}" if seg and seg["gain"] is not None else "-",
                 "PROPOSE" if passed else "no")
        if passed:
            p = _make_proposal(d, rec)
            if p:
                new_props.append(p)

    # pending proposals that no longer pass are withdrawn
    passing = {(r["kind"], r["target"], r["value"]) for r in tested if r["passed"]}
    tested_keys = {(r["kind"], r["target"], r["value"]) for r in tested}
    for p in d["proposals"]:
        key = (p["kind"], p["target"], p.get("value"))
        if p["status"] == "pending" and key in tested_keys and key not in passing:
            p["status"], p["closed"] = "expired", now()
            event(d, "expired", f"{p['id']} no longer improves the model on the latest data, withdrawn.")

    d["experiments"].append({"at": now(), "champion": {k: round(_mae(champ, k), 3) for k in TARGETS + ["margin", "total"]},
                             "groups": enabled, "params": cfg["params"], "tested": tested})
    n_pass = sum(r["passed"] for r in tested)
    event(d, "experiment", f"Tested {len(tested)} possible changes against the current model. "
                           f"{n_pass} improved accuracy on games the model hadn't seen"
                           + (f"; {len(new_props)} new proposal(s) waiting for your decision." if new_props else "."))
    return new_props


def _cooling_down(d, kind, target, value):
    for p in d["proposals"]:
        if (p["kind"], p["target"], p.get("value")) == (kind, target, value) and p["status"] == "rejected":
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(p.get("closed", p["created"]))).days
            if age < REJECT_COOLDOWN_DAYS:
                return True
    return False


def _describe(kind, target, value):
    if kind == "add_group":
        return f"Add factor: {target.replace('_', ' ')}", GROUPS[target][0]
    if kind == "remove_group":
        return f"Remove factor: {target.replace('_', ' ')}", "The model is as good or better without it, so it's just noise now. " + GROUPS[target][0]
    why = next((w for k, v, w in PARAM_IDEAS if k == target and v == value), "")
    return f"Change setting: {target.replace('_', ' ')} to {value}", why


def _make_proposal(d, rec):
    key = (rec["kind"], rec["target"], rec["value"])
    for p in d["proposals"]:
        if p["status"] == "pending" and (p["kind"], p["target"], p.get("value")) == key:
            p["evidence"] = rec  # refresh numbers
            p["updated"] = now()
            return None
    n = len(d["proposals"]) + 1
    title, why = _describe(*key)
    p = {"id": f"P-{n:04d}", "created": now(), "status": "pending", "kind": rec["kind"], "target": rec["target"],
         "value": rec["value"], "title": title, "why": why, "evidence": rec}
    d["proposals"].append(p)
    if rec.get("basis") == "targeted":
        sg = rec["segment"]
        msg = f"{p['id']}: {title}. Cuts the miss by {100 * sg['gain']:.1f}% in {sg['label']} ({sg['n']:,} test games)."
    else:
        msg = f"{p['id']}: {title}. Cuts the average miss by {100 * rec['gain']:.2f}% on games it hadn't seen."
    event(d, "proposal", msg, proposal=p["id"])
    return p


# ================================================================== applying decisions
def apply(d: dict, action: str, pid: str, who: str = "owner") -> str:
    cfg = models.load_config()
    if action == "undo":
        ch = next((c for c in reversed(d["changes"]) if c["id"] == pid or c.get("proposal") == pid), None)
        if not ch or ch.get("undone"):
            return f"Nothing to undo for {pid}."
        models.save_config(ch["config_before"])
        ch["undone"] = now()
        event(d, "undo", f"Undid {ch['proposal']} ({ch['title']}). Model settings restored.", proposal=ch["proposal"])
        return f"Undid {ch['proposal']}."
    p = next((x for x in d["proposals"] if x["id"] == pid), None)
    if not p:
        return f"No proposal {pid}."
    if p["status"] != "pending":
        return f"{pid} is already {p['status']}."
    if action == "reject":
        p["status"], p["closed"] = "rejected", now()
        event(d, "rejected", f"You rejected {pid} ({p['title']}). It won't be proposed again for {REJECT_COOLDOWN_DAYS} days.", proposal=pid)
        return f"Rejected {pid}."
    before = json.loads(json.dumps(cfg))
    if p["kind"] == "add_group" and p["target"] not in cfg["groups"]:
        cfg["groups"].append(p["target"])
    elif p["kind"] == "remove_group":
        cfg["groups"] = [g for g in cfg["groups"] if g != p["target"]]
    elif p["kind"] == "set_param":
        cfg["params"][p["target"]] = p["value"]
    cfg["version"] = int(cfg.get("version", 1)) + 1
    models.save_config(cfg)
    p["status"], p["closed"] = "approved", now()
    cid = f"C-{len(d['changes']) + 1:04d}"
    d["changes"].append({"id": cid, "proposal": pid, "at": now(), "title": p["title"], "config_before": before,
                         "config_after": cfg, "gain": p["evidence"]["gain"]})
    event(d, "approved", f"You approved {pid} ({p['title']}). Model version {cfg['version']} trains with it from the next run.",
          proposal=pid)
    return f"Approved {pid}."


def site_view(d: dict, cfg: dict) -> dict:
    """What the Learning page shows."""
    return {
        "proposals": [{k: v for k, v in p.items()} for p in d["proposals"] if p["status"] == "pending"],
        "decided": [{k: p[k] for k in ("id", "title", "status", "closed") if k in p}
                    for p in d["proposals"] if p["status"] != "pending"][-30:][::-1],
        "changes": [{k: c[k] for k in ("id", "proposal", "at", "title", "gain", "undone") if k in c} for c in d["changes"]][::-1],
        "postmortems": d["postmortems"][-14:][::-1],
        "patterns": d["patterns"],
        "experiments": [{k: e[k] for k in ("at", "champion", "groups", "tested")} for e in d["experiments"][-3:]][::-1],
        "events": d["events"][-80:][::-1],
        "config": {"groups": cfg["groups"], "params": cfg["params"], "version": cfg.get("version", 1)},
        "groups": {k: v[0] for k, v in GROUPS.items()},
    }
