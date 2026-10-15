"""Day-to-day learning on top of the trained models.

1. News: read ESPN's NBA news feed and the injury-report notes every run, and tag players with things
   that change how they play (minutes limit, returning, moved to the bench, rest, illness, personal leave...).
2. Player corrections: if a player keeps beating (or missing) his projection, nudge his next projection.
   Shrunk toward zero so a couple of lucky games barely move it; fades as games get older.
3. News-tag effects: measure how players with each tag actually did against their projection, and apply
   that learned effect next time the tag shows up (only once there are enough cases).
4. Team corrections: the same idea for game margins and totals.

All corrections are measured against the *base* model projection (before corrections), so they can't
feed on themselves. Everything is logged and shown on the Learning page.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import espn
from .config import DATA_DIR, LOG_DIR

log = logging.getLogger(__name__)
NEWS = DATA_DIR / "news.csv"
GAME_PROJ = LOG_DIR / "game_projections.csv"
STATS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk"]
UNIT = {"min": "min", "pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "threes", "stl": "steals", "blk": "blocks"}

PLAYER_K = 5.0      # pseudo-games of "no correction" each player starts with (shrinkage)
PLAYER_HL = 10.0    # a game 10 games ago counts half as much as the latest
TAG_K = 15.0        # shrinkage for news-tag effects
TAG_MIN_N = 10      # cases needed before a news tag changes projections
TEAM_K = 8.0

# ---------------------------------------------------------------- news tags (free keyword rules)
TAGS = {
    "minutes_limit": ("On a minutes limit", r"minutes? (restriction|limit)|limited minutes|on a minutes|ramp(ing)? up"),
    "returning": ("Returning", r"(set|expected|cleared|plans?|slated|poised) to (return|make (his|a|the) (season )?debut)|"
                               r"will return|season debut|back in the lineup|return(s|ing)? (to action|from)|(his|'s) return\b"),
    "starting": ("Starting", r"will start\b(?! to)|(is|are) starting(?! to)|enter(s|ing)? the starting lineup|"
                             r"(draw|get|earn)s? the start|starting in place of|join(s|ing)? the starting"),
    "bench": ("Moved to the bench", r"(moved|moving|demoted|relegated|sent) to the bench|out of the starting lineup|"
                                    r"come off the bench|bench role|lose(s)? (his )?starting"),
    "rest": ("Rest / load management", r"\brest(ing|ed)?\b|load management|maintenance"),
    "illness": ("Illness", r"illness|\bsick\b|\bflu\b|virus|non-covid"),
    "personal": ("Personal leave", r"personal (reasons|matter)|family matter|bereavement|birth of"),
    "suspension": ("Suspension", r"suspend"),
    "doubt": ("Questionable", r"questionable|doubtful|game-time decision|day-to-day"),
    "new_injury": ("New injury", r"sprain|strain|fracture|torn|soreness|tendinitis|contusion|concussion|surgery|hamstring|ankle|knee"),
    "trade": ("Traded / signed", r"\btrade[sd]?\b|acquire[sd]?|\bwaive[sd]?\b|sign(s|ed)? (with|a )"),
    "big_game": ("Coming off a big game", r"career[- ]high|season[- ]high|\b[3-6]\d[- ]point (game|night|performance|outing|effort)"),
}
TEAM_TAGS = {"coach_change": ("Coaching change", r"fired|dismiss(ed)?|interim (head )?coach|new head coach")}
_RX = {k: re.compile(rx, re.I) for k, (_, rx) in TAGS.items() if rx.strip("|")}
_TRX = {k: re.compile(rx, re.I) for k, (_, rx) in TEAM_TAGS.items()}


def tag_text(text: str) -> list[str]:
    return [k for k, rx in _RX.items() if rx.search(text or "")]


def parse_news(js: dict) -> list[dict]:
    out = []
    for a in js.get("articles", []) or []:
        cats = a.get("categories", []) or []
        athletes = [str(c.get("athleteId") or (c.get("athlete") or {}).get("id")) for c in cats
                    if c.get("type") == "athlete" and (c.get("athleteId") or c.get("athlete"))]
        teams = [str(c.get("teamId") or (c.get("team") or {}).get("id")) for c in cats
                 if c.get("type") == "team" and (c.get("teamId") or c.get("team"))]
        text = f"{a.get('headline', '')}. {a.get('description', '')}"
        link = ((a.get("links") or {}).get("web") or {}).get("href", "")
        out.append({"id": str(a.get("id") or a.get("headline")), "published": a.get("published") or a.get("lastModified"),
                    "headline": a.get("headline", ""), "description": a.get("description", "") or "", "link": link,
                    "athletes": ";".join(athletes), "teams": ";".join(teams), "tags": ";".join(tag_text(text)),
                    "team_tags": ";".join(k for k, rx in _TRX.items() if rx.search(text))})
    return out


def update_news() -> pd.DataFrame:
    """Fetch the latest news, add to the stored file (kept for 150 days)."""
    try:
        rows = parse_news(espn._get(f"{espn.BASE}/news", {"limit": 100}))
    except RuntimeError as e:
        log.warning("news failed: %s", e)
        rows = []
    old = load_news()
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    if df.empty:
        return df
    df = df.drop_duplicates("id", keep="last")
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df[df["published"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=150)].sort_values("published")
    df.to_csv(NEWS, index=False)
    log.info("news: %d new articles, %d stored", len(rows), len(df))
    return df


def load_news() -> pd.DataFrame:
    if not NEWS.exists():
        return pd.DataFrame(columns=["id", "published", "headline", "description", "link", "athletes", "teams", "tags", "team_tags"])
    df = pd.read_csv(NEWS, dtype=str).fillna("")
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    return df


def player_news(news: pd.DataFrame, days: int = 3) -> tuple[dict, dict]:
    """({player_id: set(tags) from the last few days}, {player_id: [recent headlines]})."""
    tags, heads = {}, {}
    if news.empty:
        return tags, heads
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    for r in news.itertuples(index=False):
        for pid in str(r.athletes).split(";"):
            if not pid:
                continue
            if r.published >= cutoff and r.tags:
                tags.setdefault(pid, set()).update(t for t in str(r.tags).split(";") if t)
            if r.published >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10):
                heads.setdefault(pid, []).append({"headline": r.headline, "link": r.link, "date": r.published.strftime("%Y-%m-%d"),
                                                  "tags": [TAGS[t][0] for t in str(r.tags).split(";") if t in TAGS]})
    for pid in heads:
        heads[pid] = sorted(heads[pid], key=lambda h: h["date"], reverse=True)[:4]
    return tags, heads


# ---------------------------------------------------------------- learning from logged projections
def _joined(proj: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    if proj.empty:
        return pd.DataFrame()
    pr = proj.copy()
    for s in STATS:
        if f"p_{s}" not in pr:
            pr[f"p_{s}"] = np.nan
        pr[f"adj_{s}"] = pd.to_numeric(pr[f"adj_{s}"], errors="coerce").fillna(0) if f"adj_{s}" in pr else 0.0
        pr[f"b_{s}"] = pr[f"p_{s}"] - pr[f"adj_{s}"]
    pr["tags"] = pr["tags"].fillna("").astype(str) if "tags" in pr else ""
    b = box[box["min"] > 0][["game_id", "player_id"] + STATS]
    j = pr.merge(b, on=["game_id", "player_id"], how="inner")
    j["date"] = pd.to_datetime(j["date"])
    return j.sort_values("date")


def player_corrections(j: pd.DataFrame) -> dict:
    """{player_id: {stat: {"adj", "n", "avg"}}} from residuals against the base projection."""
    out = {}
    if j.empty:
        return out
    for pid, d in j.groupby("player_id"):
        d = d.tail(25)
        n = len(d)
        if n < 3:
            continue
        w = 0.5 ** (np.arange(n)[::-1] / PLAYER_HL)
        rec = {}
        for s in STATS:
            r = (d[s] - d[f"b_{s}"]).to_numpy(dtype=float)
            ok = ~np.isnan(r)
            if ok.sum() < 3:
                continue
            adj = float((w[ok] * r[ok]).sum() / (w[ok].sum() + PLAYER_K))
            rec[s] = {"adj": round(adj, 2), "n": int(ok.sum()), "avg": round(float(r[ok].mean()), 2)}
        out[str(pid)] = rec
    return out


def tag_effects(j: pd.DataFrame) -> dict:
    """{tag: {stat: {"adj", "n", "avg"}}}: how players with each news tag did vs their base projection."""
    out = {}
    if j.empty or "tags" not in j:
        return out
    for t in TAGS:
        d = j[j["tags"].astype(str).str.contains(rf"\b{t}\b")]
        if d.empty:
            continue
        rec = {}
        for s in STATS:
            r = (d[s] - d[f"b_{s}"]).dropna()
            rec[s] = {"adj": round(float(r.sum() / (len(r) + TAG_K)), 2), "n": int(len(r)), "avg": round(float(r.mean()), 2)}
        out[t] = rec
    return out


def team_corrections(games: pd.DataFrame) -> dict:
    """{team_id: {"margin": adj, "total": adj, "n"}} from logged game projections vs results."""
    if not GAME_PROJ.exists():
        return {}
    gp = pd.read_csv(GAME_PROJ, dtype={"game_id": str, "home_id": str, "away_id": str})
    done = games[games["completed"].astype(bool)][["game_id", "home_score", "away_score"]]
    j = gp.merge(done, on="game_id")
    if j.empty:
        return {}
    j["r_m"] = (j["home_score"] - j["away_score"]) - (j["p_margin"] - j["adj_margin"].fillna(0))
    j["r_t"] = (j["home_score"] + j["away_score"]) - (j["p_total"] - j["adj_total"].fillna(0))
    rows = pd.concat([pd.DataFrame({"team": j["home_id"], "m": j["r_m"], "t": j["r_t"], "date": j["date"]}),
                      pd.DataFrame({"team": j["away_id"], "m": -j["r_m"], "t": j["r_t"], "date": j["date"]})])
    out = {}
    for tid, d in rows.sort_values("date").groupby("team"):
        d = d.tail(20)
        n = len(d)
        w = 0.5 ** (np.arange(n)[::-1] / 10)
        out[str(tid)] = {"margin": round(float((w * d["m"]).sum() / (w.sum() + TEAM_K)) / 2, 2),
                         "total": round(float((w * d["t"]).sum() / (w.sum() + TEAM_K)) / 2, 2), "n": int(n)}
    return out


# ---------------------------------------------------------------- applying
def _cap(adj: float, base: float) -> float:
    lim = 0.25 * max(base, 0) + 0.3
    return float(np.clip(adj, -lim, lim))


def apply_players(pp: pd.DataFrame, corr: dict, effects: dict, ptags: dict, inj_tags: dict) -> pd.DataFrame:
    """Add learned corrections to projections; record them in adj_* columns and in the reasons."""
    if pp.empty:
        return pp
    pp = pp.copy()
    tags_col, why_col = [], []
    for s in STATS:
        pp[f"adj_{s}"] = 0.0
    for i, r in pp.iterrows():
        pid = str(r["player_id"])
        tags = sorted(set(ptags.get(pid, set())) | set(inj_tags.get(pid, set())))
        tags_col.append(";".join(tags))
        why = r.get("why") if isinstance(r.get("why"), dict) else None
        add_items = {s: [] for s in STATS}
        for s in STATS:
            if f"p_{s}" not in pp:
                continue
            base = float(r[f"p_{s}"])
            total = 0.0
            c = corr.get(pid, {}).get(s)
            if c and abs(c["adj"]) >= 0.05:
                total += c["adj"]
                add_items[s].append({"label": "Learned from his recent games",
                                     "detail": f"Has {'beaten' if c['avg'] > 0 else 'fallen short of'} his projection by "
                                               f"{abs(c['avg']):.1f} {UNIT[s]} on average over {c['n']} games",
                                     "amt": c["adj"], "cat": "learned"})
            for t in tags:
                e = effects.get(t, {}).get(s)
                if e and e["n"] >= TAG_MIN_N and abs(e["adj"]) >= 0.05:
                    total += e["adj"]
                    add_items[s].append({"label": f"News: {TAGS[t][0].lower()}",
                                         "detail": f"Players with this in the news have gone {e['avg']:+.1f} {UNIT[s]} vs projection "
                                                   f"({e['n']} cases)", "amt": e["adj"], "cat": "news"})
            total = _cap(total, base)
            if total:
                pp.at[i, f"p_{s}"] = max(base + total, 0.0)
                pp.at[i, f"adj_{s}"] = round(total, 2)
                for vcol in (f"v_{s}", f"vv_{s}"):
                    if vcol in pp and base > 0:
                        pp.at[i, vcol] = float(r[vcol]) * max(base + total, 0.05) / base
        if why:
            why = {k: dict(v, items=list(v["items"])) for k, v in why.items()}
            for s, items in add_items.items():
                if s in why and items:
                    scale = pp.at[i, f"adj_{s}"] / sum(it["amt"] for it in items) if sum(it["amt"] for it in items) else 0
                    for it in items:
                        why[s]["items"].append({**it, "amt": round(it["amt"] * scale, 1)})
                    why[s]["pred"] = round(float(pp.at[i, f"p_{s}"]), 1)
            if "pra" in why:
                extra = [it for s in ("pts", "reb", "ast") for it in why.get(s, {}).get("items", []) if it.get("cat") in ("learned", "news")]
                merged = {}
                for it in extra:
                    k = it["label"]
                    merged.setdefault(k, {**it, "amt": 0.0})
                    merged[k]["amt"] = round(merged[k]["amt"] + it["amt"], 1)
                why["pra"]["items"] += [m for m in merged.values() if abs(m["amt"]) >= 0.1]
            pp.at[i, "why"] = why
    pp["tags"] = tags_col
    pp["p_pra"] = pp["p_pts"] + pp["p_reb"] + pp["p_ast"]
    if "why" in pp:
        for i in pp.index:
            w = pp.at[i, "why"]
            if isinstance(w, dict) and "pra" in w:
                w["pra"]["pred"] = round(float(pp.at[i, "p_pra"]), 1)
    return pp


def injury_tags(cand_comments: dict) -> dict:
    """Tags from injury-report notes: {player_id: set(tags)}."""
    return {pid: set(tag_text(txt)) for pid, txt in cand_comments.items() if txt}


def apply_games(gp: pd.DataFrame, tcorr: dict, sd_margin: float) -> pd.DataFrame:
    from scipy import stats as sps
    if gp.empty or not tcorr:
        gp = gp.copy()
        gp["adj_margin"], gp["adj_total"] = 0.0, 0.0
        return gp
    gp = gp.copy()
    am, at = [], []
    for i, g in gp.iterrows():
        h, a = tcorr.get(str(g["home_id"]), {}), tcorr.get(str(g["away_id"]), {})
        dm = float(np.clip(h.get("margin", 0) - a.get("margin", 0), -4, 4))
        dt = float(np.clip(h.get("total", 0) + a.get("total", 0), -6, 6))
        am.append(round(dm, 2))
        at.append(round(dt, 2))
        gp.at[i, "p_margin"] = g["p_margin"] + dm
        gp.at[i, "p_total"] = g["p_total"] + dt
        w = g.get("why")
        if isinstance(w, dict):
            w = {k: dict(v, items=list(v["items"])) for k, v in w.items()}
            if abs(dm) >= 0.1:
                w["margin"]["items"].append({"label": "Learned from recent results", "amt": round(dm, 1), "cat": "learned",
                                             "detail": f"{g['home_abbr']} and {g['away_abbr']} have been beating or missing their projections lately"})
                w["margin"]["pred"] = round(float(gp.at[i, "p_margin"]), 1)
            if abs(dt) >= 0.1:
                w["total"]["items"].append({"label": "Learned from recent results", "amt": round(dt, 1), "cat": "learned",
                                            "detail": "Recent games involving these teams have gone over or under projections"})
                w["total"]["pred"] = round(float(gp.at[i, "p_total"]), 1)
            gp.at[i, "why"] = w
    gp["adj_margin"], gp["adj_total"] = am, at
    gp["p_home_win"] = sps.norm.cdf(gp["p_margin"] / sd_margin)
    gp["p_home_score"] = (gp["p_total"] + gp["p_margin"]) / 2
    gp["p_away_score"] = (gp["p_total"] - gp["p_margin"]) / 2
    return gp


def log_games(slate_date, gp: pd.DataFrame) -> None:
    if gp.empty:
        return
    new = gp[["game_id", "home_id", "away_id", "p_margin", "p_total", "adj_margin", "adj_total"]].copy()
    new.insert(0, "date", str(slate_date))
    old = pd.read_csv(GAME_PROJ, dtype={"game_id": str, "home_id": str, "away_id": str}) if GAME_PROJ.exists() else None
    if old is not None:
        old = old[~old["game_id"].isin(set(new["game_id"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.round(2).to_csv(GAME_PROJ, index=False)


# ---------------------------------------------------------------- what the Learning page shows
def summary(j: pd.DataFrame, corr: dict, effects: dict, tcorr: dict, news: pd.DataFrame, names: dict, abbr: dict) -> dict:
    months = []
    if not j.empty:
        j = j.copy()
        j["month"] = j["date"].dt.strftime("%Y-%m")
        for m, d in j.groupby("month"):
            row = {"month": m, "n": int(len(d))}
            for s in ("pts", "reb", "ast", "min"):
                row[s] = round(float((d[s] - d[f"p_{s}"]).abs().mean()), 2)
                row[f"b_{s}"] = round(float((d[s] - d[f"b_{s}"]).abs().mean()), 2)
            months.append(row)
    top = []
    for pid, rec in corr.items():
        if "pts" in rec:
            top.append({"player": names.get(pid, pid), **{s: rec[s]["adj"] for s in rec}, "n": rec["pts"]["n"]})
    top.sort(key=lambda x: -abs(x.get("pts", 0)))
    eff = [{"tag": t, "label": TAGS[t][0], "n": v["pts"]["n"], "applied": v["pts"]["n"] >= TAG_MIN_N,
            **{s: v[s]["avg"] for s in ("min", "pts", "reb", "ast")}} for t, v in effects.items()]
    eff.sort(key=lambda x: -x["n"])
    teams = sorted([{"team": abbr.get(t, t), **v} for t, v in tcorr.items()], key=lambda x: -abs(x["margin"]))
    recent = []
    if not news.empty:
        tagged = news[news["tags"].astype(str) != ""].sort_values("published", ascending=False).head(25)
        recent = [{"date": r.published.strftime("%Y-%m-%d"), "headline": r.headline, "link": r.link,
                   "tags": [TAGS[t][0] for t in str(r.tags).split(";") if t in TAGS]} for r in tagged.itertuples()]
    return {"months": months, "players": top[:20], "tags": eff, "teams": teams[:10], "news": recent,
            "settings": {"player_k": PLAYER_K, "tag_min_n": TAG_MIN_N}}
