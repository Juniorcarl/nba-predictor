"""Daily pipeline entry point.

  python scripts/run.py backfill          # first time: pull ~3 past seasons (30-60 min)
  python scripts/run.py daily             # update data, retrain, predict, fetch odds, build site data
  python scripts/run.py daily --no-odds   # same, without spending Odds API credits (injury refresh)
  python scripts/run.py daily --offline   # skip all network calls (demo / local testing)
"""
import argparse
import json
import os
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred import adapt, espn, explain, learning, models, odds, picks, risky, site, store  # noqa: E402
from nbapred.config import EDGE_THRESHOLD, HISTORY_SEASONS, LOG_DIR, MAX_EDGE, SEASON_TYPES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run")


def backfill(max_games=None):
    today = store.today_et()
    seasons = sorted(set(HISTORY_SEASONS + [store.season_for(today)]))
    for s in seasons:
        start, end = store.season_window(s)
        have = store.games_path(s).exists()
        end = min(end, today + timedelta(days=7))
        if start > end:
            continue
        if have and s != store.season_for(today):
            log.info("season %s schedule already stored", s)
            continue
        log.info("schedule %s: %s -> %s", s, start, end)
        store.update_schedule(start, end)
    n = store.update_boxscores(max_games=max_games)
    log.info("fetched %d box scores", n)


def next_slate(games: pd.DataFrame, today):
    """Earliest date from today on with unfinished games.

    Returns (date, games not yet started, ids of games on that date already under way).
    """
    fut = games[(games["date"] >= today) & (~games["completed"].astype(bool))
                & games["season_type"].isin(list(SEASON_TYPES))]  # no preseason / All-Star
    if fut.empty:
        return None, fut, set()
    d = fut["date"].min()
    day = fut[fut["date"] == d]
    pre = day[day["state"].fillna("pre") == "pre"]
    return d, pre, set(day["game_id"]) - set(pre["game_id"])


def keep_started(slate: dict, started: set) -> dict:
    """Games that tipped off since the last run keep their pre-game predictions on the site."""
    prev_path = site.SITE_DATA / "slate.json"
    if not started or not prev_path.exists():
        return slate
    try:
        prev = json.loads(prev_path.read_text())
    except ValueError:
        return slate
    if prev.get("date") != slate["date"]:
        return slate
    kept = [{**g, "started": True} for g in prev.get("games", []) if g["id"] in started]
    slate["games"] = sorted(slate["games"] + kept, key=lambda g: g.get("start") or "")
    return slate


ODDS_CACHE = LOG_DIR / "odds_cache.json"


def save_odds(slate_date, glines, plines):
    ODDS_CACHE.write_text(json.dumps({"date": str(slate_date), "glines": glines,
                                      "plines": {g: {f"{k[0]}|{k[1]}": v for k, v in d.items()} for g, d in plines.items()}}))


def load_odds(slate_date):
    if not ODDS_CACHE.exists():
        return {}, {}
    c = json.loads(ODDS_CACHE.read_text())
    if c.get("date") != str(slate_date):
        return {}, {}
    plines = {g: {tuple(k.split("|", 1)): v for k, v in d.items()} for g, d in c["plines"].items()}
    log.info("reusing cached lines for %d games", len(c["glines"]))
    return c["glines"], plines


def daily(use_odds=True, offline=False, demo_odds=False, learn=False):
    today = store.today_et()
    if not offline:
        if store.load_box().empty:
            log.info("No history stored yet - running the one-time backfill first")
            backfill()
        store.update_schedule(today - timedelta(days=3), today + timedelta(days=7))
        store.update_boxscores()
    games, box = store.load_games(), store.load_box()
    if box.empty:
        log.error("No box scores stored yet - run `backfill` first.")
        sys.exit(1)

    n = picks.grade(games, box)
    log.info("graded %d picks, %d bold calls", n, risky.grade_bold(games, box))

    # learning loop, step 1: review finished slates
    L = learning.load()
    projections = pd.read_csv(picks.PROJ, dtype={"game_id": str, "player_id": str}) if picks.PROJ.exists() else pd.DataFrame()
    reviewed = learning.postmortem(L, projections, box, games)
    if reviewed:
        log.info("post-mortem written for %s", ", ".join(reviewed))

    slate_date, upcoming, started = next_slate(games, today)
    injuries = []
    if not offline:
        try:
            injuries = espn.parse_injuries(espn.fetch_injuries())
            log.info("injury report: %d entries", len(injuries))
        except RuntimeError as e:
            log.error("injuries failed: %s", e)

    rosters = {}
    if not offline:  # also keeps the Players page on current teams between slates
        recent = games[games["season"] == games["season"].max()]
        bio_rows = []
        rosters = espn.fetch_rosters(set(recent["home_id"]) | set(recent["away_id"]), bio_rows)
        store.update_bio(bio_rows)
        log.info("current rosters for %d teams, bio for %d players", len(rosters), len(bio_rows))

    hist_games = games[games["completed"].astype(bool)]
    PF, G = models.assemble(box, hist_games, upcoming, injuries, rosters)
    cfg = models.load_config()
    pm, pinfo = models.train_player_models(PF, cfg)
    gm, ginfo = models.train_game_models(G, cfg)
    valid = pinfo.pop("_valid")
    tails = pinfo.get("_tails", {"vol": {}, "tails": {}, "check": {}})

    # learning loop, step 2: where is the model consistently wrong?
    pats = learning.find_patterns(valid, cfg["groups"], "test games")
    live = learning.live_frame(projections, PF)
    if len(live) >= 300:
        pats = learning.find_patterns(live, cfg["groups"], "this season") + pats
    L["patterns"] = {"updated": learning.now(), "items": pats[:15]}

    # learning loop, step 3 (weekly, or on request): test possible changes and propose the ones that help
    if learn or not L["experiments"]:
        learning.run_experiments(L, PF, G)
    learning.save(L)
    site._write("learning.json", learning.site_view(L, cfg))
    pp = models.predict_players(pm, pinfo, PF)
    gp = models.predict_games(gm, ginfo, G)
    # plain-English reasons behind every number
    abbr = dict(zip(games["home_id"], games["home_abbr"])) | dict(zip(games["away_id"], games["away_abbr"]))
    if len(pp):
        pp["why"] = explain.explain_players(pm, pinfo, pp, abbr)
    if len(gp):
        gp["why"] = explain.explain_games(gm, ginfo, gp)

    # day-to-day learning: news tags + corrections learned from how recent projections turned out
    news = adapt.update_news() if not offline else adapt.load_news()
    ptags, heads = adapt.player_news(news)
    joined = adapt._joined(projections, box)
    pcorr, teffects = adapt.player_corrections(joined), adapt.tag_effects(joined)
    tcorr = adapt.team_corrections(games)
    if len(pp):
        inj_notes = dict(zip(pp["player_id"].astype(str), pp["inj_comment"].fillna("").astype(str)))
        pp = adapt.apply_players(pp, pcorr, teffects, ptags, adapt.injury_tags(inj_notes))
        pp["news"] = pp["player_id"].astype(str).map(heads)
    if len(gp):
        gp = adapt.apply_games(gp, tcorr, ginfo["margin"]["sd"])
    log.info("learning: %d player corrections, %d news tags measured, %d team corrections, %d players in the news",
             len(pcorr), len(teffects), len(tcorr), len(ptags))

    glines, plines = ({}, {})
    if use_odds and not offline and len(gp):
        glines, plines = odds.fetch_all(gp)
        if glines:
            save_odds(slate_date, glines, plines)
    elif not offline and len(gp):
        glines, plines = load_odds(slate_date)  # reuse this morning's lines (no credits spent)
    elif demo_odds and len(gp):  # synthetic lines for local UI testing only
        from make_demo_data import fake_odds
        evs, props = fake_odds(gp, pp)
        matched = odds.match_events(evs, gp)
        glines = {gid: odds.game_lines(ev) for gid, ev in matched.items()}
        plines = {gid: odds.prop_lines(p) for gid, p in props.items()}

    slate, rows = site.build_slate(slate_date, gp, pp, glines, plines, ginfo, pinfo)
    slate = keep_started(slate, started)
    if slate_date is not None:
        picks.log_picks(slate_date, rows)
        picks.log_projections(slate_date, pp)
        adapt.log_games(slate_date, gp)
    track = picks.track_record(box)
    players = site.players_index(box, games, pp, rosters)

    # bold page: calibrated big-night calls, upsets and same-game multis
    try:
        corr = risky.fit_corr(valid, pinfo["var"], games)
    except Exception as e:  # noqa: BLE001
        log.warning("correlation fit failed: %s", e)
        corr = None
    bold = risky.build(slate_date, pp, gp, glines, pinfo, tails, corr, players, abbr)
    bold["record"] = risky.bold_record()
    site._write("bold.json", bold)
    pinfo.pop("_tails", None)
    log.info("bold: %d calls, %d upsets, %d multis", len(bold["calls"]), len(bold["upsets"]), len(bold["multis"]))
    names = {p["id"]: p["name"] for p in players}
    site._write("adapt.json", adapt.summary(joined, pcorr, teffects, tcorr, news, names, abbr))
    site.write_all(slate, pinfo, ginfo, track, players, {
        "slate_date": str(slate_date) if slate_date else None, "edge_threshold": EDGE_THRESHOLD, "max_edge": MAX_EDGE, "odds_credits_remaining": odds.credits["remaining"],
        "injuries_count": len(injuries), "rosters_loaded": len(rosters), "has_odds": bool(glines), "has_props": bool(plines),
        "games_in_db": int(len(hist_games)), "repo": os.environ.get("GITHUB_REPOSITORY", "Juniorcarl/nba-predictor"), "player_games_in_db": int((box["min"] > 0).sum()),
    })
    log.info("slate %s: %d games, %d player projections, %d flagged picks",
             slate_date, len(gp), len(pp), sum(1 for r in rows if r.get("flagged")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["backfill", "daily"])
    ap.add_argument("--no-odds", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--demo-odds", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--learn", action="store_true", help="run the weekly experiments now")
    a = ap.parse_args()
    if a.mode == "backfill":
        backfill(a.max_games)
    else:
        daily(use_odds=not a.no_odds, offline=a.offline, demo_odds=a.demo_odds, learn=a.learn)
