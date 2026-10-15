import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NBAPRED_DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("NBAPRED_SITE_DIR", tempfile.mkdtemp())

import pandas as pd  # noqa: E402

from nbapred import espn, models, odds, picks  # noqa: E402

FX = Path(__file__).parent / "fixtures"
load = lambda n: json.loads((FX / n).read_text())  # noqa: E731


def test_scoreboard():
    g = espn.parse_scoreboard(load("scoreboard.json"))
    assert len(g) == 2
    a, b = g
    assert a["completed"] and a["home_abbr"] == "LAL" and a["home_score"] == 112 and a["away_score"] == 108
    assert a["season"] == 2026 and a["season_type"] == 2
    assert not b["completed"] and b["home_score"] is None and b["state"] == "pre"


def test_boxscore():
    rows = espn.parse_boxscore(load("summary.json"), "401810001")
    by = {r["player"]: r for r in rows}
    t = by["Jayson Tatum"]
    assert t["min"] == 38 and t["pts"] == 31 and t["reb"] == 9 and t["ast"] == 6 and t["fg3m"] == 4 and t["fga"] == 22
    assert abs(by["Jaylen Brown"]["min"] - 36.5) < 1e-9
    assert by["Bench Guy"]["dnp"] and by["Bench Guy"]["min"] == 0
    assert by["LeBron James"]["ast"] == 9 and by["LeBron James"]["team_id"] == "13"


def test_injuries():
    inj = espn.parse_injuries(load("injuries.json"))
    assert inj[0]["player_id"] == "3917376" and inj[0]["status"] == "Out"
    assert inj[1]["player_id"] == "4065648"


def test_odds_math_and_lines():
    assert abs(odds.implied(-110) - 0.5238) < 1e-3 and abs(odds.payout(150) - 1.5) < 1e-9
    assert odds.nickname("LA Clippers") == odds.nickname("Los Angeles Clippers")
    assert odds.norm_player("Jaren Jackson Jr.") == odds.norm_player("Jaren Jackson")
    ev = {"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics", "bookmakers": [
        {"key": "dk", "title": "DraftKings", "markets": [
            {"key": "spreads", "outcomes": [{"name": "Los Angeles Lakers", "price": -110, "point": 3.5},
                                            {"name": "Boston Celtics", "price": -110, "point": -3.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -105, "point": 225.5},
                                           {"name": "Under", "price": -115, "point": 225.5}]}]}]}
    gl = odds.game_lines(ev)
    assert gl["spread"]["point"] == 3.5  # home spread
    assert gl["total"]["sides"]["over"]["fair"] < 0.5 and gl["total"]["sides"]["over"]["book"] == "DraftKings"


def test_count_probs_sum_to_one():
    p = models.count_probs(24.0, 40.0, 24.5)
    assert abs(p["over"] + p["under"] - 1) < 1e-9 and 0.4 < p["over"] < 0.6
    p = models.count_probs(5.0, 8.0, 5)
    assert abs(p["over"] + p["under"] + p["push"] - 1) < 1e-9 and p["push"] > 0.1


def test_grading(tmp_path, monkeypatch):
    monkeypatch.setattr(picks, "PICKS", tmp_path / "picks.csv")
    games = pd.DataFrame([{"game_id": "1", "completed": True, "home_score": 110, "away_score": 104}])
    box = pd.DataFrame([{"game_id": "1", "player_id": "7", "dnp": False, "min": 30, "pts": 25, "reb": 8, "ast": 3, "fg3m": 2},
                        {"game_id": "1", "player_id": "8", "dnp": True, "min": 0, "pts": 0, "reb": 0, "ast": 0, "fg3m": 0}])
    base = {"matchup": "BOS @ LAL", "book": "x", "model_prob": .6, "fair_prob": .5, "edge": .1, "ev": .1, "flagged": True}
    rows = [
        {**base, "game_id": "1", "market": "spread", "side": "home", "line": -5.5, "price": -110},   # 6 > 5.5 win
        {**base, "game_id": "1", "market": "spread", "side": "away", "line": 5.5, "price": -110},    # loss
        {**base, "game_id": "1", "market": "spread", "side": "home", "line": -6.0, "price": -110},   # push
        {**base, "game_id": "1", "market": "total", "side": "under", "line": 220.5, "price": -110},  # 214 win
        {**base, "game_id": "1", "market": "moneyline", "side": "away", "line": None, "price": 150},  # loss
        {**base, "game_id": "1", "market": "pts", "player_id": "7", "side": "over", "line": 24.5, "price": 100},  # win
        {**base, "game_id": "1", "market": "pra", "player_id": "7", "side": "under", "line": 35.5, "price": -120},  # 36 loss
        {**base, "game_id": "1", "market": "pts", "player_id": "8", "side": "over", "line": 10.5, "price": -110},  # void
    ]
    picks.log_picks("2026-01-15", rows)
    assert picks.grade(games, box) == 8
    df = pd.read_csv(picks.PICKS)
    assert list(df["result"]) == ["win", "loss", "push", "win", "loss", "win", "loss", "void"]
    assert abs(df["profit"].iloc[0] - 100 / 110) < 1e-9 and df["profit"].iloc[5] == 1.0


def test_betway_line_extracted():
    ev = {"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics", "bookmakers": [
        {"key": "pinnacle", "title": "Pinnacle", "markets": [
            {"key": "totals", "outcomes": [{"name": "Over", "price": -105, "point": 225.5},
                                           {"name": "Under", "price": -105, "point": 225.5}]}]},
        {"key": "betway", "title": "Betway", "markets": [
            {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 226.5},
                                           {"name": "Under", "price": -110, "point": 226.5}]}]},
        {"key": "fanduel", "title": "FanDuel", "markets": [
            {"key": "totals", "outcomes": [{"name": "Over", "price": -112, "point": 225.5},
                                           {"name": "Under", "price": -108, "point": 225.5}]}]}]}
    t = odds.game_lines(ev)["total"]
    assert t["point"] == 225.5 and t["betway"] == {"point": 226.5, "over": -110, "under": -110}


def test_roster_parse_flat_and_grouped():
    flat = espn.parse_roster(load("roster.json"))
    assert [p["player_id"] for p in flat] == ["3917376", "5105565"] and flat[0]["pos"] == "SG"
    grouped = espn.parse_roster({"athletes": [{"position": "guards", "items": [{"id": "1", "displayName": "A"}]},
                                              {"position": "forwards", "items": [{"id": "2", "fullName": "B"}]}]})
    assert [p["player"] for p in grouped] == ["A", "B"]


def test_offseason_trade_uses_current_roster():
    """A player whose last game was for team 1 but is now on team 2's roster is projected for team 2."""
    from nbapred import features as F
    st = F.RosterState(team_idx={"1": 82, "2": 82}, players={
        "p1": {"min": 30.0, "fga": 15.0, "team": "1", "last_idx": 82, "player": "Traded Star", "pos": "G",
               "last_date": pd.Timestamp("2026-04-12")},
        "p2": {"min": 25.0, "fga": 10.0, "team": "2", "last_idx": 82, "player": "Stays Put", "pos": "F",
               "last_date": pd.Timestamp("2026-04-12")}})
    up = pd.DataFrame([{"game_id": "g", "date": pd.Timestamp("2026-10-21").date(), "home_id": "2", "away_id": "1"}])
    cand, _ = F.upcoming_candidates(st, up, [], rosters={"2": ["p1", "p2"], "1": ["newcomer"]})
    assert set(cand[cand.team_id == "2"]["player_id"]) == {"p1", "p2"}
    assert cand[cand.team_id == "1"].empty
    cand2, _ = F.upcoming_candidates(st, up, [], rosters=None)  # fallback: last team played for
    assert set(cand2[cand2.team_id == "1"]["player_id"]) == {"p1"}


def test_postmortem_reasons(tmp_path, monkeypatch):
    from nbapred import learning
    monkeypatch.setattr(learning, "LEARN", tmp_path / "learning.json")
    games = pd.DataFrame([{"game_id": "1", "completed": True, "home_score": 130, "away_score": 100, "home_id": "A", "away_id": "B"}])
    base = {"game_id": "1", "team_id": "A", "fg3m": 0, "reb": 5, "ast": 3}
    box = pd.DataFrame([
        {**base, "player_id": "blow", "player": "Blow", "min": 20, "pts": 8, "fgm": 3, "fga": 7, "dnp": False},
        {**base, "player_id": "hot", "player": "Hot", "min": 30, "pts": 30, "fgm": 12, "fga": 16, "dnp": False},
        {**base, "player_id": "norm", "player": "Norm", "min": 30, "pts": 15, "fgm": 6, "fga": 13, "dnp": False}])
    pr = pd.DataFrame([
        {"date": "2026-11-01", "game_id": "1", "player_id": pid, "player": pid, "team_id": "A",
         "p_min": pm, "p_pts": pp, "p_reb": 5, "p_ast": 3, "p_fg3m": 0, "p_pra": pp + 8}
        for pid, pm, pp in [("blow", 34, 18), ("hot", 30, 18), ("norm", 30, 15), ("ghost", 25, 10)]])
    d = learning.load()
    assert learning.postmortem(d, pr, box, games) == ["2026-11-01"]
    rep = d["postmortems"][0]
    why = {b["player"]: b["reason"] for b in rep["biggest"]}
    assert why["blow"] == "Blowout" and why["hot"] == "Hot shooting" and why["ghost"] == "Did not play"
    assert learning.postmortem(d, pr, box, games) == []  # not reviewed twice


def test_proposal_approve_and_undo(tmp_path, monkeypatch):
    from nbapred import learning, models
    monkeypatch.setattr(learning, "LEARN", tmp_path / "learning.json")
    monkeypatch.setattr(models, "config_path", lambda: tmp_path / "cfg.json")
    d = learning.load()
    p = learning._make_proposal(d, {"kind": "add_group", "target": "fatigue_travel", "value": None, "gain": 0.004,
                                    "per": {}, "halves": [0.003, 0.005], "passed": True})
    assert p["id"] == "P-0001" and p["status"] == "pending"
    assert "Approved" in learning.apply(d, "approve", "P-0001")
    assert models.load_config()["groups"] == ["fatigue_travel"]
    assert "already" in learning.apply(d, "approve", "P-0001")
    assert "Undid" in learning.apply(d, "undo", "P-0001")
    assert models.load_config()["groups"] == []


def test_reasons_add_up():
    """The starting point plus every reason equals the prediction."""
    import numpy as np
    from xgboost import XGBRegressor
    from nbapred import explain
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"pts_e10": rng.normal(15, 5, 400), "miss_min": rng.uniform(0, 60, 400), "home": rng.integers(0, 2, 400),
                      "o_t_def_rel": rng.normal(1, .05, 400), "opp_id": "2"})
    y = X["pts_e10"] + 0.05 * X["miss_min"] + X["home"] + rng.normal(0, 2, 400)
    feats = ["pts_e10", "miss_min", "home", "o_t_def_rel"]
    m = XGBRegressor(n_estimators=50, max_depth=3).fit(X[feats], y)
    up = X.head(3).copy()
    for t in ("pts", "reb", "ast", "min", "fg3m"):
        up[f"p_{t}"] = m.predict(up[feats])
    up["p_pra"] = up["p_pts"] * 3
    ex = explain.explain_players({t: m for t in ("pts", "reb", "ast", "min", "fg3m")}, {"features": feats}, up, {"2": "BOS"})
    w = ex[0]["pts"]
    assert abs(w["level"] + sum(i["amt"] for i in w["items"]) + w["other"] - w["pred"]) < 0.11
    assert {i["cat"] for i in w["items"]} <= {"teammates", "home", "opp_defense"}


def test_joint_probability_and_kelly():
    from nbapred import risky
    S = risky.CORR_STATS
    zero = [[1.0 if i == j else 0.0 for j in range(len(S))] for i in range(len(S))]
    none = [[0.0] * len(S) for _ in S]
    corr = {"stats": S, "same": zero, "mate": none, "opp": none, "win": {s: 0.0 for s in S}}
    a = {"kind": "player", "player_id": "1", "team_id": "A", "stat": "pts", "p": 0.5}
    b = {"kind": "player", "player_id": "2", "team_id": "A", "stat": "pts", "p": 0.4}
    assert abs(risky.joint_prob([a, b], corr) - 0.2) < 0.01           # independent: multiply
    same = [[1.0] * len(S) for _ in S]
    corr2 = {**corr, "same": same}
    c = {**a, "player_id": "1", "stat": "pra", "p": 0.4}
    assert abs(risky.joint_prob([a, c], corr2) - 0.4) < 0.01          # perfectly linked: the harder leg
    assert risky.kelly(0.25, 3.0) == 0.0                                # no value at fair-ish odds
    assert 0 < risky.kelly(0.25, 6.0) < 0.25 * 0.25 + 1e-9


def test_news_tags():
    from nbapred import adapt
    js = {"articles": [
        {"id": 1, "headline": "Embiid to play under minutes restriction in season debut", "description": "",
         "published": "2026-10-20T15:00Z", "categories": [{"type": "athlete", "athleteId": 3059318}, {"type": "team", "teamId": 20}]},
        {"id": 2, "headline": "Bucks sign veteran guard", "description": "Milwaukee signed a deal.", "published": "2026-10-20T15:00Z",
         "categories": [{"type": "athlete", "athlete": {"id": 77}}]},
        {"id": 3, "headline": "Celtics coach fired", "description": "Interim coach named.", "published": "2026-10-20T15:00Z", "categories": []}]}
    rows = adapt.parse_news(js)
    assert set(rows[0]["tags"].split(";")) >= {"minutes_limit", "returning"} and rows[0]["athletes"] == "3059318"
    assert rows[2]["team_tags"] == "coach_change"
    assert "trade" not in adapt.tag_text("Scored 30 in a win")  # no accidental matches


def test_corrections_learn_from_base_not_final():
    from nbapred import adapt
    # player beat his base projection by +6 in each of 8 games; earlier corrections were already applied
    proj = pd.DataFrame([{"date": f"2026-11-{d:02d}", "game_id": str(d), "player_id": "7", "p_pts": 20 + 3, "adj_pts": 3,
                          "p_min": 30, "p_reb": 5, "p_ast": 3, "tags": ""} for d in range(1, 9)])
    box = pd.DataFrame([{"game_id": str(d), "player_id": "7", "min": 31, "pts": 26, "reb": 5, "ast": 3, "fg3m": 1, "stl": 1, "blk": 0}
                        for d in range(1, 9)])
    j = adapt._joined(proj, box)
    c = adapt.player_corrections(j)["7"]["pts"]
    assert c["avg"] == 6.0 and 2.0 < c["adj"] < 6.0   # measured vs base (20), shrunk toward 0
    pp = pd.DataFrame([{"player_id": "7", "p_pts": 20.0, "p_reb": 5.0, "p_ast": 3.0, "p_min": 30.0, "p_fg3m": 1.0,
                        "p_stl": 1.0, "p_blk": .3, "v_pts": 30.0, "inj_comment": ""}])
    out = adapt.apply_players(pp, {"7": {"pts": c}}, {}, {}, {})
    assert abs(out.loc[0, "p_pts"] - (20 + min(c["adj"], 0.25 * 20 + 0.3))) < 1e-6 and out.loc[0, "adj_pts"] > 0
