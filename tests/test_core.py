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
