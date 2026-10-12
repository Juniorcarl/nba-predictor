"""Sportsbook lines from The Odds API (https://the-odds-api.com).

Free tier: 500 credits / month. Cost = markets x regions per call, where every
10 named bookmakers count as one region.
  - game lines (h2h, spreads, totals): 3 credits per call, all games at once
  - player props: 1 credit per market per game
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict

import requests

from .config import BOOKMAKERS, MAX_PROP_EVENTS, ODDS_API_KEY, ODDS_REGIONS

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
PROP_MARKETS = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
                "player_threes": "fg3m", "player_points_rebounds_assists": "pra"}

credits = {"remaining": None, "used": None}


def _get(url, params):
    params = {**params, "apiKey": ODDS_API_KEY}
    r = requests.get(url, params=params, timeout=25)
    credits["remaining"] = r.headers.get("x-requests-remaining", credits["remaining"])
    credits["used"] = r.headers.get("x-requests-used", credits["used"])
    if r.status_code != 200:
        raise RuntimeError(f"Odds API {r.status_code}: {r.text[:200]}")
    return r.json()


def _book_params():
    # Up to 10 named bookmakers cost the same as one region.
    return {"bookmakers": BOOKMAKERS} if BOOKMAKERS else {"regions": ODDS_REGIONS}


def fetch_game_odds() -> list[dict]:
    if not ODDS_API_KEY:
        return []
    return _get(f"{BASE}/odds", {**_book_params(), "markets": "h2h,spreads,totals", "oddsFormat": "american"})


def fetch_prop_odds(event_id: str) -> dict:
    return _get(f"{BASE}/events/{event_id}/odds",
                {**_book_params(), "markets": ",".join(PROP_MARKETS), "oddsFormat": "american"})


# ---------------------------------------------------------------- math
def implied(price: float) -> float:
    price = float(price)
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def payout(price: float) -> float:
    """Profit per 1 unit staked."""
    price = float(price)
    return price / 100 if price > 0 else 100 / -price


def no_vig(p_a: float, p_b: float) -> tuple[float, float]:
    s = p_a + p_b
    return p_a / s, p_b / s


# ---------------------------------------------------------------- matching
def nickname(team_name: str) -> str:
    """'LA Clippers' and 'Los Angeles Clippers' -> 'clippers'."""
    return re.sub(r"[^a-z0-9]", "", str(team_name).lower().split()[-1])


def norm_player(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\.?$", "", s.strip())
    return re.sub(r"[^a-z]", "", s)


def match_events(odds_events: list[dict], games) -> dict:
    """ESPN game_id -> odds event."""
    key = {(nickname(g.home_name), nickname(g.away_name)): g.game_id for g in games.itertuples()}
    out = {}
    for ev in odds_events:
        gid = key.get((nickname(ev.get("home_team")), nickname(ev.get("away_team"))))
        if gid:
            out[gid] = ev
    return out


# ---------------------------------------------------------------- consensus lines
def _consensus_two_way(entries):
    """entries: list of (point, side, price, book). Returns consensus line for the most common point."""
    if not entries:
        return None
    pts = Counter(e[0] for e in entries)
    point = pts.most_common(1)[0][0]
    by_book = defaultdict(dict)
    for p, side, price, book in entries:
        if p == point:
            by_book[book][side] = price
    sides = sorted({s for b in by_book.values() for s in b})
    if len(sides) != 2:
        return None
    a, b = sides
    fair_a = []
    best = {a: None, b: None}
    best_book = {a: None, b: None}
    for book, sp in by_book.items():
        if a in sp and b in sp:
            fair_a.append(no_vig(implied(sp[a]), implied(sp[b]))[0])
        for s in (a, b):
            if s in sp and (best[s] is None or payout(sp[s]) > payout(best[s])):
                best[s], best_book[s] = sp[s], book
    if not fair_a:
        return None
    fa = sum(fair_a) / len(fair_a)
    out = {"point": point, "sides": {a: {"fair": fa, "price": best[a], "book": best_book[a]},
                                     b: {"fair": 1 - fa, "price": best[b], "book": best_book[b]}},
           "books": len(fair_a)}
    bw = _betway(entries, (a, b))
    if bw:
        out["betway"] = bw
    return out


def _betway(entries, sides):
    """Betway's own line: {'point': x, a: price, b: price} at its main (most common) point."""
    mine = [e for e in entries if "betway" in str(e[3]).lower()]
    if not mine:
        return None
    by_pt = defaultdict(dict)
    for p, side, price, _ in mine:
        by_pt[p][side] = price
    full = [p for p, d in by_pt.items() if all(s in d for s in sides)]
    if not full:
        return None
    pt = Counter(e[0] for e in mine if e[0] in full).most_common(1)[0][0]
    return {"point": pt, **by_pt[pt]}


def game_lines(ev: dict) -> dict:
    """Consensus moneyline, spread (home perspective) and total."""
    home, away = ev["home_team"], ev["away_team"]
    ml, spread, total = [], [], []
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            for o in m.get("outcomes", []):
                if m["key"] == "h2h":
                    ml.append((0, "home" if o["name"] == home else "away", o["price"], bk.get("title") or bk["key"]))
                elif m["key"] == "spreads":
                    side = "home" if o["name"] == home else "away"
                    hp = o["point"] if side == "home" else -o["point"]  # store as home spread
                    spread.append((hp, side, o["price"], bk.get("title") or bk["key"]))
                elif m["key"] == "totals":
                    total.append((o["point"], o["name"].lower(), o["price"], bk.get("title") or bk["key"]))
    return {"ml": _consensus_two_way(ml), "spread": _consensus_two_way(spread), "total": _consensus_two_way(total)}


def prop_lines(ev_odds: dict) -> dict:
    """{(normalized player, stat): consensus line}"""
    raw = defaultdict(list)
    names = {}
    for bk in ev_odds.get("bookmakers", []):
        for m in bk.get("markets", []):
            stat = PROP_MARKETS.get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                who = o.get("description")
                if not who or o.get("point") is None:
                    continue
                k = (norm_player(who), stat)
                names[k] = who
                raw[k].append((o["point"], o["name"].lower(), o["price"], bk.get("title") or bk["key"]))
    out = {}
    for k, entries in raw.items():
        c = _consensus_two_way(entries)
        if c:
            c["name"] = names[k]
            out[k] = c
    return out


def fetch_all(games):
    """Return ({game_id: game lines}, {game_id: prop lines}) for upcoming ESPN games."""
    if not ODDS_API_KEY or games.empty:
        log.info("No ODDS_API_KEY set - skipping sportsbook lines")
        return {}, {}
    try:
        events = fetch_game_odds()
    except Exception as e:  # noqa: BLE001
        log.error("game odds failed: %s", e)
        return {}, {}
    matched = match_events(events, games)
    glines = {gid: game_lines(ev) for gid, ev in matched.items()}
    plines = {}
    from .config import FETCH_PROPS
    if FETCH_PROPS:
        order = games.sort_values("start_utc")["game_id"]
        for gid in [g for g in order if g in matched][:MAX_PROP_EVENTS]:
            try:
                plines[gid] = prop_lines(fetch_prop_odds(matched[gid]["id"]))
            except Exception as e:  # noqa: BLE001
                log.error("props for %s failed: %s", gid, e)
    log.info("odds: %d games matched, props for %d, credits remaining %s", len(glines), len(plines), credits["remaining"])
    return glines, plines
