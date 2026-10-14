"""Shared settings and paths."""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NBAPRED_DATA_DIR", ROOT / "data"))
BOX_DIR = DATA_DIR / "boxscores"      # one append-only CSV per season
GAMES_DIR = DATA_DIR / "games"        # one CSV per season (schedule + results)
LOG_DIR = DATA_DIR / "logs"           # prediction + pick logs
SITE_DIR = Path(os.environ.get("NBAPRED_SITE_DIR", ROOT / "docs"))
SITE_DATA = SITE_DIR / "data"

ET = ZoneInfo("America/New_York")     # NBA schedule dates are US Eastern

# ESPN labels a season by the year it ends: 2025-26 -> 2026.
HISTORY_SEASONS = [2024, 2025, 2026]
SEASON_TYPES = {2: "regular", 3: "playoffs"}  # skip preseason (1) / all-star

# Prop markets we model. PRA is derived from the other three.
PROP_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk"]
ALL_TARGETS = PROP_STATS + ["min"]

# Odds API (https://the-odds-api.com). Free tier = 500 credits / month.
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_REGIONS = os.environ.get("ODDS_REGIONS", "us")
# Named bookmakers (max 10 = cost of one region). Pinnacle is the sharpest book, so it anchors fair prices;
# Betway (UK) is included so its prices show next to each pick.
BOOKMAKERS = os.environ.get("BOOKMAKERS", "pinnacle,betway,draftkings,fanduel,betmgm,williamhill_us,"
                                          "betrivers,bovada,betonlineag,lowvig")
# Player props cost ~1 credit per market per event. Keep this small on the free tier.
MAX_PROP_EVENTS = int(os.environ.get("MAX_PROP_EVENTS", "2"))
FETCH_PROPS = os.environ.get("FETCH_PROPS", "0") == "1"

# A pick is flagged when model probability beats the no-vig market probability by this much.
EDGE_THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "0.05"))
# Bigger disagreements than this usually mean the model is missing news (injury, role change),
# so they are shown as "check the news" instead of being flagged.
MAX_EDGE = float(os.environ.get("MAX_EDGE", "0.20"))

for d in (BOX_DIR, GAMES_DIR, LOG_DIR, SITE_DATA):
    d.mkdir(parents=True, exist_ok=True)
