# Courtside Edge: NBA predictor

Daily NBA projections for every player and game, compared with sportsbook lines.
Runs for free on GitHub: GitHub Actions updates the data and models every day, and GitHub Pages hosts the site.

## What it does

- **Player projections** for everyone in tonight's rotations: minutes, points, rebounds, assists, 3-pointers, and PRA.
  Each projection comes with a spread of likely outcomes, so you get a real over/under probability for any line.
- **Game predictions**: score, win chance, spread, and total.
- **Edges**: sportsbook lines from [The Odds API](https://the-odds-api.com) are converted to no-vig probabilities
  and compared with the model's. Picks are flagged when the model is at least 5 points more confident and the bet has positive expected value.
  Very large disagreements (over 20 points), questionable players, and players with little history are listed separately under "check the news first".
- **Line checker**: type any line and price on a player page to get the over/under chance, fair odds, and value.
- **My bets**: type in props or game lines straight from your Betway app (decimal odds like 1.85 work).
  The app strips out the bookmaker's margin and shows which side, if any, has value. Saved on your device.
- **Betway prices**: the odds feed includes Betway (UK) next to Pinnacle and the big US books, so each pick shows Betway's own line and its value.
  Betway Botswana prices can differ, which is what My bets is for.
- **Track record**: every flagged pick is logged before tip-off and graded after the game at the logged price.
  The page also shows test-set accuracy against a simple last-10-games average, and a calibration table.

## How it works

| Step | File |
| --- | --- |
| Schedule, box scores, and injury report from ESPN's public feed | `nbapred/espn.py`, `nbapred/store.py` |
| Features: recent form (weighted), season averages, per-minute rates, rest, home/away, opponent defence and pace, and minutes missing from injured teammates | `nbapred/features.py` |
| XGBoost models for each stat plus game margin/total, time-based validation, and a variance model for probabilities (negative binomial) | `nbapred/models.py` |
| Sportsbook lines, consensus no-vig probabilities | `nbapred/odds.py` |
| Edges, pick log, grading, track record | `nbapred/picks.py` |
| JSON for the site | `nbapred/site.py` |
| Static site | `docs/` |

Data is stored as plain CSV in `data/` (one file per season), so the repo is the database.

## Setup

1. **Enable Pages:** Settings → Pages → Source: **GitHub Actions**.
2. **Add the odds key (optional but recommended):** sign up for the free plan at the-odds-api.com, then go to
   Settings → Secrets and variables → Actions → New repository secret, named `ODDS_API_KEY`.
   Without it, everything still works except automatic edges. The line checker works without it.
3. **Run it once:** Actions → Daily update → Run workflow. The first run downloads three seasons of history
   (about 30 to 60 minutes). After that, it runs automatically twice a day:
   - 15:00 UTC (17:00 in Botswana): new results, retraining, fresh lines, picks
   - Every hour from 21:30 to 01:30 UTC (23:30 to 03:30 in Botswana): injury and late-scratch refresh before tip-offs,
     reusing the morning's lines so no extra credits are spent. Games that have already started keep their pre-game predictions.
4. Your site will be at `https://<username>.github.io/nba-predictor/`.

### Free-tier odds budget

The morning run uses about 13 credits: 3 for game lines plus 5 for props on each of 2 games.
That comes to about 400 of the free 500 credits a month. Optional repository **variables** (not secrets) adjust this:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FETCH_PROPS` | `1` | Set to `0` to skip player props |
| `MAX_PROP_EVENTS` | `2` | Games per day to fetch props for (5 credits each) |
| `EDGE_THRESHOLD` | `0.05` | Minimum probability edge to flag a pick |
| `BOOKMAKERS` | Pinnacle, Betway, 8 US books | Up to 10 bookmaker keys, comma-separated (10 cost the same as one region) |

## Run locally

```bash
pip install -r requirements.txt
python -m pytest -q tests
python scripts/run.py daily              # first run triggers the backfill
python -m http.server -d docs 8000       # open http://localhost:8000
```

Offline demo with synthetic data (no network):

```bash
NBAPRED_DATA_DIR=demo/data NBAPRED_SITE_DIR=demo/site python scripts/make_demo_data.py
NBAPRED_DATA_DIR=demo/data NBAPRED_SITE_DIR=demo/site python scripts/run.py daily --offline --demo-odds
cp docs/index.html docs/app.js docs/style.css demo/site/ && python -m http.server -d demo/site 8000
```

## Honest limits

- ESPN's feed is unofficial and could change without notice. If the daily run fails, the parsers in `nbapred/espn.py` are the first place to look.
- Injuries are the biggest source of error. The model accounts for teammates listed as out,
  but news after the last hourly run (01:30 UTC) won't be reflected.
- Players traded or signed recently are matched to their new team only after their first game there.
- Sportsbooks are very good at this. Treat flagged picks as a starting point, and trust the track record only after a few hundred graded picks.
