# Project Ticker — data robot

A free data robot for the **market-move-explainer** Claude skill.

Every weekday at 6:17 am Singapore time, GitHub Actions runs `daily_data.py`.
It downloads a year of prices for every asset in `watchlist.csv`, computes
each day's move, how unusual it was (sigma) and the market / sector /
specific split, then saves the results in `data/`:

| File | What it holds |
| --- | --- |
| `data/latest.json` | The most recent US session |
| `data/YYYY-MM-DD.json` | One file per session (the archive) |
| `data/index.json` | The list of sessions available |

Claude reads these files from the raw URL, e.g.
`https://raw.githubusercontent.com/Ansel-lau/project-ticker/main/data/latest.json`.

## Sources (all free)
- Prices: Yahoo Finance via `yfinance` (unofficial; update the library if it breaks)
- Rates: US Treasury daily yield curves and the New York Fed (FRED as a fallback).
  The 2-year yield uses the Treasury curve (`fred:DGS2`), not Yahoo's `2YY=F`,
  which is rarely updated.
- Headlines and earnings dates: Finnhub (free key, stored as the
  `FINNHUB_API_KEY` repository secret, never in a file)

## Heatmap video
After the data, `heatmap_video.py` makes a 12-second 16:9 video of one US session
of the S&P 500: it starts at the previous close (every tile grey), then replays
the day in one-hour steps from the 9:30 open to the 4 pm close, with that day's
15-minute chart and the change in total market value. Tiles are sized by
market value (shares outstanding x previous close, refreshed weekly into
`data/sp500_meta.csv`) and grouped by GICS sector. Videos are published to the
`heatmap-videos` branch, which is rewritten each day and keeps the last 30
sessions, so the main branch never grows. Each video is under 4.5 MB so the
digest can put it into Notion:
- one day: `https://raw.githubusercontent.com/Ansel-lau/project-ticker/heatmap-videos/heatmap-YYYY-MM-DD.mp4`
  (a `.png` still of the close sits beside it)
- newest: `.../heatmap-videos/heatmap-latest.mp4`
- test without real prices: `python heatmap_video.py --demo`

## Common jobs
- **Add an asset:** add a row to `watchlist.csv` (Yahoo symbol in `symbol`),
  and a row in the Notion Watchlist.
- **Rebuild a past day:** Actions > Project Ticker daily > Run workflow, type the date.
- **Fill history:** Run workflow with `backfill` = 20.
- **Test on your Mac:** `pip install -r requirements.txt` then `python daily_data.py`.

The method is explained in the skill's `reference/attribution-method.md`.
