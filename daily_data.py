#!/usr/bin/env python3
"""
Daily data feed for the market-move-explainer skill.

Run after the US close (GitHub Actions does this every weekday). It writes
  data/latest.json          the most recent US session
  data/<YYYY-MM-DD>.json    an archive copy for that session
  data/index.json           the list of sessions available

For every asset in watchlist.csv it records the close, the move, how unusual
the move was (sigma), the market/sector/specific split (or NA with the
reason), peer moves and, for stocks, recent headlines and earnings dates.

Free sources only
  prices     Yahoo Finance via yfinance (no key)
  rates      US Treasury yield curves and New York Fed (no key), FRED as fallback
  headlines  Finnhub (free key in the FINNHUB_API_KEY environment variable;
             skipped if it is missing)

Usage
  python daily_data.py                       # latest US session
  python daily_data.py --date 2026-10-30     # one past session
  python daily_data.py --backfill 20         # the last 20 sessions
  python daily_data.py --prices-csv test.csv --no-network   # offline test
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from attribution import SIGMA_WINDOW, attribute, _naive

HERE = Path(__file__).resolve().parent
MARKET_CALENDAR = "SPY"          # a US session = a day SPY traded
RATE_SYMBOLS = {"^TNX", "^FVX", "^TYX", "^IRX", "2YY=F"}   # quoted as a yield in %

BACKDROP = [("S&P 500 (SPY)", "SPY"), ("Nasdaq 100 (QQQ)", "QQQ"), ("VIX", "^VIX"),
            ("US 10Y yield", "^TNX"), ("US Dollar Index", "DX-Y.NYB"),
            ("Brent", "BZ=F"), ("WTI", "CL=F"), ("Gold", "GC=F"), ("Bitcoin", "BTC-USD")]
SECTOR_ETFS = {"XLK": "Technology", "XLC": "Communication services",
               "XLY": "Consumer discretionary", "XLP": "Consumer staples",
               "XLF": "Financials", "XLV": "Health care", "XLI": "Industrials",
               "XLE": "Energy", "XLB": "Materials", "XLRE": "Real estate",
               "XLU": "Utilities"}
FRED_SERIES = {"DGS2": "US 2Y Treasury yield, %", "DGS10": "US 10Y Treasury yield, %",
               "DFII10": "US 10Y real (TIPS) yield, %",
               "DFEDTARU": "Fed funds target, upper bound, %",
               "DFEDTARL": "Fed funds target, lower bound, %"}
FINNHUB = "https://finnhub.io/api/v1"
FINNHUB_MIN_GAP = 1.1        # seconds between calls: stays under 60 calls/minute
FINNHUB_BUDGET = 300         # stop asking Finnhub after 5 minutes in total
T0 = time.time()


def log(msg: str) -> None:
    """Print with elapsed time, flushed at once so GitHub shows progress live."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------- helpers ---
def clean(x):
    """Make a value JSON-safe: NaN/inf -> None, numpy -> python, round floats."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def load_watchlist(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("ticker", "").strip() and not r["ticker"].startswith("#")]
    for r in rows:
        r["peers"] = (r.get("peers") or "").split()
        for k in ("symbol", "market", "sector", "class", "news"):
            r[k] = (r.get(k) or "").strip()
    return rows


def all_symbols(watch: list[dict]) -> list[str]:
    syms = [MARKET_CALENDAR] + [s for _, s in BACKDROP] + list(SECTOR_ETFS)
    for r in watch:
        syms += [r["symbol"], r["market"], r["sector"], *r["peers"]]
    return list(dict.fromkeys(s for s in syms if s and not s.startswith("fred:")))


def download(symbols: list[str], end: pd.Timestamp | None, attempts: int = 3) -> pd.DataFrame:
    import yfinance as yf
    start = (end or pd.Timestamp.today()) - pd.Timedelta(days=640)
    stop = end + pd.Timedelta(days=1) if end is not None else None
    last_err = None
    for i in range(attempts):
        try:
            log(f"Downloading {len(symbols)} symbols from Yahoo (try {i + 1})")
            raw = yf.download(symbols, start=start, end=stop, auto_adjust=True,
                              progress=False, threads=True, timeout=30)
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            closes.index = _naive(closes.index)
            if closes.dropna(how="all").empty:
                raise RuntimeError("Yahoo returned no data")
            return to_daily(closes)
        except Exception as e:  # network hiccups, rate limits
            last_err = e
            log(f"Yahoo download problem: {e}")
            time.sleep(15 * (i + 1))
    raise RuntimeError(f"price download failed after {attempts} tries: {last_err}")


def to_daily(closes: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar date. Yahoo can stamp 24/7 assets (crypto) at a
    different time of day from US stocks; without this, BTC's row and SPY's row
    for the same date would not line up."""
    closes = closes.copy()
    closes.index = pd.DatetimeIndex(closes.index).normalize()
    return closes.sort_index().groupby(level=0).last()


def refetch_stale(prices: pd.DataFrame, end: pd.Timestamp | None) -> pd.DataFrame:
    """Yahoo's batch download sometimes returns an out-of-date history for a few
    symbols (on 6 Nov 2026: GOOG, META, NBIS, CRWV, CHA, BUG and BTC stopped at
    the day before). Re-download any symbol whose last price is older than
    SPY's, one at a time, and keep the fresher series."""
    import yfinance as yf
    cal = prices[MARKET_CALENDAR].dropna()
    if cal.empty:
        return prices
    last = cal.index[-1]
    stale = [s for s in prices.columns
             if prices[s].dropna().empty or prices[s].dropna().index[-1] < last]
    if not stale:
        return prices
    log(f"Re-downloading {len(stale)} out-of-date symbol(s) one by one: {', '.join(stale)}")
    start = (end or pd.Timestamp.today()) - pd.Timedelta(days=640)
    stop = end + pd.Timedelta(days=1) if end is not None else None
    for s in stale:
        for attempt in range(2):
            try:
                h = yf.Ticker(s).history(start=start, end=stop, auto_adjust=True, timeout=30)
                if h.empty:
                    raise RuntimeError("empty")
                fresh = h["Close"].copy()
                fresh.index = _naive(fresh.index).normalize()
                fresh = fresh.groupby(level=0).last()
                if fresh.index[-1] > (prices[s].dropna().index[-1]
                                      if not prices[s].dropna().empty else pd.Timestamp.min):
                    prices = prices.reindex(prices.index.union(fresh.index))
                    prices[s] = fresh.reindex(prices.index).combine_first(prices[s])
                break
            except Exception as e:
                if attempt:
                    log(f"  {s}: re-download failed ({e})")
                time.sleep(2)
        time.sleep(1)
    return prices


def drop_unfinished_session(prices: pd.DataFrame) -> pd.DataFrame:
    """If New York's session is still open (before 4:30 pm New York time), drop
    rows dated today: Yahoo's 'close' for today would be a live mid-session price."""
    ny = datetime.now(ZoneInfo("America/New_York"))
    if (ny.hour, ny.minute) >= (16, 30):
        return prices
    today = pd.Timestamp(ny.date())
    n = int((prices.index >= today).sum())
    if n:
        log(f"New York time is {ny:%H:%M}: today's session is not finished, "
            f"so {n} row(s) dated {today.date()} are ignored")
    return prices.loc[prices.index < today]


def report_stale(prices: pd.DataFrame) -> None:
    """Log any symbol whose latest price is older than the US calendar's latest day."""
    cal = prices[MARKET_CALENDAR].dropna()
    if cal.empty:
        return
    last = cal.index[-1]
    stale = {s: prices[s].dropna().index[-1].date() for s in prices.columns
             if not prices[s].dropna().empty and prices[s].dropna().index[-1] < last}
    if stale:
        log(f"Latest price older than {MARKET_CALENDAR}'s ({last.date()}): " +
            ", ".join(f"{s} {d}" for s, d in sorted(stale.items())))


def fix_rate_scale(prices: pd.DataFrame) -> None:
    """Some feeds quote ^TNX etc. as yield x 10; convert to plain % if so."""
    for s in RATE_SYMBOLS & set(prices.columns):
        col = prices[s].dropna()
        if len(col) and col.median() > 20:
            prices[s] = prices[s] / 10


def change(series: pd.Series, t: pd.Timestamp, kind: str):
    """Move on t vs the previous price in the asset's own series.
    kind 'pct' -> % change; 'bp' -> basis-point change of a yield."""
    s = series.dropna().loc[:t]
    if len(s) < 2 or s.index[-1] != t:
        return None
    if kind == "bp":
        diffs = s.diff() * 100
    else:
        diffs = s.pct_change() * 100
    move = float(diffs.iloc[-1])
    hist = diffs.iloc[1:-1].tail(SIGMA_WINDOW)
    sd = hist.std(ddof=1)
    sigma = move / sd if len(hist) >= 20 and sd and not np.isnan(sd) else None
    return {"close": float(s.iloc[-1]), "prev_date": str(s.index[-2].date()),
            "move": move, "sigma": sigma}


def rate_sigma(cache: dict | None, sid: str, t: pd.Timestamp):
    """Sigma for an official rate series: today's bp change divided by the
    standard deviation of its previous SIGMA_WINDOW daily bp changes."""
    got = (cache or {}).get(sid)
    if not got or isinstance(got, Exception):
        return None
    s = got[0].dropna().loc[:t]
    diffs = s.diff() * 100
    hist = diffs.iloc[1:-1].tail(SIGMA_WINDOW)
    sd = hist.std(ddof=1)
    if len(hist) < 20 or not sd or np.isnan(sd):
        return None
    return float(diffs.iloc[-1] / sd)


def unit_for(row_class: str, symbol: str) -> str:
    return "bp" if row_class == "rate" or symbol in RATE_SYMBOLS else "pct"


# ------------------------------------------------------- free web data ---
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TREASURY_CSV = ("https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/daily-treasury-rates.csv/{year}/all")


def _get(url: str, params: dict | None = None, timeout: int = 20):
    import requests
    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": BROWSER_UA, "Accept": "*/*"})
    r.raise_for_status()
    return r


def treasury_curve(kind: str, since: pd.Timestamp) -> pd.DataFrame:
    """US Treasury daily yield curve (kind: 'daily_treasury_yield_curve' or
    'daily_treasury_real_yield_curve'), all years from `since` to today."""
    frames = []
    for year in range(since.year, pd.Timestamp.today().year + 1):
        r = _get(TREASURY_CSV.format(year=year),
                 {"type": kind, "field_tdr_date_value": year, "page": "", "_format": "csv"})
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
        frames.append(df.set_index("date"))
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated()]


def nyfed_target(since: pd.Timestamp) -> pd.DataFrame:
    """Fed funds target range and effective rate from the New York Fed."""
    r = _get("https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json",
             {"startDate": since.strftime("%Y-%m-%d"),
              "endDate": pd.Timestamp.today().strftime("%Y-%m-%d")})
    rows = r.json().get("refRates", [])
    df = pd.DataFrame(rows)
    df["effectiveDate"] = pd.to_datetime(df["effectiveDate"])
    return df.set_index("effectiveDate").sort_index()


def fetch_fred(series_id: str, since: pd.Timestamp) -> pd.Series:
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv",
             {"id": series_id, "cosd": since.strftime("%Y-%m-%d")}, timeout=15)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    return pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["date"]))


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    s = pd.to_numeric(df[name], errors="coerce").dropna()
    if s.empty:
        raise RuntimeError(f"column {name} is empty")
    return s


def load_rates(since: pd.Timestamp) -> dict:
    """Official sources first (Treasury, New York Fed), FRED as the fallback.
    Returns {series_id: (Series, source) or Exception}."""
    cache: dict = {}
    try:
        nom = treasury_curve("daily_treasury_yield_curve", since)
        cache["DGS2"] = (_col(nom, "2 yr"), "US Treasury daily yield curve")
        cache["DGS10"] = (_col(nom, "10 yr"), "US Treasury daily yield curve")
    except Exception as e:
        log(f"Treasury yield curve unavailable: {e}")
    try:
        real = treasury_curve("daily_treasury_real_yield_curve", since)
        cache["DFII10"] = (_col(real, "10 yr"), "US Treasury daily real yield curve")
    except Exception as e:
        log(f"Treasury real yield curve unavailable: {e}")
    try:
        ny = nyfed_target(since)
        cache["DFEDTARU"] = (_col(ny, "targetRateTo"), "Federal Reserve Bank of New York")
        cache["DFEDTARL"] = (_col(ny, "targetRateFrom"), "Federal Reserve Bank of New York")
    except Exception as e:
        log(f"New York Fed rates unavailable: {e}")
    for sid in FRED_SERIES:
        if sid not in cache:
            try:
                cache[sid] = (fetch_fred(sid, since), "FRED, St. Louis Fed")
            except Exception as e:
                cache[sid] = e
    return cache


def fred_block(t: pd.Timestamp, cache: dict, errors: list) -> dict:
    out = {}
    for sid, label in FRED_SERIES.items():
        try:
            got = cache.get(sid)
            if isinstance(got, Exception) or got is None:
                raise RuntimeError(got)
            s, source = got
            s = s.loc[:t]
            if len(s) < 2:
                raise RuntimeError("not enough observations")
            out[sid] = {"label": label, "date": str(s.index[-1].date()),
                        "value": float(s.iloc[-1]),
                        "change_bp": float((s.iloc[-1] - s.iloc[-2]) * 100),
                        "source": source,
                        "note": "official end-of-day data; may lag the session by a day"}
        except Exception as e:
            errors.append(f"Rates {sid}: {e}")
    return out


_last_call = [0.0]
_finnhub_time = [0.0]


def finnhub(path: str, params: dict, key: str):
    import requests
    if _finnhub_time[0] > FINNHUB_BUDGET:
        raise RuntimeError("Finnhub time budget used up; skipped")
    started = time.time()
    try:
        for i in range(3):
            wait = FINNHUB_MIN_GAP - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
            r = requests.get(f"{FINNHUB}/{path}", params={**params, "token": key}, timeout=20)
            if r.status_code == 429:          # free tier: 60 calls/minute
                time.sleep(10)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Finnhub rate limit")
    finally:
        _finnhub_time[0] += time.time() - started


def load_company_news(symbol: str, first: pd.Timestamp, last: pd.Timestamp, key: str) -> list:
    """One call per stock for the whole run; sessions are filtered later."""
    return finnhub("company-news", {"symbol": symbol,
                                    "from": (first - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                                    "to": (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")}, key) or []


def load_earnings(symbol: str, first: pd.Timestamp, last: pd.Timestamp, key: str) -> list:
    data = finnhub("calendar/earnings", {"symbol": symbol,
                                         "from": (first - pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
                                         "to": (last + pd.Timedelta(days=30)).strftime("%Y-%m-%d")}, key)
    return (data or {}).get("earningsCalendar", [])


def headlines(items: list, t: pd.Timestamp) -> list[dict]:
    lo = (t - pd.Timedelta(days=3)).timestamp()
    hi = (t + pd.Timedelta(days=1, hours=12)).timestamp()
    items = [i for i in items if lo <= i.get("datetime", 0) <= hi and i.get("headline")]
    items = sorted(items, key=lambda x: x.get("datetime", 0), reverse=True)[:10]
    return [{"time_utc": datetime.fromtimestamp(i["datetime"], timezone.utc).strftime("%Y-%m-%d %H:%M"),
             "source": i.get("source"), "headline": i.get("headline"), "url": i.get("url")}
            for i in items]


def earnings(cal: list, t: pd.Timestamp) -> list[dict]:
    hour = {"bmo": "before market open", "amc": "after market close", "dmh": "during market hours"}
    lo, hi = t - pd.Timedelta(days=4), t + pd.Timedelta(days=30)
    out = []
    for e in cal:
        try:
            d = pd.Timestamp(e.get("date"))
        except Exception:
            continue
        if lo <= d <= hi:
            out.append({"date": e.get("date"), "time": hour.get(e.get("hour"), e.get("hour") or "unknown"),
                        "eps_estimate": e.get("epsEstimate"), "eps_actual": e.get("epsActual")})
    return out


def market_news(t: pd.Timestamp, key: str) -> list[dict]:
    items = finnhub("news", {"category": "general"}, key) or []
    cutoff = (t - pd.Timedelta(days=1)).timestamp()
    items = [i for i in items if i.get("datetime", 0) >= cutoff][:15]
    return [{"time_utc": datetime.fromtimestamp(i["datetime"], timezone.utc).strftime("%Y-%m-%d %H:%M"),
             "source": i.get("source"), "headline": i.get("headline"), "url": i.get("url")}
            for i in items]


# --------------------------------------------------------- one session ---
def build_session(prices: pd.DataFrame, watch: list[dict], t: pd.Timestamp,
                  fred_cache: dict | None, news: dict, is_latest: bool,
                  general_news: list) -> dict:
    errors: list[str] = []
    out = {
        "schema": 1,
        "session_date": str(t.date()),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "sources": {"prices": "Yahoo Finance via yfinance (adjusted closes)",
                    "rates": "US Treasury and New York Fed (FRED as fallback); see each series' source",
                    "headlines": "Finnhub" if news else "not fetched"},
        "method": {"move": "close on session_date vs the asset's previous close",
                   "sigma_window_days": SIGMA_WINDOW,
                   "attribution": "two-factor OLS on the 252 common trading days before the "
                                  "session; NA if < 120 days or R^2 < 0.10",
                   "units": "move_pct in %, move_bp in basis points, components in "
                            "percentage points (pp)"},
        "backdrop": [], "sectors": [], "assets": [], "fred": {}, "market_news": [],
        "errors": errors,
    }

    for label, sym in BACKDROP:
        if sym not in prices:
            continue
        c = change(prices[sym], t, unit_for("", sym))
        if c:
            key = "move_bp" if unit_for("", sym) == "bp" else "move_pct"
            out["backdrop"].append({"name": label, "symbol": sym, "close": c["close"],
                                    key: c["move"]})

    for etf, name in SECTOR_ETFS.items():
        if etf in prices:
            c = change(prices[etf], t, "pct")
            if c:
                out["sectors"].append({"etf": etf, "sector": name, "move_pct": c["move"]})
    out["sectors"].sort(key=lambda r: r["move_pct"], reverse=True)

    if fred_cache is not None:
        out["fred"] = fred_block(t, fred_cache, errors)

    for r in watch:
        a = {"ticker": r["ticker"], "name": r["name"], "class": r["class"],
             "symbol": r["symbol"], "notes": []}
        sym, cls = r["symbol"], r["class"]
        if sym.startswith("fred:"):
            sid = sym.split(":", 1)[1]
            f = out["fred"].get(sid)
            a["attribution"] = {"na_reason": "no split for rates, FX, policy or the market itself"}
            if not f:
                a["notes"].append(f"Official rate data ({sid}) unavailable in this run.")
            elif cls == "policy":
                a.update({"close": f["value"], "as_of": f["date"], "move_bp": f["change_bp"],
                          "sigma": None, "source": f["source"]})
                a["notes"].append(f"Policy rate from {f['source']}; changes only at FOMC decisions.")
            elif f["date"] != str(t.date()):
                # Official curve not published for this session yet: never show
                # an older day's change as if it were this session's move.
                a.update({"close": f["value"], "as_of": f["date"], "move_bp": None,
                          "sigma": None, "source": f["source"]})
                a["notes"].append(f"{f['source']} has not published {t.date()} yet; the latest "
                                  f"value is from {f['date']}, so no move is computed.")
            else:
                sig = rate_sigma(fred_cache, sid, t)
                a.update({"close": f["value"], "as_of": f["date"], "move_bp": f["change_bp"],
                          "sigma": sig, "source": f["source"],
                          "above_2_sigma": (abs(sig) >= 2) if sig is not None else None})
                a["notes"].append(f"Official end-of-day yield from {f['source']}.")
            out["assets"].append(a)
            continue
        if sym not in prices or prices[sym].dropna().empty:
            a["notes"].append("No price data from Yahoo for this symbol.")
            errors.append(f"{r['ticker']}: no data for {sym}")
            out["assets"].append(a)
            continue

        unit = unit_for(cls, sym)
        c = change(prices[sym], t, unit)
        if c is None:
            a["notes"].append(f"No price on {t.date()} (holiday or not yet updated).")
            if r["class"] in ("stock", "etf"):
                # a US-listed asset with no price on a US session is a data failure,
                # not a holiday: flag it so the digest falls back to the web for it
                errors.append(f"{r['ticker']}: Yahoo has no {t.date()} price ({sym}); use web data")
            out["assets"].append(a)
            continue
        a.update({"close": c["close"], "prev_date": c["prev_date"],
                  ("move_bp" if unit == "bp" else "move_pct"): c["move"],
                  "sigma": c["sigma"],
                  "above_2_sigma": (abs(c["sigma"]) >= 2) if c["sigma"] is not None else None})

        # market / sector / specific split
        if cls in ("stock", "etf", "commodity", "crypto") and r["market"]:
            if any(x not in prices for x in (r["market"], r["sector"]) if x):
                a["attribution"] = {"na_reason": "benchmark data missing"}
            else:
                at = attribute(prices, sym, r["market"], r["sector"] or None, t)
                a["attribution"] = {k: getattr(at, k) for k in (
                    "market", "sector", "move_pct", "interval_start", "market_pp",
                    "sector_pp", "specific_pp", "beta_market", "beta_sector", "r2",
                    "observations", "na_reason", "notes")}
                if at.move_pct is not None and abs(at.move_pct - c["move"]) > 0.05:
                    a["notes"].append(
                        f"Split measured from {at.interval_start} (last day all benchmarks "
                        f"traded), so its move ({at.move_pct:+.2f}%) differs from the headline move.")
        else:
            a["attribution"] = {"na_reason": "no split for rates, FX, policy or the market itself"}

        # peers
        peers = {}
        for p in r["peers"]:
            if p in prices:
                pc = change(prices[p], t, unit_for("", p))
                if pc:
                    peers[p] = {("move_bp" if unit_for("", p) == "bp" else "move_pct"): pc["move"]}
        a["peers"] = peers

        if cls == "commodity":
            a["notes"].append("Continuous front-month futures: check for a contract roll if "
                              "the move looks out of line with news.")
        if cls == "crypto":
            a["notes"].append("Crypto trades 24/7; the move is close-to-close on Yahoo's UTC "
                              "daily bars and the latest bar may be partial.")

        if r["news"] == "yes" and sym in news:
            n = news[sym]
            if isinstance(n.get("news"), list):
                a["headlines"] = headlines(n["news"], t)
            else:
                errors.append(f"Finnhub news {sym}: {n.get('news')}")
            if isinstance(n.get("earnings"), list):
                a["earnings"] = earnings(n["earnings"], t)
            else:
                errors.append(f"Finnhub earnings {sym}: {n.get('earnings')}")
        out["assets"].append(a)

    if is_latest:
        out["market_news"] = general_news
    return clean(out)


# ---------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="session date YYYY-MM-DD (default: latest)")
    ap.add_argument("--backfill", type=int, default=0, help="also build the N sessions before")
    ap.add_argument("--watchlist", default=str(HERE / "watchlist.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "data"))
    ap.add_argument("--prices-csv", help="offline test: read prices from a CSV instead of Yahoo")
    ap.add_argument("--no-network", action="store_true", help="skip FRED and Finnhub")
    args = ap.parse_args()

    watch = load_watchlist(Path(args.watchlist))
    syms = all_symbols(watch)
    end = pd.Timestamp(args.date) if args.date else None
    if args.prices_csv:
        prices = pd.read_csv(args.prices_csv, parse_dates=["date"]).set_index("date").sort_index()
    else:
        prices = download(syms, end)
        prices = refetch_stale(prices, end)
        prices = drop_unfinished_session(prices)
    log("Price download finished")
    fix_rate_scale(prices)
    report_stale(prices)

    cal = prices[MARKET_CALENDAR].dropna()
    if end is not None:
        if end not in cal.index:
            sys.exit(f"{end.date()} was not a completed US trading session (no {MARKET_CALENDAR} close; "
                     "a session still trading in New York is ignored until 4:30 pm New York time).")
        sessions = cal.loc[:end].index
    else:
        sessions = cal.index
    targets = list(sessions[-(args.backfill + 1):])
    latest = cal.index[-1]

    log(f"Prices ready: {prices.shape[1]} symbols, sessions {targets[0].date()} to {targets[-1].date()}")

    network = not args.no_network
    fred_cache = None
    if network:
        # 130 calendar days gives 60+ daily changes for the rate sigma
        fred_cache = load_rates(targets[0] - pd.Timedelta(days=130))
        log("Interest rates downloaded")

    key = os.environ.get("FINNHUB_API_KEY") or None
    news: dict = {}
    general_news: list = []
    if network and key:
        for r in watch:
            if r["news"] != "yes":
                continue
            entry = {}
            for name, fn in (("news", load_company_news), ("earnings", load_earnings)):
                try:
                    entry[name] = fn(r["symbol"], targets[0], targets[-1], key)
                except Exception as e:
                    entry[name] = str(e)
            news[r["symbol"]] = entry
        if not args.date:
            try:
                general_news = market_news(latest, key)
            except Exception as e:
                log(f"Finnhub market news skipped: {e}")
        log("Finnhub headlines and earnings downloaded")
    elif network:
        log("No FINNHUB_API_KEY set: headlines skipped")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in targets:
        is_latest = (t == latest) and not args.date
        data = build_session(prices, watch, t, fred_cache, news, is_latest,
                             general_news if is_latest else [])
        text = json.dumps(data, indent=1, ensure_ascii=False)
        (out_dir / f"{t.date()}.json").write_text(text)
        if t == latest and not args.date:      # a past --date never overwrites latest
            (out_dir / "latest.json").write_text(text)
        n_err = len(data["errors"])
        log(f"{t.date()}: {len(data['assets'])} assets written, {n_err} warnings")
        for e in data["errors"]:
            log(f"  warning: {e}")

    dates = sorted(p.stem for p in out_dir.glob("20??-??-??.json"))
    (out_dir / "index.json").write_text(json.dumps(
        {"latest": dates[-1] if dates else None, "sessions": dates}, indent=1))


if __name__ == "__main__":
    main()
