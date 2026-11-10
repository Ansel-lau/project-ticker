#!/usr/bin/env python3
"""
S&P 500 heatmap replay video for the market-move system.

Makes a 16:9 MP4 (1920x1080, 30 fps, about 13 s) of the most recent completed
US session:
  * opens on YESTERDAY's map (close-to-close),
  * then replays TODAY in one-hour steps: 9:30 open, 10:30, 11:30 ... close,
  * left panel: S&P 500 15-minute candles revealed in step with the map,
    plus the change in the index's total market value and the biggest $ moves.

Tiles are sized by market value (shares outstanding x previous close) and
grouped by GICS sector. Colour = % move versus the previous close.

Free sources only
  constituents  github.com/datasets/s-and-p-500-companies (CSV, refreshed weekly)
  shares        Yahoo Finance via yfinance (cached for 7 days)
  prices        Yahoo Finance via yfinance (daily, 60-minute and 15-minute bars)

Usage
  python heatmap_video.py                  # latest completed session
  python heatmap_video.py --date 2026-11-05
  python heatmap_video.py --demo           # simulated prices, no network for prices
"""
from __future__ import annotations

import argparse
import io
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NY = ZoneInfo("America/New_York")
CONSTITUENTS_URL = ("https://raw.githubusercontent.com/datasets/"
                    "s-and-p-500-companies/main/data/constituents.csv")
META_CACHE = HERE / "data" / "sp500_meta.csv"
META_MAX_AGE_DAYS = 7
INDEX = "^GSPC"

W, H, FPS = 1920, 1080, 30
# timeline (seconds)
T_YEST_HOLD = 1.8
T_TRANS = 0.6          # one hour-step transition (tiles sweep left to right)
T_TILE = 0.3           # each tile's own colour change inside a transition
T_HOLD = 0.6           # hold on each hourly state
T_END_HOLD = 2.2       # hold on the close

BG = "#0b0d12"
PANEL = "#12151d"
GRID = "#232838"
TXT = "#e8eaf0"
MUTED = "#8a90a2"
UP = "#30cc5a"
DOWN = "#f63538"
# % move -> colour (finviz-style diverging scale, clipped at +-3%)
SCALE = [(-3, "#f63538"), (-2, "#bf4045"), (-1, "#8b444e"), (0, "#414554"),
         (1, "#35764e"), (2, "#2f9e4f"), (3, "#30cc5a")]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# =============================================================== data model ===
@dataclass
class Scene:
    tickers: list[str]
    sectors: list[str]
    shares: np.ndarray            # (N,)
    layout_cap: np.ndarray        # (N,) market value used for tile size
    yday: pd.Timestamp            # previous session date
    today: pd.Timestamp           # session being replayed
    yday_pct: np.ndarray          # (N,) yesterday close-to-close %
    yday_dollar: np.ndarray       # (N,) yesterday $ change per stock
    yday_index_pct: float
    key_labels: list[str]         # e.g. ["OPEN 9:30", "10:30", ..., "CLOSE 16:00"]
    key_minutes: list[float]      # minutes after 9:30 for each keyframe
    key_pct: np.ndarray           # (K, N) % vs previous close at each keyframe
    key_dollar: np.ndarray        # (K, N) $ change vs previous close
    key_index_pct: np.ndarray     # (K,)
    candles_y: pd.DataFrame       # 15-min OHLC of yesterday (column 'm' = minutes after 9:30)
    candles_t: pd.DataFrame       # 15-min OHLC of today
    ref_y: float                  # index close the day before yesterday
    ref_t: float                  # index close yesterday
    demo: bool = False
    notes: list[str] = field(default_factory=list)


# ================================================================ helpers ===
def pct_colour(p: np.ndarray) -> np.ndarray:
    """Vector of % moves -> (N, 4) RGBA array."""
    from matplotlib.colors import to_rgba
    xs = np.array([s[0] for s in SCALE], float)
    cs = np.array([to_rgba(s[1]) for s in SCALE])
    p = np.nan_to_num(np.clip(p, xs[0], xs[-1]), nan=0.0)
    out = np.empty((len(p), 4))
    for k in range(4):
        out[:, k] = np.interp(p, xs, cs[:, k])
    return out


def fmt_pct(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{'+' if p >= 0 else '−'}{abs(p):.2f}%"


def fmt_money(v: float) -> str:
    """Dollars -> '+$1.23T' / '−$412B'."""
    sign = "+" if v >= 0 else "−"
    a = abs(v)
    if a >= 1e12:
        return f"{sign}${a / 1e12:.2f}T"
    if a >= 1e9:
        return f"{sign}${a / 1e9:.0f}B" if a >= 1e10 else f"{sign}${a / 1e9:.1f}B"
    return f"{sign}${a / 1e6:.0f}M"


def smooth(x: np.ndarray | float):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def yahoo_symbol(s: str) -> str:
    return s.replace(".", "-")          # BRK.B -> BRK-B


# =========================================================== real-data path ===
def load_constituents() -> pd.DataFrame:
    import requests
    r = requests.get(CONSTITUENTS_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"})
    df["symbol"] = df["symbol"].map(yahoo_symbol)
    return df[["symbol", "name", "sector"]]


def _shares_one(sym: str):
    import yfinance as yf
    for attempt in range(3):
        try:
            fi = yf.Ticker(sym).fast_info
            sh = fi.get("shares")
            if not sh:
                mc, lp = fi.get("marketCap"), fi.get("lastPrice")
                sh = mc / lp if mc and lp else None
            return float(sh) if sh else None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def load_meta() -> pd.DataFrame:
    """Constituents + sector + shares outstanding, cached for a week."""
    cached = None
    if META_CACHE.exists():
        cached = pd.read_csv(META_CACHE)
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(cached["updated_utc"].iloc[0])).days
        if age < META_MAX_AGE_DAYS and cached["shares"].notna().mean() > 0.9:
            log(f"Using cached constituents and shares ({age} days old)")
            return cached
    try:
        cons = load_constituents()
    except Exception as e:
        if cached is not None:
            log(f"Constituents download failed ({e}); using the older cache")
            return cached
        raise
    log(f"Fetching shares outstanding for {len(cons)} stocks (weekly refresh)")
    with ThreadPoolExecutor(max_workers=4) as ex:
        cons["shares"] = list(ex.map(_shares_one, cons["symbol"]))
    missing = cons["shares"].isna().sum()
    if missing and cached is not None:      # keep last week's number rather than drop a tile
        old = cached.set_index("symbol")["shares"]
        cons["shares"] = cons["shares"].fillna(cons["symbol"].map(old))
    log(f"Shares fetched; {int(cons['shares'].isna().sum())} still missing")
    cons["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    META_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cons.to_csv(META_CACHE, index=False)
    return cons


def _ny_naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(NY).tz_localize(None)


def _download(symbols, **kw) -> pd.DataFrame:
    import yfinance as yf
    last = None
    for i in range(3):
        try:
            raw = yf.download(symbols, progress=False, threads=True, timeout=30,
                              auto_adjust=False, **kw)
            if raw is not None and not raw.dropna(how="all").empty:
                return raw
            last = "empty result"
        except Exception as e:
            last = e
        log(f"Yahoo download problem ({kw.get('interval')}): {last}")
        time.sleep(15 * (i + 1))
    raise RuntimeError(f"Yahoo download failed: {last}")


def refetch_missing(d: pd.DataFrame, days: list, limit: int = 150) -> pd.DataFrame:
    """Yahoo's batch download sometimes returns out-of-date history for some
    symbols (on 6 Nov 2026, 7 of 20 watchlist symbols had no 6 Nov price).
    Re-download those one at a time, as daily_data.py's refetch_stale does."""
    import yfinance as yf
    miss = [s for s in d.columns if s != INDEX and d.loc[days, s].isna().any()]
    if not miss:
        return d
    log(f"{len(miss)} stock(s) missing a daily close; re-downloading one by one"
        + (f" (first {limit})" if len(miss) > limit else ""))
    d = d.copy()
    got = 0
    for s in miss[:limit]:
        try:
            h = yf.Ticker(s).history(period="15d", interval="1d", auto_adjust=False, timeout=20)
            if h.empty:
                continue
            c = h["Close"].copy()
            c.index = pd.DatetimeIndex(_ny_naive(c.index)).normalize()
            c = c.groupby(level=0).last()
            d[s] = d[s].combine_first(c.reindex(d.index))
            got += 1
        except Exception as e:
            log(f"  {s}: re-download failed ({e})")
        time.sleep(0.3)
    still = [s for s in miss if d.loc[days, s].isna().any()]
    log(f"Re-downloaded {got}; still missing a close: {len(still)}")
    return d


def completed_session(daily_close: pd.Series, want: pd.Timestamp | None) -> pd.Timestamp:
    s = daily_close.dropna()
    ny = datetime.now(NY)
    if (ny.hour, ny.minute) < (16, 30):                 # today's session not finished
        s = s[s.index < pd.Timestamp(ny.date())]
    if want is not None:
        if want not in s.index:
            sys.exit(f"{want.date()} is not a completed US session in the data")
        return want
    return s.index[-1]


def build_real(date: str | None) -> Scene:
    meta = load_meta().dropna(subset=["shares"])
    syms = list(meta["symbol"])
    want = pd.Timestamp(date) if date else None

    log("Downloading daily closes")
    d = _download(syms + [INDEX], period="15d", interval="1d")["Close"]
    d.index = pd.DatetimeIndex(d.index).normalize()
    d = d.groupby(level=0).last()
    t = completed_session(d[INDEX], want)
    pos = d.index.get_loc(t)
    if pos < 2:
        sys.exit("not enough daily history")
    y, y2 = d.index[pos - 1], d.index[pos - 2]
    d = refetch_missing(d, [t, y, y2])
    close_t, close_y, close_y2 = d.loc[t], d.loc[y], d.loc[y2]

    log("Downloading 60-minute bars")
    hr = _download(syms + [INDEX], period="5d", interval="60m", prepost=False)
    hr.index = _ny_naive(hr.index)
    hr = hr[hr.index.normalize() == t]
    # last resort for a daily close that is still missing: the last hourly price
    gap = [s for s in syms if not np.isfinite(close_t.get(s, np.nan))]
    notes = []
    if gap and not hr.empty:
        last_hr = hr["Close"][gap].ffill().iloc[-1]
        filled = last_hr.dropna()
        if len(filled):
            close_t = close_t.copy()
            close_t[filled.index] = filled
            log(f"{len(filled)} stock(s) use their last hourly price as the close: "
                f"{' '.join(filled.index[:20])}")
            notes.append(f"{len(filled)} close(s) from the last hourly price")
    if hr.empty:
        sys.exit(f"no hourly bars for {t.date()}")
    bar_starts = sorted(set(hr.index))
    # keyframes: open, then the close of each hourly bar except the last, then the daily close
    key_prices, labels, minutes = [], [], []
    key_prices.append(hr["Open"].loc[bar_starts[0]])
    labels.append(f"OPEN {bar_starts[0]:%-H:%M}")
    minutes.append(0.0)
    for b in bar_starts[:-1]:
        end = b + pd.Timedelta(hours=1)
        key_prices.append(hr["Close"].loc[b])
        labels.append(f"{end:%-H:%M}")
        minutes.append((end - bar_starts[0]).total_seconds() / 60)
    last_end = min(bar_starts[-1] + pd.Timedelta(hours=1), t + pd.Timedelta(hours=16))
    key_prices.append(close_t)
    labels.append(f"CLOSE {last_end:%-H:%M}")
    minutes.append((last_end - bar_starts[0]).total_seconds() / 60)

    P = pd.DataFrame(key_prices).reset_index(drop=True)
    P = P.ffill()                                        # a stock with no trade in a bar keeps its last price
    shares = meta.set_index("symbol")["shares"]

    log("Downloading 15-minute index bars")
    q = _download(INDEX, period="5d", interval="15m", prepost=False)
    if isinstance(q.columns, pd.MultiIndex):
        q.columns = q.columns.get_level_values(0)
    q.index = _ny_naive(q.index)

    def candles(day):
        c = q[q.index.normalize() == day][["Open", "High", "Low", "Close"]].dropna()
        c["m"] = [(ix - ix.normalize() - pd.Timedelta(hours=9, minutes=30)).total_seconds() / 60
                  for ix in c.index]
        return c.reset_index(drop=True)

    keep = [s for s in syms if np.isfinite(close_y.get(s, np.nan)) and np.isfinite(close_t.get(s, np.nan))]
    dropped = sorted(set(syms) - set(keep))
    if dropped:
        log(f"No price for {len(dropped)} stocks, left out: {' '.join(dropped[:20])}")
    m = meta.set_index("symbol").loc[keep]
    sh = shares.loc[keep].to_numpy(float)
    py, py2 = close_y[keep].to_numpy(float), close_y2[keep].to_numpy(float)
    Pk = P[keep].to_numpy(float)
    Pk = np.where(np.isfinite(Pk), Pk, py)                # still missing -> flat
    idx_k = P[INDEX].to_numpy(float)

    return Scene(
        tickers=keep, sectors=list(m["sector"]), shares=sh, layout_cap=sh * py,
        yday=y, today=t,
        yday_pct=(py / py2 - 1) * 100, yday_dollar=sh * (py - py2),
        yday_index_pct=float((close_y[INDEX] / close_y2[INDEX] - 1) * 100),
        key_labels=labels, key_minutes=minutes,
        key_pct=(Pk / py - 1) * 100, key_dollar=sh * (Pk - py),
        key_index_pct=(idx_k / close_y[INDEX] - 1) * 100,
        candles_y=candles(y), candles_t=candles(t),
        ref_y=float(close_y2[INDEX]), ref_t=float(close_y[INDEX]),
        notes=notes + ([f"{len(dropped)} stock(s) without a price left out"] if dropped else []),
    )


# ============================================================ demo-data path ===
def build_demo(constituents_csv: str | None) -> Scene:
    """Simulated prices on the real constituent list, for layout testing only."""
    rng = np.random.default_rng(7)
    if constituents_csv:
        cons = pd.read_csv(constituents_csv).rename(
            columns={"Symbol": "symbol", "GICS Sector": "sector"})
        cons["symbol"] = cons["symbol"].map(yahoo_symbol)
    else:
        sect = ["Information Technology", "Financials", "Health Care", "Industrials",
                "Consumer Discretionary", "Communication Services", "Consumer Staples",
                "Energy", "Utilities", "Real Estate", "Materials"]
        cons = pd.DataFrame({"symbol": [f"S{i:03d}" for i in range(500)],
                             "sector": rng.choice(sect, 500)})
    n = len(cons)
    beta_sector = {"Information Technology": 1.6, "Communication Services": 1.2,
                   "Consumer Discretionary": 1.1, "Financials": 0.9, "Industrials": 0.9,
                   "Materials": 0.8, "Health Care": 0.6, "Real Estate": 0.6,
                   "Energy": 0.4, "Consumer Staples": 0.3, "Utilities": 0.3}
    beta = cons["sector"].map(beta_sector).fillna(1.0).to_numpy() * rng.uniform(0.6, 1.4, n)
    price_y2 = rng.uniform(30, 600, n)
    cap = np.exp(rng.normal(np.log(4e10), 1.15, n))
    shares = cap / price_y2

    y_mkt = 1.55
    py = price_y2 * (1 + (beta * y_mkt + rng.normal(0, 1.1, n)) / 100)
    # market path for today, hourly (open, 10:30 ... close): steady slide into the close
    mkt = np.array([0.35, -0.6, -1.4, -2.1, -2.6, -3.3, -3.9, -4.4])
    idio = np.cumsum(rng.normal(0, 0.45, (len(mkt), n)), axis=0) + rng.normal(0, 0.6, n)
    pct = mkt[:, None] * beta[None, :] + idio
    Pk = py[None, :] * (1 + pct / 100)
    labels = ["OPEN 9:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "CLOSE 16:00"]
    minutes = [0, 60, 120, 180, 240, 300, 360, 390]

    def path(start, end_pct, wobble):
        m = np.arange(0, 390, 15)
        level = start * (1 + (np.interp(m, [0, 390], [0, end_pct])
                              + np.cumsum(rng.normal(0, wobble, len(m)))) / 100)
        o = np.r_[start, level[:-1]]
        hi = np.maximum(o, level) * (1 + rng.uniform(0, 0.0012, len(m)))
        lo = np.minimum(o, level) * (1 - rng.uniform(0, 0.0012, len(m)))
        return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": level, "m": m})

    ref_y = 6780.0
    ref_t = ref_y * (1 + y_mkt / 100)
    cy = path(ref_y, y_mkt, 0.12)
    ct = path(ref_t * 1.0035, -4.75, 0.14)
    idx_pct = (np.array([ct["Open"].iloc[0]] + list(ct["Close"].iloc[[3, 7, 11, 15, 19, 23]])
                        + [ct["Close"].iloc[-1]]) / ref_t - 1) * 100
    return Scene(
        tickers=list(cons["symbol"]), sectors=list(cons["sector"]), shares=shares,
        layout_cap=shares * py, yday=pd.Timestamp("2026-11-04"), today=pd.Timestamp("2026-11-05"),
        yday_pct=(py / price_y2 - 1) * 100, yday_dollar=shares * (py - price_y2),
        yday_index_pct=y_mkt, key_labels=labels, key_minutes=minutes,
        key_pct=pct, key_dollar=shares[None, :] * (Pk - py[None, :]), key_index_pct=idx_pct,
        candles_y=cy, candles_t=ct, ref_y=ref_y, ref_t=ref_t, demo=True,
    )


# =================================================================== layout ===
def treemap(scene: Scene, x0, y0, w, h):
    """Return sector boxes and one rectangle per stock (pixel coordinates, y down)."""
    import squarify
    df = pd.DataFrame({"i": range(len(scene.tickers)), "sector": scene.sectors,
                       "cap": scene.layout_cap})
    df = df[np.isfinite(df["cap"]) & (df["cap"] > 0)]
    by = df.groupby("sector")["cap"].sum().sort_values(ascending=False)
    boxes = squarify.squarify(squarify.normalize_sizes(by.values, w, h), x0, y0, w, h)
    sectors, rects = [], {}
    pad, head = 3, 20
    for name, b in zip(by.index, boxes):
        bx, by_, bw, bh = b["x"] + pad, b["y"] + pad, b["dx"] - 2 * pad, b["dy"] - 2 * pad
        show_head = bh > 60 and bw > 90
        iy = by_ + (head if show_head else 0)
        ih = bh - (head if show_head else 0)
        sub = df[df["sector"] == name].sort_values("cap", ascending=False)
        rs = squarify.squarify(squarify.normalize_sizes(sub["cap"].values, bw, ih), bx, iy, bw, ih)
        for i, r in zip(sub["i"], rs):
            rects[int(i)] = (r["x"], r["y"], r["dx"], r["dy"])
        members = sub["i"].to_numpy()
        sectors.append({"name": name, "box": (bx, by_, bw, bh), "head": show_head,
                        "members": members})
    return sectors, rects


SECTOR_SHORT = {"Information Technology": "TECHNOLOGY", "Communication Services": "COMMUNICATION",
                "Consumer Discretionary": "CONSUMER DISC.", "Consumer Staples": "STAPLES",
                "Health Care": "HEALTH CARE", "Financials": "FINANCIALS",
                "Industrials": "INDUSTRIALS", "Energy": "ENERGY", "Utilities": "UTILITIES",
                "Real Estate": "REAL ESTATE", "Materials": "MATERIALS"}


# ================================================================== render ===
class Renderer:
    def __init__(self, scene: Scene):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import FancyBboxPatch, Rectangle

        plt.rcParams["font.family"] = "DejaVu Sans"
        self.s = scene
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        self.FancyBboxPatch = FancyBboxPatch

        # one full-figure axes in pixel coordinates (y down) for everything but the chart
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        self.ax = ax

        # ---- treemap
        TX, TY, TW, TH = 624, 116, 1264, 916
        self.sectors, rects = treemap(scene, TX, TY, TW, TH)
        self.order = np.array(sorted(rects))
        patches = [Rectangle(rects[i][:2], rects[i][2], rects[i][3]) for i in self.order]
        self.tiles = PatchCollection(patches, edgecolor=BG, linewidth=0.7, zorder=2)
        ax.add_collection(self.tiles)
        xs = np.array([rects[i][0] + rects[i][2] / 2 for i in self.order])
        self.delay = (xs - TX) / TW * (T_TRANS - T_TILE)     # left-to-right sweep

        # tile labels, measured so they always fit inside their tile
        self.labels = []
        rend = self.fig.canvas.get_renderer()
        PT = 100 / 72                                    # pixels per point at dpi 100
        for j, i in enumerate(self.order):
            x, y, w, h = rects[i]
            tk = scene.tickers[i]
            fs = min(h / 2.0 / PT, 26)                   # start from the height limit
            t1 = ax.text(x + w / 2, y + h / 2, tk, ha="center", va="center", color="white",
                         fontsize=fs, fontweight="bold", zorder=3)
            tw = t1.get_window_extent(rend).width
            if tw > 0.86 * w:
                fs *= 0.86 * w / tw
            if fs < 5.5:                                 # too small to read: colour only
                t1.remove()
                continue
            t1.set_fontsize(fs)
            pfs = fs * 0.6
            show_pct = h > (fs + pfs) * PT * 1.55 and pfs >= 5.5 and w > pfs * PT * 4.6
            if not show_pct and h > 34 and w > 44:      # shrink the ticker to make room for the %
                fs2 = min(fs, h / (PT * 1.6 * 1.55), w / (PT * 0.6 * 4.6))
                if fs2 >= 9:
                    fs, pfs, show_pct = fs2, fs2 * 0.6, True
                    t1.set_fontsize(fs)
            t2 = None
            if show_pct:
                gap = (fs + pfs) * PT * 0.42
                t1.set_y(y + h / 2 - gap * 0.55)
                t2 = ax.text(x + w / 2, y + h / 2 + gap * 0.75, "", ha="center", va="center",
                             color=(1, 1, 1, 0.88), fontsize=pfs, zorder=3)
            self.labels.append((j, i, t2))
        # sector headers
        self.sector_txt = []
        for sct in self.sectors:
            bx, by_, bw, bh = sct["box"]
            if sct["head"]:
                name = SECTOR_SHORT.get(sct["name"], sct["name"].upper())
                t = ax.text(bx + 4, by_ + 10, name, color=MUTED, fontsize=10.5,
                            fontweight="bold", va="center", zorder=3)
                if t.get_window_extent(rend).width + 78 < bw:     # room for the sector %
                    tp = ax.text(bx + bw - 4, by_ + 10, "", color=MUTED, fontsize=10.5,
                                 fontweight="bold", va="center", ha="right", zorder=3)
                    self.sector_txt.append((sct, tp))

        # ---- header
        ax.text(32, 40, "S&P 500 HEATMAP", color=TXT, fontsize=27, fontweight="bold", va="center")
        ax.text(32, 80, f"{len(rects)} stocks · tile size = market value · colour = move vs previous close",
                color=MUTED, fontsize=13, va="center")
        self.pill = FancyBboxPatch((624, 24), 380, 44, boxstyle="round,pad=0,rounding_size=8",
                                   linewidth=0, facecolor="#2a2f3d", zorder=3)
        ax.add_patch(self.pill)
        self.pill_txt = ax.text(624 + 190, 46, "", color="white", fontsize=17,
                                fontweight="bold", ha="center", va="center", zorder=4)
        self.clock = ax.text(1024, 46, "", color=TXT, fontsize=22, fontweight="bold", va="center")
        ax.text(1888, 30, "S&P 500", color=MUTED, fontsize=13, ha="right", va="center")
        self.idx_txt = ax.text(1888, 70, "", color=TXT, fontsize=30, fontweight="bold",
                               ha="right", va="center")

        # ---- left panel: chart
        self.cax = self.fig.add_axes([88 / W, 1 - 636 / H, 466 / W, 512 / H], facecolor=PANEL)
        for sp in self.cax.spines.values():
            sp.set_visible(False)
        self.cax.tick_params(colors=MUTED, labelsize=10, length=0)
        self.cax.grid(axis="y", color=GRID, linewidth=0.8)
        self.cax.set_axisbelow(True)
        ny_ = len(scene.candles_y)
        self.x_y = np.arange(ny_)
        self.x_t = np.arange(len(scene.candles_t)) + ny_ + 2
        allc = pd.concat([scene.candles_y, scene.candles_t])
        lo, hi = allc["Low"].min(), allc["High"].max()
        pad = (hi - lo) * 0.10
        self.cax.set_ylim(lo - pad, hi + pad)
        self.cax.set_xlim(-1.5, self.x_t[-1] + 1.5 if len(self.x_t) else ny_ + 30)
        self.cax.set_xticks([ny_ / 2, ny_ + 2 + 13])
        self.cax.set_xticklabels([f"{scene.yday:%a %-d %b}", f"{scene.today:%a %-d %b}"])
        self.cax.yaxis.set_major_formatter(
            __import__("matplotlib").ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        self.cax.axvline(ny_ + 0.5, color=GRID, linewidth=1)
        self.chart_artists = []

        # ---- left panel: counter and biggest moves
        ax.text(32, 676, "S&P 500 MARKET VALUE vs PREVIOUS CLOSE (est.)", color=MUTED,
                fontsize=11.5, fontweight="bold", va="center")
        self.money = ax.text(32, 732, "", color=TXT, fontsize=50, fontweight="bold", va="center")
        ax.text(32, 806, "BIGGEST $ DRAGS", color=MUTED, fontsize=11.5, fontweight="bold", va="center")
        ax.text(312, 806, "BIGGEST $ LIFTS", color=MUTED, fontsize=11.5, fontweight="bold", va="center")
        self.drag_txt = [ax.text(32, 840 + 32 * k, "", color=TXT, fontsize=15, va="center",
                                 family="DejaVu Sans Mono") for k in range(4)]
        self.lift_txt = [ax.text(312, 840 + 32 * k, "", color=TXT, fontsize=15, va="center",
                                 family="DejaVu Sans Mono") for k in range(4)]

        # ---- footer
        ax.text(32, 1062, "Prices: Yahoo Finance (yfinance) · hourly steps · market value = shares "
                "outstanding × price · not investment advice", color=MUTED, fontsize=10.5, va="center")
        if scene.notes and not scene.demo:
            ax.text(1888, 1062, " · ".join(scene.notes), color=MUTED, fontsize=10.5,
                    ha="right", va="center")
        if scene.demo:
            ax.text(1888, 1062, "SIMULATED PRICES · LAYOUT TEST", color=DOWN, fontsize=12,
                    fontweight="bold", ha="right", va="center")
            ax.text(TX + TW / 2, TY + TH / 2, "SIMULATED PRICES", color=(1, 1, 1, 0.13),
                    fontsize=88, fontweight="bold", ha="center", va="center", rotation=18, zorder=5)
            self.fig.text(320 / W, 1 - 380 / H, "SIMULATED", color=(1, 1, 1, 0.13), fontsize=52,
                          fontweight="bold", ha="center", va="center", rotation=18, zorder=50)

        self._last_label = {}

    # --- per-state numbers -------------------------------------------------
    def state(self, k: int):
        """k = 0 is yesterday; k >= 1 is today's keyframe k-1."""
        s = self.s
        if k == 0:
            return s.yday_pct, s.yday_dollar, s.yday_index_pct
        return s.key_pct[k - 1], s.key_dollar[k - 1], s.key_index_pct[k - 1]

    def draw_chart(self, k_a: int, k_b: int, e: float):
        s = self.s
        for a in self.chart_artists:
            a.remove()
        self.chart_artists = []
        today_phase = not (k_a == 0 and k_b == 0)
        y_alpha = 1.0 if not today_phase else (1 - 0.65 * e if k_a == 0 else 0.35)
        reveal = -1.0
        if today_phase:
            ma = -1.0 if k_a == 0 else s.key_minutes[k_a - 1]
            mb = s.key_minutes[k_b - 1]
            reveal = ma + (mb - ma) * e
        self._candles(s.candles_y, self.x_y, None, y_alpha)
        if reveal >= 0:
            self._candles(s.candles_t, self.x_t, reveal, 1.0)
        # reference line: previous close of the session being shown
        use_today = today_phase and (k_a != 0 or e >= 0.5)
        ref = s.ref_t if use_today else s.ref_y
        ln = self.cax.axhline(ref, color="#d9b45a", linewidth=1, linestyle=(0, (4, 4)))
        tx = self.cax.text(0.99, ref, f"prev close {ref:,.0f}", color="#d9b45a", fontsize=9.5,
                           va="bottom", ha="right", transform=self.cax.get_yaxis_transform())
        self.chart_artists += [ln, tx]

    def _candles(self, c: pd.DataFrame, xs, reveal, alpha):
        if c.empty:
            return
        from matplotlib.collections import LineCollection, PolyCollection
        vis = np.ones(len(c), bool) if reveal is None else (c["m"].to_numpy() + 15 <= reveal + 1e-6)
        if not vis.any():
            return
        c, xs = c[vis], xs[vis]
        up = (c["Close"] >= c["Open"]).to_numpy()
        col = np.where(up, UP, DOWN)
        from matplotlib.colors import to_rgba
        rgba = [to_rgba(x, alpha) for x in col]
        wicks = LineCollection([[(x, l), (x, h)] for x, l, h in zip(xs, c["Low"], c["High"])],
                               colors=rgba, linewidths=1.1)
        bodies = PolyCollection([[(x - 0.36, o), (x + 0.36, o), (x + 0.36, cl), (x - 0.36, cl)]
                                 for x, o, cl in zip(xs, c["Open"], c["Close"])],
                                facecolors=rgba, edgecolors=rgba, linewidths=0.6)
        self.cax.add_collection(wicks)
        self.cax.add_collection(bodies)
        self.chart_artists += [wicks, bodies]
        if reveal is not None:                       # live price tag at the right edge
            last = c["Close"].iloc[-1]
            ref = self.s.ref_t
            tag = self.cax.text(1.0, last, f" {last:,.0f} ", transform=self.cax.get_yaxis_transform(),
                                color="white", fontsize=9.5, fontweight="bold", va="center", ha="left",
                                bbox=dict(boxstyle="square,pad=0.15", fc=UP if last >= ref else DOWN, lw=0),
                                clip_on=False)
            self.chart_artists.append(tag)

    def frame(self, k_a: int, k_b: int, t: float, dur: float):
        """Draw one frame of the transition k_a -> k_b at time t (0..dur)."""
        s = self.s
        pa, da, ia = self.state(k_a)
        pb, db, ib = self.state(k_b)
        e = float(smooth(t / dur)) if dur > 0 else 1.0
        if k_a == k_b:
            ei = np.ones(len(self.order))
        else:
            ei = smooth((t - self.delay) / T_TILE)
        oa, ob = pa[self.order], pb[self.order]
        self.tiles.set_facecolor(pct_colour(oa + (ob - oa) * ei))
        # tile % labels snap once each tile has passed half way
        for j, i, tp in self.labels:
            if tp is None:
                continue
            txt = fmt_pct(pb[i] if ei[j] >= 0.5 else pa[i])
            if self._last_label.get(i) != txt:
                tp.set_text(txt)
                self._last_label[i] = txt
        cur = pb if e >= 0.5 else pa
        cur_d = db if e >= 0.5 else da
        for sct, tp in self.sector_txt:
            mm = sct["members"]
            w_ = s.layout_cap[mm]
            sp = float(np.nansum(cur[mm] * w_) / np.nansum(w_))
            tp.set_text(fmt_pct(sp))
            tp.set_color(UP if sp >= 0 else DOWN)
        # header
        kk = k_b if e >= 0.5 else k_a
        if kk == 0:
            self.pill_txt.set_text(f"YESTERDAY · {s.yday:%a %-d %b}".upper())
            self.pill.set_facecolor("#2a2f3d")
            self.clock.set_text("CLOSE")
        else:
            self.pill_txt.set_text(f"TODAY · {s.today:%a %-d %b}".upper())
            self.pill.set_facecolor("#b8323a" if s.key_index_pct[-1] < 0 else "#23874a")
            self.clock.set_text(f"{s.key_labels[kk - 1]} ET")
        ip = ia + (ib - ia) * e
        self.idx_txt.set_text(fmt_pct(ip))
        self.idx_txt.set_color(UP if ip >= 0 else DOWN)
        money = float(np.nansum(da) + (np.nansum(db) - np.nansum(da)) * e)
        self.money.set_text(fmt_money(money).replace("T", " trillion").replace("B", " billion"))
        self.money.set_color(UP if money >= 0 else DOWN)
        order = np.argsort(np.nan_to_num(cur_d))
        drags = [i for i in order[:4] if cur_d[i] < 0]
        lifts = [i for i in order[::-1][:4] if cur_d[i] > 0]
        for k, tt in enumerate(self.drag_txt):
            tt.set_text(f"{s.tickers[drags[k]]:<6}{fmt_money(cur_d[drags[k]]):>8}" if k < len(drags) else "")
            tt.set_color(DOWN)
        for k, tt in enumerate(self.lift_txt):
            tt.set_text(f"{s.tickers[lifts[k]]:<6}{fmt_money(cur_d[lifts[k]]):>8}" if k < len(lifts) else "")
            tt.set_color(UP)
        self.draw_chart(k_a, k_b, e)

    def rgb(self) -> bytes:
        self.fig.canvas.draw()
        return bytes(self.fig.canvas.buffer_rgba())

    def png(self, path: Path):
        self.fig.savefig(path, dpi=100, facecolor=BG)


def timeline(scene: Scene):
    """List of (k_a, k_b, duration) segments."""
    K = len(scene.key_labels)
    seg = [(0, 0, T_YEST_HOLD), (0, 1, T_TRANS)]
    for k in range(1, K + 1):
        seg.append((k, k, T_HOLD if k < K else T_END_HOLD))
        if k < K:
            seg.append((k, k + 1, T_TRANS))
    return seg


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


MAX_BYTES = 4_500_000   # Notion's free plan accepts files up to 5 MiB


def shrink_to_fit(out: Path) -> None:
    """Re-encode at lower quality until the file fits under MAX_BYTES, so the
    digest can always put it into Notion."""
    for crf in (24, 28, 32):
        if out.stat().st_size <= MAX_BYTES:
            return
        log(f"Video is {out.stat().st_size / 1e6:.1f} MB; re-encoding at CRF {crf}")
        tmp = out.with_name(out.stem + ".small.mp4")
        subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(out), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(out)


def render(scene: Scene, out: Path, poster: Path | None = None) -> None:
    r = Renderer(scene)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for k_a, k_b, dur in timeline(scene):
        frames = max(1, round(dur * FPS))
        for f in range(frames):
            r.frame(k_a, k_b, (f + 1) / frames * dur, dur)
            proc.stdin.write(r.rgb())
            n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    tmp.replace(out)
    shrink_to_fit(out)
    log(f"Video written: {out} ({n} frames, {n / FPS:.1f} s, {out.stat().st_size / 1e6:.1f} MB)")
    if poster:
        r.png(poster)
        log(f"Poster written: {poster}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="session date YYYY-MM-DD (default: latest completed)")
    ap.add_argument("--out-dir", default=str(HERE / "video"))
    ap.add_argument("--demo", action="store_true", help="simulated prices (layout test)")
    ap.add_argument("--constituents-csv", help="demo only: local constituents CSV")
    ap.add_argument("--frame", type=float, nargs="*",
                    help="save still PNGs at these seconds instead of a video")
    args = ap.parse_args()

    scene = build_demo(args.constituents_csv) if args.demo else build_real(args.date)
    out_dir = Path(args.out_dir)
    stem = f"heatmap-{scene.today.date()}" + ("-demo" if scene.demo else "")
    if args.frame is not None:
        r = Renderer(scene)
        segs = timeline(scene)
        for sec in args.frame:
            acc = 0.0
            for k_a, k_b, dur in segs:
                if sec <= acc + dur:
                    r.frame(k_a, k_b, sec - acc, dur)
                    break
                acc += dur
            p = out_dir / f"{stem}-{sec:05.2f}s.png"
            out_dir.mkdir(parents=True, exist_ok=True)
            r.png(p)
            log(f"Still written: {p}")
        return
    render(scene, out_dir / f"{stem}.mp4", out_dir / f"{stem}.png")
    (out_dir / "latest.txt").write_text(f"{stem}.mp4\n")


if __name__ == "__main__":
    main()
