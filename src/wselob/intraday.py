from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .data_io import STOCKS, day_keys, order_path, read_order_day, read_trade_day
from .events import build_events, reconstruct
from .trade_matching import match_trades


WARSAW = ZoneInfo("Europe/Warsaw")


@dataclass(frozen=True, slots=True)
class Profile:
    minute: np.ndarray
    mean_normalized_rate: np.ndarray
    mean_rate: np.ndarray
    coverage: np.ndarray
    n_stock_days: int
    bin_minutes: int


@dataclass(frozen=True, slots=True)
class SelectedWindow:
    start_minute: int
    end_minute: int
    coefficient_of_variation: float
    relative_slope: float
    relative_range: float
    n_bins: int

    @staticmethod
    def clock(minute: int) -> str:
        return f"{minute // 60:02d}:{minute % 60:02d}"

    @property
    def start(self) -> str:
        return self.clock(self.start_minute)

    @property
    def end(self) -> str:
        return self.clock(self.end_minute)


def _minute_of_day(time_ns: int) -> float:
    dt = datetime.fromtimestamp(time_ns * 1e-9, tz=WARSAW)
    return dt.hour * 60 + dt.minute + (dt.second + dt.microsecond * 1e-6) / 60


def _stock_day_rates(job: tuple[str, str, str, int]) -> np.ndarray:
    data_dir_raw, stock, day, bin_minutes = job
    data_dir = Path(data_dir_raw)
    n_bins = 24 * 60 // bin_minutes
    rows = read_order_day(data_dir, stock, day)
    messages, _ = reconstruct(rows)
    trades = read_trade_day(data_dir, stock, day)
    matches = match_trades(messages, trades)
    events, _ = build_events(messages, "qi_continuous", 1, trade_matches=matches)
    counts = np.zeros(n_bins, dtype=float)
    for event in events:
        ix = int(_minute_of_day(event.time_ns) // bin_minutes)
        if 0 <= ix < n_bins:
            counts[ix] += 1
    return counts / (60 * bin_minutes)


def estimate_profile(
    data_dir: str | Path,
    stocks: tuple[str, ...] = STOCKS,
    training_days: int = 150,
    day_step: int = 1,
    bin_minutes: int = 5,
    workers: int = 1,
) -> Profile:
    """Estimate the Hawkes-event intraday profile using training days only."""
    data_dir = Path(data_dir)
    n_bins = 24 * 60 // bin_minutes
    jobs: list[tuple[str, str, str, int]] = []
    for stock in stocks:
        days = day_keys(order_path(data_dir, stock))[:training_days:day_step]
        for day in days:
            jobs.append((str(data_dir), stock, day, bin_minutes))
    if workers == 1:
        rates = [_stock_day_rates(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rates = list(pool.map(_stock_day_rates, jobs, chunksize=1))
    matrix = np.asarray(rates)
    active = matrix > 0
    coverage = active.mean(axis=0)
    mean_rate = matrix.mean(axis=0)
    normalized = np.full_like(matrix, np.nan)
    for row_ix, row in enumerate(matrix):
        positive = row[row > 0]
        scale = np.median(positive) if len(positive) else np.nan
        if np.isfinite(scale) and scale > 0:
            normalized[row_ix] = np.where(row > 0, row / scale, np.nan)
    valid_counts = np.sum(np.isfinite(normalized), axis=0)
    mean_norm = np.divide(
        np.nansum(normalized, axis=0), valid_counts,
        out=np.full(n_bins, np.nan), where=valid_counts > 0,
    )
    minutes = np.arange(n_bins) * bin_minutes
    return Profile(minutes, mean_norm, mean_rate, coverage, len(matrix), bin_minutes)


def select_stationary_window(
    profile: Profile,
    min_minutes: int = 120,
    min_coverage: float = 0.9,
    max_cv: float = 0.15,
    max_relative_slope: float = 0.20,
    max_relative_range: float = 0.35,
) -> SelectedWindow:
    """Choose the longest well-covered interval passing level and slope tests."""
    min_bins = int(np.ceil(min_minutes / profile.bin_minutes))
    candidates: list[SelectedWindow] = []
    n = len(profile.minute)
    for left in range(n):
        for right in range(left + min_bins, n + 1):
            values = profile.mean_normalized_rate[left:right]
            coverage = profile.coverage[left:right]
            if np.any(~np.isfinite(values)) or np.min(coverage) < min_coverage:
                continue
            mean = float(np.mean(values))
            cv = float(np.std(values) / mean)
            x = np.arange(len(values), dtype=float)
            slope = float(np.polyfit(x, values, 1)[0])
            relative_slope = abs(slope * max(len(values) - 1, 1) / mean)
            relative_range = float((np.max(values) - np.min(values)) / mean)
            if (cv <= max_cv and relative_slope <= max_relative_slope
                    and relative_range <= max_relative_range):
                candidates.append(SelectedWindow(
                    int(profile.minute[left]),
                    int(profile.minute[right - 1] + profile.bin_minutes),
                    cv, relative_slope, relative_range, right - left,
                ))
    if not candidates:
        raise ValueError("no interval satisfies the stationarity criteria")
    return max(candidates, key=lambda item: (item.n_bins, -item.coefficient_of_variation))
