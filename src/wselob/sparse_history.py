from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SparseHistory:
    current: np.ndarray
    past: np.ndarray
    lag: np.ndarray
    cutoff_seconds: float
    n_observed: int

    @property
    def n_pairs(self) -> int:
        return len(self.current)


def build_sparse_history(times: np.ndarray, cutoff_seconds: float) -> SparseHistory:
    """Encode all strict-past source/target pairs within a time cutoff.

    Equal timestamps are excluded from mutual excitation. Events before zero
    remain available as history for observed events.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or np.any(np.diff(times) < 0):
        raise ValueError("times must be a sorted one-dimensional array")
    if cutoff_seconds <= 0:
        raise ValueError("cutoff_seconds must be positive")
    observed = np.flatnonzero(times >= 0)
    counts = np.zeros(len(observed), dtype=np.int64)
    lefts = np.searchsorted(times, times[observed] - cutoff_seconds, side="left")
    rights = np.searchsorted(times, times[observed], side="left")
    counts[:] = rights - lefts
    total = int(counts.sum())
    current = np.empty(total, dtype=np.int32)
    past = np.empty(total, dtype=np.int32)
    lag = np.empty(total, dtype=np.float64)
    offset = 0
    for i, (target, left, right) in enumerate(zip(observed, lefts, rights)):
        size = int(right - left)
        current[offset:offset + size] = target
        past[offset:offset + size] = np.arange(left, right)
        lag[offset:offset + size] = times[target] - times[left:right]
        offset += size
    return SparseHistory(current, past, lag, cutoff_seconds, len(observed))


def omitted_intensity_bound(
    times: np.ndarray, cutoff_seconds: float,
    max_amplitude: float, min_decay: float,
) -> float:
    """Conservative per-event bound on all omitted historical excitation."""
    if min_decay <= 0 or max_amplitude < 0:
        raise ValueError("positive decay and nonnegative amplitude required")
    return len(times) * max_amplitude * np.exp(-min_decay * cutoff_seconds)

