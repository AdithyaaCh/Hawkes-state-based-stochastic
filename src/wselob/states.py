from __future__ import annotations

import numpy as np

from .types import BookSnapshot


QI_EDGES = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
QI_LABELS = ("sell++", "sell+", "neutral", "buy+", "buy++")


def qi_bin(qi: float) -> int:
    if not -1.0 <= qi <= 1.0:
        raise ValueError(f"QI outside [-1, 1]: {qi}")
    return min(int(np.searchsorted(QI_EDGES, qi, side="right") - 1), 4)


def spread_state(snapshot: BookSnapshot, tick_raw: int) -> int:
    if not snapshot.valid or snapshot.spread_raw is None:
        raise ValueError("spread state requires a valid uncrossed book")
    if tick_raw <= 0 or snapshot.spread_raw % tick_raw:
        raise ValueError("spread is not an integer number of inferred ticks")
    return 0 if snapshot.spread_raw == tick_raw else 1


def spread_ticks(snapshot: BookSnapshot, tick_raw: int) -> float:
    if not snapshot.valid or snapshot.spread_raw is None:
        raise ValueError("spread requires a valid uncrossed book")
    return snapshot.spread_raw / tick_raw

