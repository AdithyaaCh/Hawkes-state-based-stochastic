from __future__ import annotations

from collections import Counter
from functools import reduce
from math import gcd
from .book import MessageResult


def infer_tick_from_prices(
    prices: list[int], min_observations: int = 20,
) -> int:
    """Estimate the minimum price increment from common observed gaps."""
    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    unique = sorted(set(prices))
    gaps = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    if len(gaps) < min_observations:
        raise ValueError(f"only {len(gaps)} positive quote gaps available")
    tick = reduce(gcd, (gap for gap, _ in Counter(gaps).most_common(20)))
    if tick <= 0:
        raise ValueError("failed to infer a positive tick")
    return tick


def infer_tick_raw(
    messages: list[MessageResult], min_observations: int = 20,
) -> int:
    """Infer the day's effective tick from valid best-quote price gaps.

    Missing price levels make the modal observed gap potentially larger than
    the permissible tick. We therefore take the GCD of the 20 most frequent
    gaps, which recovers the common divisor without letting every rare outlier
    dominate. The result remains a diagnostic that must be checked against the
    exchange's price-band tick schedule for final analysis.
    """
    prices: list[int] = []
    for msg in messages:
        if msg.post.valid:
            prices.extend((msg.post.best_bid, msg.post.best_ask))  # type: ignore[arg-type]
    return infer_tick_from_prices(prices, min_observations)
