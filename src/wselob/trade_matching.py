from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np

from .book import MessageResult
from .types import EventType


@dataclass(frozen=True, slots=True)
class MatchedMarketEvent:
    time_ns: int
    event_type: EventType
    message: MessageResult
    trade_rows: tuple[int, ...]
    source_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TradeMatchResult:
    market_events: tuple[MatchedMarketEvent, ...]
    executed_message_rows: frozenset[int]
    matched_trades: int
    unmatched_trades: int


def match_trades(
    messages: list[MessageResult],
    trades: np.ndarray,
    tolerance_ns: int = 100_000,
) -> TradeMatchResult:
    """Associate book-origin trades with nearby resting-order reductions.

    Matching requires equal raw price and proximity in feed time. The resting
    side supplies aggressor direction: consuming asks is a bid event and
    consuming bids is an ask event. This deterministic matcher is deliberately
    auditable; its tolerance must be validated per stock in the data audit.
    """
    reductions: dict[int, list[MessageResult]] = defaultdict(list)
    for msg in messages:
        if msg.error or msg.action not in {"M", "D"} or msg.old_order is None:
            continue
        if msg.action == "M" and msg.volume >= msg.old_order.volume:
            continue
        reductions[msg.old_order.price].append(msg)
    time_index: dict[int, list[int]] = {}
    for price, values in reductions.items():
        values.sort(key=lambda m: (m.time_ns, m.row_number))
        time_index[price] = [m.time_ns for m in values]

    groups: dict[tuple[int, EventType], list[tuple[int, MessageResult]]] = defaultdict(list)
    unmatched = 0
    for trade_row, trade in enumerate(trades):
        if bytes(trade["trade_origin"]) != b"B":
            continue
        price = int(trade["price"])
        candidates = reductions.get(price, [])
        times = time_index.get(price, [])
        trade_time = int(trade["time"])
        left = bisect_left(times, trade_time - tolerance_ns)
        right = bisect_right(times, trade_time + tolerance_ns)
        if left == right:
            unmatched += 1
            continue
        msg = min(candidates[left:right], key=lambda m: (abs(m.time_ns - trade_time), m.row_number))
        resting_side = msg.old_order.side  # type: ignore[union-attr]
        event_type = EventType.BID if resting_side == 2 else EventType.ASK
        # Aggregate fills with the same exchange trade timestamp and direction.
        event_time = int(trade["trading_datetime"])
        groups[(event_time, event_type)].append((trade_row, msg))

    market: list[MatchedMarketEvent] = []
    executed: set[int] = set()
    for (time_ns, event_type), pairs in groups.items():
        msg = max((pair[1] for pair in pairs), key=lambda m: m.row_number)
        source_rows = tuple(sorted({pair[1].row_number for pair in pairs}))
        executed.update(source_rows)
        market.append(MatchedMarketEvent(
            time_ns, event_type, msg,
            tuple(pair[0] for pair in pairs), source_rows,
        ))
    market.sort(key=lambda e: (e.time_ns, int(e.event_type)))
    return TradeMatchResult(
        tuple(market), frozenset(executed),
        sum(len(e.trade_rows) for e in market), unmatched,
    )

