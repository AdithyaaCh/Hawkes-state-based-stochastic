import numpy as np

from wselob.book import MessageResult, RestingOrder
from wselob.trade_matching import match_trades
from wselob.types import BookSnapshot, EventType


TRADE_DTYPE = np.dtype([
    ("time", "<i8"), ("trading_datetime", "<i8"), ("price", "<i8"),
    ("trade_origin", "S1"),
])


def test_trade_matches_nearby_ask_reduction_as_bid_event():
    pre = BookSnapshot(99, 100, 10, 10)
    post = BookSnapshot(99, 100, 10, 5)
    msg = MessageResult(7, 1_002_000, "M", 2, "2", 100, 5,
                        RestingOrder(2, 100, 10, "2"), pre, post)
    trades = np.array([(1_000_000, 1_000_000, 100, b"B")], dtype=TRADE_DTYPE)
    result = match_trades([msg], trades, tolerance_ns=5_000)
    assert result.matched_trades == 1
    assert result.market_events[0].event_type == EventType.BID
    assert result.executed_message_rows == frozenset({7})

