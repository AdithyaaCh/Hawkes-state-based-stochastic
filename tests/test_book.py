import numpy as np

from wselob.book import OrderBook


DTYPE = np.dtype([
    ("time", "<i8"), ("order_date", "<i8"), ("order_id", "<i8"),
    ("price", "<i8"), ("volume", "<i8"), ("side", "i1"),
    ("order_type", "S2"), ("action_type", "S1"),
])


def row(t, oid, price, volume, side, action, order_type=b"2"):
    return np.array((t, 1, oid, price, volume, side, order_type, action), dtype=DTYPE)[()]


def test_book_add_modify_delete_and_best_levels():
    book = OrderBook()
    book.apply(row(1, 1, 100, 10, 1, b"A"), 0)
    book.apply(row(2, 2, 102, 20, 2, b"A"), 1)
    book.apply(row(3, 3, 99, 5, 1, b"A"), 2)
    snap = book.snapshot()
    assert (snap.best_bid, snap.best_ask, snap.bid_volume, snap.ask_volume) == (100, 102, 10, 20)
    result = book.apply(row(4, 1, 100, 4, 1, b"M"), 3)
    assert result.old_order.volume == 10
    assert book.snapshot().bid_volume == 4
    book.apply(row(5, 1, 100, 4, 1, b"D"), 4)
    assert book.snapshot().best_bid == 99


def test_clear_and_retransmit_upsert():
    book = OrderBook()
    book.apply(row(1, 1, 100, 10, 1, b"A"), 0)
    book.apply(row(2, 0, 0, 0, -1, b"F", b"-1"), 1)
    assert not book.orders
    book.apply(row(3, 1, 101, 7, 1, b"Y"), 2)
    assert book.snapshot().best_bid == 101
    book.apply(row(4, 1, 101, 5, 1, b"Y"), 3)
    assert book.snapshot().bid_volume == 5

