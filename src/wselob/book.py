from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Optional

import numpy as np

from .data_io import decode_byte
from .types import BookSnapshot


@dataclass(slots=True)
class RestingOrder:
    side: int
    price: int
    volume: int
    order_type: str


@dataclass(frozen=True, slots=True)
class MessageResult:
    row_number: int
    time_ns: int
    action: str
    side: int
    order_type: str
    price: int
    volume: int
    old_order: Optional[RestingOrder]
    pre: BookSnapshot
    post: BookSnapshot
    error: Optional[str] = None


class OrderBook:
    """Order-level book with O(log L) best-price updates.

    Price heaps are lazily cleaned. Volumes are aggregated by raw integer
    price, while order identities follow the WSE documentation.
    """

    def __init__(self) -> None:
        self.orders: dict[tuple[int, int], RestingOrder] = {}
        self.bid_volume: dict[int, int] = {}
        self.ask_volume: dict[int, int] = {}
        self._bid_heap: list[int] = []
        self._ask_heap: list[int] = []

    @staticmethod
    def _key(row: np.void) -> tuple[int, int]:
        return int(row["order_date"]), int(row["order_id"])

    @staticmethod
    def _normal_side(side: int) -> int:
        if side == 1:
            return 1
        if side in (2, 5):
            return 2
        raise ValueError(f"unsupported side {side}")

    def clear(self) -> None:
        self.orders.clear()
        self.bid_volume.clear()
        self.ask_volume.clear()
        self._bid_heap.clear()
        self._ask_heap.clear()

    def _levels(self, side: int) -> dict[int, int]:
        return self.bid_volume if side == 1 else self.ask_volume

    def _adjust_level(self, side: int, price: int, delta: int) -> None:
        levels = self._levels(side)
        before = levels.get(price, 0)
        after = before + delta
        if after < 0:
            raise ValueError(f"negative aggregate at side={side}, price={price}")
        if after == 0:
            levels.pop(price, None)
        else:
            levels[price] = after
            if before == 0:
                heapq.heappush(self._bid_heap if side == 1 else self._ask_heap,
                               -price if side == 1 else price)

    def _best(self, side: int) -> tuple[Optional[int], int]:
        heap = self._bid_heap if side == 1 else self._ask_heap
        levels = self._levels(side)
        while heap:
            price = -heap[0] if side == 1 else heap[0]
            if levels.get(price, 0) > 0:
                return price, levels[price]
            heapq.heappop(heap)
        return None, 0

    def snapshot(self) -> BookSnapshot:
        bid, bid_q = self._best(1)
        ask, ask_q = self._best(2)
        return BookSnapshot(bid, ask, bid_q, ask_q)

    def _insert(self, key: tuple[int, int], order: RestingOrder) -> None:
        if key in self.orders:
            raise ValueError(f"duplicate order {key}")
        self.orders[key] = order
        self._adjust_level(order.side, order.price, order.volume)

    def _remove(self, key: tuple[int, int]) -> RestingOrder:
        old = self.orders.pop(key)
        self._adjust_level(old.side, old.price, -old.volume)
        return old

    def apply(self, row: np.void, row_number: int) -> MessageResult:
        pre = self.snapshot()
        action = decode_byte(row["action_type"])
        side_raw = int(row["side"])
        side = side_raw if side_raw == -1 else self._normal_side(side_raw)
        order_type = decode_byte(row["order_type"])
        price, volume = int(row["price"]), int(row["volume"])
        old: Optional[RestingOrder] = None
        error: Optional[str] = None
        key = self._key(row)

        try:
            if action == "F":
                self.clear()
            elif action == "A":
                self._insert(key, RestingOrder(side, price, volume, order_type))
            elif action == "Y":
                if key in self.orders:
                    old = self._remove(key)
                self._insert(key, RestingOrder(side, price, volume, order_type))
            elif action == "M":
                old = self._remove(key)
                new_side = old.side if side == -1 else side
                new_price = old.price if price == -1 else price
                new_volume = old.volume if volume == -1 else volume
                new_type = old.order_type if order_type == "-1" else order_type
                self._insert(key, RestingOrder(new_side, new_price, new_volume, new_type))
            elif action == "D":
                old = self._remove(key)
            else:
                raise ValueError(f"unsupported action {action}")
        except (KeyError, ValueError) as exc:
            error = str(exc)

        return MessageResult(
            row_number=row_number,
            time_ns=int(row["time"]),
            action=action,
            side=side,
            order_type=order_type,
            price=price,
            volume=volume,
            old_order=old,
            pre=pre,
            post=self.snapshot(),
            error=error,
        )

