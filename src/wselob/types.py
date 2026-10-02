from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class EventType(IntEnum):
    ASK = 0
    BID = 1


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    best_bid: Optional[int]
    best_ask: Optional[int]
    bid_volume: int
    ask_volume: int

    @property
    def valid(self) -> bool:
        return (
            self.best_bid is not None
            and self.best_ask is not None
            and self.bid_volume > 0
            and self.ask_volume > 0
            and self.best_bid < self.best_ask
        )

    @property
    def spread_raw(self) -> Optional[int]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def qi(self) -> Optional[float]:
        total = self.bid_volume + self.ask_volume
        return (self.bid_volume - self.ask_volume) / total if total else None


@dataclass(frozen=True, slots=True)
class Event:
    time_ns: int
    event_type: EventType
    state_value: float
    pre_state_value: float
    source_action: str
    source_side: int
    classification: str
    source_rows: tuple[int, ...]

