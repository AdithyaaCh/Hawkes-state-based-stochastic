from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .book import MessageResult, OrderBook
from .states import qi_bin, spread_state, spread_ticks
from .types import Event, EventType
from .trade_matching import TradeMatchResult


StateKind = Literal["spread_discrete", "qi_discrete", "spread_continuous", "qi_continuous"]


@dataclass(frozen=True, slots=True)
class Candidate:
    message: MessageResult
    event_type: EventType
    classification: str


@dataclass(slots=True)
class ReconstructionAudit:
    rows: int = 0
    errors: int = 0
    invalid_books: int = 0
    crossed_books: int = 0
    tied_rows: int = 0
    candidates: int = 0
    ambiguous_reductions: int = 0


def reconstruct(rows: np.ndarray) -> tuple[list[MessageResult], ReconstructionAudit]:
    book = OrderBook()
    out: list[MessageResult] = []
    audit = ReconstructionAudit(rows=len(rows))
    previous_time: int | None = None
    for number, row in enumerate(rows):
        msg = book.apply(row, number)
        out.append(msg)
        audit.errors += msg.error is not None
        audit.invalid_books += not msg.post.valid
        audit.crossed_books += (
            msg.post.best_bid is not None
            and msg.post.best_ask is not None
            and msg.post.best_bid >= msg.post.best_ask
        )
        audit.tied_rows += previous_time == msg.time_ns
        previous_time = msg.time_ns
    return out, audit


def _is_level_one_add(msg: MessageResult) -> bool:
    if msg.action != "A" or msg.order_type not in {"2", "K", "P"}:
        return False
    if msg.side == 1:
        return msg.pre.best_bid is None or msg.price >= msg.pre.best_bid
    if msg.side == 2:
        return msg.pre.best_ask is None or msg.price <= msg.pre.best_ask
    return False


def classify_candidate(msg: MessageResult, executed_rows: frozenset[int] = frozenset()) -> Candidate | None:
    if msg.error or msg.action in {"F", "Y"} or not msg.post.valid:
        return None
    if _is_level_one_add(msg):
        return Candidate(msg, EventType.BID if msg.side == 1 else EventType.ASK,
                         "level1_limit")
    if msg.action == "D" and msg.old_order is not None and msg.row_number not in executed_rows:
        old = msg.old_order
        if old.side == 1 and old.price == msg.pre.best_bid:
            return Candidate(msg, EventType.ASK, "level1_buy_cancel")
        if old.side == 2 and old.price == msg.pre.best_ask:
            return Candidate(msg, EventType.BID, "level1_sell_cancel")
    if msg.action == "M" and msg.old_order is not None and msg.row_number not in executed_rows:
        old = msg.old_order
        reduced = msg.volume >= 0 and msg.volume < old.volume
        if reduced and old.side == 1 and old.price == msg.pre.best_bid:
            return Candidate(msg, EventType.ASK, "ambiguous_buy_reduction")
        if reduced and old.side == 2 and old.price == msg.pre.best_ask:
            return Candidate(msg, EventType.BID, "ambiguous_sell_reduction")
    return None


def _state(msg: MessageResult, kind: StateKind, tick_raw: int) -> tuple[float, float]:
    if kind == "spread_discrete":
        return float(spread_state(msg.post, tick_raw)), float(spread_state(msg.pre, tick_raw))
    if kind == "spread_continuous":
        return spread_ticks(msg.post, tick_raw), spread_ticks(msg.pre, tick_raw)
    # All model variants must be fitted to the same marked point process.
    # QI is algebraically defined for some empty/crossed snapshots, but those
    # are not valid LOB states and the spread variants necessarily reject them.
    if not msg.post.valid or not msg.pre.valid:
        raise ValueError("QI state requires valid uncrossed pre/post books")
    post_qi, pre_qi = msg.post.qi, msg.pre.qi
    if post_qi is None or pre_qi is None:
        raise ValueError("QI state requires two nonempty best queues")
    if kind == "qi_discrete":
        return float(qi_bin(post_qi)), float(qi_bin(pre_qi))
    return post_qi, pre_qi


def build_events(
    messages: Iterable[MessageResult],
    state_kind: StateKind,
    tick_raw: int,
    include_ambiguous_reductions: bool = False,
    trade_matches: TradeMatchResult | None = None,
) -> tuple[list[Event], ReconstructionAudit]:
    events: list[Event] = []
    audit = ReconstructionAudit()
    for msg in messages:
        executed = trade_matches.executed_message_rows if trade_matches else frozenset()
        candidate = classify_candidate(msg, executed)
        if candidate is None:
            continue
        ambiguous = candidate.classification.startswith("ambiguous_")
        audit.ambiguous_reductions += ambiguous
        if ambiguous and not include_ambiguous_reductions:
            continue
        try:
            post, pre = _state(msg, state_kind, tick_raw)
        except ValueError:
            continue
        events.append(Event(
            time_ns=msg.time_ns,
            event_type=candidate.event_type,
            state_value=post,
            pre_state_value=pre,
            source_action=msg.action,
            source_side=msg.side,
            classification=candidate.classification,
            source_rows=(msg.row_number,),
        ))
    if trade_matches:
        for market in trade_matches.market_events:
            try:
                post, pre = _state(market.message, state_kind, tick_raw)
            except ValueError:
                continue
            events.append(Event(
                time_ns=market.time_ns,
                event_type=market.event_type,
                state_value=post,
                pre_state_value=pre,
                source_action="TRADE",
                source_side=2 if market.event_type == EventType.BID else 1,
                classification="buy_market" if market.event_type == EventType.BID else "sell_market",
                source_rows=market.source_rows,
            ))
    events.sort(key=lambda e: (e.time_ns, e.source_rows))
    audit.candidates = len(events)
    return events, audit


def tied_event_groups(events: list[Event]) -> list[list[Event]]:
    groups: list[list[Event]] = []
    for event in events:
        if not groups or groups[-1][0].time_ns != event.time_ns:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups
