from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import time
from pathlib import Path

import numpy as np

from .config import load_intraday_window
from .data_io import read_order_day, read_trade_day
from .events import build_events, reconstruct, tied_event_groups
from .splines import quantile_basis
from .ticks import infer_tick_raw
from .time_windows import local_window_ns
from .trade_matching import match_trades


@dataclass(frozen=True, slots=True)
class DailyArrays:
    times: np.ndarray
    types: np.ndarray
    spread: np.ndarray
    qi: np.ndarray
    spread_state: np.ndarray
    qi_state: np.ndarray
    spread_pre_state: np.ndarray
    qi_pre_state: np.ndarray
    spread_knots: np.ndarray
    qi_knots: np.ndarray
    spline_degree: int
    end: float
    stock: str
    day: str
    tick_raw: int
    tied_groups: int
    mixed_tied_groups: int
    unmatched_trades: int
    ambiguous_reductions: int
    reconstruction_errors: int

    @property
    def observed_count(self) -> int:
        return int(np.sum(self.times >= 0))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **{field.name: getattr(self, field.name) for field in fields(self)})
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "DailyArrays":
        with np.load(path, allow_pickle=False) as raw:
            values = {name: raw[name] for name in cls.__dataclass_fields__}
        for name in ("spline_degree", "end", "stock", "day", "tick_raw",
                     "tied_groups", "mixed_tied_groups", "unmatched_trades",
                     "ambiguous_reductions", "reconstruction_errors"):
            values[name] = values[name].item()
        return cls(**values)


def _aligned(reference, other) -> bool:
    if len(reference) != len(other):
        return False
    return all(
        a.time_ns == b.time_ns and a.event_type == b.event_type
        and a.source_rows == b.source_rows
        for a, b in zip(reference, other)
    )


def prepare_daily_arrays(
    data_dir: str | Path, stock: str, day: str,
    config_path: str | Path = "config/intraday_window.json",
    n_basis: int = 6, degree: int = 3,
) -> DailyArrays:
    config = load_intraday_window(config_path)
    start_time = time.fromisoformat(config["window_start"])
    end_time = time.fromisoformat(config["window_end"])
    start_ns, end_ns = local_window_ns(day, start_time, end_time)
    rows = read_order_day(data_dir, stock, day)
    messages, audit = reconstruct(rows)
    trades = read_trade_day(data_dir, stock, day)
    matches = match_trades(messages, trades)
    tick_raw = infer_tick_raw(messages)
    variants = {}
    event_audit = None
    for kind in ("qi_continuous", "spread_continuous", "qi_discrete", "spread_discrete"):
        events, candidate_audit = build_events(
            messages, kind, tick_raw, trade_matches=matches,
        )
        # Keep all same-day history before the stationary observation window.
        # It is conditioning information, not part of the scored event term.
        variants[kind] = [e for e in events if e.time_ns < end_ns]
        event_audit = candidate_audit
    reference = variants["qi_continuous"]
    for kind, events in variants.items():
        if not _aligned(reference, events):
            raise ValueError(f"{stock} {day}: {kind} event stream does not align")
    if not reference or not any(e.time_ns >= start_ns for e in reference):
        raise ValueError(f"{stock} {day}: no events in selected window")
    observed = np.asarray([e.time_ns >= start_ns for e in reference])
    spread = np.asarray([e.state_value for e in variants["spread_continuous"]])
    qi = np.asarray([e.state_value for e in reference])
    spread_basis = quantile_basis(spread[observed], n_basis, degree)
    qi_basis = quantile_basis(qi[observed], n_basis, degree)
    groups = [group for group in tied_event_groups(reference) if len(group) > 1]
    return DailyArrays(
        times=np.asarray([(e.time_ns - start_ns) * 1e-9 for e in reference]),
        types=np.asarray([int(e.event_type) for e in reference], dtype=np.int8),
        spread=spread, qi=qi,
        spread_state=np.asarray([int(e.state_value) for e in variants["spread_discrete"]], dtype=np.int8),
        qi_state=np.asarray([int(e.state_value) for e in variants["qi_discrete"]], dtype=np.int8),
        spread_pre_state=np.asarray([int(e.pre_state_value) for e in variants["spread_discrete"]], dtype=np.int8),
        qi_pre_state=np.asarray([int(e.pre_state_value) for e in variants["qi_discrete"]], dtype=np.int8),
        spread_knots=spread_basis.knots, qi_knots=qi_basis.knots,
        spline_degree=degree, end=(end_ns - start_ns) * 1e-9,
        stock=stock, day=day, tick_raw=tick_raw,
        tied_groups=len(groups),
        mixed_tied_groups=sum(len({e.event_type for e in group}) > 1 for group in groups),
        unmatched_trades=matches.unmatched_trades,
        ambiguous_reductions=event_audit.ambiguous_reductions if event_audit else 0,
        reconstruction_errors=audit.errors,
    )
