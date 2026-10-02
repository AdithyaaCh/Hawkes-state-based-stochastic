#!/usr/bin/env python3
"""Reconstruct one stock-day and export continuous-state events for M2/M3."""
from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np

from wselob.config import load_intraday_window
from wselob.data_io import read_order_day, read_trade_day
from wselob.events import build_events, reconstruct
from wselob.splines import quantile_basis
from wselob.ticks import infer_tick_raw
from wselob.time_windows import local_window_ns
from wselob.trade_matching import match_trades


def subtract_minutes(value: time, minutes: int) -> time:
    anchor = datetime.combine(datetime(2000, 1, 1), value)
    return (anchor - timedelta(minutes=minutes)).time()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--stock", required=True,
                        choices=("KGHM", "PKNORLEN", "PKOBP", "PZU", "PEKAO"))
    parser.add_argument("--day", required=True, help="HDF5 day key, e.g. d20170110")
    parser.add_argument("--state", required=True, choices=("spread", "qi"))
    parser.add_argument("--n-basis", type=int, default=6)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_intraday_window()
    start_clock = time.fromisoformat(config["window_start"])
    end_clock = time.fromisoformat(config["window_end"])
    warmup_clock = subtract_minutes(start_clock, int(config["warmup_minutes"]))
    start_ns, end_ns = local_window_ns(args.day, start_clock, end_clock)
    history_ns, _ = local_window_ns(args.day, warmup_clock, end_clock)

    rows = read_order_day(args.data_dir, args.stock, args.day)
    messages, book_audit = reconstruct(rows)
    trades = read_trade_day(args.data_dir, args.stock, args.day)
    matches = match_trades(messages, trades)
    tick_raw = infer_tick_raw(messages) if args.state == "spread" else 1
    state_kind = "spread_continuous" if args.state == "spread" else "qi_continuous"
    events, event_audit = build_events(
        messages, state_kind, tick_raw, trade_matches=matches
    )
    selected = [event for event in events if history_ns <= event.time_ns <= end_ns]
    observed = [event for event in selected if event.time_ns >= start_ns]
    if not observed:
        raise RuntimeError("no modeled events in configured observation window")

    # Knot placement uses the observation window. In final chronological
    # experiments, a shared basis must be learned from training days and reused.
    observed_values = np.asarray([event.state_value for event in observed], dtype=float)
    basis = quantile_basis(observed_values, args.n_basis, args.degree)
    times = np.asarray([(event.time_ns - start_ns) * 1e-9 for event in selected])
    types = np.asarray([int(event.event_type) for event in selected], dtype=np.int8)
    values = np.asarray([event.state_value for event in selected], dtype=float)
    output = args.output or Path("artifacts/events") / (
        f"{args.stock}_{args.day}_{args.state}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        times=times,
        types=types,
        values=values,
        end=np.asarray((end_ns - start_ns) * 1e-9),
        knots=basis.knots,
        degree=np.asarray(basis.degree),
        tick_raw=np.asarray(tick_raw),
        window_start=np.asarray(config["window_start"]),
        window_end=np.asarray(config["window_end"]),
        warmup_minutes=np.asarray(config["warmup_minutes"]),
    )
    print(
        f"WROTE {output} observed_events={len(observed)} "
        f"warmup_events={len(selected) - len(observed)} "
        f"book_errors={book_audit.errors} ambiguous_excluded={event_audit.ambiguous_reductions}",
        flush=True,
    )


if __name__ == "__main__":
    main()
