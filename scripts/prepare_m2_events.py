#!/usr/bin/env python3
"""Reconstruct one WSELOB day and export M2-S or M2-QI GPU input."""
from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np

from wselob.config import load_intraday_window
from wselob.data_io import STOCKS, read_order_day, read_trade_day
from wselob.events import build_events, reconstruct
from wselob.splines import quantile_basis
from wselob.ticks import infer_tick_raw
from wselob.time_windows import WARSAW, local_window_ns
from wselob.trade_matching import match_trades


def subtract_minutes(value: time, minutes: int) -> time:
    anchor = datetime.combine(datetime(2000, 1, 1), value)
    return (anchor - timedelta(minutes=minutes)).time()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--stock", choices=STOCKS, required=True)
    parser.add_argument("--day", required=True, help="HDF5 key, for example d20170110")
    parser.add_argument("--state", choices=("qi", "spread"), required=True)
    parser.add_argument("--n-basis", type=int, default=6)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/intraday_window.json"))
    args = parser.parse_args()

    config = load_intraday_window(args.config)
    window_start = time.fromisoformat(config["window_start"])
    window_end = time.fromisoformat(config["window_end"])
    warmup_minutes = int(config["warmup_minutes"])
    history_start = subtract_minutes(window_start, warmup_minutes)
    start_ns, end_ns = local_window_ns(args.day, window_start, window_end)
    history_ns, _ = local_window_ns(args.day, history_start, window_end)

    print(f"Reading and reconstructing {args.stock} {args.day}...", flush=True)
    rows = read_order_day(args.data_dir, args.stock, args.day)
    messages, audit = reconstruct(rows)
    trades = read_trade_day(args.data_dir, args.stock, args.day)
    matches = match_trades(messages, trades)
    tick_raw = infer_tick_raw(messages)
    state_kind = "qi_continuous" if args.state == "qi" else "spread_continuous"
    events, event_audit = build_events(
        messages, state_kind, tick_raw, trade_matches=matches
    )
    selected = [event for event in events if history_ns <= event.time_ns <= end_ns]
    observed_values = np.asarray([
        event.state_value for event in selected if event.time_ns >= start_ns
    ], dtype=np.float64)
    if len(observed_values) == 0:
        raise RuntimeError("no events found inside the configured observation window")
    basis = quantile_basis(observed_values, n_basis=args.n_basis, degree=args.degree)

    output = args.output or Path("artifacts") / (
        f"{args.stock}_{args.day}_m2_{args.state}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        times=np.asarray([(event.time_ns - start_ns) * 1e-9 for event in selected]),
        types=np.asarray([int(event.event_type) for event in selected], dtype=np.int8),
        values=np.asarray([event.state_value for event in selected], dtype=np.float64),
        end=np.asarray((end_ns - start_ns) * 1e-9),
        knots=basis.knots,
        degree=np.asarray(basis.degree),
        stock=np.asarray(args.stock),
        day=np.asarray(args.day),
        state=np.asarray(args.state),
        tick_raw=np.asarray(tick_raw),
        timezone=np.asarray(str(WARSAW)),
        window_start=np.asarray(config["window_start"]),
        window_end=np.asarray(config["window_end"]),
        warmup_minutes=np.asarray(warmup_minutes),
    )
    observed = sum(event.time_ns >= start_ns for event in selected)
    print(
        f"Wrote {output} with {observed} observed events and "
        f"{len(selected) - observed} warm-up events; "
        f"reconstruction_errors={audit.errors}, "
        f"unmatched_trades={matches.unmatched_trades}, "
        f"ambiguous_reductions_excluded={event_audit.ambiguous_reductions}",
        flush=True,
    )


if __name__ == "__main__":
    main()
