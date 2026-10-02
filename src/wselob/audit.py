from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import time
from pathlib import Path

import numpy as np

from .data_io import read_order_day, read_trade_day_from_zip
from .events import build_events, reconstruct, tied_event_groups
from .ticks import infer_tick_raw
from .time_windows import local_window_ns
from .trade_matching import match_trades
from .config import load_intraday_window


def audit_day(data_dir: Path, stock: str, day: str,
              window_start: time = time(10, 45),
              window_end: time = time(14, 30)) -> dict[str, object]:
    rows = read_order_day(data_dir, stock, day)
    messages, book_audit = reconstruct(rows)
    tick = infer_tick_raw(messages)
    trades = read_trade_day_from_zip(data_dir / "WSELOB-2017.zip", stock, day)
    matches = match_trades(messages, trades)
    qi_events, event_audit = build_events(messages, "qi_discrete", tick, trade_matches=matches)
    spread_events, _ = build_events(messages, "spread_discrete", tick, trade_matches=matches)
    start_ns, end_ns = local_window_ns(day, window_start, window_end)
    qi_window = [e for e in qi_events if start_ns <= e.time_ns < end_ns]
    spread_window = [e for e in spread_events if start_ns <= e.time_ns < end_ns]
    tied = [g for g in tied_event_groups(qi_window) if len(g) > 1]
    valid = [m.post for m in messages if m.post.valid]
    spreads = np.asarray([s.spread_raw / tick for s in valid])
    return {
        "stock": stock,
        "day": day.removeprefix("/"),
        "tick_raw_empirical": tick,
        "price_scale": int(np.unique(rows["price_level"])[0]),
        "book": asdict(book_audit),
        "events": {
            "qi_primary_full_day": len(qi_events),
            "spread_primary_full_day": len(spread_events),
            "window": f"{window_start.isoformat(timespec='minutes')}-{window_end.isoformat(timespec='minutes')}",
            "qi_primary_window": len(qi_window),
            "spread_primary_window": len(spread_window),
            "ambiguous_reductions_excluded": event_audit.ambiguous_reductions,
            "matched_trade_rows": matches.matched_trades,
            "unmatched_trade_rows": matches.unmatched_trades,
            "tied_event_groups_window": len(tied),
            "classifications_window": dict(Counter(e.classification for e in qi_window)),
            "types_window": dict(Counter(e.event_type.name for e in qi_window)),
        },
        "spread_ticks": {
            "median": float(np.median(spreads)),
            "p95": float(np.quantile(spreads, 0.95)),
            "one_tick_share": float(np.mean(spreads == 1)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one reconstructed WSELOB day")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--stock", default="KGHM")
    parser.add_argument("--day", default="d20170110")
    configured = load_intraday_window()
    parser.add_argument("--window-start", type=time.fromisoformat,
                        default=time.fromisoformat(configured["window_start"]))
    parser.add_argument("--window-end", type=time.fromisoformat,
                        default=time.fromisoformat(configured["window_end"]))
    args = parser.parse_args()
    report = audit_day(args.data_dir, args.stock, args.day, args.window_start, args.window_end)
    import pprint
    pprint.pp(report, sort_dicts=False)


if __name__ == "__main__":
    main()
