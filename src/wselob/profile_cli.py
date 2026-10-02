from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .intraday import estimate_profile, select_stationary_window


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the WSE intraday Hawkes-event profile")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--training-days", type=int, default=150)
    parser.add_argument("--day-step", type=int, default=1,
                        help="Use every nth training day; 1 is required for the final diagnostic")
    parser.add_argument("--bin-minutes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(
        f"EXECUTION pid={os.getpid()} device=cpu "
        f"workers={args.workers} task=order_book_reconstruction",
        flush=True,
    )
    profile = estimate_profile(args.data_dir, training_days=args.training_days,
                               day_step=args.day_step, bin_minutes=args.bin_minutes,
                               workers=args.workers)
    window = select_stationary_window(profile)
    print({
        "stock_days": profile.n_stock_days,
        "bin_minutes": profile.bin_minutes,
        "selected_start": window.start,
        "selected_end": window.end,
        "duration_minutes": window.n_bins * profile.bin_minutes,
        "cv": window.coefficient_of_variation,
        "relative_slope": window.relative_slope,
        "relative_range": window.relative_range,
    })
    for minute, normalized, rate, coverage in zip(
        profile.minute, profile.mean_normalized_rate, profile.mean_rate, profile.coverage
    ):
        if coverage >= 0.5:
            print(f"{minute//60:02.0f}:{minute%60:02.0f} {normalized:.6f} {rate:.6f} {coverage:.4f}")


if __name__ == "__main__":
    main()
