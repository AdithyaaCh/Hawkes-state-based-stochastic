#!/usr/bin/env python3
"""Print one on-demand progress and remaining-time estimate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wselob.data_io import STOCKS, day_keys, order_path
from wselob.daily_fit import FIT_VERSION


MODELS = tuple(f"M{family}-{state}" for family in range(4) for state in ("S", "QI"))


def _complete(output: Path, stock: str, day: str) -> bool:
    root = output / stock / day
    for model in MODELS:
        path = root / f"{model}.npz"
        if not path.is_file():
            return False
        try:
            with np.load(path, allow_pickle=False) as raw:
                metadata = json.loads(str(raw["metadata_json"].item()))
            if metadata.get("implementation_version") != FIT_VERSION:
                return False
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            return False
    return True


def _records(path: Path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return latest
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "stock" in record and "day" in record:
            latest[(record["stock"], record["day"])] = record
    return latest


def _scaled_pilot_seconds(path: Path, target_steps: int) -> list[float]:
    """Return timing estimates from complete pilot records.

    Pilot manifests are timing inputs, not reusable fit artifacts.  In
    particular, manifests written before FIT_VERSION was introduced do not
    carry ``implementation_version`` on each fit, but their elapsed time and
    step counts are still sufficient for an ETA.  Artifact compatibility is
    enforced separately by :func:`_complete`.
    """
    estimates = []
    for record in _records(path).values():
        if record.get("status") != "complete" or not record.get("fits"):
            continue
        measured = 0.0
        scaled = 0.0
        usable = True
        for fit in record["fits"]:
            elapsed = float(fit.get("elapsed_seconds", 0.0))
            measured += elapsed
            if fit["model"].startswith("M0-"):
                scaled += elapsed
            else:
                steps = int(fit.get("steps", 0))
                if steps <= 0:
                    usable = False
                    break
                scaled += elapsed * target_steps / steps
        if usable:
            estimates.append(
                max(float(record["elapsed_seconds"]) - measured, 0.0) + scaled
            )
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/training_fits"))
    parser.add_argument("--training-days", type=int, default=150)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--benchmark-manifest", type=Path,
        default=Path("artifacts/training_fits_test/manifest.jsonl"),
    )
    args = parser.parse_args()
    scheduled = [
        (stock, day)
        for stock in STOCKS
        for day in day_keys(order_path(args.data_dir, stock))[:args.training_days]
    ]
    completed = [item for item in scheduled if _complete(args.output, *item)]
    remaining = len(scheduled) - len(completed)
    records = _records(args.output / "manifest.jsonl")
    actual = [
        float(record["elapsed_seconds"])
        for key, record in records.items()
        if key in completed and record.get("status") == "complete"
        and "elapsed_seconds" in record
    ]
    source = "completed full-run days"
    durations = actual
    if not durations:
        durations = _scaled_pilot_seconds(args.benchmark_manifest, args.steps)
        source = "scaled pilot days"
    failures = sum(record.get("status") == "failed" for record in records.values())
    completed_fits = len(completed) * 8
    total_fits = len(scheduled) * 8
    print(f"COMPLETED_STOCK_DAYS {len(completed)}/{len(scheduled)}")
    print(f"COMPLETED_FITS {completed_fits}/{total_fits}")
    print(f"REMAINING_STOCK_DAYS {remaining}")
    print(f"LATEST_UNRESOLVED_FAILURES {failures}")
    if durations and remaining:
        mean_day = sum(durations) / len(durations)
        hours = mean_day * remaining / args.workers / 3600
        print(f"ESTIMATED_HOURS_REMAINING {hours:.2f}")
        print(f"ESTIMATE_SOURCE {source} samples={len(durations)} "
              f"mean_seconds_per_stock_day={mean_day:.1f} workers={args.workers}")
    elif remaining == 0:
        print("ESTIMATED_HOURS_REMAINING 0.00")
    else:
        print("ESTIMATED_HOURS_REMAINING unavailable")


if __name__ == "__main__":
    main()
