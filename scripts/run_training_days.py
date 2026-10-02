#!/usr/bin/env python3
"""Resumable training-day scheduler for 8 daily fits per stock-day.

Each worker owns one complete stock-day, reconstructs/caches it once, and fits
S then QI (M0→M1 and M2→M3). Multiple workers therefore parallelize across
independent days without violating nested initialization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import numpy as np

from wselob.daily_data import prepare_daily_arrays
from wselob.daily_fit import FIT_VERSION, FitSettings, fit_branch
from wselob.data_io import STOCKS, day_keys, order_path


MODELS = tuple(f"M{family}-{state}" for family in range(4) for state in ("S", "QI"))


def _benchmark_day_seconds(path: Path, target_steps: int) -> tuple[float, int] | None:
    """Scale pilot optimizer time while retaining measured preparation/M0 time."""
    if not path.is_file():
        return None
    latest: dict[tuple[str, str], dict] = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "complete" and record.get("fits"):
            latest[(record["stock"], record["day"])] = record
    estimates = []
    for record in latest.values():
        measured_fit = 0.0
        scaled_fit = 0.0
        usable = True
        for fit in record["fits"]:
            elapsed = float(fit.get("elapsed_seconds", 0.0))
            measured_fit += elapsed
            if fit["model"].startswith("M0-"):
                scaled_fit += elapsed
            else:
                pilot_steps = int(fit.get("steps", 0))
                if pilot_steps <= 0:
                    usable = False
                    break
                scaled_fit += elapsed * target_steps / pilot_steps
        if usable:
            overhead = max(float(record["elapsed_seconds"]) - measured_fit, 0.0)
            estimates.append(overhead + scaled_fit)
    if not estimates:
        return None
    return sum(estimates) / len(estimates), len(estimates)


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


def _run_day(task: tuple) -> dict:
    data_dir, output, cache_dir, stock, day, device_name, settings = task
    started = time.monotonic()
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{stock} {day}: CUDA unavailable in worker {os.getpid()}")
        torch.cuda.set_device(0)
    cache = cache_dir / f"{stock}_{day}_daily_v2.npz"
    if not cache.exists():
        daily = prepare_daily_arrays(data_dir, stock, day)
        daily.save(cache)
    fits = []
    for state in ("S", "QI"):
        fits.extend(fit_branch(cache, state, output, device_name, settings))
        if device_name == "cuda":
            torch.cuda.empty_cache()
    if len(fits) != 8:
        raise RuntimeError(f"{stock} {day}: expected 8 fits, got {len(fits)}")
    summary = {
        "stock": stock, "day": day, "pid": os.getpid(), "device": device_name,
        "elapsed_seconds": time.monotonic() - started, "fits": fits,
    }
    root = output / stock / day
    temporary = root / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(root / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/training_fits"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/daily_cache"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--training-days", type=int, default=150)
    parser.add_argument("--limit-days", type=int, default=None,
                        help="Limit total stock-days for a pilot; omit for all 750")
    parser.add_argument("--max-events", type=int, default=1_000_000_000)
    parser.add_argument("--history-events", type=int, default=1_000_000_000)
    parser.add_argument("--cutoff", type=float, default=30.0,
                        help="Deprecated compatibility option; all past events are used")
    parser.add_argument("--event-batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--m0-maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--benchmark-manifest", type=Path,
        default=Path("artifacts/training_fits_test/manifest.jsonl"),
        help="Completed pilot manifest used for the one-time runtime estimate",
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.training_days < 1 or args.event_batch_size < 1:
        parser.error("workers, training-days, and event-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    settings = FitSettings(
        max_events=args.max_events, history_events=args.history_events,
        cutoff_seconds=args.cutoff, steps=args.steps,
        m0_maxiter=args.m0_maxiter, seed=args.seed,
        event_batch_size=args.event_batch_size,
    )
    days = []
    for stock in STOCKS:
        for day in day_keys(order_path(args.data_dir, stock))[:args.training_days]:
            days.append((stock, day))
    days.sort(key=lambda item: (item[1], item[0]))
    if args.limit_days is not None:
        days = days[:args.limit_days]
    pending = [item for item in days if args.rerun or not _complete(args.output, *item)]
    skipped = len(days) - len(pending)
    print(f"SCHEDULE stock_days={len(days)} pending={len(pending)} skipped={skipped} "
          f"workers={args.workers} fits_pending={8 * len(pending)}", flush=True)
    print(f"LIKELIHOOD history=all_strict_past event_batch_size={args.event_batch_size} "
          f"implementation={FIT_VERSION}", flush=True)
    benchmark = _benchmark_day_seconds(args.benchmark_manifest, args.steps)
    if benchmark and pending:
        seconds_per_day, samples = benchmark
        expected_hours = seconds_per_day * len(pending) / args.workers / 3600
        print(
            f"EXPECTED_RUNTIME_HOURS {expected_hours:.2f} "
            f"for_stock_days={len(pending)} models={8 * len(pending)} "
            f"workers={args.workers} benchmark_days={samples}",
            flush=True,
        )
    elif pending:
        print(f"EXPECTED_RUNTIME_HOURS unavailable benchmark={args.benchmark_manifest}",
              flush=True)
    if args.device == "cuda":
        print(f"GPU {torch.cuda.get_device_name(0)}; all workers share cuda:0", flush=True)
    tasks = [
        (args.data_dir, args.output, args.cache_dir, stock, day,
         args.device, settings) for stock, day in pending
    ]
    failures = []
    run_started = time.monotonic()
    manifest = args.output / "manifest.jsonl"
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_run_day, task): (task[3], task[4]) for task in tasks}
        for future in as_completed(futures):
            stock, day = futures[future]
            try:
                result = future.result()
                with manifest.open("a") as handle:
                    handle.write(json.dumps({"status": "complete", **result}) + "\n")
                print(f"DONE {stock} {day} seconds={result['elapsed_seconds']:.1f}", flush=True)
            except Exception as error:
                failure = {"status": "failed", "stock": stock, "day": day,
                           "error": repr(error)}
                failures.append(failure)
                with manifest.open("a") as handle:
                    handle.write(json.dumps(failure) + "\n")
                print(f"FAILED {stock} {day}: {error!r}", flush=True)
    print(f"FINISHED completed={len(pending)-len(failures)} failed={len(failures)} "
          f"skipped={skipped} elapsed_hours={(time.monotonic()-run_started)/3600:.2f}",
          flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
