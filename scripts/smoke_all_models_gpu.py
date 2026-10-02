#!/usr/bin/env python3
"""Fast one-day, eight-model CUDA smoke run (not a scientific fit).

The two independent state branches run in separate spawned processes. Each
branch fits M0→M1 and M2→M3, preserving the intended nested initialization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from wselob.daily_data import prepare_daily_arrays
from wselob.daily_fit import FitSettings, fit_branch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--stock", default="KGHM")
    parser.add_argument("--day", default="d20170110")
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke_8_models"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--history-events", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--event-batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--m0-maxiter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.workers, args.max_events, args.history_events,
           args.steps, args.m0_maxiter) <= 0 or args.cutoff <= 0:
        parser.error("workers, event counts, steps, iterations, and cutoff must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python process; check the active GPU venv")

    start = time.monotonic()
    print(f"DEVICE requested={args.device} available={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if args.device == 'cuda' else 'none'} "
          f"parent_pid={os.getpid()}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / f"{args.stock}_{args.day}_daily_v2.npz"
    if not cache.exists():
        print(f"PREPARE {args.stock} {args.day}: reconstructing book and matching trades", flush=True)
        daily = prepare_daily_arrays(args.data_dir, args.stock, args.day)
        daily.save(cache)
    else:
        print(f"PREPARE using cached {cache}", flush=True)
    print(f"PARALLEL workers={min(args.workers, 2)} branches=S,QI "
          f"events_per_branch={args.max_events} device={args.device}", flush=True)
    settings = FitSettings(
        max_events=args.max_events, history_events=args.history_events,
        cutoff_seconds=args.cutoff, steps=args.steps,
        m0_maxiter=args.m0_maxiter, seed=args.seed,
        event_batch_size=args.event_batch_size,
    )
    results = []
    # Spawn is required: forking after CUDA initialization is unsafe.
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=min(args.workers, 2),
                             mp_context=mp.get_context("spawn")) as pool:
        futures = {
            pool.submit(fit_branch, cache, kind, args.output, args.device, settings): kind
            for kind in ("S", "QI")
        }
        for future in as_completed(futures):
            branch = futures[future]
            fitted = future.result()
            for item in fitted:
                print(f"FIT {item['model']} objective={item['objective']:.3f} "
                      f"kind={item['objective_kind']} device={item['device']} "
                      f"seconds={item['elapsed_seconds']:.2f} "
                      f"peak_gpu_mib={item.get('peak_gpu_mib', 0.0):.1f}", flush=True)
            results.extend(fitted)
            print(f"BRANCH_DONE {branch}", flush=True)
    if len(results) != 8:
        raise RuntimeError(f"expected 8 model fits, found {len(results)}")
    summary = {
        "stock": args.stock, "day": args.day, "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "workers": min(args.workers, 2), "smoke_only": True,
        "max_events": args.max_events, "steps": args.steps,
        "elapsed_seconds": time.monotonic() - start,
        "fits": sorted(results, key=lambda row: row["model"]),
    }
    target = args.output / args.stock / args.day / "summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(target)
    print(f"SUCCESS 8/8 fits saved under {target.parent}", flush=True)
    print(f"SUMMARY {target}", flush=True)
    print(f"TOTAL_SECONDS {summary['elapsed_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
