#!/usr/bin/env python3
"""Show live NVIDIA GPU processes and memory, optionally filtering by PID."""
from __future__ import annotations

import argparse
import subprocess
import time


def snapshot(pid: int | None) -> tuple[str, bool]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if pid is not None:
        lines = [line for line in lines if line.split(",", 1)[0].strip() == str(pid)]
    return "\n".join(lines) if lines else "No matching GPU compute process", bool(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, help="PID printed by the fitting script")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    try:
        while True:
            output, active = snapshot(args.pid)
            print(f"{time.strftime('%H:%M:%S')} {output}", flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except FileNotFoundError:
        parser.exit(2, "nvidia-smi is not installed or not on PATH\n")
    except RuntimeError as exc:
        parser.exit(2, f"{exc}\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
