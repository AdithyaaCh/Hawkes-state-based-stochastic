#!/usr/bin/env python3
"""Fail-fast CUDA visibility and allocation check."""
from __future__ import annotations

import argparse
import json

import torch

from wselob.gpu_continuous import device_report, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = require_cuda(args.device)
    # Allocate and execute a real matrix multiplication, then synchronize so a
    # successful report proves CUDA execution rather than mere driver discovery.
    x = torch.randn((2048, 2048), device=device)
    y = x @ x
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = device_report(device)
    report["matmul_device"] = str(y.device)
    report["matmul_checksum"] = float(y[0, 0].item())
    report["max_allocated_mib"] = (
        round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        if device.type == "cuda" else 0.0
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
