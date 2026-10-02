#!/usr/bin/env python3
"""Fit M2-S or M2-QI with CUDA and autograd.

This script intentionally accepts a compact NPZ event file produced after the
book/event audit. Arrays: times (seconds relative to window start), types
(0=ask, 1=bid), values (continuous post-event state), end (seconds), and knots.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from wselob.continuous import ContinuousData
from wselob.gpu_continuous import (
    TorchContinuousData, device_report, parameters_from_torch, require_cuda,
    torch_continuous_log_likelihood,
)
from wselob.splines import SplineBasis


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("events", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--smoothness", type=float, default=1e-2)
    p.add_argument("--chunk-size", type=int, default=2048)
    args = p.parse_args()
    device = require_cuda(args.device)
    print(f"EXECUTION pid={os.getpid()} device={device} gpu={device.type == 'cuda'}", flush=True)
    print("DEVICE", device_report(device), flush=True)
    raw = np.load(args.events)
    basis = SplineBasis(raw["knots"], int(raw["degree"]))
    data = ContinuousData(raw["times"], raw["types"], raw["values"],
                          basis.design(raw["values"]), float(raw["end"]), basis)
    td = TorchContinuousData.from_numpy(data, device)
    print(
        f"TENSORS times={td.times.device} types={td.types.device} "
        f"basis={td.basis.device}",
        flush=True,
    )
    r = basis.n_basis
    rate = np.bincount(data.types[data.times >= 0], minlength=2) / data.end
    log_nu = torch.nn.Parameter(torch.log(torch.as_tensor(np.maximum(rate * 0.5, 1e-4),
                                                               device=device, dtype=torch.float64)))
    alpha = torch.nn.Parameter(torch.full((2, 2, r), -2.0, device=device, dtype=torch.float64))
    beta = torch.nn.Parameter(torch.full((2, 2, r), 2.0, device=device, dtype=torch.float64))
    optimizer = torch.optim.Adam([log_nu, alpha, beta], lr=args.lr)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        ll = torch_continuous_log_likelihood(
            td, log_nu, alpha, beta, args.cutoff, args.smoothness, args.chunk_size
        )
        (-ll).backward()
        torch.nn.utils.clip_grad_norm_([log_nu, alpha, beta], 100.0)
        optimizer.step()
        if step % 50 == 0 or step + 1 == args.steps:
            memory = ""
            if device.type == "cuda":
                memory = (
                    f" allocated_mib={torch.cuda.memory_allocated(device)/2**20:.1f}"
                    f" max_allocated_mib={torch.cuda.max_memory_allocated(device)/2**20:.1f}"
                )
            print(f"step={step:04d} penalized_loglik={ll.item():.6f}{memory}", flush=True)
    fitted = parameters_from_torch(log_nu, alpha, beta)
    np.savez(args.events.with_suffix(".m2-fit.npz"), nu=fitted.nu,
             alpha_coef=fitted.alpha_coef, beta_coef=fitted.beta_coef)


if __name__ == "__main__":
    main()
