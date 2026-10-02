"""Small, independent per-day fits for the eight specified model variants.

The stochastic objective is a Gaussian mean-shift variational ELBO, not the
exact marginal likelihood. This module deliberately reports that distinction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import time

import numpy as np
import torch

from .daily_data import DailyArrays
from .gpu_ou import variational_ou_elbo, variational_ou_elbo_chunked
from .gpu_sparse import (
    TorchEventData, exact_chunked_log_likelihood, exact_minibatch_log_likelihood,
)
from .m0 import M0Data, fit_m0
from .splines import SplineBasis

FIT_VERSION = "all-history-minibatch-v3"


@dataclass(frozen=True, slots=True)
class FitSettings:
    max_events: int = 128
    history_events: int = 64
    cutoff_seconds: float = 3.0
    steps: int = 3
    m0_maxiter: int = 5
    learning_rate: float = 0.01
    smoothness: float = 0.01
    seed: int = 2026
    event_batch_size: int = 256


def _slice_day(daily: DailyArrays, settings: FitSettings):
    observed = np.flatnonzero(daily.times >= 0)
    if len(observed) == 0:
        raise ValueError("daily data has no observed events")
    last = observed[min(settings.max_events, len(observed)) - 1]
    first = max(0, observed[0] - settings.history_events)
    take = slice(first, last + 1)
    end = min(float(daily.end), float(daily.times[last]) + 0.001)
    return take, end


def _spline_initial_from_m0(
    basis_values: np.ndarray, types: np.ndarray, discrete_states: np.ndarray,
    observed: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the discrete M0 log-kernels onto the continuous spline basis."""
    r = basis_values.shape[1]
    alpha_coef = np.empty((2, 2, r), dtype=float)
    beta_coef = np.empty((2, 2, r), dtype=float)
    differences = np.diff(np.eye(r), n=2, axis=0)
    penalty = ridge * differences.T @ differences + 1e-8 * np.eye(r)
    for source in range(2):
        mask = observed & (types == source)
        design = basis_values[mask]
        states = discrete_states[mask]
        if len(design) == 0:
            design = np.ones((1, r)) / r
            states = np.zeros(1, dtype=int)
        system = design.T @ design + penalty
        for target in range(2):
            alpha_response = np.log(np.maximum(alpha[source, states, target], 1e-12))
            beta_response = np.log(np.maximum(beta[source, states, target], 1e-12))
            alpha_coef[source, target] = np.linalg.solve(
                system, design.T @ alpha_response,
            )
            beta_coef[source, target] = np.linalg.solve(
                system, design.T @ beta_response,
            )
    return np.clip(alpha_coef, -20, 20), np.clip(beta_coef, -20, 20)


def _fit_torch(
    data: TorchEventData, initial: tuple[np.ndarray, np.ndarray, np.ndarray],
    *, stochastic: bool, settings: FitSettings, seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | str | int]]:
    device = data.times.device
    dtype = torch.float64
    log_nu = torch.nn.Parameter(torch.tensor(initial[0], device=device, dtype=dtype))
    alpha = torch.nn.Parameter(torch.tensor(initial[1], device=device, dtype=dtype))
    beta = torch.nn.Parameter(torch.tensor(initial[2], device=device, dtype=dtype))
    parameters = [log_nu, alpha, beta]
    extra = {}
    if stochastic:
        log_kappa = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        eta_raw = torch.nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
        gamma = torch.nn.Parameter(torch.zeros((2, 2), device=device, dtype=dtype))
        # An OU path has exactly one latent value at a physical instant.  LOB
        # data can contain several events with the same timestamp, so giving
        # each row an independent variational mean makes the zero-time OU
        # transition singular. Optimize one mean per unique timestamp instead.
        latent_times, event_to_latent = torch.unique_consecutive(
            data.times, return_inverse=True,
        )
        latent_mean = torch.nn.Parameter(
            torch.zeros(len(latent_times), device=device, dtype=dtype)
        )
        parameters.extend((log_kappa, eta_raw, gamma, latent_mean))
        extra = dict(log_kappa=log_kappa, eta_raw=eta_raw,
                     gamma=gamma, mean_shift_unique=latent_mean,
                     latent_times=latent_times)
    optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(settings.steps, 1), eta_min=settings.learning_rate * 0.01,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    value = float("nan")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.monotonic()
    objective_trace = np.empty(settings.steps, dtype=np.float64)
    for step in range(settings.steps):
        optimizer.zero_grad(set_to_none=True)
        batch_size = min(settings.event_batch_size, len(data.observed))
        choice = torch.randint(
            len(data.observed), (batch_size,), generator=generator, device=device,
        )
        event_indices = data.observed[choice]
        if stochastic:
            unique_noise = torch.randn(
                (1, len(latent_times)), generator=generator,
                device=device, dtype=dtype,
            )
            result = variational_ou_elbo(
                data, log_nu, alpha, beta, log_kappa, eta_raw, gamma,
                latent_mean, unique_noise, settings.smoothness,
                latent_times=latent_times, event_to_latent=event_to_latent,
                event_indices=event_indices,
            )
            objective = result.elbo
        else:
            objective = exact_minibatch_log_likelihood(
                data, event_indices, log_nu, alpha, beta,
                settings.smoothness,
            )
        if not bool(torch.isfinite(objective)):
            detail = ""
            if stochastic:
                detail = (
                    f" event={float(result.event_term.detach().cpu())}"
                    f" baseline_comp={float(result.baseline_compensator.detach().cpu())}"
                    f" excitation_comp={float(result.excitation_compensator.detach().cpu())}"
                    f" kl={float(result.kl.detach().cpu())}"
                )
            raise FloatingPointError(
                f"non-finite fitting objective at step {step};{detail}"
            )
        (-objective).backward()
        bad_gradients = [
            index for index, parameter in enumerate(parameters)
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        if bad_gradients:
            raise FloatingPointError(
                f"non-finite gradients at step {step}, parameter indices {bad_gradients}"
            )
        torch.nn.utils.clip_grad_norm_(parameters, 100.0)
        optimizer.step()
        scheduler.step()
        value = float(objective.detach().cpu())
        objective_trace[step] = value
        # Keep exponentiated parameters inside a broad numerical domain. These
        # are safeguards, not fixed decay rates or state discretization.
        with torch.no_grad():
            log_nu.clamp_(-20.0, 20.0)
            alpha.clamp_(-20.0, 20.0)
            beta.clamp_(-20.0, 20.0)
            if stochastic:
                log_kappa.clamp_(np.log(1e-4), np.log(1e3))
                eta_raw.clamp_(-5.0, 5.0)
                gamma.clamp_(-5.0, 5.0)
                latent_mean.clamp_(-10.0, 10.0)
    with torch.no_grad():
        if stochastic and eta_raw[0] < 0:
            eta_raw.mul_(-1)
            gamma.mul_(-1)
            latent_mean.mul_(-1)
        if stochastic:
            evaluation_samples = 2
            evaluation_noise = torch.randn(
                (evaluation_samples, len(latent_times)), generator=generator,
                device=device, dtype=dtype,
            )
            evaluation = variational_ou_elbo_chunked(
                data, log_nu, alpha, beta, log_kappa, eta_raw, gamma,
                latent_mean, evaluation_noise, latent_times=latent_times,
                event_to_latent=event_to_latent, smoothness=settings.smoothness,
                batch_size=settings.event_batch_size,
            )
            evaluation_value = float(evaluation.elbo.cpu())
        else:
            evaluation_samples = 0
            evaluation = None
            evaluation_value = float(exact_chunked_log_likelihood(
                data, log_nu, alpha, beta, settings.smoothness,
                settings.event_batch_size,
            ).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with torch.no_grad():
        saved = {
            "log_nu": log_nu.cpu().numpy().copy(),
            "alpha_raw": alpha.cpu().numpy().copy(),
            "beta_raw": beta.cpu().numpy().copy(),
            "objective_trace": objective_trace,
        }
        saved.update({key: tensor.cpu().numpy().copy() for key, tensor in extra.items()})
        if stochastic:
            saved["mean_shift"] = latent_mean[event_to_latent].cpu().numpy().copy()
    return saved, {
        "implementation_version": FIT_VERSION,
        "objective": evaluation_value,
        "training_last_minibatch_objective": value,
        "objective_kind": ("full_history_MC_variational_ELBO" if stochastic
                           else "full_history_penalized_loglik"),
        "evaluation_mc_samples": evaluation_samples,
        "steps": settings.steps,
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
        "history_mode": "all_strict_past_minibatch",
        "event_batch_size": min(settings.event_batch_size, len(data.observed)),
        "objective_last_100_mean": float(np.mean(objective_trace[-100:])),
        "objective_previous_100_mean": (
            float(np.mean(objective_trace[-200:-100]))
            if settings.steps >= 200 else None
        ),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "peak_gpu_mib": (float(torch.cuda.max_memory_allocated(device) / 2**20)
                         if device.type == "cuda" else 0.0),
    }


def _write_fit(path: Path, params: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **params, metadata_json=json.dumps(metadata))
    temporary.replace(path)


def fit_branch(
    cache_path: str | Path, state_kind: str, output_dir: str | Path,
    device_name: str, settings: FitSettings,
) -> list[dict]:
    """Fit M0→M1 and M2→M3 for one state kind on one day."""
    if state_kind not in ("S", "QI"):
        raise ValueError("state_kind must be S or QI")
    torch.set_num_threads(1)
    daily = DailyArrays.load(cache_path)
    take, end = _slice_day(daily, settings)
    times = daily.times[take]
    types = daily.types[take]
    discrete = daily.spread_state[take] if state_kind == "S" else daily.qi_state[take]
    pre_state = daily.spread_pre_state[take] if state_kind == "S" else daily.qi_pre_state[take]
    values = daily.spread[take] if state_kind == "S" else daily.qi[take]
    knots = daily.spread_knots if state_kind == "S" else daily.qi_knots
    n_states = 2 if state_kind == "S" else 5
    device = torch.device(device_name)
    print(f"WORKER pid={os.getpid()} branch={state_kind} device={device} "
          f"gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none'}",
          flush=True)
    discrete_data = TorchEventData.from_full_arrays(
        times, types, discrete, end, device,
    )
    basis = SplineBasis(knots, daily.spline_degree).design(values)
    continuous_data = TorchEventData.from_full_arrays(
        times, types, np.zeros(len(times), dtype=np.int8), end,
        device, basis=basis,
    )
    root = Path(output_dir) / daily.stock / daily.day
    results = []

    m0_data = M0Data(times, types, discrete, 0.0, end, n_states)
    started = time.monotonic()
    print(f"START M0-{state_kind} pid={os.getpid()} events={m0_data.n_observed}", flush=True)
    m0 = fit_m0(m0_data, maxiter=settings.m0_maxiter)
    counts = np.zeros((2, n_states, n_states), dtype=np.int64)
    for typ, before, after, timestamp in zip(types, pre_state, discrete, times):
        if timestamp >= 0 and 0 <= before < n_states and 0 <= after < n_states:
            counts[int(typ), int(before), int(after)] += 1
    phi = np.full(counts.shape, np.nan)
    np.divide(counts, counts.sum(axis=-1, keepdims=True), out=phi,
              where=counts.sum(axis=-1, keepdims=True) > 0)
    metadata = {
        "implementation_version": FIT_VERSION,
        "model": "M0-" + state_kind, "objective": -float(m0.fun),
        "objective_kind": "exact_discrete_loglik", "success": bool(m0.success),
        "message": str(m0.message), "iterations": int(m0.nit),
        "elapsed_seconds": time.monotonic() - started,
        "events": int(np.sum(times >= 0)), "total_events": len(times),
        "device": "cpu", "smoke_truncated": settings.max_events < daily.observed_count,
    }
    _write_fit(root / f"M0-{state_kind}.npz", {
        "nu": m0.parameters.nu, "alpha": m0.parameters.alpha,
        "beta": m0.parameters.beta, "phi": phi, "phi_counts": counts,
    }, metadata)
    results.append(metadata)

    m0_initial = (
        np.log(m0.parameters.nu), np.log(m0.parameters.alpha),
        np.log(m0.parameters.beta),
    )
    print(f"START M1-{state_kind} pid={os.getpid()} device={device}", flush=True)
    m1_params, m1_meta = _fit_torch(
        discrete_data, m0_initial, stochastic=True,
        settings=settings, seed=settings.seed + (0 if state_kind == "S" else 1),
    )
    m1_meta.update(model="M1-" + state_kind, events=metadata["events"],
                   smoke_truncated=metadata["smoke_truncated"])
    _write_fit(root / f"M1-{state_kind}.npz", m1_params, m1_meta)
    results.append(m1_meta)

    alpha_initial, beta_initial = _spline_initial_from_m0(
        basis, types, discrete, times >= 0,
        m0.parameters.alpha, m0.parameters.beta,
    )
    m2_initial = (
        np.log(m0.parameters.nu),
        alpha_initial, beta_initial,
    )
    print(f"START M2-{state_kind} pid={os.getpid()} device={device}", flush=True)
    m2_params, m2_meta = _fit_torch(
        continuous_data, m2_initial, stochastic=False, settings=settings,
        seed=settings.seed + 2,
    )
    m2_meta.update(model="M2-" + state_kind, events=metadata["events"],
                   smoke_truncated=metadata["smoke_truncated"])
    _write_fit(root / f"M2-{state_kind}.npz", m2_params | {"knots": knots}, m2_meta)
    results.append(m2_meta)

    m3_initial = tuple(m2_params[key] for key in ("log_nu", "alpha_raw", "beta_raw"))
    print(f"START M3-{state_kind} pid={os.getpid()} device={device}", flush=True)
    m3_params, m3_meta = _fit_torch(
        continuous_data, m3_initial, stochastic=True, settings=settings,
        seed=settings.seed + 3,
    )
    m3_meta.update(model="M3-" + state_kind, events=metadata["events"],
                   smoke_truncated=metadata["smoke_truncated"])
    _write_fit(root / f"M3-{state_kind}.npz", m3_params | {"knots": knots}, m3_meta)
    results.append(m3_meta)
    return results
