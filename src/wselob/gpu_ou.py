from __future__ import annotations

from dataclasses import dataclass

import torch

from .gpu_sparse import (
    TorchEventData, all_history_event_log_intensity, event_log_intensity, excitation_compensator,
    source_coefficients,
)


@dataclass(frozen=True, slots=True)
class VariationalResult:
    elbo: torch.Tensor
    event_term: torch.Tensor
    baseline_compensator: torch.Tensor
    excitation_compensator: torch.Tensor
    kl: torch.Tensor


def _ou_path(
    times: torch.Tensor, kappa: torch.Tensor,
    mean_shift: torch.Tensor, noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample stationary OU at event times and return transition persistence."""
    if len(times) == 0:
        raise ValueError("at least one event is required")
    dt = torch.diff(times)
    if bool(torch.any(dt < 0)):
        raise ValueError("times must be sorted")
    persistence = torch.exp(-kappa * dt)
    innovations = torch.sqrt(torch.clamp(1.0 - persistence.square(), min=0.0))
    # Parallel prefix composition of affine maps z_n = a_n z_{n-1} + b_n.
    # This uses O(log N) GPU launches instead of one launch per event.
    a = torch.cat((torch.zeros_like(noise[:1]), persistence))
    b = torch.cat((noise[:1], innovations * noise[1:]))
    step = 1
    while step < len(times):
        next_a = torch.cat((a[:step], a[step:] * a[:-step]))
        next_b = torch.cat((b[:step], b[step:] + a[step:] * b[:-step]))
        a, b = next_a, next_b
        step *= 2
    return b + mean_shift, persistence


def _mean_shift_kl(mean_shift: torch.Tensor, persistence: torch.Tensor) -> torch.Tensor:
    first = mean_shift[0].square()
    residual = mean_shift[1:] - persistence * mean_shift[:-1]
    variance = torch.clamp(1.0 - persistence.square(), min=1e-12)
    return 0.5 * (first + (residual.square() / variance).sum())


def _baseline_compensator(
    data: TorchEventData, log_nu: torch.Tensor, eta: torch.Tensor,
    kappa: torch.Tensor, mean_shift: torch.Tensor,
    quadrature_nodes: int = 8,
) -> torch.Tensor:
    """Expected integrated stochastic baseline under an OU mean-shift family.

    The variational covariance equals the stationary OU prior covariance, so
    every interpolated Z_t has variance one. Gauss-Legendre quadrature only
    approximates the smooth one-dimensional time integral.
    """
    if quadrature_nodes not in (4, 8, 16, 32):
        raise ValueError("quadrature_nodes must be 4, 8, 16, or 32")
    from numpy.polynomial.legendre import leggauss
    nodes_np, weights_np = leggauss(quadrature_nodes)
    nodes = torch.as_tensor((nodes_np + 1) * 0.5, dtype=data.times.dtype,
                            device=data.times.device)
    weights = torch.as_tensor(weights_np * 0.5, dtype=data.times.dtype,
                              device=data.times.device)
    times = data.times
    # Integrate every interval between successive observed latent points,
    # clipping to [0, end]. History before zero contributes only as excitation.
    starts = torch.maximum(times[:-1], torch.zeros_like(times[:-1]))
    ends = torch.minimum(times[1:], torch.full_like(times[1:], data.end))
    valid = ends > starts
    if bool(torch.any(valid)):
        left_t, right_t = times[:-1][valid], times[1:][valid]
        span = right_t - left_t
        dt = ends[valid] - starts[valid]
        point_t = starts[valid, None] + dt[:, None] * nodes[None, :]
        u = point_t - left_t[:, None]
        v = right_t[:, None] - point_t
        r_u = torch.exp(-kappa * u)
        r_v = torch.exp(-kappa * v)
        r = torch.exp(-kappa * span)[:, None]
        denom = torch.clamp(1.0 - r.square(), min=1e-12)
        a = (r_u - r * r_v) / denom
        b = (r_v - r * r_u) / denom
        mean = a * mean_shift[:-1][valid, None] + b * mean_shift[1:][valid, None]
        values = torch.exp(log_nu[None, None, :] + eta[None, None, :] * mean[:, :, None]
                           + 0.5 * eta[None, None, :].square())
        inside = (values.sum(dim=2) * weights[None, :] * dt[:, None]).sum()
    else:
        inside = torch.zeros((), dtype=times.dtype, device=times.device)

    # The first interval after zero may begin before the first event. Use the
    # stationary prior mean there. Normally a 30-minute warm-up supplies events.
    first_gap = torch.clamp(times[0], min=0.0)
    before = first_gap * torch.exp(log_nu + 0.5 * eta.square()).sum()

    last_start = torch.maximum(times[-1], torch.zeros_like(times[-1]))
    last_dt = torch.clamp(data.end - last_start, min=0.0)
    future_u = last_dt * nodes
    future_mean = torch.exp(-kappa * future_u) * mean_shift[-1]
    future_values = torch.exp(log_nu[None, :] + eta[None, :] * future_mean[:, None]
                              + 0.5 * eta[None, :].square())
    after = last_dt * (future_values.sum(dim=1) * weights).sum()
    return before + inside + after


def variational_ou_elbo(
    data: TorchEventData,
    log_nu: torch.Tensor,
    alpha_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    log_kappa: torch.Tensor,
    eta_raw: torch.Tensor,
    gamma: torch.Tensor,
    mean_shift: torch.Tensor,
    noise: torch.Tensor,
    smoothness: float = 0.0,
    quadrature_nodes: int = 8,
    latent_times: torch.Tensor | None = None,
    event_to_latent: torch.Tensor | None = None,
    event_indices: torch.Tensor | None = None,
    scale_event_term: bool = True,
) -> VariationalResult:
    """A pathwise Gaussian variational bound for M1 or M3.

    By default ``noise`` has shape [samples, events]. For data with tied event
    times, pass the strictly increasing ``latent_times`` grid and the event-row
    mapping; then ``noise`` and ``mean_shift`` live on that unique-time grid.
    The q covariance is exactly the OU prior covariance; q differs only by an
    optimized mean path. This retains M0/M2 as the zero-loading special case.
    """
    kappa = torch.exp(log_kappa)
    # Sign symmetry is canonicalized after fitting; leaving eta unconstrained
    # makes eta=gamma=0 an exact finite representation of the nested baseline.
    eta = eta_raw
    alpha, beta = source_coefficients(data, alpha_raw, beta_raw)
    if (latent_times is None) != (event_to_latent is None):
        raise ValueError("latent_times and event_to_latent must be supplied together")
    path_times = data.times if latent_times is None else latent_times
    expanded_mean = mean_shift if event_to_latent is None else mean_shift[event_to_latent]
    event_terms = []
    persistence = torch.exp(-kappa * torch.diff(path_times))
    for sample in noise:
        z_path, _ = _ou_path(path_times, kappa, mean_shift, sample)
        z = z_path if event_to_latent is None else z_path[event_to_latent]
        if event_indices is None:
            term = event_log_intensity(
                data, log_nu, alpha, beta, z=z, eta=eta, gamma=gamma,
            )
        else:
            term = all_history_event_log_intensity(
                data, event_indices, log_nu, alpha, beta,
                z=z, eta=eta, gamma=gamma,
            )
            if scale_event_term:
                term = term * (len(data.observed) / len(event_indices))
        event_terms.append(term)
    event_term = torch.stack(event_terms).mean()
    baseline_comp = _baseline_compensator(
        data, log_nu, eta, kappa, expanded_mean, quadrature_nodes,
    )
    expected_amplitude = torch.exp(
        gamma[data.types, :] * expanded_mean[:, None]
        + 0.5 * gamma[data.types, :].square()
    )
    excitation_comp = excitation_compensator(
        data, alpha, beta, expected_exp_gamma=expected_amplitude,
    )
    kl = _mean_shift_kl(mean_shift, persistence)
    penalty = torch.zeros_like(kl)
    if data.basis is not None and smoothness:
        penalty = smoothness * (
            torch.diff(alpha_raw, n=2, dim=-1).square().sum()
            + torch.diff(beta_raw, n=2, dim=-1).square().sum()
        )
    elbo = event_term - baseline_comp - excitation_comp - kl - penalty
    return VariationalResult(elbo, event_term, baseline_comp, excitation_comp, kl)


def variational_ou_elbo_chunked(
    data: TorchEventData, log_nu: torch.Tensor, alpha_raw: torch.Tensor,
    beta_raw: torch.Tensor, log_kappa: torch.Tensor, eta: torch.Tensor,
    gamma: torch.Tensor, mean_shift: torch.Tensor, noise: torch.Tensor,
    *, latent_times: torch.Tensor | None = None,
    event_to_latent: torch.Tensor | None = None,
    smoothness: float = 0.0, quadrature_nodes: int = 8,
    batch_size: int = 256,
) -> VariationalResult:
    """Evaluate the complete variational objective in bounded pair memory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (latent_times is None) != (event_to_latent is None):
        raise ValueError("latent_times and event_to_latent must be supplied together")
    kappa = torch.exp(log_kappa)
    alpha, beta = source_coefficients(data, alpha_raw, beta_raw)
    path_times = data.times if latent_times is None else latent_times
    expanded_mean = mean_shift if event_to_latent is None else mean_shift[event_to_latent]
    sample_terms = []
    for sample in noise:
        z_path, _ = _ou_path(path_times, kappa, mean_shift, sample)
        z = z_path if event_to_latent is None else z_path[event_to_latent]
        term = torch.zeros((), dtype=data.times.dtype, device=data.times.device)
        for indices in data.observed.split(batch_size):
            term = term + all_history_event_log_intensity(
                data, indices, log_nu, alpha, beta, z=z, eta=eta, gamma=gamma,
            )
        sample_terms.append(term)
    event_term = torch.stack(sample_terms).mean()
    baseline_comp = _baseline_compensator(
        data, log_nu, eta, kappa, expanded_mean, quadrature_nodes,
    )
    expected_amplitude = torch.exp(
        gamma[data.types, :] * expanded_mean[:, None]
        + 0.5 * gamma[data.types, :].square()
    )
    excitation_comp = excitation_compensator(
        data, alpha, beta, expected_exp_gamma=expected_amplitude,
    )
    persistence = torch.exp(-kappa * torch.diff(path_times))
    kl = _mean_shift_kl(mean_shift, persistence)
    penalty = torch.zeros_like(kl)
    if data.basis is not None and smoothness:
        penalty = smoothness * (
            torch.diff(alpha_raw, n=2, dim=-1).square().sum()
            + torch.diff(beta_raw, n=2, dim=-1).square().sum()
        )
    elbo = event_term - baseline_comp - excitation_comp - kl - penalty
    return VariationalResult(elbo, event_term, baseline_comp, excitation_comp, kl)
