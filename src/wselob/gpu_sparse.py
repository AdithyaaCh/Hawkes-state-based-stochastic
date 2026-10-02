from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .sparse_history import SparseHistory


@dataclass(frozen=True, slots=True)
class TorchEventData:
    times: torch.Tensor
    types: torch.Tensor
    states: torch.Tensor
    basis: torch.Tensor | None
    end: float
    current: torch.Tensor
    past: torch.Tensor
    lag: torch.Tensor
    observed: torch.Tensor
    cutoff_seconds: float

    @classmethod
    def from_arrays(
        cls, times: np.ndarray, types: np.ndarray, states: np.ndarray,
        end: float, history: SparseHistory, device: torch.device,
        basis: np.ndarray | None = None,
    ) -> "TorchEventData":
        dtype = torch.float64
        return cls(
            torch.as_tensor(times, dtype=dtype, device=device),
            torch.as_tensor(types, dtype=torch.long, device=device),
            torch.as_tensor(states, dtype=torch.long, device=device),
            None if basis is None else torch.as_tensor(basis, dtype=dtype, device=device),
            float(end),
            torch.as_tensor(history.current, dtype=torch.long, device=device),
            torch.as_tensor(history.past, dtype=torch.long, device=device),
            torch.as_tensor(history.lag, dtype=dtype, device=device),
            torch.nonzero(torch.as_tensor(times >= 0, device=device), as_tuple=False).flatten(),
            history.cutoff_seconds,
        )

    @classmethod
    def from_full_arrays(
        cls, times: np.ndarray, types: np.ndarray, states: np.ndarray,
        end: float, device: torch.device, basis: np.ndarray | None = None,
    ) -> "TorchEventData":
        """Create data for exact all-history minibatches without stored pairs."""
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        history = SparseHistory(empty_i, empty_i, empty_f, float("inf"),
                                int(np.sum(np.asarray(times) >= 0)))
        return cls.from_arrays(times, types, states, end, history, device, basis)


def source_coefficients(
    data: TorchEventData, alpha_raw: torch.Tensor, beta_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Event-specific amplitude and decay, shape [event, target]."""
    if data.basis is None:
        alpha = torch.exp(alpha_raw[data.types, data.states, :])
        beta = torch.exp(beta_raw[data.types, data.states, :])
    else:
        alpha = torch.exp(torch.einsum("nr,ntr->nt", data.basis, alpha_raw[data.types]))
        beta = torch.exp(torch.einsum("nr,ntr->nt", data.basis, beta_raw[data.types]))
    return alpha, beta


def event_log_intensity(
    data: TorchEventData, log_nu: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor,
    z: torch.Tensor | None = None,
    eta: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum log intensities using sparse strict-past pairs."""
    observed = data.observed
    target_types = data.types[observed]
    if z is None:
        baseline = torch.exp(log_nu[target_types])
    else:
        assert eta is not None and gamma is not None
        baseline = torch.exp(log_nu[target_types] + eta[target_types] * z[observed])
    past = data.past
    current = data.current
    pair_target = data.types[current]
    pair_alpha = alpha[past, pair_target]
    pair_beta = beta[past, pair_target]
    if z is not None:
        pair_alpha = pair_alpha * torch.exp(gamma[data.types[past], pair_target] * z[past])
    contribution = pair_alpha * torch.exp(-pair_beta * data.lag)
    total = torch.zeros(len(data.times), dtype=data.times.dtype, device=data.times.device)
    total = total.scatter_add(0, current, contribution)
    return torch.log(baseline + total[observed]).sum()


def all_history_event_log_intensity(
    data: TorchEventData, event_indices: torch.Tensor, log_nu: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor,
    z: torch.Tensor | None = None, eta: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact event log term for selected targets using every strict-past event.

    This constructs only the selected target/source pairs, avoiding both a
    fixed history cutoff and the full O(N²) pair matrix. Equal timestamps are
    excluded with a left search, matching the paper's strict ``T_n < t`` rule.
    """
    event_indices = event_indices.to(device=data.times.device, dtype=torch.long)
    if event_indices.numel() == 0:
        return torch.zeros((), dtype=data.times.dtype, device=data.times.device)
    target_times = data.times[event_indices]
    counts = torch.searchsorted(data.times, target_times, right=False)
    total_pairs = int(counts.sum().item())
    batch_ids = torch.repeat_interleave(
        torch.arange(len(event_indices), device=data.times.device), counts,
    )
    offsets = torch.cumsum(counts, dim=0) - counts
    past = torch.arange(total_pairs, device=data.times.device)
    past = past - torch.repeat_interleave(offsets, counts)
    current = event_indices[batch_ids]
    target_types = data.types[event_indices]
    if z is None:
        baseline = torch.exp(log_nu[target_types])
    else:
        assert eta is not None and gamma is not None
        baseline = torch.exp(log_nu[target_types] + eta[target_types] * z[event_indices])
    pair_target = data.types[current]
    contribution = alpha[past, pair_target] * torch.exp(
        -beta[past, pair_target] * (data.times[current] - data.times[past])
    )
    if z is not None:
        contribution = contribution * torch.exp(
            gamma[data.types[past], pair_target] * z[past]
        )
    excitation = torch.zeros(
        len(event_indices), dtype=data.times.dtype, device=data.times.device,
    ).scatter_add(0, batch_ids, contribution)
    return torch.log(baseline + excitation).sum()


def exact_minibatch_log_likelihood(
    data: TorchEventData, event_indices: torch.Tensor, log_nu: torch.Tensor,
    alpha_raw: torch.Tensor, beta_raw: torch.Tensor,
    smoothness: float = 0.0, scale_event_term: bool = True,
) -> torch.Tensor:
    """Unbiased minibatch objective with an exact full compensator."""
    alpha, beta = source_coefficients(data, alpha_raw, beta_raw)
    event_term = all_history_event_log_intensity(
        data, event_indices, log_nu, alpha, beta,
    )
    if scale_event_term:
        event_term = event_term * (len(data.observed) / len(event_indices))
    ll = event_term - data.end * torch.exp(log_nu).sum()
    ll -= excitation_compensator(data, alpha, beta)
    if data.basis is not None and smoothness:
        ll -= smoothness * (
            torch.diff(alpha_raw, n=2, dim=-1).square().sum()
            + torch.diff(beta_raw, n=2, dim=-1).square().sum()
        )
    return ll


def exact_chunked_log_likelihood(
    data: TorchEventData, log_nu: torch.Tensor,
    alpha_raw: torch.Tensor, beta_raw: torch.Tensor,
    smoothness: float = 0.0, batch_size: int = 256,
) -> torch.Tensor:
    """Evaluate the complete untruncated likelihood in bounded memory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    alpha, beta = source_coefficients(data, alpha_raw, beta_raw)
    event_term = torch.zeros((), dtype=data.times.dtype, device=data.times.device)
    for indices in data.observed.split(batch_size):
        event_term = event_term + all_history_event_log_intensity(
            data, indices, log_nu, alpha, beta,
        )
    ll = event_term - data.end * torch.exp(log_nu).sum()
    ll -= excitation_compensator(data, alpha, beta)
    if data.basis is not None and smoothness:
        ll -= smoothness * (
            torch.diff(alpha_raw, n=2, dim=-1).square().sum()
            + torch.diff(beta_raw, n=2, dim=-1).square().sum()
        )
    return ll


def excitation_compensator(
    data: TorchEventData, alpha: torch.Tensor, beta: torch.Tensor,
    z: torch.Tensor | None = None, gamma: torch.Tensor | None = None,
    expected_exp_gamma: torch.Tensor | None = None,
) -> torch.Tensor:
    lower = torch.clamp(-data.times, min=0.0)[:, None]
    horizon = (data.end - data.times)[:, None]
    if expected_exp_gamma is not None:
        factor = expected_exp_gamma
    elif z is not None:
        assert gamma is not None
        factor = torch.exp(gamma[data.types, :] * z[:, None])
    else:
        factor = 1.0
    span = torch.clamp(horizon - lower, min=0.0)
    integral = alpha * factor / beta * torch.exp(-beta * lower) * (
        -torch.expm1(-beta * span)
    )
    return integral.sum()


def deterministic_log_likelihood(
    data: TorchEventData, log_nu: torch.Tensor,
    alpha_raw: torch.Tensor, beta_raw: torch.Tensor,
    smoothness: float = 0.0,
) -> torch.Tensor:
    alpha, beta = source_coefficients(data, alpha_raw, beta_raw)
    ll = event_log_intensity(data, log_nu, alpha, beta)
    ll -= data.end * torch.exp(log_nu).sum()
    ll -= excitation_compensator(data, alpha, beta)
    if data.basis is not None and smoothness:
        ll -= smoothness * (
            torch.diff(alpha_raw, n=2, dim=-1).square().sum()
            + torch.diff(beta_raw, n=2, dim=-1).square().sum()
        )
    return ll
