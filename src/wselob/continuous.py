from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .splines import SplineBasis, second_difference_penalty
from .types import Event


@dataclass(frozen=True, slots=True)
class ContinuousData:
    times: np.ndarray
    types: np.ndarray
    values: np.ndarray
    basis_values: np.ndarray
    end: float
    basis: SplineBasis

    @classmethod
    def from_events(
        cls, events: list[Event], basis: SplineBasis,
        start_ns: int, end_ns: int, history_start_ns: int,
    ) -> "ContinuousData":
        chosen = [e for e in events if history_start_ns <= e.time_ns <= end_ns]
        if not any(e.time_ns >= start_ns for e in chosen):
            raise ValueError("no observed continuous-state events")
        times = np.asarray([(e.time_ns - start_ns) * 1e-9 for e in chosen])
        values = np.asarray([e.state_value for e in chosen])
        return cls(
            times, np.asarray([int(e.event_type) for e in chosen], dtype=np.int8),
            values, basis.design(values), (end_ns - start_ns) * 1e-9, basis,
        )


@dataclass(frozen=True, slots=True)
class ContinuousParameters:
    nu: np.ndarray                    # [target]
    alpha_coef: np.ndarray            # [source, target, basis]
    beta_coef: np.ndarray             # [source, target, basis]


def continuous_log_likelihood(
    data: ContinuousData,
    params: ContinuousParameters,
    max_history_seconds: float | None = None,
    smoothness: float = 0.0,
) -> tuple[float, float]:
    """M2 conditional log likelihood and conservative omitted-intensity bound.

    With ``max_history_seconds=None`` this is the direct exact O(N^2)
    reference. A finite cutoff accelerates fitting without changing the spline
    parameterization. The returned bound is a sum of omitted kernel values at
    event times and is used to validate the cutoff empirically.
    """
    if np.any(params.nu <= 0):
        return -np.inf, np.inf
    n = len(data.times)
    log_alpha = np.einsum("nr,str->nst", data.basis_values, params.alpha_coef)
    log_beta = np.einsum("nr,str->nst", data.basis_values, params.beta_coef)
    alpha, beta = np.exp(log_alpha), np.exp(log_beta)
    ll_event = 0.0
    omitted = 0.0
    for current in range(n):
        time = data.times[current]
        if time < 0:
            continue
        left = 0
        if max_history_seconds is not None:
            left = int(np.searchsorted(data.times, time - max_history_seconds, side="left"))
        right = int(np.searchsorted(data.times, time, side="left"))
        intensity = params.nu.copy()
        for past in range(left, right):
            source = int(data.types[past])
            lag = time - data.times[past]
            intensity += alpha[past, source, :] * np.exp(-beta[past, source, :] * lag)
        if left:
            # This computes the actual omitted tail for diagnostics. It can be
            # disabled later once a validated global cutoff has been frozen.
            for past in range(left):
                source = int(data.types[past])
                lag = time - data.times[past]
                omitted += float(np.sum(
                    alpha[past, source, :] * np.exp(-beta[past, source, :] * lag)
                ))
        value = intensity[int(data.types[current])]
        if value <= 0 or not np.isfinite(value):
            return -np.inf, np.inf
        ll_event += np.log(value)

    compensator = data.end * float(np.sum(params.nu))
    for past in range(n):
        source = int(data.types[past])
        lower = max(-data.times[past], 0.0)
        horizon = data.end - data.times[past]
        if horizon <= lower:
            continue
        a, b = alpha[past, source, :], beta[past, source, :]
        compensator += float(np.sum(a / b * (np.exp(-b * lower) - np.exp(-b * horizon))))
    penalty = smoothness * (
        second_difference_penalty(params.alpha_coef)
        + second_difference_penalty(params.beta_coef)
    )
    return ll_event - compensator - penalty, omitted

