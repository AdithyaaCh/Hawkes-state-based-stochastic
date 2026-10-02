from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize

from .types import Event


@dataclass(frozen=True, slots=True)
class M0Data:
    times: np.ndarray
    types: np.ndarray
    states: np.ndarray
    start: float
    end: float
    n_states: int

    @classmethod
    def from_events(
        cls,
        events: list[Event],
        start_ns: int | None = None,
        end_ns: int | None = None,
        n_states: int | None = None,
        history_start_ns: int | None = None,
    ) -> "M0Data":
        if not events:
            raise ValueError("at least one event is required")
        origin = start_ns if start_ns is not None else events[0].time_ns
        finish = end_ns if end_ns is not None else events[-1].time_ns
        history_start = history_start_ns if history_start_ns is not None else origin
        selected = [e for e in events if history_start <= e.time_ns <= finish]
        observed = [e for e in selected if e.time_ns >= origin]
        if not observed:
            raise ValueError("no events in observation window")
        states = np.asarray([int(e.state_value) for e in selected], dtype=np.int16)
        inferred = int(states.max()) + 1
        return cls(
            times=np.asarray([(e.time_ns - origin) * 1e-9 for e in selected]),
            types=np.asarray([int(e.event_type) for e in selected], dtype=np.int8),
            states=states,
            start=0.0,
            end=(finish - origin) * 1e-9,
            n_states=n_states or inferred,
        )

    @property
    def observed_mask(self) -> np.ndarray:
        return self.times >= self.start

    @property
    def n_observed(self) -> int:
        return int(self.observed_mask.sum())


@dataclass(frozen=True, slots=True)
class M0Parameters:
    nu: np.ndarray                 # [target]
    alpha: np.ndarray              # [source, state, target]
    beta: np.ndarray               # [source, state, target]


def unpack(raw: np.ndarray, n_states: int) -> M0Parameters:
    count = 2 * n_states * 2
    nu = np.exp(raw[:2])
    alpha = np.exp(raw[2:2 + count]).reshape(2, n_states, 2)
    beta = np.exp(raw[2 + count:2 + 2 * count]).reshape(2, n_states, 2)
    return M0Parameters(nu, alpha, beta)


def pack(params: M0Parameters) -> np.ndarray:
    return np.concatenate([
        np.log(params.nu).ravel(),
        np.log(params.alpha).ravel(),
        np.log(params.beta).ravel(),
    ])


def default_parameters(data: M0Data) -> M0Parameters:
    duration = max(data.end - data.start, 1e-6)
    rates = np.bincount(data.types, minlength=2) / duration
    nu = np.maximum(0.5 * rates, 1e-4)
    beta = np.full((2, data.n_states, 2), 10.0)
    # Integral alpha/beta starts at a conservative 0.1 per source-state.
    alpha = 0.1 * beta / max(2 * data.n_states, 1)
    return M0Parameters(nu, alpha, beta)


def log_likelihood(data: M0Data, params: M0Parameters) -> float:
    """Conditional M0 event log likelihood using the exact recursion."""
    if np.any(params.nu <= 0) or np.any(params.alpha < 0) or np.any(params.beta <= 0):
        return -np.inf
    r = np.zeros((2, data.n_states, 2), dtype=float)
    ll_events = 0.0
    previous = float(data.times[0]) if data.times[0] < data.start else data.start
    n = 0
    while n < len(data.times):
        time = data.times[n]
        dt = float(time - previous)
        if dt < 0:
            raise ValueError("event times must be sorted")
        r *= np.exp(-params.beta * dt)
        intensity = params.nu + np.sum(params.alpha * r, axis=(0, 1))
        if not np.all(np.isfinite(intensity)) or np.any(intensity <= 0):
            return -np.inf
        stop = int(np.searchsorted(data.times, time, side="right"))
        if time >= data.start:
            for k in range(n, stop):
                ll_events += np.log(intensity[int(data.types[k])])
        for k in range(n, stop):
            r[int(data.types[k]), int(data.states[k]), :] += 1.0
        previous = float(time)
        n = stop

    duration = data.end - data.start
    compensator = duration * float(params.nu.sum())
    # Each source event contributes independently to both target integrals.
    remaining = data.end - data.times
    for source in range(2):
        for state in range(data.n_states):
            mask = (data.types == source) & (data.states == state)
            if not np.any(mask):
                continue
            source_times = data.times[mask, None]
            horizon = remaining[mask, None]
            beta = params.beta[source, state, :][None, :]
            alpha = params.alpha[source, state, :][None, :]
            lower = np.maximum(data.start - source_times, 0.0)
            compensator += float(np.sum(
                alpha / beta * (np.exp(-beta * lower) - np.exp(-beta * horizon))
            ))
    return ll_events - compensator


@njit(cache=True)
def _negative_log_likelihood_raw(raw, times, types, states, end, n_states):
    block = 4 * n_states
    nu0, nu1 = np.exp(raw[0]), np.exp(raw[1])
    r = np.zeros((2, n_states, 2))
    ll_events = 0.0
    previous = times[0] if times[0] < 0.0 else 0.0
    n = 0
    while n < len(times):
        time = times[n]
        dt = time - previous
        intensity0, intensity1 = nu0, nu1
        for source in range(2):
            for state in range(n_states):
                for target in range(2):
                    ix = (source * n_states + state) * 2 + target
                    alpha = np.exp(raw[2 + ix])
                    beta = np.exp(raw[2 + block + ix])
                    r[source, state, target] *= np.exp(-beta * dt)
                    if target == 0:
                        intensity0 += alpha * r[source, state, target]
                    else:
                        intensity1 += alpha * r[source, state, target]
        stop = n + 1
        while stop < len(times) and times[stop] == time:
            stop += 1
        if time >= 0.0:
            for k in range(n, stop):
                ll_events += np.log(intensity0 if int(types[k]) == 0 else intensity1)
        for k in range(n, stop):
            source_n = int(types[k])
            state_n = int(states[k])
            r[source_n, state_n, 0] += 1.0
            r[source_n, state_n, 1] += 1.0
        previous = time
        n = stop

    compensator = end * (nu0 + nu1)
    for n in range(len(times)):
        source = int(types[n])
        state = int(states[n])
        lower = -times[n] if times[n] < 0.0 else 0.0
        horizon = end - times[n]
        for target in range(2):
            ix = (source * n_states + state) * 2 + target
            alpha = np.exp(raw[2 + ix])
            beta = np.exp(raw[2 + block + ix])
            compensator += alpha / beta * (
                np.exp(-beta * lower) - np.exp(-beta * horizon)
            )
    return compensator - ll_events


def negative_log_likelihood_raw(raw: np.ndarray, data: M0Data) -> float:
    return float(_negative_log_likelihood_raw(
        np.asarray(raw, dtype=float), data.times, data.types, data.states,
        data.end, data.n_states,
    ))


@njit(cache=True)
def _negative_log_likelihood_and_gradient(raw, times, types, states, end, n_states):
    """Exact tied-batch M0 likelihood and analytic gradient in log parameters."""
    block = 4 * n_states
    nu = np.exp(raw[:2])
    alpha = np.exp(raw[2:2 + block]).reshape((2, n_states, 2))
    beta = np.exp(raw[2 + block:2 + 2 * block]).reshape((2, n_states, 2))
    r = np.zeros((2, n_states, 2))
    q = np.zeros((2, n_states, 2))
    gradient = np.zeros(len(raw))
    value = 0.0
    previous = times[0] if times[0] < 0.0 else 0.0
    n = 0
    while n < len(times):
        t = times[n]
        dt = t - previous
        intensity = nu.copy()
        for source in range(2):
            for state in range(n_states):
                for target in range(2):
                    old_r = r[source, state, target]
                    decay = np.exp(-beta[source, state, target] * dt)
                    r[source, state, target] = old_r * decay
                    q[source, state, target] = (q[source, state, target] + dt * old_r) * decay
                    intensity[target] += alpha[source, state, target] * r[source, state, target]
        stop = n + 1
        while stop < len(times) and times[stop] == t:
            stop += 1
        if t >= 0.0:
            for k in range(n, stop):
                target = int(types[k])
                value -= np.log(intensity[target])
                inverse = 1.0 / intensity[target]
                gradient[target] -= nu[target] * inverse
                for source in range(2):
                    for state in range(n_states):
                        index = (source * n_states + state) * 2 + target
                        amplitude = alpha[source, state, target]
                        decay_rate = beta[source, state, target]
                        gradient[2 + index] -= amplitude * r[source, state, target] * inverse
                        gradient[2 + block + index] += (
                            amplitude * decay_rate * q[source, state, target] * inverse
                        )
        for k in range(n, stop):
            source = int(types[k])
            state = int(states[k])
            for target in range(2):
                r[source, state, target] += 1.0
        previous = t
        n = stop

    value += end * (nu[0] + nu[1])
    gradient[0] += end * nu[0]
    gradient[1] += end * nu[1]
    for n in range(len(times)):
        source = int(types[n])
        state = int(states[n])
        lower = -times[n] if times[n] < 0.0 else 0.0
        horizon = end - times[n]
        for target in range(2):
            index = (source * n_states + state) * 2 + target
            a = alpha[source, state, target]
            b = beta[source, state, target]
            e_lower = np.exp(-b * lower)
            e_upper = np.exp(-b * horizon)
            integral = a / b * (e_lower - e_upper)
            value += integral
            gradient[2 + index] += integral
            gradient[2 + block + index] += (
                -integral + a * (horizon * e_upper - lower * e_lower)
            )
    return value, gradient


def negative_log_likelihood_and_gradient(raw: np.ndarray, data: M0Data):
    return _negative_log_likelihood_and_gradient(
        np.asarray(raw, dtype=float), data.times, data.types, data.states,
        data.end, data.n_states,
    )


def fit_m0(data: M0Data, initial: M0Parameters | None = None, maxiter: int = 500):
    start = pack(initial or default_parameters(data))
    bounds = [(-20.0, 20.0)] * len(start)
    result = minimize(
        lambda raw: negative_log_likelihood_and_gradient(raw, data),
        start,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10},
    )
    result.parameters = unpack(result.x, data.n_states)
    return result


def transition_matrices(events: list[Event], n_states: int) -> tuple[np.ndarray, np.ndarray]:
    """Return MLE transition probabilities and raw counts [event, pre, post]."""
    counts = np.zeros((2, n_states, n_states), dtype=np.int64)
    for event in events:
        before, after = int(event.pre_state_value), int(event.state_value)
        if 0 <= before < n_states and 0 <= after < n_states:
            counts[int(event.event_type), before, after] += 1
    totals = counts.sum(axis=2, keepdims=True)
    phi = np.full(counts.shape, np.nan, dtype=float)
    np.divide(counts, totals, out=phi, where=totals > 0)
    return phi, counts
