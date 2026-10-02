from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .m0 import M0Data, M0Parameters


@dataclass(frozen=True, slots=True)
class OUParameters:
    kappa: float
    eta: np.ndarray                 # [target]
    gamma: np.ndarray               # [source, target]

    @property
    def sigma(self) -> float:
        """Diffusion fixed by the Var(Z)=1 identification constraint."""
        return float(np.sqrt(2.0 * self.kappa))

    def validate(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.eta.shape != (2,) or self.gamma.shape != (2, 2):
            raise ValueError("eta and gamma have incompatible shapes")
        if self.eta[0] < 0:
            raise ValueError("eta[0] must be nonnegative to fix OU sign symmetry")


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions)


def _ou_step(z: np.ndarray, dt: float, kappa: float, rng: np.random.Generator) -> np.ndarray:
    if dt <= 0:
        return z.copy()
    persistence = np.exp(-kappa * dt)
    sd = np.sqrt(max(1.0 - persistence * persistence, 0.0))
    return persistence * z + sd * rng.standard_normal(len(z))


def m1_particle_log_likelihood(
    data: M0Data,
    hawkes: M0Parameters,
    ou: OUParameters,
    n_particles: int = 512,
    seed: int = 0,
    resample_threshold: float = 0.5,
) -> float:
    """Bootstrap particle estimate of the M1 conditional log likelihood.

    The OU process has stationary mean zero and variance one. Baseline survival
    uses trapezoidal quadrature over each event interval; excitation survival
    is exact between events. History at negative times is assimilated and then
    conditioned upon at time zero.
    """
    ou.validate()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_particles)
    excitation = np.zeros((n_particles, 2, data.n_states, 2))
    logw = np.full(n_particles, -np.log(n_particles))
    loglik = 0.0
    previous = float(data.times[0]) if data.times[0] < 0 else 0.0
    in_observation = previous >= 0

    def propagate(to_time: float, accumulate: bool) -> None:
        nonlocal z, excitation, logw, loglik, previous
        dt = to_time - previous
        if dt < 0:
            raise ValueError("event times must be sorted")
        old_z = z
        new_z = _ou_step(old_z, dt, ou.kappa, rng)
        baseline_integral = np.zeros(n_particles)
        for target in range(2):
            baseline_integral += hawkes.nu[target] * dt * 0.5 * (
                np.exp(ou.eta[target] * old_z) + np.exp(ou.eta[target] * new_z)
            )
        excitation_integral = np.zeros(n_particles)
        for source in range(2):
            for state in range(data.n_states):
                for target in range(2):
                    b = hawkes.beta[source, state, target]
                    excitation_integral += excitation[:, source, state, target] / b * (1 - np.exp(-b * dt))
                    excitation[:, source, state, target] *= np.exp(-b * dt)
        logw -= baseline_integral + excitation_integral
        z = new_z
        normalizer = logsumexp(logw)
        logw -= normalizer
        if accumulate:
            loglik += float(normalizer)
        previous = to_time

    for time, source_raw, state_raw in zip(data.times, data.types, data.states):
        time = float(time)
        if previous < 0 <= time:
            propagate(0.0, False)
            in_observation = True
        propagate(time, in_observation)
        source, state = int(source_raw), int(state_raw)
        intensity = hawkes.nu[source] * np.exp(ou.eta[source] * z)
        intensity += np.sum(excitation[:, :, :, source], axis=(1, 2))
        logw += np.log(np.maximum(intensity, np.finfo(float).tiny))
        normalizer = logsumexp(logw)
        logw -= normalizer
        if in_observation:
            loglik += float(normalizer)
        amplitude_factor = np.exp(ou.gamma[source, :] * z[:, None])
        excitation[:, source, state, :] += hawkes.alpha[source, state, :][None, :] * amplitude_factor
        weights = np.exp(logw)
        ess = 1.0 / np.sum(weights * weights)
        if ess < resample_threshold * n_particles:
            indices = _systematic_resample(weights, rng)
            z = z[indices]
            excitation = excitation[indices]
            logw.fill(-np.log(n_particles))

    if previous < data.end:
        propagate(data.end, True)
    return loglik

