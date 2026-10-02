import numpy as np
import pytest

from wselob.latent_ou import OUParameters, m1_particle_log_likelihood
from wselob.m0 import M0Data, M0Parameters, log_likelihood


def test_zero_loadings_nest_m0_exactly():
    data = M0Data(
        times=np.array([0.2, 0.7, 1.4]), types=np.array([0, 1, 0]),
        states=np.array([0, 0, 0]), start=0.0, end=2.0, n_states=1,
    )
    hawkes = M0Parameters(
        nu=np.array([1.2, 0.8]), alpha=np.full((2, 1, 2), 0.3),
        beta=np.full((2, 1, 2), 2.0),
    )
    ou = OUParameters(1.0, np.zeros(2), np.zeros((2, 2)))
    particle = m1_particle_log_likelihood(data, hawkes, ou, n_particles=64)
    assert particle == pytest.approx(log_likelihood(data, hawkes), abs=1e-10)


def test_ou_normalization_and_sign_constraint():
    ou = OUParameters(2.0, np.array([0.1, -0.2]), np.zeros((2, 2)))
    assert ou.sigma == pytest.approx(2.0)
    with pytest.raises(ValueError):
        OUParameters(1.0, np.array([-0.1, 0.0]), np.zeros((2, 2))).validate()
