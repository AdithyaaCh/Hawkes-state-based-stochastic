import numpy as np
import pytest

from wselob.continuous import ContinuousData, ContinuousParameters, continuous_log_likelihood
from wselob.splines import quantile_basis


def test_continuous_constant_functions_match_manual_hawkes():
    basis = quantile_basis(np.linspace(-1, 1, 20), n_basis=4)
    values = np.array([-0.5, 0.5])
    data = ContinuousData(
        times=np.array([0.0, 1.0]), types=np.array([0, 0]), values=values,
        basis_values=basis.design(values), end=2.0, basis=basis,
    )
    params = ContinuousParameters(
        nu=np.ones(2),
        alpha_coef=np.zeros((2, 2, 4)),  # alpha = 1
        beta_coef=np.zeros((2, 2, 4)),   # beta = 1
    )
    ll, omitted = continuous_log_likelihood(data, params)
    event_term = np.log(1.0) + np.log(1.0 + np.exp(-1.0))
    compensator = 4.0 + 2 * (1 - np.exp(-2.0)) + 2 * (1 - np.exp(-1.0))
    assert ll == pytest.approx(event_term - compensator)
    assert omitted == 0.0

