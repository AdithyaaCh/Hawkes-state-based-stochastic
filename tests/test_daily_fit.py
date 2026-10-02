import numpy as np
import torch

from wselob.daily_fit import FitSettings, _fit_torch, _spline_initial_from_m0
from wselob.gpu_sparse import TorchEventData
from wselob.sparse_history import build_sparse_history


def test_stochastic_fit_shares_variational_mean_at_tied_timestamps():
    times = np.array([-0.5, 0.1, 0.1, 0.4, 0.4, 0.8])
    types = np.array([0, 0, 1, 0, 1, 1])
    states = np.array([0, 0, 1, 1, 0, 1])
    history = build_sparse_history(times, 2.0)
    data = TorchEventData.from_arrays(
        times, types, states, 1.0, history, torch.device("cpu")
    )
    initial = (
        np.log(np.array([1.0, 1.0])),
        np.log(np.full((2, 2, 2), 0.2)),
        np.log(np.full((2, 2, 2), 2.0)),
    )
    parameters, metadata = _fit_torch(
        data, initial, stochastic=True,
        settings=FitSettings(steps=20, learning_rate=0.02), seed=7,
    )
    assert np.isfinite(metadata["objective"])
    assert len(parameters["mean_shift_unique"]) == len(np.unique(times))
    assert parameters["mean_shift"][1] == parameters["mean_shift"][2]
    assert parameters["mean_shift"][3] == parameters["mean_shift"][4]


def test_stochastic_fit_is_reproducible_for_fixed_seed():
    times = np.array([-0.5, 0.1, 0.1, 0.4, 0.4, 0.8])
    types = np.array([0, 0, 1, 0, 1, 1])
    states = np.array([0, 0, 1, 1, 0, 1])
    history = build_sparse_history(times, 2.0)
    data = TorchEventData.from_arrays(
        times, types, states, 1.0, history, torch.device("cpu")
    )
    initial = (
        np.log(np.array([1.0, 1.0])),
        np.log(np.full((2, 2, 2), 0.2)),
        np.log(np.full((2, 2, 2), 2.0)),
    )
    settings = FitSettings(steps=20, learning_rate=0.02)

    first_parameters, first_metadata = _fit_torch(
        data, initial, stochastic=True, settings=settings, seed=7,
    )
    second_parameters, second_metadata = _fit_torch(
        data, initial, stochastic=True, settings=settings, seed=7,
    )

    assert first_parameters.keys() == second_parameters.keys()
    for key in first_parameters:
        np.testing.assert_array_equal(first_parameters[key], second_parameters[key])
    assert first_metadata["objective"] == second_metadata["objective"]


def test_spline_initialization_preserves_constant_m0_kernel():
    basis = np.array([
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ] * 2)
    types = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    states = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    alpha = np.full((2, 2, 2), 3.0)
    beta = np.full((2, 2, 2), 5.0)
    a, b = _spline_initial_from_m0(
        basis, types, states, np.ones(8, dtype=bool), alpha, beta,
    )
    np.testing.assert_allclose(a, np.log(3.0), atol=1e-7)
    np.testing.assert_allclose(b, np.log(5.0), atol=1e-7)
