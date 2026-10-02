import numpy as np
import pytest
import torch

from wselob.gpu_sparse import (
    TorchEventData, deterministic_log_likelihood, exact_minibatch_log_likelihood,
)
from wselob.m0 import M0Data, M0Parameters, log_likelihood
from wselob.sparse_history import build_sparse_history


def test_sparse_gpu_backend_matches_exact_m0_with_long_cutoff():
    times = np.array([-0.4, 0.2, 0.7, 0.7, 1.3])
    types = np.array([0, 1, 0, 1, 0])
    states = np.array([1, 0, 1, 1, 0])
    data = M0Data(times, types, states, 0.0, 2.0, 2)
    params = M0Parameters(
        np.array([1.2, 0.8]),
        np.array([[[0.3, 0.2], [0.4, 0.5]], [[0.1, 0.3], [0.2, 0.4]]]),
        np.array([[[2.0, 3.0], [1.0, 4.0]], [[5.0, 2.0], [3.0, 1.0]]]),
    )
    sparse = build_sparse_history(times, 100.0)
    td = TorchEventData.from_arrays(times, types, states, 2.0, sparse, torch.device("cpu"))
    actual = deterministic_log_likelihood(
        td, torch.tensor(np.log(params.nu)),
        torch.tensor(np.log(params.alpha)), torch.tensor(np.log(params.beta)),
    )
    # Both backends use the same left-limit intensity for simultaneous events.
    untied = M0Data(times[[0, 1, 2, 4]], types[[0, 1, 2, 4]],
                    states[[0, 1, 2, 4]], 0.0, 2.0, 2)
    sparse2 = build_sparse_history(untied.times, 100.0)
    td2 = TorchEventData.from_arrays(untied.times, untied.types, untied.states,
                                     2.0, sparse2, torch.device("cpu"))
    actual2 = deterministic_log_likelihood(
        td2, torch.tensor(np.log(params.nu)),
        torch.tensor(np.log(params.alpha)), torch.tensor(np.log(params.beta)),
    )
    assert np.isfinite(actual.item())
    assert actual.item() == pytest.approx(log_likelihood(data, params), abs=1e-10)
    assert actual2.item() == pytest.approx(log_likelihood(untied, params), abs=1e-10)


def test_all_history_batch_matches_exact_likelihood_without_cutoff():
    times = np.array([-10.0, -0.4, 0.2, 0.7, 0.7, 1.3])
    types = np.array([1, 0, 1, 0, 1, 0])
    states = np.array([0, 1, 0, 1, 1, 0])
    params = M0Parameters(
        np.array([1.2, 0.8]),
        np.array([[[0.3, 0.2], [0.4, 0.5]], [[0.1, 0.3], [0.2, 0.4]]]),
        np.array([[[0.02, 3.0], [1.0, 4.0]], [[5.0, 2.0], [3.0, 1.0]]]),
    )
    exact = M0Data(times, types, states, 0.0, 2.0, 2)
    td = TorchEventData.from_full_arrays(
        times, types, states, 2.0, torch.device("cpu")
    )
    actual = exact_minibatch_log_likelihood(
        td, td.observed, torch.tensor(np.log(params.nu)),
        torch.tensor(np.log(params.alpha)), torch.tensor(np.log(params.beta)),
        scale_event_term=False,
    )
    assert actual.item() == pytest.approx(log_likelihood(exact, params), abs=1e-10)
