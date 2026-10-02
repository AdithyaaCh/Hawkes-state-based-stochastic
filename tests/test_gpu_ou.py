import numpy as np
import pytest
import torch

from wselob.gpu_ou import variational_ou_elbo
from wselob.gpu_sparse import (
    TorchEventData, deterministic_log_likelihood, exact_minibatch_log_likelihood,
)
from wselob.sparse_history import build_sparse_history


def test_zero_loadings_exactly_recover_deterministic_likelihood():
    times = np.array([-1.0, -0.2, 0.3, 0.8, 1.2])
    types = np.array([0, 1, 0, 1, 0])
    states = np.array([0, 1, 0, 1, 0])
    history = build_sparse_history(times, 5.0)
    data = TorchEventData.from_arrays(times, types, states, 2.0, history, torch.device("cpu"))
    log_nu = torch.tensor([0.1, -0.2], dtype=torch.float64)
    alpha = torch.full((2, 2, 2), -1.0, dtype=torch.float64)
    beta = torch.full((2, 2, 2), 0.5, dtype=torch.float64)
    deterministic = deterministic_log_likelihood(data, log_nu, alpha, beta)
    stochastic = variational_ou_elbo(
        data, log_nu, alpha, beta, torch.tensor(0.0),
        torch.zeros(2, dtype=torch.float64), torch.zeros((2, 2), dtype=torch.float64),
        torch.zeros(len(times), dtype=torch.float64),
        torch.randn((2, len(times)), dtype=torch.float64),
    )
    assert stochastic.elbo.item() == pytest.approx(deterministic.item(), abs=1e-8)


def test_zero_loadings_nest_all_history_minibatch_objective():
    times = np.array([-5.0, -0.2, 0.3, 0.8, 1.2])
    types = np.array([0, 1, 0, 1, 0])
    states = np.array([0, 1, 0, 1, 0])
    data = TorchEventData.from_full_arrays(
        times, types, states, 2.0, torch.device("cpu")
    )
    log_nu = torch.tensor([0.1, -0.2], dtype=torch.float64)
    alpha = torch.full((2, 2, 2), -1.0, dtype=torch.float64)
    beta = torch.full((2, 2, 2), -2.0, dtype=torch.float64)
    deterministic = exact_minibatch_log_likelihood(
        data, data.observed, log_nu, alpha, beta, scale_event_term=False,
    )
    stochastic = variational_ou_elbo(
        data, log_nu, alpha, beta, torch.tensor(0.0),
        torch.zeros(2, dtype=torch.float64), torch.zeros((2, 2), dtype=torch.float64),
        torch.zeros(len(times), dtype=torch.float64),
        torch.randn((2, len(times)), dtype=torch.float64),
        event_indices=data.observed, scale_event_term=False,
    )
    assert stochastic.elbo.item() == pytest.approx(deterministic.item(), abs=1e-8)
