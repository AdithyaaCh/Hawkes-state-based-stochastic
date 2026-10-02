import numpy as np
import pytest
import torch

from wselob.continuous import ContinuousData, ContinuousParameters, continuous_log_likelihood
from wselob.gpu_continuous import TorchContinuousData, torch_continuous_log_likelihood
from wselob.splines import quantile_basis


def test_torch_backend_matches_cpu_reference_and_has_gradients():
    basis = quantile_basis(np.linspace(-1, 1, 20), n_basis=4)
    values = np.array([-0.5, 0.0, 0.5])
    data = ContinuousData(
        np.array([-0.2, 0.3, 1.0]), np.array([0, 1, 0]), values,
        basis.design(values), 1.5, basis,
    )
    params = ContinuousParameters(
        np.array([1.2, 0.8]), np.zeros((2, 2, 4)), np.zeros((2, 2, 4)),
    )
    expected, _ = continuous_log_likelihood(data, params)
    td = TorchContinuousData.from_numpy(data, torch.device("cpu"))
    log_nu = torch.tensor(np.log(params.nu), dtype=torch.float64, requires_grad=True)
    a = torch.zeros((2, 2, 4), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((2, 2, 4), dtype=torch.float64, requires_grad=True)
    actual = torch_continuous_log_likelihood(td, log_nu, a, b, chunk_size=2)
    assert actual.item() == pytest.approx(expected)
    (-actual).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()

