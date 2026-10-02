import numpy as np

from wselob.splines import quantile_basis


def test_spline_uses_constant_boundary_extension():
    basis = quantile_basis(np.linspace(-1.0, 1.0, 21), n_basis=6)
    design = basis.design(np.array([-100.0, -1.0, 1.0, 100.0]))
    np.testing.assert_allclose(design[0], design[1])
    np.testing.assert_allclose(design[2], design[3])
    np.testing.assert_allclose(design.sum(axis=1), 1.0)
