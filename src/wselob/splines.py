from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True, slots=True)
class SplineBasis:
    knots: np.ndarray
    degree: int

    @property
    def n_basis(self) -> int:
        return len(self.knots) - self.degree - 1

    def design(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        # Knots are learned from the scored training window. Conditioning
        # history can contain states outside that support; polynomial spline
        # extrapolation is numerically unbounded and scientifically unjustified.
        # Use constant boundary extension while keeping the learned function
        # fully cubic and state-dependent inside its observed support.
        lower = self.knots[self.degree]
        upper = self.knots[-self.degree - 1]
        values = np.clip(values, lower, upper)
        return BSpline.design_matrix(
            values, self.knots, self.degree, extrapolate=False,
        ).toarray()


def quantile_basis(values: np.ndarray, n_basis: int = 6, degree: int = 3) -> SplineBasis:
    values = np.asarray(values, dtype=float)
    if n_basis < degree + 1:
        raise ValueError("n_basis must be at least degree + 1")
    lower, upper = float(np.min(values)), float(np.max(values))
    if not lower < upper:
        raise ValueError("spline state needs nonzero support")
    n_interior = n_basis - degree - 1
    if n_interior:
        probs = np.arange(1, n_interior + 1) / (n_interior + 1)
        interior = np.unique(np.quantile(values, probs))
        # Degenerate empirical quantiles are nudged only in knot construction.
        if len(interior) != n_interior:
            interior = np.linspace(lower, upper, n_interior + 2)[1:-1]
    else:
        interior = np.empty(0)
    knots = np.concatenate([
        np.repeat(lower, degree + 1), interior,
        np.repeat(upper, degree + 1),
    ])
    return SplineBasis(knots, degree)


def second_difference_penalty(coefficients: np.ndarray) -> float:
    return float(np.sum(np.diff(coefficients, n=2, axis=-1) ** 2))
