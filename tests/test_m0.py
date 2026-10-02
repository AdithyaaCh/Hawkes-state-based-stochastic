import numpy as np

from wselob.m0 import M0Data, M0Parameters, log_likelihood, transition_matrices, pack, negative_log_likelihood_raw
from wselob.types import Event, EventType


def make_event(t, typ, pre, post):
    return Event(t, typ, float(post), float(pre), "A", 1, "test", (0,))


def test_poisson_limit_matches_closed_form():
    data = M0Data(
        times=np.array([1.0, 2.0, 4.0]),
        types=np.array([0, 1, 0]), states=np.array([0, 0, 0]),
        start=0.0, end=5.0, n_states=1,
    )
    params = M0Parameters(
        nu=np.array([2.0, 3.0]),
        alpha=np.zeros((2, 1, 2)),
        beta=np.ones((2, 1, 2)),
    )
    expected = 2 * np.log(2.0) + np.log(3.0) - 25.0
    assert log_likelihood(data, params) == pytest.approx(expected)


def test_transition_mle():
    events = [
        make_event(1, EventType.BID, 0, 1),
        make_event(2, EventType.BID, 0, 0),
        make_event(3, EventType.ASK, 1, 0),
    ]
    phi, counts = transition_matrices(events, 2)
    assert counts[1, 0].tolist() == [1, 1]
    assert phi[1, 0].tolist() == [0.5, 0.5]
    assert phi[0, 1, 0] == 1.0


def test_warmup_event_affects_intensity_and_compensator():
    data = M0Data(
        times=np.array([-1.0, 1.0]), types=np.array([0, 0]),
        states=np.array([0, 0]), start=0.0, end=2.0, n_states=1,
    )
    params = M0Parameters(
        nu=np.ones(2), alpha=np.ones((2, 1, 2)), beta=np.ones((2, 1, 2)),
    )
    # Only the event at t=1 contributes a log-intensity term.
    intensity = 1.0 + np.exp(-2.0)
    comp = 4.0 + 2 * (np.exp(-1.0) - np.exp(-3.0)) + 2 * (1 - np.exp(-1.0))
    assert log_likelihood(data, params) == pytest.approx(np.log(intensity) - comp)
    assert negative_log_likelihood_raw(pack(params), data) == pytest.approx(comp - np.log(intensity))


def test_tied_events_use_same_left_limit_intensity():
    data = M0Data(
        times=np.array([0.5, 0.5]), types=np.array([0, 1]),
        states=np.array([0, 0]), start=0.0, end=1.0, n_states=1,
    )
    params = M0Parameters(np.ones(2), np.ones((2, 1, 2)), np.ones((2, 1, 2)))
    compensator = 2.0 + 4.0 * (1.0 - np.exp(-0.5))
    assert log_likelihood(data, params) == pytest.approx(-compensator)
    assert negative_log_likelihood_raw(pack(params), data) == pytest.approx(compensator)


import pytest
