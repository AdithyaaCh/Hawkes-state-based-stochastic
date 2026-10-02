import numpy as np

from wselob.intraday import Profile, select_stationary_window


def test_stationary_window_avoids_open_and_close_spikes():
    values = np.r_[np.full(2, 2.0), np.ones(8), np.full(2, 2.0)]
    profile = Profile(np.arange(12) * 30, values, values, np.ones(12), 10, 30)
    selected = select_stationary_window(profile, min_minutes=120, max_cv=0.05,
                                        max_relative_range=0.1)
    assert selected.start_minute == 60
    assert selected.end_minute == 300
