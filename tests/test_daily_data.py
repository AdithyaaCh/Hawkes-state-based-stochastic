from pathlib import Path

import numpy as np

from wselob.daily_data import DailyArrays


def test_daily_arrays_round_trip(tmp_path: Path):
    arr = np.array([0.0, 1.0])
    daily = DailyArrays(
        arr, np.array([0, 1]), arr, arr, np.array([0, 1]),
        np.array([0, 1]), np.array([0, 0]), np.array([0, 0]),
        np.arange(10, dtype=float), np.arange(10, dtype=float),
        3, 2.0, "KGHM", "d20170110", 1, 0, 0, 0, 0, 0,
    )
    target = tmp_path / "day.npz"
    daily.save(target)
    restored = DailyArrays.load(target)
    assert restored.stock == "KGHM"
    assert restored.observed_count == 2
    np.testing.assert_array_equal(restored.types, daily.types)

