import pytest

from wselob.ticks import infer_tick_from_prices


def test_gcd_tick_detects_increment_smaller_than_modal_gap():
    # Most observed adjacent levels are five raw units apart, but valid four-
    # and six-unit gaps prove that the underlying minimum increment is one.
    prices = [100]
    for gap in [5] * 20 + [4, 6, 10]:
        prices.append(prices[-1] + gap)
    assert infer_tick_from_prices(prices) == 1


def test_tick_requires_enough_observed_gaps():
    with pytest.raises(ValueError, match="only 3 positive quote gaps"):
        infer_tick_from_prices([0, 1, 2, 3])
