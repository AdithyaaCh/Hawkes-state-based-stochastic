import pytest

from wselob.book import MessageResult
from wselob.events import _state
from wselob.states import qi_bin, spread_state
from wselob.types import BookSnapshot


@pytest.mark.parametrize("value, expected", [
    (-1.0, 0), (-0.6, 1), (-0.2, 2), (0.2, 3), (0.6, 4), (1.0, 4),
])
def test_exact_qi_edges(value, expected):
    assert qi_bin(value) == expected


def test_spread_states():
    assert spread_state(BookSnapshot(100, 101, 1, 1), 1) == 0
    assert spread_state(BookSnapshot(100, 103, 1, 1), 1) == 1


def test_qi_rejects_crossed_pre_book_to_match_spread_event_stream():
    message = MessageResult(
        0, 1, "M", 1, "2", 100, 1, None,
        BookSnapshot(101, 100, 10, 10),
        BookSnapshot(99, 100, 10, 10),
    )
    with pytest.raises(ValueError, match="valid uncrossed"):
        _state(message, "qi_continuous", 1)
