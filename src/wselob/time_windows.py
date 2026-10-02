from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np


WARSAW = ZoneInfo("Europe/Warsaw")


def local_window_ns(day_key: str, start: time, end: time) -> tuple[int, int]:
    date = datetime.strptime(day_key.removeprefix("d"), "%Y%m%d").date()
    start_dt = datetime.combine(date, start, tzinfo=WARSAW)
    end_dt = datetime.combine(date, end, tzinfo=WARSAW)
    return int(start_dt.timestamp() * 1e9), int(end_dt.timestamp() * 1e9)


def filter_time(rows: np.ndarray, start_ns: int, end_ns: int) -> np.ndarray:
    return rows[(rows["time"] >= start_ns) & (rows["time"] < end_ns)]

