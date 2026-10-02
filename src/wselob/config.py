from __future__ import annotations

import json
from pathlib import Path


def load_intraday_window(path: str | Path = "config/intraday_window.json") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
