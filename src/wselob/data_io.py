from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import io
import zipfile

import h5py
import numpy as np


STOCKS = ("KGHM", "PKNORLEN", "PKOBP", "PZU", "PEKAO")


def order_path(data_dir: str | Path, stock: str) -> Path:
    return Path(data_dir) / f"{stock}_lob_2017_zlib.h5"


def day_keys(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(handle.keys())


def read_day(path: str | Path, day: str) -> np.ndarray:
    key = day.removeprefix("/")
    with h5py.File(path, "r") as handle:
        return handle[key]["table"][:]


def read_order_day(data_dir: str | Path, stock: str, day: str) -> np.ndarray:
    return read_day(order_path(data_dir, stock), day)


def _trade_member(zf: zipfile.ZipFile, stock: str) -> str:
    suffix = f"/trades/{stock}_trades_2017_zlib.h5"
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one trade file for {stock}, found {matches}")
    return matches[0]


def read_trade_day_from_zip(
    archive: str | Path, stock: str, day: str
) -> np.ndarray:
    """Read one trade day without requiring users to extract the archive.

    The compressed trade member is small enough to buffer, but callers should
    cache the result when processing multiple stages for the same stock.
    """
    key = day.removeprefix("/")
    with zipfile.ZipFile(archive) as zf:
        raw = zf.read(_trade_member(zf, stock))
    with h5py.File(io.BytesIO(raw), "r") as handle:
        return handle[key]["table"][:]


def read_trade_day(data_dir: str | Path, stock: str, day: str) -> np.ndarray:
    extracted = Path(data_dir) / "trades" / f"{stock}_trades_2017_zlib.h5"
    if extracted.exists():
        return read_day(extracted, day)
    return read_trade_day_from_zip(Path(data_dir) / "WSELOB-2017.zip", stock, day)


def decode_byte(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("ascii")
    return str(value)
