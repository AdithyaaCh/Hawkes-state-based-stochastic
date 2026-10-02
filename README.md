# State-Dependent Hawkes Models for WSELOB-2017

Research code for reconstructing limit order books from the WSELOB-2017 dataset and fitting state-dependent Hawkes models. It builds bid and ask event streams, derives spread and queue-imbalance states, and supports CPU and CUDA fitting workflows.

## Features

- Limit-order-book reconstruction from WSELOB order messages
- Trade matching and bid/ask event classification
- Spread and queue-imbalance state representations
- Discrete, continuous, and latent-state Hawkes model variants
- CPU and CUDA fitting workflows
- Resumable multi-day training with progress and runtime estimates

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer
- SciPy 1.10 or newer
- h5py 3.8 or newer
- Numba 0.57 or newer
- PyTorch 2.2 or newer for GPU workflows
- pytest 7 or newer for tests

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy h5py numba pytest
python -m pip install torch  # required for fitting modules and GPU workflows
export PYTHONPATH="$PWD/src"
```

Install the PyTorch build appropriate for your operating system and CUDA version if you plan to train on a GPU.

## Dataset

This project expects the WSELOB-2017 order-book files for `KGHM`, `PKNORLEN`, `PKOBP`, `PZU`, and `PEKAO` in a local `data` directory. Trade data may be supplied as extracted HDF5 files or read directly from the original `WSELOB-2017.zip` archive. The dataset is not included in this repository.

The preparation and training commands also expect `config/intraday_window.json`:

```json
{
  "window_start": "10:45:00",
  "window_end": "14:30:00",
  "warmup_minutes": 30
}
```

Times are interpreted in the `Europe/Warsaw` time zone.

## Quick start

Run the test suite:

```bash
PYTHONPATH=src pytest -q
```

Check that CUDA is visible and can allocate tensors:

```bash
PYTHONPATH=src python scripts/check_gpu.py
```

Prepare one stock-day of continuous spread events:

```bash
PYTHONPATH=src python scripts/prepare_events.py \
  --stock KGHM \
  --day d20170110 \
  --state spread
```

The default output is written to `artifacts/events/` as a compressed NumPy archive.

Fit the continuous-state M2 model on the prepared events:

```bash
PYTHONPATH=src python scripts/fit_m2_gpu.py \
  artifacts/events/KGHM_d20170110_spread.npz
```

## Eight-model workflow

Each stock-day can be fit with eight variants: model families M0 through M3, each conditioned on spread (`S`) or queue imbalance (`QI`). Before a long run, use the smoke script on a single day:

```bash
PYTHONPATH=src python scripts/smoke_all_models_gpu.py \
  --data-dir data \
  --stock KGHM \
  --day d20170110 \
  --device cuda
```

Run the resumable multi-day scheduler:

```bash
PYTHONPATH=src python scripts/run_training_days.py \
  --data-dir data \
  --device cuda \
  --workers 4 \
  --training-days 150
```

Completed fits are stored under `artifacts/training_fits/<stock>/<day>/`. Existing compatible results are skipped unless `--rerun` is supplied. Daily reconstruction caches are stored under `artifacts/daily_cache/`.

Check progress and estimate the remaining run time:

```bash
PYTHONPATH=src python scripts/training_status.py \
  --data-dir data \
  --workers 4 \
  --training-days 150
```

Use `--help` on any script to see all available options.

## Notes

- Day identifiers use HDF5 keys such as `d20170110`.
- The supported stocks are `KGHM`, `PKNORLEN`, `PKOBP`, `PZU`, and `PEKAO`.
- Generated datasets, caches, and fit results are written under `artifacts/` by default.
- GPU workers share `cuda:0`; tune the worker count to fit the available GPU memory.
