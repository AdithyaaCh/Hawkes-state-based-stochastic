import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training_status.py"
SPEC = importlib.util.spec_from_file_location("training_status", SCRIPT)
training_status = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(training_status)


def test_scaled_pilot_seconds_accepts_legacy_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    record = {
        "status": "complete",
        "stock": "TEST",
        "day": "d1",
        "elapsed_seconds": 16.0,
        "fits": [
            {"model": "M0-S", "elapsed_seconds": 2.0},
            {"model": "M1-S", "elapsed_seconds": 4.0, "steps": 100},
            {"model": "M0-QI", "elapsed_seconds": 2.0},
            {"model": "M1-QI", "elapsed_seconds": 4.0, "steps": 100},
        ],
    }
    manifest.write_text(json.dumps(record) + "\n")

    # Four seconds of non-fit overhead + two unscaled M0 fits + two neural
    # fits doubled from 100 to 200 steps.
    assert training_status._scaled_pilot_seconds(manifest, 200) == [24.0]


def test_scaled_pilot_seconds_ignores_incomplete_or_unscalable_records(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {
            "status": "failed",
            "stock": "TEST",
            "day": "d1",
            "elapsed_seconds": 10.0,
            "fits": [{"model": "M0-S", "elapsed_seconds": 1.0}],
        },
        {
            "status": "complete",
            "stock": "TEST",
            "day": "d2",
            "elapsed_seconds": 10.0,
            "fits": [{"model": "M1-S", "elapsed_seconds": 5.0}],
        },
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert training_status._scaled_pilot_seconds(manifest, 200) == []
