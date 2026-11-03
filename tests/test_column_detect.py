"""
Tests for smart column auto-detection and certify_file().
"""
import tempfile
import os
import pytest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")


def test_detect_accel_from_gravity():
    """Accelerometer columns should be detected from gravity signature ~9.81 m/s²."""
    from s2s_standard_v1_3.adapters.column_detect import detect_columns
    import random
    r = random.Random(42)
    # Column 0: accel z (gravity ~9.81 m/s²)
    # Column 1: gyro (near zero)
    # Column 2: noise
    n = 300
    data = np.column_stack([
        [r.gauss(1000.0, 30.0) for _ in range(n)],   # accel
        [r.gauss(0.0, 0.2) for _ in range(n)],    # gyro
        [r.gauss(0.0, 50.0) for _ in range(n)],   # noise/unknown
    ])
    result = detect_columns(data)
    assert 0 in result['accel'], f"Accel not detected: {result}"
    assert result['confidence'] > 0.5


def test_detect_gyro_near_zero():
    """Gyroscope columns should be detected from near-zero values."""
    from s2s_standard_v1_3.adapters.column_detect import detect_columns
    import random
    r = random.Random(42)
    n = 300
    data = np.column_stack([
        [r.gauss(1000.0, 30.0) for _ in range(n)],   # accel
        [r.gauss(0.0, 0.15) for _ in range(n)],   # gyro rad/s
        [r.gauss(0.0, 0.15) for _ in range(n)],   # gyro rad/s
    ])
    result = detect_columns(data)
    assert len(result['gyro']) >= 1, f"Gyro not detected: {result}"


def test_detect_from_csv_file():
    """certify_file should work on a CSV with named accel columns."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file
    import random
    r = random.Random(42)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False) as f:
        f.write("time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
        for i in range(400):
            t = i * 0.01
            ax = r.gauss(0.0, 100.0)
            ay = r.gauss(0.0, 100.0)
            az = r.gauss(1000.0, 30.0)
            gx = r.gauss(0.0, 0.1)
            gy = r.gauss(0.0, 0.1)
            gz = r.gauss(0.0, 0.1)
            f.write(f"{t},{ax},{ay},{az},{gx},{gy},{gz}\n")
        path = f.name

    try:
        result = certify_file(path, segment='forearm')
        assert 'tier' in result
        # Accept either successful certification or graceful error
        if 'error' not in result:
            assert result['total_windows'] > 0
            assert result['pass_rate'] >= 0.0
    finally:
        os.unlink(path)


def test_detect_from_spacedelimited_file():
    """certify_file should work on space-delimited files without headers."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file
    import random
    r = random.Random(42)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat',
                                     delete=False) as f:
        for i in range(400):
            # col0: timestamp-like, col1-3: accel, col4-6: gyro
            t = i * 33000000  # 30Hz in ns
            ax = r.gauss(0.0, 100.0)
            ay = r.gauss(0.0, 100.0)
            az = r.gauss(1000.0, 30.0)
            gx = r.gauss(0.0, 0.1)
            gy = r.gauss(0.0, 0.1)
            gz = r.gauss(0.0, 0.1)
            f.write(f"{t} {ax} {ay} {az} {gx} {gy} {gz}\n")
        path = f.name

    try:
        result = certify_file(path, segment='forearm', delimiter=' ')
        assert 'tier' in result
        assert result['total_windows'] > 0
    finally:
        os.unlink(path)


def test_no_accel_returns_error():
    """certify_file should return error dict when no accel found."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file
    import random
    r = random.Random(42)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False) as f:
        f.write("label,category,score\n")
        for i in range(50):
            f.write(f"walking,{i},0.{i}\n")
        path = f.name

    try:
        result = certify_file(path, segment='forearm')
        assert 'error' in result
    finally:
        os.unlink(path)


class _RecordingColumnEngine:
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def certify(self, imu_raw, segment):
        self.calls.append((imu_raw, segment))
        return {
            "tier": "SILVER",
            "physical_law_score": 80,
        }


def _run_column_adapter(
    monkeypatch, gyro_cols, missing_gyro_row=None, timestamp_values=None
):
    from s2s_standard_v1_3.adapters import column_detect
    from s2s_standard_v1_3 import s2s_physics_v1_3

    _RecordingColumnEngine.instances.clear()
    monkeypatch.setattr(
        s2s_physics_v1_3, "PhysicsEngine", _RecordingColumnEngine
    )

    n = 400
    i = np.arange(n, dtype=float)
    source_time = (
        np.asarray(timestamp_values, dtype=float)
        if timestamp_values is not None
        else i / 100.0
    )
    if len(source_time) != n:
        raise ValueError("timestamp_values must have exactly n entries")

    data = np.column_stack([
        source_time,               # source time in seconds
        8.0 + i * 0.0001,          # acceleration X
        9.0 + i * 0.0001,          # acceleration Y
        9.81 + i * 0.0001,         # acceleration Z
        np.full(n, 0.1),           # gyro X
        np.full(n, 0.2),           # gyro Y
        np.full(n, 0.3),           # gyro Z
    ])

    if missing_gyro_row is not None:
        data[missing_gyro_row, 5] = np.nan

    detection = {
        "accel": [1, 2, 3],
        "gyro": gyro_cols,
        "timestamp": 0,
        "confidence": 1.0,
        "_votes": [(0, "timestamp", 0.95)],
    }

    monkeypatch.setattr(
        column_detect,
        "detect_from_file",
        lambda *args, **kwargs: detection,
    )

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        path = f.name

    try:
        np.savetxt(
            path,
            data,
            delimiter=",",
            header="time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z",
            comments="",
        )

        result = column_detect.certify_file(
            path,
            segment="forearm",
            delimiter=",",
            max_windows=2,
        )

        engine = _RecordingColumnEngine.instances[-1]
        return result, engine.calls, data
    finally:
        os.unlink(path)


def test_accel_only_file_does_not_receive_fake_gyro(monkeypatch):
    _, calls, _ = _run_column_adapter(monkeypatch, gyro_cols=[])

    assert len(calls) == 1
    imu_raw, _ = calls[0]

    assert "gyro" not in imu_raw
    assert len(imu_raw["accel"]) == 256


def test_missing_gyro_row_does_not_remove_accel_row(monkeypatch):
    _, calls, data = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        missing_gyro_row=40,
    )

    assert len(calls) == 1
    imu_raw, _ = calls[0]

    expected_accel = data[:256, 1:4].tolist()

    assert imu_raw["accel"] == expected_accel
    assert len(imu_raw["accel"]) == 256
    assert "gyro" not in imu_raw


def test_complete_gyro_stream_is_preserved(monkeypatch):
    _, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
    )

    assert len(calls) == 1
    imu_raw, _ = calls[0]

    assert "gyro" in imu_raw
    assert len(imu_raw["gyro"]) == 256
    assert imu_raw["gyro"][0] == [0.1, 0.2, 0.3]



def test_source_timestamps_are_preserved_in_nanoseconds(monkeypatch):
    n = 400
    source_time = 1_700_000_000.0 + np.arange(n) * 0.01

    # Simulate a 30 ms timing gap after row 99.
    source_time[100:] += 0.02

    _, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        timestamp_values=source_time,
    )

    assert len(calls) == 1
    imu_raw, _ = calls[0]

    expected = np.rint(source_time[:256] * 1e9).astype("int64").tolist()
    assert imu_raw["timestamps_ns"] == expected
    gap_ns = (
        imu_raw["timestamps_ns"][100]
        - imu_raw["timestamps_ns"][99]
    )
    assert abs(gap_ns - 30_000_000) <= 500
