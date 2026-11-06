"""
Tests for smart column auto-detection and certify_file().
"""
import csv
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
        result = certify_file(
            path,
            segment='forearm',
            accel_columns=[1, 2, 3],
            gyro_columns=[4, 5, 6],
            accel_unit="milli-g",
            gyro_unit="rad/s",
        )
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
        result = certify_file(
            path,
            segment='forearm',
            delimiter=' ',
            accel_columns=[1, 2, 3],
            gyro_columns=[4, 5, 6],
            accel_unit="milli-g",
            gyro_unit="rad/s",
        )
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
    monkeypatch,
    gyro_cols,
    missing_gyro_row=None,
    timestamp_values=None,
    accel_unit="m/s2",
    gyro_unit="rad/s",
    header_names=None,
    accel_columns=(1, 2, 3),
    detection_override=None,
    corrupt_all_gyro=False,
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

    if corrupt_all_gyro:
        # Set all gyro columns (4, 5, 6) to NaN
        data[:, 4:7] = np.nan

    detection = detection_override if detection_override is not None else {
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
        names = header_names or [
            "time", "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
        ]
        np.savetxt(
            path,
            data,
            delimiter=",",
            header=",".join(names),
            comments="",
        )

        result = column_detect.certify_file(
            path,
            segment="forearm",
            delimiter=",",
            max_windows=2,
            accel_unit=accel_unit,
            gyro_unit=gyro_unit,
            accel_columns=accel_columns,
            gyro_columns=gyro_cols,
        )

        engine = (
            _RecordingColumnEngine.instances[-1]
            if _RecordingColumnEngine.instances else None
        )
        calls = engine.calls if engine is not None else []
        return result, calls, data
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



def test_header_units_are_inferred_only_when_all_axes_agree(monkeypatch):
    names = [
        "time",
        "acc_x [m/s^2]",
        "acc_y [m/s^2]",
        "acc_z [m/s^2]",
        "gyro_x [rad/s]",
        "gyro_y [rad/s]",
        "gyro_z [rad/s]",
    ]

    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit=None,
        gyro_unit=None,
        header_names=names,
    )

    assert "error" not in result
    assert result["unit_metadata"]["accel"]["unit"] == "m/s2"
    assert result["unit_metadata"]["accel"]["status"] == (
        "inferred_from_header"
    )
    assert result["unit_metadata"]["gyro"]["unit"] == "rad/s"
    assert result["unit_metadata"]["gyro"]["status"] == (
        "inferred_from_header"
    )
    assert len(calls) == 1
    assert "gyro" in calls[0][0]


def test_ambiguous_acceleration_unit_refuses_certification(monkeypatch):
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit=None,
        gyro_unit="rad/s",
    )

    assert result["tier"] == "N/A"
    assert "unit unknown" in result["error"].lower()
    assert result["unit_metadata"]["accel"]["status"] == "unit_unknown"
    assert calls == []


def test_mixed_acceleration_header_units_refuse_certification(monkeypatch):
    names = [
        "time",
        "acc_x [m/s^2]",
        "acc_y [m/s^2]",
        "acc_z [g]",
        "gyro_x [rad/s]",
        "gyro_y [rad/s]",
        "gyro_z [rad/s]",
    ]

    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit=None,
        gyro_unit=None,
        header_names=names,
    )

    assert result["tier"] == "N/A"
    assert result["unit_metadata"]["accel"]["status"] == "unit_unknown"
    assert calls == []


def test_unknown_gyro_unit_is_omitted_but_accel_is_assessed(monkeypatch):
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit=None,
    )

    assert "error" not in result
    assert len(calls) == 1
    assert "gyro" not in calls[0][0]
    assert result["unit_metadata"]["gyro"]["status"] == "unit_unknown"
    assert result["gyro_window_availability"]["unit_unknown"] > 0


def test_explicit_millig_acceleration_conversion(monkeypatch):
    _, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="milli-g",
        gyro_unit="rad/s",
    )

    assert len(calls) == 1
    actual = np.asarray(calls[0][0]["accel"][0])
    expected = np.asarray([8.0, 9.0, 9.81]) * 0.00980665
    assert np.allclose(actual, expected)


def test_unsupported_explicit_acceleration_unit_fails_closed(monkeypatch):
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="ADC counts",
        gyro_unit="rad/s",
    )

    assert calls == []
    assert result["tier"] == "N/A"
    assert "unsupported explicit unit" in result["error"].lower()
    assert result["unit_metadata"]["accel"]["status"] == "invalid_explicit"


def test_partial_auto_accel_detection_requires_explicit_mapping(monkeypatch):
    detection = {
        "accel": [3],
        "gyro": [4, 5, 6],
        "timestamp": 0,
        "confidence": 0.5,
        "_votes": [],
    }

    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="rad/s",
        accel_columns=None,
        detection_override=detection,
    )

    assert result["tier"] == "N/A"
    assert "provide accel_columns" in result["error"].lower()
    assert calls == []



def test_pamap2_column_maps_select_requested_axes(monkeypatch):
    from s2s_standard_v1_3.adapters import column_detect
    from s2s_standard_v1_3 import s2s_physics_v1_3

    _RecordingColumnEngine.instances.clear()
    monkeypatch.setattr(
        s2s_physics_v1_3,
        "PhysicsEngine",
        _RecordingColumnEngine,
    )

    n_rows = 256
    n_columns = 54
    data = np.zeros((n_rows, n_columns), dtype=float)

    row = np.arange(n_rows, dtype=float)
    data[:, 0] = row / 100.0

    # Each source column has a distinct value, allowing us to verify
    # exactly which columns the adapter passes to the physics engine.
    for column in range(1, n_columns):
        data[:, column] = column + row * 0.00001

    detection = {
        "accel": [],
        "gyro": [],
        "timestamp": 0,
        "confidence": 1.0,
        "_votes": [(0, "timestamp", 0.95)],
    }

    monkeypatch.setattr(
        column_detect,
        "detect_from_file",
        lambda *args, **kwargs: detection,
    )

    mappings = [
        ("hand_16g", [4, 5, 6], [10, 11, 12]),
        ("hand_6g", [7, 8, 9], [10, 11, 12]),
        ("chest_16g", [21, 22, 23], [27, 28, 29]),
        ("chest_6g", [24, 25, 26], [27, 28, 29]),
        ("ankle_16g", [38, 39, 40], [44, 45, 46]),
        ("ankle_6g", [41, 42, 43], [44, 45, 46]),
    ]

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False
    ) as f:
        path = f.name

    try:
        np.savetxt(path, data, delimiter=" ", fmt="%.8f")

        for name, accel_cols, gyro_cols in mappings:
            result = column_detect.certify_file(
                path,
                segment="forearm",
                delimiter=" ",
                max_windows=1,
                accel_columns=accel_cols,
                gyro_columns=gyro_cols,
                accel_unit="m/s2",
                gyro_unit="rad/s",
            )

            assert "error" not in result, (name, result)
            assert result["total_windows"] == 1
            assert result["gyro_columns_usable"]

            engine = _RecordingColumnEngine.instances[-1]
            assert len(engine.calls) == 1

            imu_raw, _ = engine.calls[0]

            np.testing.assert_allclose(
                imu_raw["accel"][0],
                data[0, accel_cols],
            )
            np.testing.assert_allclose(
                imu_raw["gyro"][0],
                data[0, gyro_cols],
            )

    finally:
        os.unlink(path)


def test_accel_m_s_minus2_unit_alias():
    from s2s_standard_v1_3.adapters.column_detect import _canonical_unit

    assert _canonical_unit("m s^-2", "accel") == "m/s2"


# ---------------------------------------------------------------------------
# Delimiter detection regression tests
#
# detect_from_file() and certify_file() currently select delimiters by file
# extension: .csv -> comma, everything else -> space.  This breaks
# semicolon-separated CSV, tab-separated files, and any other valid delimited
# data.  The same delimiter must be used consistently for detection, header
# parsing, data loading, and certification.
# ---------------------------------------------------------------------------


def _write_delimiter_test_file(path, delimiter, n_rows=400, header=True):
    """
    Write realistic sensor data: timestamp, 3 accel axes, 3 gyro axes.

    Values are chosen so detect_columns() can classify them when the
    delimiter is parsed correctly:
      - timestamp: monotonically increasing seconds (100 Hz)
      - accel_z: ~9.81 m/s^2 (gravity signature)
      - accel_x/y: small motion around zero
      - gyro: near-zero rad/s at rest
    """
    import random
    r = random.Random(42)
    lines = []
    if header:
        lines.append(delimiter.join([
            "time_s", "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
        ]))
    for i in range(n_rows):
        t = i * 0.01
        ax = r.gauss(0.5, 0.3)
        ay = r.gauss(-0.3, 0.3)
        az = r.gauss(9.81, 0.2)
        gx = r.gauss(0.0, 0.05)
        gy = r.gauss(0.0, 0.05)
        gz = r.gauss(0.0, 0.05)
        row = [f"{v:.6f}" for v in [t, ax, ay, az, gx, gy, gz]]
        lines.append(delimiter.join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _certify_kwargs():
    """Explicit column mappings and units to isolate delimiter behavior."""
    return dict(
        segment="forearm",
        accel_columns=[1, 2, 3],
        gyro_columns=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="rad/s",
        max_windows=2,
    )


# --- detect_from_file() delimiter cases ------------------------------------


def test_detect_from_file_comma_csv():
    """Comma-separated .csv should detect 7 columns."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, ",", header=True)
        result = detect_from_file(path)
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
    finally:
        os.unlink(path)


def test_detect_from_file_semicolon_csv():
    """Semicolon-separated .csv should detect 7 columns (currently fails)."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, ";", header=True)
        result = detect_from_file(path)
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
    finally:
        os.unlink(path)


def test_detect_from_file_tab_csv():
    """Tab-separated .csv should detect 7 columns (currently fails)."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, "\t", header=True)
        result = detect_from_file(path)
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
    finally:
        os.unlink(path)


def test_detect_from_file_tab_dat():
    """Tab-separated .dat should detect 7 columns (currently fails)."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, "\t", header=True)
        result = detect_from_file(path)
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
    finally:
        os.unlink(path)


def test_detect_from_file_whitespace_dat():
    """Whitespace-separated .dat should detect 7 columns."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, " ", header=True)
        result = detect_from_file(path)
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
    finally:
        os.unlink(path)


def test_detect_from_file_explicit_delimiter():
    """Explicit delimiter override should work for any file type."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, ";", header=True)
        result = detect_from_file(path, delimiter=";")
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
    finally:
        os.unlink(path)


# --- certify_file() delimiter cases ----------------------------------------


def test_certify_file_comma_csv():
    """certify_file should process comma-separated .csv correctly."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, ",", header=True)
        result = certify_file(path, **_certify_kwargs())
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        assert result["total_windows"] > 0, (
            f"Expected windows processed, got {result}"
        )
    finally:
        os.unlink(path)


def test_certify_file_semicolon_csv():
    """certify_file should process semicolon-separated .csv (currently fails)."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, ";", header=True)
        result = certify_file(path, **_certify_kwargs())
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        assert result["total_windows"] > 0, (
            f"Expected windows processed, got {result}"
        )
    finally:
        os.unlink(path)


def test_certify_file_tab_csv():
    """certify_file should process tab-separated .csv (currently fails)."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, "\t", header=True)
        result = certify_file(path, **_certify_kwargs())
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        assert result["total_windows"] > 0, (
            f"Expected windows processed, got {result}"
        )
    finally:
        os.unlink(path)


def test_certify_file_tab_dat():
    """certify_file should process tab-separated .dat (currently fails)."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, "\t", header=True)
        result = certify_file(path, **_certify_kwargs())
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        assert result["total_windows"] > 0, (
            f"Expected windows processed, got {result}"
        )
    finally:
        os.unlink(path)


def test_certify_file_whitespace_dat():
    """certify_file should process whitespace-separated .dat correctly."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, " ", header=True)
        result = certify_file(path, **_certify_kwargs())
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        assert result["total_windows"] > 0, (
            f"Expected windows processed, got {result}"
        )
    finally:
        os.unlink(path)


def test_certify_file_explicit_delimiter():
    """certify_file should honor explicit delimiter override."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, ";", header=True)
        kwargs = _certify_kwargs()
        kwargs["delimiter"] = ";"
        result = certify_file(path, **kwargs)
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        assert result["total_windows"] > 0, (
            f"Expected windows processed, got {result}"
        )
    finally:
        os.unlink(path)


# --- Consistency between detect_from_file and certify_file ------------------


def test_detect_and_certify_use_same_delimiter_explicit():
    """
    When an explicit delimiter is provided, both detect_from_file and
    certify_file must use it consistently.
    """
    from s2s_standard_v1_3.adapters.column_detect import (
        detect_from_file, certify_file,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, ";", header=True)

        detection = detect_from_file(path, delimiter=";")
        assert detection["shape"][1] == 7

        kwargs = _certify_kwargs()
        kwargs["delimiter"] = ";"
        result = certify_file(path, **kwargs)
        assert "error" not in result
        assert result["total_windows"] > 0
    finally:
        os.unlink(path)


def test_detect_and_certify_auto_delimiter_consistency():
    """
    Verify that detect_from_file and certify_file use the same auto-detected
    delimiter consistently. With content-based detection, semicolon-separated
    CSV should be parsed correctly by both functions.
    """
    from s2s_standard_v1_3.adapters.column_detect import (
        detect_from_file, certify_file,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_delimiter_test_file(path, ";", header=True)

        detection = detect_from_file(path)
        result = certify_file(path, **_certify_kwargs())

        # Both should succeed consistently: detection finds 7 columns,
        # certification processes windows without error.
        assert detection["shape"][1] == 7, (
            f"Auto-detection failed for semicolon CSV: {detection['shape']}"
        )
        assert "error" not in detection, (
            f"detect_from_file returned error: {detection.get('error')}"
        )
        assert "error" not in result, (
            f"certify_file returned error for semicolon CSV: {result.get('error')}"
        )
        assert result["total_windows"] > 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Quoted CSV delimiter robustness tests
#
# _sniff_delimiter() uses raw line.count() which counts delimiter characters
# inside quoted fields. These tests verify whether the detector correctly
# ignores quoted content or is fooled by competing delimiters inside quotes.
# ---------------------------------------------------------------------------


def test_sniff_delimiter_ignores_semicolons_in_quoted_comma_csv():
    """
    Comma-separated CSV with quoted fields containing multiple semicolons.
    Expected delimiter: comma (not semicolon).
    """
    from s2s_standard_v1_3.adapters.column_detect import _sniff_delimiter

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        # Write comma-separated CSV with ALL fields quoted
        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)
            # Header
            writer.writerow(["time", "acc_x", "acc_y", "acc_z", "note"])
            # Data rows with quoted fields containing MANY semicolons
            # 5 columns = 4 commas per line, but quoted field has 6 semicolons
            for i in range(20):
                writer.writerow([
                    i * 0.01,
                    0.5 + i * 0.001,
                    -0.3 + i * 0.001,
                    9.81 + i * 0.001,
                    "a;b;c;d;e;f",  # Quoted field with 6 semicolons > 4 commas
                ])

        # Verify the file actually contains quoted fields with semicolons
        with open(path, "r") as f:
            content = f.read()
        assert '"a;b;c;d;e;f"' in content, (
            "Test file does not contain properly quoted field with semicolons"
        )

        detected = _sniff_delimiter(path)
        assert detected == ",", (
            f"Expected comma delimiter, got {detected!r}. "
            "_sniff_delimiter may be counting semicolons inside quoted fields."
        )
    finally:
        os.unlink(path)


def test_sniff_delimiter_ignores_commas_in_quoted_semicolon_csv():
    """
    Semicolon-separated CSV with quoted fields containing multiple commas.
    Expected delimiter: semicolon (not comma).
    """
    from s2s_standard_v1_3.adapters.column_detect import _sniff_delimiter

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        # Write semicolon-separated CSV with ALL fields quoted
        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";", quoting=csv.QUOTE_ALL)
            # Header
            writer.writerow(["time", "acc_x", "acc_y", "acc_z", "note"])
            # Data rows with quoted fields containing MANY commas
            # 5 columns = 4 semicolons per line, but quoted field has 6 commas
            for i in range(20):
                writer.writerow([
                    i * 0.01,
                    0.5 + i * 0.001,
                    -0.3 + i * 0.001,
                    9.81 + i * 0.001,
                    "a,b,c,d,e,f",  # Quoted field with 6 commas > 4 semicolons
                ])

        # Verify the file actually contains quoted fields with commas
        with open(path, "r") as f:
            content = f.read()
        assert '"a,b,c,d,e,f"' in content, (
            "Test file does not contain properly quoted field with commas"
        )

        detected = _sniff_delimiter(path)
        assert detected == ";", (
            f"Expected semicolon delimiter, got {detected!r}. "
            "_sniff_delimiter may be counting commas inside quoted fields."
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Scientific notation header detection regression tests
#
# detect_from_file() and certify_file() use alphabetical-character detection
# to identify headers: any(c.isalpha() for c in first_line). This incorrectly
# identifies scientific notation like 1.0e-3 as textual header content because
# 'e' is alphabetic. These tests verify that valid numeric sensor rows are
# never discarded due to scientific notation.
# ---------------------------------------------------------------------------


def _write_sci_notation_test_file(path, delimiter=",", n_rows=256,
                                  first_row_sci=True, header=False):
    """
    Write sensor data with optional scientific notation in first row.

    Format: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    First row uses scientific notation if first_row_sci=True.
    """
    import random
    r = random.Random(42)
    lines = []
    if header:
        lines.append(delimiter.join([
            "time_s", "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
        ]))
    for i in range(n_rows):
        t = i * 0.01
        ax = r.gauss(0.5, 0.3)
        ay = r.gauss(-0.3, 0.3)
        az = r.gauss(9.81, 0.2)
        gx = r.gauss(0.0, 0.05)
        gy = r.gauss(0.0, 0.05)
        gz = r.gauss(0.0, 0.05)
        if i == 0 and first_row_sci:
            # First row with scientific notation
            row = [
                f"{t:.2e}",      # 0.00e+00
                f"{ax:.2e}",     # scientific notation
                f"{ay:.2e}",
                f"{az:.2e}",
                f"{gx:.2e}",
                f"{gy:.2e}",
                f"{gz:.2e}",
            ]
        else:
            row = [f"{v:.6f}" for v in [t, ax, ay, az, gx, gy, gz]]
        lines.append(delimiter.join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _write_uppercase_sci_test_file(path, delimiter=",", n_rows=256):
    """Write sensor data with uppercase scientific notation (E-4) in first row."""
    import random
    r = random.Random(42)
    lines = []
    for i in range(n_rows):
        t = i * 0.01
        ax = r.gauss(0.5, 0.3)
        ay = r.gauss(-0.3, 0.3)
        az = r.gauss(9.81, 0.2)
        gx = r.gauss(0.0, 0.05)
        gy = r.gauss(0.0, 0.05)
        gz = r.gauss(0.0, 0.05)
        if i == 0:
            # Uppercase scientific notation: E-4, E+00, etc.
            row = [
                f"{t:.4E}",      # 0.0000E+00
                f"{ax:.4E}",
                f"{ay:.4E}",
                f"{az:.4E}",
                f"{gx:.4E}",
                f"{gy:.4E}",
                f"{gz:.4E}",
            ]
        else:
            row = [f"{v:.6f}" for v in [t, ax, ay, az, gx, gy, gz]]
        lines.append(delimiter.join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _write_nan_inf_test_file(path, delimiter=",", n_rows=256):
    """Write sensor data with NaN and infinity in first row."""
    import random
    r = random.Random(42)
    lines = []
    for i in range(n_rows):
        t = i * 0.01
        ax = r.gauss(0.5, 0.3)
        ay = r.gauss(-0.3, 0.3)
        az = r.gauss(9.81, 0.2)
        gx = r.gauss(0.0, 0.05)
        gy = r.gauss(0.0, 0.05)
        gz = r.gauss(0.0, 0.05)
        if i == 0:
            # First row with NaN and infinity
            row = [
                f"{t:.6f}",
                "nan",           # NaN value
                f"{ay:.6f}",
                "inf",           # Infinity value
                f"{gx:.6f}",
                "-inf",          # Negative infinity
                f"{gz:.6f}",
            ]
        else:
            row = [f"{v:.6f}" for v in [t, ax, ay, az, gx, gy, gz]]
        lines.append(delimiter.join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _write_nan_inf_certify_test_file(path, delimiter=",", n_rows=256):
    """
    Write sensor data with NaN/Inf in an extra non-critical column.

    Format: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, extra
    The extra column (index 7) contains NaN/Inf in the first row only.
    All accel and gyro columns contain finite values.
    """
    import random
    r = random.Random(42)
    lines = []
    for i in range(n_rows):
        t = i * 0.01
        ax = r.gauss(0.5, 0.3)
        ay = r.gauss(-0.3, 0.3)
        az = r.gauss(9.81, 0.2)
        gx = r.gauss(0.0, 0.05)
        gy = r.gauss(0.0, 0.05)
        gz = r.gauss(0.0, 0.05)
        extra = r.gauss(0.0, 0.1)
        if i == 0:
            # First row: NaN/Inf in extra column only
            row = [
                f"{t:.6f}",
                f"{ax:.6f}",
                f"{ay:.6f}",
                f"{az:.6f}",
                f"{gx:.6f}",
                f"{gy:.6f}",
                f"{gz:.6f}",
                "nan",           # NaN in extra column
            ]
        else:
            row = [f"{v:.6f}" for v in [t, ax, ay, az, gx, gy, gz, extra]]
        lines.append(delimiter.join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --- detect_from_file() scientific notation tests ---------------------------


def test_detect_from_file_preserves_sci_notation_first_row():
    """detect_from_file must not discard first row with scientific notation."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_sci_notation_test_file(path, ",", n_rows=256, first_row_sci=True)
        result = detect_from_file(path)
        # Should detect 7 columns and preserve all 256 rows
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
        # max_rows=500 in detect_from_file, so should get all 256 rows
        assert result["shape"][0] == 256, (
            f"Expected 256 rows, got {result['shape'][0]}. "
            "First row may have been incorrectly discarded as header."
        )
    finally:
        os.unlink(path)


def test_detect_from_file_uppercase_sci_notation():
    """detect_from_file must handle uppercase scientific notation (E-4)."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_uppercase_sci_test_file(path, ",", n_rows=256)
        result = detect_from_file(path)
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
        assert result["shape"][0] == 256, (
            f"Expected 256 rows, got {result['shape'][0]}. "
            "First row with uppercase scientific notation may have been discarded."
        )
    finally:
        os.unlink(path)


def test_detect_from_file_nan_inf_first_row():
    """detect_from_file must treat NaN and infinity as numeric, not headers."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_nan_inf_test_file(path, ",", n_rows=256)
        result = detect_from_file(path)
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
        assert result["shape"][0] == 256, (
            f"Expected 256 rows, got {result['shape'][0]}. "
            "First row with NaN/inf may have been incorrectly discarded as header."
        )
    finally:
        os.unlink(path)


def test_detect_from_file_genuine_header_still_detected():
    """detect_from_file must still recognize genuine textual headers."""
    from s2s_standard_v1_3.adapters.column_detect import detect_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_sci_notation_test_file(path, ",", n_rows=256,
                                      first_row_sci=False, header=True)
        result = detect_from_file(path)
        assert result["shape"][1] == 7, (
            f"Expected 7 columns, got shape {result['shape']}"
        )
        # With header, should get 256 data rows (header skipped)
        assert result["shape"][0] == 256, (
            f"Expected 256 data rows (header skipped), got {result['shape'][0]}"
        )
    finally:
        os.unlink(path)


# --- certify_file() scientific notation tests -------------------------------


def test_certify_file_preserves_sci_notation_first_row():
    """
    certify_file must not discard first row with scientific notation.
    With 256 rows and max_windows=1, one complete window requires all 256 samples.
    """
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_sci_notation_test_file(path, ",", n_rows=256, first_row_sci=True)
        kwargs = _certify_kwargs()
        kwargs["max_windows"] = 1  # Exactly one window of 256 samples
        result = certify_file(path, **kwargs)
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        # Should have exactly 1 window (256 samples)
        assert result["total_windows"] == 1, (
            f"Expected 1 window (256 samples), got {result['total_windows']}. "
            "First row may have been incorrectly discarded, leaving only 255 samples."
        )
    finally:
        os.unlink(path)


def test_certify_file_uppercase_sci_notation():
    """certify_file must handle uppercase scientific notation (E-4)."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_uppercase_sci_test_file(path, ",", n_rows=256)
        kwargs = _certify_kwargs()
        kwargs["max_windows"] = 1
        result = certify_file(path, **kwargs)
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        assert result["total_windows"] == 1, (
            f"Expected 1 window, got {result['total_windows']}. "
            "First row with uppercase scientific notation may have been discarded."
        )
    finally:
        os.unlink(path)


def test_certify_file_nan_inf_first_row():
    """
    certify_file must treat NaN and infinity as numeric, not headers.
    NaN/Inf are placed in an extra non-critical column (index 7) so that
    all accel and gyro columns contain finite values.
    """
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_nan_inf_certify_test_file(path, ",", n_rows=256)
        kwargs = _certify_kwargs()
        kwargs["max_windows"] = 1
        result = certify_file(path, **kwargs)
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        assert result["total_windows"] == 1, (
            f"Expected 1 window, got {result['total_windows']}. "
            "First row with NaN/inf in non-critical column may have been incorrectly discarded."
        )
    finally:
        os.unlink(path)


def test_certify_file_genuine_header_still_detected():
    """certify_file must still recognize genuine textual headers."""
    from s2s_standard_v1_3.adapters.column_detect import certify_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_sci_notation_test_file(path, ",", n_rows=256,
                                      first_row_sci=False, header=True)
        kwargs = _certify_kwargs()
        kwargs["max_windows"] = 1
        result = certify_file(path, **kwargs)
        assert "error" not in result, (
            f"Unexpected error: {result.get('error')}"
        )
        # With header skipped, should still have 256 data rows = 1 window
        assert result["total_windows"] == 1, (
            f"Expected 1 window, got {result['total_windows']}"
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Unit normalization and SI conversion regression tests
#
# Tests for _canonical_unit(), _unit_from_header(), _resolve_unit(), and
# actual SI conversion in certify_file(). Verifies that all supported unit
# aliases resolve correctly, header units are recognized, and actual data
# arrays are converted to SI units before passing to PhysicsEngine.
# ---------------------------------------------------------------------------


# --- A. Explicit unit aliases ------------------------------------------------


@pytest.mark.parametrize("unit,expected", [
    # Acceleration aliases
    ("m/s2", "m/s2"),
    ("m/sec2", "m/s2"),
    ("mps2", "m/s2"),
    ("ms2", "m/s2"),
    ("g", "g"),
    ("mg", "milli-g"),
    ("milli-g", "milli-g"),
    ("milligravity", "milli-g"),
])
def test_canonical_unit_accel_aliases(unit, expected):
    """All supported acceleration unit aliases resolve to canonical units."""
    from s2s_standard_v1_3.adapters.column_detect import _canonical_unit
    assert _canonical_unit(unit, "accel") == expected


@pytest.mark.parametrize("unit,expected", [
    # Gyroscope aliases
    ("rad/s", "rad/s"),
    ("rads", "rad/s"),
    ("radsec", "rad/s"),
    ("radian/s", "rad/s"),
    ("radians/s", "rad/s"),
    ("deg/s", "deg/s"),
    ("degree/s", "deg/s"),
    ("degrees/s", "deg/s"),
    ("dps", "deg/s"),
    ("degsec", "deg/s"),
    ("mrad/s", "mrad/s"),
    ("mrads", "mrad/s"),
    ("millirad/s", "mrad/s"),
    ("mdps", "mdps"),
    ("millideg/s", "mdps"),
    ("millidegree/s", "mdps"),
    ("millidegrees/s", "mdps"),
])
def test_canonical_unit_gyro_aliases(unit, expected):
    """All supported gyroscope unit aliases resolve to canonical units."""
    from s2s_standard_v1_3.adapters.column_detect import _canonical_unit
    assert _canonical_unit(unit, "gyro") == expected


# --- B. Header unit recognition ----------------------------------------------


@pytest.mark.parametrize("header,expected", [
    # Acceleration header formats
    ("acc_x [m/s2]", "m/s2"),
    ("acc_x [m/s^2]", "m/s2"),
    ("acc_x [m/s²]", "m/s2"),
    ("acc_x [m s^-2]", "m/s2"),  # Known gap: may fail
    ("acc_x [g]", "g"),
    ("acc_x [mg]", "milli-g"),
    ("acc_x [milli-g]", "milli-g"),
])
def test_unit_from_header_accel(header, expected):
    """Acceleration units are recognized from column headers."""
    from s2s_standard_v1_3.adapters.column_detect import _unit_from_header
    result = _unit_from_header(header, "accel")
    assert result == expected, (
        f"Header {header!r} should resolve to {expected!r}, got {result!r}"
    )


@pytest.mark.parametrize("header,expected", [
    # Gyroscope header formats
    ("gyro_x [rad/s]", "rad/s"),
    ("gyro_x [deg/s]", "deg/s"),
    ("gyro_x [°/s]", "deg/s"),
    ("gyro_x [degree/s]", "deg/s"),
    ("gyro_x [degrees/s]", "deg/s"),
    ("gyro_x [mrad/s]", "mrad/s"),
    ("gyro_x [mdps]", "mdps"),
])
def test_unit_from_header_gyro(header, expected):
    """Gyroscope units are recognized from column headers."""
    from s2s_standard_v1_3.adapters.column_detect import _unit_from_header
    result = _unit_from_header(header, "gyro")
    assert result == expected, (
        f"Header {header!r} should resolve to {expected!r}, got {result!r}"
    )


# --- C. Actual SI conversion --------------------------------------------------


def test_si_conversion_accel_g(monkeypatch):
    """Acceleration in g is multiplied by 9.80665 before PhysicsEngine."""
    _, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="g",
        gyro_unit="rad/s",
    )

    assert len(calls) == 1
    actual = np.asarray(calls[0][0]["accel"][0])
    # Raw values from _run_column_adapter: [8.0, 9.0, 9.81]
    expected = np.asarray([8.0, 9.0, 9.81]) * 9.80665
    assert np.allclose(actual, expected), (
        f"Expected g conversion (x9.80665), got {actual}"
    )


def test_si_conversion_accel_milli_g(monkeypatch):
    """Acceleration in milli-g is multiplied by 0.00980665 before PhysicsEngine."""
    _, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="milli-g",
        gyro_unit="rad/s",
    )

    assert len(calls) == 1
    actual = np.asarray(calls[0][0]["accel"][0])
    expected = np.asarray([8.0, 9.0, 9.81]) * 0.00980665
    assert np.allclose(actual, expected), (
        f"Expected milli-g conversion (x0.00980665), got {actual}"
    )


def test_si_conversion_gyro_deg_per_s(monkeypatch):
    """Gyroscope in deg/s is multiplied by pi/180 before PhysicsEngine."""
    _, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="deg/s",
    )

    assert len(calls) == 1
    actual = np.asarray(calls[0][0]["gyro"][0])
    # Raw gyro values from _run_column_adapter: [0.1, 0.2, 0.3]
    expected = np.asarray([0.1, 0.2, 0.3]) * (np.pi / 180.0)
    assert np.allclose(actual, expected), (
        f"Expected deg/s conversion (xpi/180), got {actual}"
    )


def test_si_conversion_gyro_mrad_per_s(monkeypatch):
    """Gyroscope in mrad/s is multiplied by 0.001 before PhysicsEngine."""
    _, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="mrad/s",
    )

    assert len(calls) == 1
    actual = np.asarray(calls[0][0]["gyro"][0])
    expected = np.asarray([0.1, 0.2, 0.3]) * 0.001
    assert np.allclose(actual, expected), (
        f"Expected mrad/s conversion (x0.001), got {actual}"
    )


def test_si_conversion_gyro_mdps(monkeypatch):
    """Gyroscope in mdps is multiplied by pi/180000 before PhysicsEngine."""
    _, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="mdps",
    )

    assert len(calls) == 1
    actual = np.asarray(calls[0][0]["gyro"][0])
    expected = np.asarray([0.1, 0.2, 0.3]) * (np.pi / 180000.0)
    assert np.allclose(actual, expected), (
        f"Expected mdps conversion (xpi/180000), got {actual}"
    )


# --- D. Provenance and precedence ---------------------------------------------


def test_explicit_unit_takes_precedence_over_header(monkeypatch):
    """Explicit unit argument overrides conflicting header declaration."""
    names = [
        "time",
        "acc_x [g]",      # Header says g
        "acc_y [g]",
        "acc_z [g]",
        "gyro_x [rad/s]",
        "gyro_y [rad/s]",
        "gyro_z [rad/s]",
    ]

    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",  # Explicit says m/s2 (conflicts with header)
        gyro_unit="rad/s",
        header_names=names,
    )

    assert "error" not in result
    # Explicit unit should win
    assert result["unit_metadata"]["accel"]["unit"] == "m/s2"
    assert result["unit_metadata"]["accel"]["status"] == "explicit"
    assert result["unit_metadata"]["accel"]["source"] == "argument"


def test_explicit_unit_metadata_status_and_source(monkeypatch):
    """Explicit unit metadata reports status='explicit' and source='argument'."""
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="g",
        gyro_unit="deg/s",
    )

    assert "error" not in result
    assert result["unit_metadata"]["accel"]["status"] == "explicit"
    assert result["unit_metadata"]["accel"]["source"] == "argument"
    assert result["unit_metadata"]["gyro"]["status"] == "explicit"
    assert result["unit_metadata"]["gyro"]["source"] == "argument"


def test_header_unit_metadata_status_and_source(monkeypatch):
    """Header-derived metadata reports status='inferred_from_header' and source='column_headers'."""
    names = [
        "time",
        "acc_x [m/s^2]",
        "acc_y [m/s^2]",
        "acc_z [m/s^2]",
        "gyro_x [rad/s]",
        "gyro_y [rad/s]",
        "gyro_z [rad/s]",
    ]

    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit=None,
        gyro_unit=None,
        header_names=names,
    )

    assert "error" not in result
    assert result["unit_metadata"]["accel"]["status"] == "inferred_from_header"
    assert result["unit_metadata"]["accel"]["source"] == "column_headers"
    assert result["unit_metadata"]["gyro"]["status"] == "inferred_from_header"
    assert result["unit_metadata"]["gyro"]["source"] == "column_headers"


def test_conflicting_gyro_axis_units_remain_unknown(monkeypatch):
    """Conflicting gyroscope axis units result in unit_unknown."""
    names = [
        "time",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x [rad/s]",
        "gyro_y [deg/s]",  # Conflicts with gyro_x
        "gyro_z [rad/s]",
    ]

    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit=None,
        header_names=names,
    )

    assert "error" not in result
    assert result["unit_metadata"]["gyro"]["status"] == "unit_unknown"
    # Gyro should not be passed to PhysicsEngine
    assert "gyro" not in calls[0][0]


def test_unknown_gyro_unit_not_passed_to_engine(monkeypatch):
    """Unknown gyroscope units are not passed to PhysicsEngine."""
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit=None,  # No explicit unit, no headers
    )

    assert "error" not in result
    assert result["unit_metadata"]["gyro"]["status"] == "unit_unknown"
    assert "gyro" not in calls[0][0]


def test_unsupported_explicit_gyro_unit_fails_closed(monkeypatch):
    """Unsupported explicit gyroscope units fail closed."""
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="invalid_unit",
    )

    assert result["tier"] == "N/A"
    assert "error" in result
    assert result["unit_metadata"]["gyro"]["status"] == "invalid_explicit"
    assert calls == []


# ---------------------------------------------------------------------------
# Cross-sensor column overlap regression tests
#
# A source column cannot be assigned to both accelerometer and gyroscope axes.
# These tests verify that certify_file() rejects overlapping column mappings.
# ---------------------------------------------------------------------------


def test_accel_gyro_overlap_rejected(monkeypatch):
    """
    certify_file must reject when the same column is assigned to both
    accelerometer and gyroscope axes.

    accel_columns=[1, 2, 3], gyro_columns=[3, 4, 5] — column 3 overlaps.
    """
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[3, 4, 5],  # Overlaps with accel column 3
        accel_unit="m/s2",
        gyro_unit="rad/s",
        accel_columns=[1, 2, 3],
    )

    # Should be rejected with an error about overlapping columns
    assert "error" in result, (
        "Expected error for overlapping accel/gyro columns, "
        f"got result: {result}"
    )
    assert result["tier"] == "N/A"
    # Should not call PhysicsEngine
    assert calls == []


def test_accel_gyro_nonoverlapping_mapping_accepted(monkeypatch):
    """
    certify_file must accept non-overlapping accel and gyro column mappings.

    accel_columns=[1, 2, 3], gyro_columns=[4, 5, 6] — six distinct columns.
    """
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="rad/s",
        accel_columns=[1, 2, 3],
    )

    # Should succeed normally
    assert "error" not in result, (
        f"Unexpected error for valid non-overlapping mapping: {result.get('error')}"
    )
    assert result["total_windows"] > 0
    assert len(calls) > 0


# ---------------------------------------------------------------------------
# Invalid gyro and modality-state regression tests
#
# Tests for gyro sample validation, missing-modality distinctions, and
# explicit partial gyro mapping rejection.
# ---------------------------------------------------------------------------


def test_complete_gyro_all_finite_samples(monkeypatch):
    """Complete three-axis gyro mapping with all finite samples is 'complete'."""
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="rad/s",
    )

    assert "error" not in result
    assert result["gyro_window_availability"]["complete"] > 0
    assert "gyro" in calls[0][0]


def test_complete_gyro_some_nan_samples_not_passed(monkeypatch):
    """Complete mapping with some NaN samples is not passed as complete gyro."""
    # Create data with NaN in gyro column
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        missing_gyro_row=40,  # One NaN in gyro Y
        accel_unit="m/s2",
        gyro_unit="rad/s",
    )

    assert "error" not in result
    # Should be partial, not complete
    assert result["gyro_window_availability"]["partial"] > 0
    assert result["gyro_window_availability"]["complete"] == 0
    # Gyro should not be in imu_raw for partial windows
    assert "gyro" not in calls[0][0]


def test_complete_gyro_all_invalid_samples_distinguishable_from_absent(monkeypatch):
    """
    Complete mapping with all invalid gyro samples must be 'invalid', not 'absent'.

    The gyro columns exist and units are known, but no samples are usable.
    This must be distinguishable from a genuinely absent gyro.
    """
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit="rad/s",
        corrupt_all_gyro=True,  # All gyro samples are NaN
    )

    assert "error" not in result

    # The gyro availability should be 'invalid', not 'absent' or 'partial'
    # This test will FAIL against current production code (which returns 'absent')
    # This is intentional: we want a failing regression test before fixing.
    assert result["gyro_window_availability"]["invalid"] > 0, (
        f"Expected 'invalid' availability for all-invalid gyro, "
        f"got: {result['gyro_window_availability']}"
    )

    # Gyro should not be passed to physics engine as valid measurement
    assert "gyro" not in calls[0][0]

    # Acceleration processing should be unaffected
    assert "accel" in calls[0][0]
    assert len(calls[0][0]["accel"]) == 256


def test_no_gyro_columns_absent_behavior(monkeypatch):
    """Dataset with no gyro columns retains 'absent' behavior."""
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[],  # No gyro columns
        accel_unit="m/s2",
        gyro_unit=None,
    )

    assert "error" not in result
    assert result["gyro_window_availability"]["absent"] > 0
    assert "gyro" not in calls[0][0]


def test_complete_gyro_unknown_units_unit_unknown(monkeypatch):
    """Complete mapping with unknown units retains 'unit_unknown' behavior."""
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5, 6],
        accel_unit="m/s2",
        gyro_unit=None,  # No explicit unit, no headers
    )

    assert "error" not in result
    assert result["unit_metadata"]["gyro"]["status"] == "unit_unknown"
    assert result["gyro_window_availability"]["unit_unknown"] > 0
    assert "gyro" not in calls[0][0]


def test_explicit_partial_gyro_mapping_rejected(monkeypatch):
    """Explicit gyro mapping with only 1-2 columns is rejected."""
    result, calls, _ = _run_column_adapter(
        monkeypatch,
        gyro_cols=[4, 5],  # Only 2 columns
        accel_unit="m/s2",
        gyro_unit="rad/s",
    )

    # Should be rejected because explicit gyro requires exactly 3 columns
    assert "error" in result
    assert result["tier"] == "N/A"
    assert calls == []
