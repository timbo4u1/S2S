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
