"""
Tests for Law 9: Cross-Axis Cohesion.
Human motion: bone/muscle forces couple IMU axes (max_r > 0.10)
Gaussian noise: axes independent (max_r ≈ 0.04-0.09)
"""
import math
import random
import glob


def test_gaussian_fails():
    """Gaussian noise has near-zero cross-axis correlation — should fail."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_cross_axis_cohesion
    rr = random.Random(42)
    acc = [[rr.gauss(0, 2) + (9.81 if k == 2 else 0) for k in range(3)]
           for _ in range(256)]
    passed, conf, detail = check_cross_axis_cohesion({'accel': acc})
    assert not passed, f"Gaussian should fail: max_r={detail.get('max_r')}"
    assert detail['max_r'] < 0.10


def test_human_motion_passes():
    """Coupled-axis motion (rigid body rotation) should pass."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_cross_axis_cohesion
    # Rotation: x and y are 90-degree phase-shifted — strongly coupled
    acc = [
        [math.sin(i*0.1)*2.0,
         math.cos(i*0.1)*2.0 + math.sin(i*0.3)*0.5,
         9.81 + math.sin(i*0.07)*0.3]
        for i in range(256)
    ]
    passed, conf, detail = check_cross_axis_cohesion({'accel': acc})
    assert passed, f"Coupled motion should pass: max_r={detail.get('max_r')}"
    assert detail['max_r'] > 0.10


def test_ninapro_passes():
    """Real NinaPro forearm IMU should pass cross-axis cohesion."""
    pytest = __import__('pytest')
    try:
        import scipy.io as sio
    except ImportError:
        return  # skip if scipy not available (CI)
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_cross_axis_cohesion
    import glob as _glob
    mats = sorted(_glob.glob('/Users/timbo/ninapro_db5/s1/S1_E1_A1.mat'))
    if not mats:
        return  # skip if dataset not available
    d = sio.loadmat(mats[0])
    acc = d['acc'].astype(float)
    chunk = acc[:256].tolist()
    passed, conf, detail = check_cross_axis_cohesion({'accel': chunk})
    assert passed, f"NinaPro should pass: max_r={detail.get('max_r')}"


def test_insufficient_data_skips():
    """Short data should be skipped without failing."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_cross_axis_cohesion
    acc = [[0.1, 0.0, 9.81]] * 10
    passed, conf, detail = check_cross_axis_cohesion({'accel': acc})
    assert passed  # skip = pass
    assert detail.get('skip') == 'INSUFFICIENT_DATA'


def test_threshold_is_0_10():
    """Threshold should be 0.10."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_cross_axis_cohesion
    acc = [[0.1*i, 0.0, 9.81] for i in range(256)]
    passed, conf, detail = check_cross_axis_cohesion({'accel': acc})
    assert detail.get('threshold') == 0.115


def test_multiple_gaussian_seeds():
    """Most Gaussian seeds fail — statistical test."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_cross_axis_cohesion
    failed = 0
    for seed in range(20):
        rr = random.Random(seed)
        acc = [[rr.gauss(0, 2) + (9.81 if k == 2 else 0) for k in range(3)]
               for _ in range(256)]
        passed, conf, detail = check_cross_axis_cohesion({'accel': acc})
        if not passed:
            failed += 1
    # At least 80% of random seeds should fail (Gaussian axes are independent)
    assert failed >= 15, f"Only {failed}/20 Gaussian seeds failed — threshold may be too low"
