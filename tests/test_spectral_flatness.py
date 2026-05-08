"""
Tests for Law 11: Spectral Flatness (Wiener Entropy).
Human motion: peaked spectrum (flatness 0.07-0.53)
Gaussian noise: flat spectrum (flatness 0.55-0.65)
"""
import math
import random


def test_gaussian_fails():
    """Gaussian noise has flat spectrum — should fail."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_spectral_flatness
    rr = random.Random(42)
    acc = [[rr.gauss(0,2)+(9.81 if k==2 else 0) for k in range(3)] for _ in range(256)]
    ts  = [int(i*1e9/100) for i in range(256)]
    passed, conf, detail = check_spectral_flatness({'accel':acc,'timestamps_ns':ts})
    assert not passed, f"Gaussian should fail: flatness={detail.get('flatness')}"
    assert detail['flatness'] > 0.54


def test_human_motion_passes():
    """Structured human motion has peaked spectrum — should pass."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_spectral_flatness
    acc = [[math.sin(i*0.1)*2+math.sin(i*0.5)*0.3,
            math.cos(i*0.1)*1.5,
            9.81+math.sin(i*0.07)*0.2] for i in range(256)]
    ts  = [int(i*1e9/100) for i in range(256)]
    passed, conf, detail = check_spectral_flatness({'accel':acc,'timestamps_ns':ts})
    assert passed, f"Human motion should pass: flatness={detail.get('flatness')}"
    assert detail['flatness'] < 0.54


def test_short_data_skips():
    """Less than 64 samples skips."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_spectral_flatness
    acc = [[0.1, 0.0, 9.81]] * 32
    ts  = [int(i*1e9/100) for i in range(32)]
    passed, conf, detail = check_spectral_flatness({'accel':acc,'timestamps_ns':ts})
    assert passed
    assert detail.get('skip')


def test_threshold_is_0_54():
    """Threshold should be 0.54."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_spectral_flatness
    acc = [[math.sin(i*0.1), 0.0, 9.81] for i in range(256)]
    ts  = [int(i*1e9/100) for i in range(256)]
    _, _, detail = check_spectral_flatness({'accel':acc,'timestamps_ns':ts})
    assert detail.get('threshold') == 0.54


def test_multiple_gaussian_seeds():
    """Most Gaussian seeds fail spectral flatness."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_spectral_flatness
    failed = 0
    for seed in range(20):
        rr = random.Random(seed)
        acc = [[rr.gauss(0,2)+(9.81 if k==2 else 0) for k in range(3)] for _ in range(256)]
        ts  = [int(i*1e9/100) for i in range(256)]
        passed, _, _ = check_spectral_flatness({'accel':acc,'timestamps_ns':ts})
        if not passed:
            failed += 1
    assert failed >= 10, f"Only {failed}/20 seeds failed — spectral flatness catches ~60% of Gaussian"


def test_ninapro_passes():
    """Real NinaPro passes spectral flatness."""
    try:
        import scipy.io as sio
        import glob
    except ImportError:
        return
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_spectral_flatness
    mats = sorted(glob.glob('/Users/timbo/ninapro_db5/s1/S1_E1_A1.mat'))
    if not mats:
        return
    d = sio.loadmat(mats[0])
    acc = d['acc'].astype(float)[:256].tolist()
    ts  = [int(i*1e9/2000) for i in range(256)]
    passed, conf, detail = check_spectral_flatness({'accel':acc,'timestamps_ns':ts})
    assert passed, f"NinaPro should pass: flatness={detail.get('flatness')}"
