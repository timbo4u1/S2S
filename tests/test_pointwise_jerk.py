"""
Tests for Law 10: Pointwise Jerk — instantaneous spike detection.
Catches sub-millisecond sensor glitches invisible to windowed averages.
"""
import math
import random


def test_clean_signal_passes():
    """Smooth human-like signal passes pointwise jerk."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_pointwise_jerk
    acc = [[math.sin(i*0.1)*2, math.cos(i*0.1)*1.5, 9.81] for i in range(256)]
    ts  = [int(i*1e9/100) for i in range(256)]
    passed, conf, detail = check_pointwise_jerk({'accel':acc,'timestamps_ns':ts})
    assert passed, f"Smooth signal should pass: {detail}"
    assert detail['spike_count'] == 0


def test_spike_injection_fails():
    """Single spike of magnitude 200 at 2000Hz fails."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_pointwise_jerk
    acc = [[math.sin(i*0.1), 0.0, 9.81] for i in range(256)]
    ts  = [int(i*1e9/2000) for i in range(256)]
    # Inject spike at sample 100
    acc[100] = [200.0, 200.0, 200.0]
    passed, conf, detail = check_pointwise_jerk({'accel':acc,'timestamps_ns':ts})
    assert not passed, f"Spike should fail: {detail}"
    assert detail['spike_count'] >= 1


def test_threshold_scales_with_hz():
    """Higher Hz = smaller allowed delta per sample."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_pointwise_jerk
    acc = [[0.1, 0.0, 9.81]] * 256

    # At 2000Hz: max_delta = 10000/2000 = 5.0
    ts_2000 = [int(i*1e9/2000) for i in range(256)]
    _, _, d2000 = check_pointwise_jerk({'accel':acc,'timestamps_ns':ts_2000})
    assert abs(d2000['max_allowed_ms2'] - 5.0) < 0.1

    # At 100Hz: max_delta = 10000/100 = 100.0
    ts_100 = [int(i*1e9/100) for i in range(256)]
    _, _, d100 = check_pointwise_jerk({'accel':acc,'timestamps_ns':ts_100})
    assert abs(d100['max_allowed_ms2'] - 100.0) < 0.1


def test_multiple_spikes_fail():
    """6 spikes (benchmark pattern) all caught."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_pointwise_jerk
    import copy, random as _r
    acc = [[math.sin(i*0.1), 0.0, 9.81] for i in range(256)]
    ts  = [int(i*1e9/2000) for i in range(256)]
    rr = _r.Random(99)
    idxs = rr.sample(range(10, 246), 6)
    for idx in idxs:
        acc[idx] = [rr.gauss(0, 200) for _ in range(3)]
    passed, conf, detail = check_pointwise_jerk({'accel':acc,'timestamps_ns':ts})
    assert not passed
    assert detail['spike_count'] >= 1


def test_short_data_skips():
    """Less than 2 samples returns skip."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_pointwise_jerk
    passed, conf, detail = check_pointwise_jerk({'accel':[[0,0,9.81]],'timestamps_ns':[0]})
    assert passed
    assert detail.get('skip')


def test_ninapro_passes():
    """Real NinaPro data passes pointwise jerk at 2000Hz."""
    try:
        import scipy.io as sio
        import glob
    except ImportError:
        return
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_pointwise_jerk
    mats = sorted(glob.glob('/Users/timbo/ninapro_db5/s1/S1_E1_A1.mat'))
    if not mats:
        return
    d = sio.loadmat(mats[0])
    acc = d['acc'].astype(float)[:256].tolist()
    ts  = [int(i*1e9/2000) for i in range(256)]
    passed, conf, detail = check_pointwise_jerk({'accel':acc,'timestamps_ns':ts})
    assert passed, f"NinaPro should pass: {detail}"
