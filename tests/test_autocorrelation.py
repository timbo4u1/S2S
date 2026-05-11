"""
Tests for Law 12: Temporal Autocorrelation (Lag-1 ACF).
Human motion: ACF[1] > 0.20 (neuromuscular time constants >100ms)
Gaussian noise: ACF[1] ≈ 0 (independent samples, iid)
"""
import math
import random


def test_gaussian_fails():
    """Gaussian noise has no temporal structure — should fail."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_autocorrelation
    rr = random.Random(50)
    acc = [[rr.gauss(0, 8) for _ in range(3)] for _ in range(256)]
    ts  = [int(i*1e9/50) for i in range(256)]
    passed, conf, detail = check_autocorrelation({'accel': acc, 'timestamps_ns': ts})
    assert not passed, f"Gaussian should fail: ACF[1]={detail.get('acf1')}"
    assert detail['acf1'] < 0.20


def test_human_motion_passes():
    """Smooth structured motion has high temporal ACF — should pass."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_autocorrelation
    acc = [
        [math.sin(i*0.1)*2 + math.sin(i*0.05)*0.5,
         math.cos(i*0.1)*1.5,
         9.81 + math.sin(i*0.07)*0.2]
        for i in range(256)
    ]
    ts = [int(i*1e9/100) for i in range(256)]
    passed, conf, detail = check_autocorrelation({'accel': acc, 'timestamps_ns': ts})
    assert passed, f"Smooth motion should pass: ACF[1]={detail.get('acf1')}"
    assert detail['acf1'] > 0.20


def test_random_walk_passes():
    """Random walk (cumulative sum) has persistent structure like human motion."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_autocorrelation
    rr = random.Random(42)
    acc = [[0.0, 0.0, 9.81]] * 256
    ax, ay = 0.0, 0.0
    for i in range(256):
        ax += rr.gauss(0, 0.1)
        ay += rr.gauss(0, 0.1)
        acc[i] = [ax, ay, 9.81]
    ts = [int(i*1e9/100) for i in range(256)]
    passed, conf, detail = check_autocorrelation({'accel': acc, 'timestamps_ns': ts})
    assert passed, f"Random walk should pass: ACF[1]={detail.get('acf1')}"


def test_short_data_skips():
    """Fewer than 32 samples returns skip."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_autocorrelation
    acc = [[0.1, 0.0, 9.81]] * 20
    ts  = [int(i*1e9/100) for i in range(20)]
    passed, conf, detail = check_autocorrelation({'accel': acc, 'timestamps_ns': ts})
    assert passed
    assert detail.get('skip')


def test_threshold_is_0_20():
    """Threshold must be 0.20."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_autocorrelation
    acc = [[math.sin(i*0.1), 0.0, 9.81] for i in range(256)]
    ts  = [int(i*1e9/100) for i in range(256)]
    _, _, detail = check_autocorrelation({'accel': acc, 'timestamps_ns': ts})
    assert detail.get('threshold') == 0.20


def test_all_benchmark_seeds_fail():
    """All 5 benchmark synthetic seeds fail ACF check."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_autocorrelation
    for i in range(5):
        rr = random.Random(i + 50)
        acc = [[rr.gauss(0, 8) for _ in range(3)] for _ in range(256)]
        ts  = [int(j*1e9/50) for j in range(256)]
        passed, _, detail = check_autocorrelation({'accel': acc, 'timestamps_ns': ts})
        assert not passed, f"Benchmark seed {i+50} should fail: ACF[1]={detail.get('acf1')}"


def test_dual_coherence_rejects_seed_64():
    """Seed 64 previously escaped — dual coherence rule must now catch it."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
    engine = PhysicsEngine()
    rr = random.Random(64)
    acc  = [[rr.gauss(0, 8) for _ in range(3)] for _ in range(256)]
    gyro = [[rr.gauss(0, 8) for _ in range(3)] for _ in range(256)]
    ts   = [int(j*1e9/50) for j in range(256)]
    r = engine.certify({"timestamps_ns": ts, "accel": acc, "gyro": gyro})
    assert r['tier'] == 'REJECTED', f"Seed 64 must be REJECTED: got {r['tier']}"


def test_wesad_wrist_passes():
    """WESAD wrist at rest (tightest real case, ACF=0.2325) passes threshold."""
    try:
        import pickle, glob
        import numpy as np
    except ImportError:
        return
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_autocorrelation
    files = sorted(glob.glob('/Users/timbo/wesad_data/WESAD/S*/S*.pkl'))[:1]
    if not files:
        return
    with open(files[0], 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    acc = d['signal']['wrist']['ACC'].astype(float)[:256].tolist()
    ts  = [int(i*1e9/32) for i in range(256)]
    passed, conf, detail = check_autocorrelation({'accel': acc, 'timestamps_ns': ts})
    assert passed, f"WESAD wrist should pass: ACF[1]={detail.get('acf1')}"
    assert detail['acf1'] > 0.20
