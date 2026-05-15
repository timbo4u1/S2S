"""
Tests for Law 13: Sensor Freeze Detection.
Correctness + non-regression + benchmark gate.
"""
import math
import time
import random


def test_sensor_freeze_detected():
    """20 consecutive identical samples → sensor_freeze in laws_failed, score penalised."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
    engine = PhysicsEngine()
    acc = [[0.0, 0.0, 9.81]] * 20 + \
          [[math.sin(i*0.1)*2, math.cos(i*0.1), 9.81] for i in range(80)]
    ts = [int(i*1e9/200) for i in range(100)]
    r = engine.certify({"timestamps_ns": ts, "accel": acc, "gyro": [[0,0,0]]*100})
    assert 'sensor_freeze' in r['laws_failed'], "Frozen segment must be detected"
    assert r['tier'] != 'GOLD', "Frozen window must not reach GOLD"


def test_clean_motion_not_flagged():
    """Smooth human motion never triggers sensor_freeze."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
    engine = PhysicsEngine()
    acc = [[math.sin(i*0.1)*2, math.cos(i*0.1)*1.5,
            9.81 + math.sin(i*0.07)*0.2] for i in range(256)]
    ts = [int(i*1e9/100) for i in range(256)]
    r = engine.certify({"timestamps_ns": ts, "accel": acc,
                        "gyro": [[0.05*math.sin(i*0.1), 0, 0] for i in range(256)]})
    assert 'sensor_freeze' not in r['laws_failed'], \
        f"Clean motion wrongly flagged: {r['laws_failed']}"


def test_low_motion_with_noise_not_frozen():
    """Low-motion with realistic ADC noise (not identical values) not flagged."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
    rr = random.Random(42)
    engine = PhysicsEngine()
    # Near-static but values vary by ≥ 0.001 m/s² (real sensor noise floor)
    acc = [[rr.gauss(0, 0.005), rr.gauss(0, 0.005),
            9.81 + rr.gauss(0, 0.002)] for _ in range(256)]
    ts = [int(i*1e9/100) for i in range(256)]
    r = engine.certify({"timestamps_ns": ts, "accel": acc,
                        "gyro": [[0, 0, 0]]*256})
    assert 'sensor_freeze' not in r['laws_failed'], \
        f"Low-motion with noise wrongly frozen: {r['laws_failed']}"


def test_existing_coherence_rejection_unaffected():
    """Gaussian noise (dual coherence failure) still rejects — no conflict."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
    engine = PhysicsEngine()
    rr = random.Random(50)
    acc  = [[rr.gauss(0, 8) for _ in range(3)] for _ in range(256)]
    gyro = [[rr.gauss(0, 8) for _ in range(3)] for _ in range(256)]
    ts   = [int(j*1e9/50) for j in range(256)]
    r = engine.certify({"timestamps_ns": ts, "accel": acc, "gyro": gyro})
    assert r['tier'] == 'REJECTED', f"Gaussian must still REJECT: {r['tier']}"
    assert 'sensor_freeze' not in r['laws_failed']  # wrong law, right result


def test_latency_budget():
    """certify() must stay under 5ms after Law 13 addition (with warmup)."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
    engine = PhysicsEngine()
    n = 256
    acc  = [[math.sin(i*0.1)*2, math.cos(i*0.1)*1.5, 9.81] for i in range(n)]
    gyro = [[0.05*math.sin(i*0.1), 0.0, 0.0] for i in range(n)]
    ts   = [int(i*1e9/100) for i in range(n)]
    for _ in range(5): engine.certify({"timestamps_ns": ts, "accel": acc, "gyro": gyro})
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        engine.certify({"timestamps_ns": ts, "accel": acc, "gyro": gyro})
    avg_ms = (time.perf_counter() - t0) / N * 1000
    print(f"\n  Law 13 latency: {avg_ms:.2f}ms/window")
    assert avg_ms < 5.0, f"Latency {avg_ms:.2f}ms exceeds 5ms budget"
