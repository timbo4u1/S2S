import random
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine


def _imu(n=256, hz=100):
    rng = random.Random(1)
    acc = [[rng.gauss(0, 1), rng.gauss(0, 1), 9.81 + rng.gauss(0, 0.3)] for _ in range(n)]
    gyro = [[rng.gauss(0, 0.05) for _ in range(3)] for _ in range(n)]
    ts = [int(i * 1e9 / hz) for i in range(n)]
    return {"timestamps_ns": ts, "accel": acc, "gyro": gyro}


def test_unknown_segment_is_not_assessed():
    r = PhysicsEngine().certify(imu_raw=_imu(), segment="waist")
    assert "resonance_frequency" in r["laws_not_assessed"]
    assert "resonance_frequency" not in r["laws_passed"]


def test_known_segment_is_still_assessed():
    r = PhysicsEngine().certify(imu_raw=_imu(), segment="forearm")
    assert "resonance_frequency" not in r["laws_not_assessed"]
