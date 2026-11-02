"""Scoring invariants: a FAILED law never raises the quality score,
and GOLD allows no failed law. All law checks are stubbed so only the
aggregation logic is under test."""
import random
import s2s_standard_v1_3.s2s_physics_v1_3 as phys

LAW_CHECKS = (
    "check_continuity", "check_resonance", "check_jerk", "check_rigid_body",
    "check_imu_consistency", "check_cross_axis_cohesion", "check_autocorrelation",
    "check_spectral_flatness", "check_pointwise_jerk", "check_sensor_freeze",
    "check_powerline_interference", "check_intra_window_splice",
    "check_innovation_kurtosis",
)


def _imu(n=256, hz=100):
    rng = random.Random(0)
    return {
        "timestamps_ns": [int(i * 1e9 / hz) for i in range(n)],
        "accel": [[rng.gauss(0, 1), rng.gauss(0, 1), 9.81] for _ in range(n)],
        "gyro": [[rng.gauss(0, 0.05) for _ in range(3)] for _ in range(n)],
    }


def _stub(monkeypatch, name, ok, conf):
    monkeypatch.setattr(phys, name, lambda *a, **k: (ok, conf, {"law": name}))


def _certify(monkeypatch, law15):
    for name in LAW_CHECKS:
        _stub(monkeypatch, name, True, 80)
    _stub(monkeypatch, "check_intra_window_splice", *law15)
    return phys.PhysicsEngine().certify(imu_raw=_imu(), segment="forearm")


def test_failed_law_never_raises_score(monkeypatch):
    passed = _certify(monkeypatch, (True, 75))
    failed = _certify(monkeypatch, (False, 90))  # detected splice, confidence 90
    assert "intra_window_splice" in failed["laws_failed"]
    assert failed["physical_law_score"] <= passed["physical_law_score"]


def test_gold_requires_no_failed_law(monkeypatch):
    assert _certify(monkeypatch, (True, 80))["tier"] == "GOLD"  # control: all pass
    r = _certify(monkeypatch, (False, 90))
    assert "intra_window_splice" in r["laws_failed"]
    assert r["tier"] != "GOLD"
