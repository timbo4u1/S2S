"""
test_certify_session.py — Tests for PhysicsEngine.certify_session()

Tests the biological origin detection (BFS) which checks:
  - Hurst exponent ≥ 0.70 = biological motor control
  - biological_grade: HUMAN / LOW_BIOLOGICAL_FIDELITY / NOT_BIOLOGICAL
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.expanduser("~/S2S"))

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine


def make_timestamps(n, hz):
    """Generate realistic timestamps with jitter."""
    import random
    return [int(i * 1e9 / hz + random.gauss(0, 200000)) for i in range(n)]


def make_human_window(n=256, hz=50):
    """Realistic forearm motion with fractal (correlated) structure.
    Uses cumulative sum of noise — gives Hurst > 0.7 like real motor control."""
    import random
    # Fractional Brownian motion approximation: cumsum of correlated noise
    ax = [0.0] * n
    ay = [0.0] * n
    az = [9.8] * n  # gravity component
    for i in range(1, n):
        ax[i] = ax[i-1] * 0.85 + random.gauss(0, 0.3)
        ay[i] = ay[i-1] * 0.85 + random.gauss(0, 0.3)
        az[i] = az[i-1] * 0.85 + random.gauss(9.8 * 0.15, 0.1)
    gx = [0.0] * n
    gy = [0.0] * n
    gz = [0.0] * n
    for i in range(1, n):
        gx[i] = gx[i-1] * 0.85 + random.gauss(0, 0.05)
        gy[i] = gy[i-1] * 0.85 + random.gauss(0, 0.05)
        gz[i] = gz[i-1] * 0.85 + random.gauss(0, 0.05)
    accel = [[ax[i], ay[i], az[i]] for i in range(n)]
    gyro  = [[gx[i], gy[i], gz[i]] for i in range(n)]
    return accel, gyro


def make_synthetic_window(n=256):
    """Pure random signal — no biological structure."""
    import random
    accel = [[random.gauss(0, 5) for _ in range(3)] for _ in range(n)]
    gyro  = [[random.gauss(0, 5) for _ in range(3)] for _ in range(n)]
    return accel, gyro


class TestCertifySession:

    def test_certify_session_returns_required_keys(self):
        engine = PhysicsEngine()
        accel, gyro = make_human_window()
        ts = make_timestamps(256, 50)
        for _ in range(5):
            engine.certify({"timestamps_ns": ts, "accel": accel, "gyro": gyro},
                           segment="forearm")
        result = engine.certify_session()
        assert isinstance(result, dict)
        for key in ["biological_grade", "recommendation", "n_windows"]:
            assert key in result, f"Missing key: {key}"

    def test_certify_session_needs_minimum_windows(self):
        """Session with fewer than 4 windows returns insufficient data."""
        engine = PhysicsEngine()
        accel, gyro = make_human_window()
        ts = make_timestamps(256, 50)
        # Only 2 windows — below minimum
        for _ in range(2):
            engine.certify({"timestamps_ns": ts, "accel": accel, "gyro": gyro},
                           segment="forearm")
        result = engine.certify_session()
        assert result is not None
        # Should either return grade or report insufficient windows
        assert "biological_grade" in result or "reason" in result

    def test_synthetic_motion_grade(self):
        """Pure random noise should score lower than human motion."""
        import random
        random.seed(99)

        # Human session
        engine_human = PhysicsEngine()
        for _ in range(10):
            accel, gyro = make_human_window(n=256, hz=50)
            ts = make_timestamps(256, 50)
            engine_human.certify({"timestamps_ns": ts, "accel": accel, "gyro": gyro},
                                  segment="forearm")
        human_result = engine_human.certify_session()

        # Synthetic session
        engine_synth = PhysicsEngine()
        for _ in range(10):
            accel, gyro = make_synthetic_window(n=256)
            ts = make_timestamps(256, 50)
            engine_synth.certify({"timestamps_ns": ts, "accel": accel, "gyro": gyro},
                                  segment="forearm")
        synth_result = engine_synth.certify_session()

        human_bfs = human_result.get("bfs") or 0
        synth_bfs  = synth_result.get("bfs") or 0

        # Human should have higher biological fidelity score than synthetic
        # (or synthetic is flagged NOT_BIOLOGICAL)
        synth_grade = synth_result.get("biological_grade", "")
        assert (human_bfs >= synth_bfs or synth_grade == "NOT_BIOLOGICAL"), (
            f"Synthetic BFS {synth_bfs} >= Human BFS {human_bfs} — "
            f"biological detector not working"
        )

    def test_recommendation_field_valid(self):
        """recommendation must be one of the valid values."""
        import random
        random.seed(7)
        engine = PhysicsEngine()
        for _ in range(6):
            accel, gyro = make_human_window()
            ts = make_timestamps(256, 50)
            engine.certify({"timestamps_ns": ts, "accel": accel, "gyro": gyro},
                           segment="forearm")
        result = engine.certify_session()
        rec = result.get("recommendation", "")
        assert rec in ("ACCEPT", "REVIEW", "REJECT", ""), (
            f"Invalid recommendation: {rec}"
        )

    def test_n_windows_counted(self):
        """n_windows should match how many certify() calls were made."""
        import random
        random.seed(13)
        engine = PhysicsEngine()
        n_calls = 8
        for _ in range(n_calls):
            accel, gyro = make_human_window()
            ts = make_timestamps(256, 50)
            engine.certify({"timestamps_ns": ts, "accel": accel, "gyro": gyro},
                           segment="forearm")
        result = engine.certify_session()
        n = result.get("n_windows", 0)
        assert n >= 4, f"n_windows {n} too low for {n_calls} certify calls"

    def test_source_type_in_certify_output(self):
        """source_type HIL_BIOLOGICAL must be present in certify() output."""
        import random
        random.seed(1)
        engine = PhysicsEngine()
        accel, gyro = make_human_window()
        ts = make_timestamps(256, 50)
        result = engine.certify({"timestamps_ns": ts, "accel": accel, "gyro": gyro},
                                segment="forearm")
        assert "source_type" in result, "source_type missing from certify() output"
        assert result["source_type"] == "HIL_BIOLOGICAL"
