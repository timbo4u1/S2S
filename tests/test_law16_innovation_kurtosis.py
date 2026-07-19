"""
tests/test_law16_innovation_kurtosis.py

Law 16 — Innovation Kurtosis / Gaussian Innovation Detection.

Validates that:
- Coupled OU signals (Gaussian innovations) are REJECTED
- Real-like impulse-modulated signals pass
- Acc-only mode (no gyro) skips gracefully — not penalized
- Short windows skip gracefully
- Flat signals fail with correct reason
- Law is present in laws_checked for full IMU input
"""
import sys, os, math, random
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s2s_standard_v1_3.s2s_physics_v1_3 import (
    PhysicsEngine,
    check_innovation_kurtosis,
)
from experiments.data_gen.ou_generator import generate_window, generate_batch


# ── helpers ──────────────────────────────────────────────────────────────────

def _real_like(n=256, hz=100.0, seed=0):
    """Impulse-modulated AR(1) signal — leptokurtic innovations."""
    random.seed(seed)
    state = [0.0, 0.0, 9.81]
    accel = []
    for _ in range(n):
        row = []
        for axis in range(3):
            base = 0.92 * state[axis] + random.gauss(0, 0.4)
            if random.random() < 0.05:
                base += random.choice([-1, 1]) * random.uniform(1.5, 4.0)
            state[axis] = base
            row.append(base)
        row[2] += 9.81
        accel.append(row)
    gyro = [[random.gauss(0, 0.05) + 0.01 for _ in range(3)] for _ in range(n)]
    return {
        "timestamps_ns": [int(i * 1e9 / hz) for i in range(n)],
        "accel": accel,
        "gyro":  gyro,
    }


def _acc_only(n=256, hz=100.0):
    """Clean signal with no gyro — acc-only mode."""
    random.seed(42)
    state = [0.0, 0.0, 9.81]
    accel = []
    for _ in range(n):
        row = []
        for axis in range(3):
            base = 0.92 * state[axis] + random.gauss(0, 0.5)
            if random.random() < 0.05:
                base += random.choice([-1, 1]) * random.uniform(1.5, 3.0)
            state[axis] = base
            row.append(base)
        row[2] += 9.81
        accel.append(row)
    return {
        "timestamps_ns": [int(i * 1e9 / hz) for i in range(n)],
        "accel": accel,
        "gyro":  [[0.0, 0.0, 0.0]] * n,
    }


# ── unit tests on check_innovation_kurtosis() directly ───────────────────────

class TestInnovationKurtosisUnit:

    def test_ou_signal_fails(self):
        """Coupled OU has Gaussian innovations — must fail."""
        for seed in range(10):
            w = generate_window(seed=seed)
            passes, conf, detail = check_innovation_kurtosis(w)
            ek = detail.get("excess_kurtosis", 0) or 0
            assert not passes, (
                f"OU seed={seed} should fail Law 16 "
                f"(excess_kurtosis={ek:.4f}, threshold=0.63)"
            )

    def test_real_like_passes(self):
        """Impulse-modulated signal has leptokurtic innovations — must pass."""
        failures = 0
        for seed in range(20):
            w = _real_like(seed=seed)
            passes, conf, detail = check_innovation_kurtosis(w)
            if not passes:
                failures += 1
        assert failures == 0, f"{failures}/20 real-like windows failed Law 16"

    def test_short_window_skips(self):
        """Windows under 30 samples skip gracefully (PASS, conf=50)."""
        w = {"accel": [[0.1 * i, 0.0, 9.81] for i in range(10)], "gyro": []}
        passes, conf, detail = check_innovation_kurtosis(w)
        assert passes is True
        assert detail["reason"] == "SKIP_SHORT"

    def test_no_gyro_skips(self):
        """Acc-only (zero gyro) skips gracefully."""
        w = _acc_only()
        passes, conf, detail = check_innovation_kurtosis(w)
        assert passes is True
        assert detail["reason"] == "SKIP_NO_GYRO"

    def test_flat_signal_fails(self):
        """Completely flat signal has zero variance — must fail."""
        w = {
            "accel": [[9.81, 0.0, 0.0]] * 256,
            "gyro":  [[0.1, 0.0, 0.0]] * 256,
        }
        passes, conf, detail = check_innovation_kurtosis(w)
        assert not passes
        assert detail["reason"] in ("FLAT_AXIS", "ZERO_VAR",
                                    "GAUSSIAN_INNOVATIONS_DETECTED")

    def test_return_structure(self):
        """Return tuple must always be (bool, int, dict) with expected keys."""
        w = generate_window(seed=0)
        result = check_innovation_kurtosis(w)
        assert isinstance(result, tuple) and len(result) == 3
        passes, conf, detail = result
        assert isinstance(passes, bool)
        assert isinstance(conf, int)
        assert isinstance(detail, dict)
        assert "reason" in detail
        assert "excess_kurtosis" in detail

    def test_ou_kurtosis_near_gaussian(self):
        """OU excess kurtosis should be near 0 (Gaussian by construction)."""
        kurtosis_vals = []
        for seed in range(20):
            w = generate_window(seed=seed)
            _, _, detail = check_innovation_kurtosis(w)
            ek = detail.get("excess_kurtosis")
            if ek is not None:
                kurtosis_vals.append(ek)
        mean_ek = sum(kurtosis_vals) / len(kurtosis_vals)
        assert abs(mean_ek) < 0.5, (
            f"OU mean excess kurtosis={mean_ek:.4f} — expected near 0"
        )

    def test_real_kurtosis_well_above_threshold(self):
        """Real-like signals should have excess kurtosis well above 0.63."""
        kurtosis_vals = []
        for seed in range(20):
            w = _real_like(seed=seed)
            _, _, detail = check_innovation_kurtosis(w)
            ek = detail.get("excess_kurtosis")
            if ek is not None:
                kurtosis_vals.append(ek)
        min_ek = min(kurtosis_vals)
        assert min_ek > 1.0, (
            f"Real-like min excess kurtosis={min_ek:.4f} — expected > 1.0"
        )


# ── integration tests through PhysicsEngine.certify() ────────────────────────

class TestLaw16Integration:

    def test_ou_rejected_by_engine(self):
        """Full engine must REJECT coupled OU signals."""
        pe = PhysicsEngine()
        rejected = 0
        for seed in range(10):
            w = generate_window(seed=seed)
            r = pe.certify(imu_raw=w, segment="forearm")
            if r["tier"] == "REJECTED":
                rejected += 1
        assert rejected == 10, (
            f"Only {rejected}/10 OU windows rejected — "
            f"Law 16 not blocking OU in full engine"
        )

    def test_innovation_kurtosis_in_laws_checked(self):
        """Law 16 must appear in laws_checked for full IMU input."""
        pe = PhysicsEngine()
        w = generate_window(seed=0)
        r = pe.certify(imu_raw=w, segment="forearm")
        assert "innovation_kurtosis" in r["laws_checked"], (
            "innovation_kurtosis missing from laws_checked"
        )

    def test_innovation_kurtosis_in_laws_failed_for_ou(self):
        """innovation_kurtosis must appear in laws_failed for OU signals."""
        pe = PhysicsEngine()
        w = generate_window(seed=0)
        r = pe.certify(imu_raw=w, segment="forearm")
        assert "innovation_kurtosis" in r["laws_failed"], (
            f"innovation_kurtosis not in laws_failed — "
            f"laws_failed={r['laws_failed']}"
        )

    def test_acc_only_not_rejected_by_law16(self):
        """Acc-only data must not be rejected solely because of Law 16."""
        pe = PhysicsEngine()
        w = _acc_only()
        r = pe.certify(imu_raw=w, segment="forearm")
        assert "innovation_kurtosis" not in r.get("laws_failed", []), (
            "Law 16 should skip for acc-only input, not fail it"
        )

    def test_real_like_not_rejected_by_law16(self):
        """Real-like signals must not be rejected by Law 16."""
        pe = PhysicsEngine()
        rejected_by_16 = 0
        for seed in range(10):
            w = _real_like(seed=seed)
            r = pe.certify(imu_raw=w, segment="forearm")
            if "innovation_kurtosis" in r.get("laws_failed", []):
                rejected_by_16 += 1
        assert rejected_by_16 == 0, (
            f"{rejected_by_16}/10 real-like windows failed Law 16 — "
            f"false positive rate too high"
        )

    def test_ou_batch_all_rejected(self):
        """Batch of 30 OU windows — all must be REJECTED."""
        pe = PhysicsEngine()
        windows = generate_batch(30)
        passed = [w for w in windows
                  if pe.certify(imu_raw=w, segment="forearm")["tier"] != "REJECTED"]
        assert len(passed) == 0, (
            f"{len(passed)}/30 OU batch windows were not rejected"
        )
