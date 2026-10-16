"""Regression tests for tri-state physics-law outcomes."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine


def _accel_only(n=256, hz=100):
    timestamps = [int(i * 1e9 / hz) for i in range(n)]
    accel = [
        [
            0.5 * math.sin(2 * math.pi * 1.5 * i / hz),
            0.3 * math.cos(2 * math.pi * 1.2 * i / hz),
            9.81 + 0.1 * math.sin(2 * math.pi * 0.8 * i / hz),
        ]
        for i in range(n)
    ]
    return {"timestamps_ns": timestamps, "accel": accel}


def test_missing_gyro_checks_are_not_assessed_not_passed():
    result = PhysicsEngine().certify(imu_raw=_accel_only(), segment="forearm")

    for law in (
        "rigid_body_kinematics",
        "imu_internal_consistency",
        "innovation_kurtosis",
    ):
        assert result["law_statuses"][law] == "NOT_ASSESSED"
        assert law in result["laws_not_assessed"]
        assert law not in result["laws_passed"]
        assert law not in result["laws_failed"]
        assert result["law_details"][law]["status"] == "NOT_ASSESSED"

    assert result["assessment_status"] == "PARTIAL"
    assert result["n_laws_checked"] == len(result["laws_checked"])
    assert result["n_laws_assessed"] == len(result["laws_passed"]) + len(result["laws_failed"])
    assert result["n_laws_not_assessed"] == len(result["laws_not_assessed"])
    assert result["n_laws_assessed"] + result["n_laws_not_assessed"] == result["n_laws_checked"]
    assert result["tier_scope"] == "ASSESSED_CHECKS_ONLY"


def test_empty_input_is_unverified_not_a_pass():
    result = PhysicsEngine().certify()

    assert result["tier"] == "UNVERIFIED"
    assert result["status"] == "FAIL"
    assert result["assessment_status"] == "UNVERIFIED"
    assert result["n_laws_assessed"] == 0
    assert result["physical_law_score"] == 0
    assert result["laws_passed"] == []
    assert result["laws_failed"] == []
    assert result["laws_not_assessed"] == result["laws_checked"]
