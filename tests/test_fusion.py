"""
S2S FusionCertifier — Pytest Test Suite
Tests multi-sensor fusion: IMU | EMG | PPG | Thermal | LiDAR
Run: pytest tests/test_fusion.py -v
"""
import time
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier, FUSION_MIN_STREAMS, SENSOR_WEIGHTS

# ── MOCK CERT FACTORIES ────────────────────────────────────────────────────────

NOW_NS = time.time_ns()
DUR_NS = 5_000_000_000  # 5 seconds


def imu_cert(tier="SILVER", cv=0.15, synthetic=False):
    return {
        "tier": tier,
        "sensor_type": "IMU",
        "flags": ["SUSPECT_SYNTHETIC"] if synthetic else [],
        "frame_start_ts_ns": NOW_NS,
        "frame_end_ts_ns":   NOW_NS + DUR_NS,
        "duration_ms":       5000.0,
        "metrics": {"cv": cv},
        "physical_law_score": 70 if tier == "SILVER" else 90,
    }


def emg_cert(tier="SILVER", burst_frac=0.20, synthetic=False):
    return {
        "tier": tier,
        "sensor_type": "EMG",
        "flags": ["SUSPECT_SYNTHETIC"] if synthetic else [],
        "frame_start_ts_ns": NOW_NS,
        "frame_end_ts_ns":   NOW_NS + DUR_NS,
        "duration_ms":       5000.0,
        "notes": {"mean_burst_frac": burst_frac},
    }


def ppg_cert(tier="SILVER", hr_bpm=70, hrv_ms=25.0, has_pulse=True, synthetic=False):
    return {
        "tier": tier,
        "sensor_type": "PPG",
        "flags": ["SUSPECT_SYNTHETIC"] if synthetic else [],
        "frame_start_ts_ns": NOW_NS,
        "frame_end_ts_ns":   NOW_NS + DUR_NS,
        "duration_ms":       5000.0,
        "per_channel": {
            "ch0": {"has_pulse": has_pulse},
            "ch1": {"has_pulse": has_pulse},
        },
        "vitals": {
            "heart_rate_bpm": hr_bpm,
            "hrv_rmssd_ms":   hrv_ms,
            "breathing_rate_hz": 0.25,
        },
    }


def thermal_cert(tier="SILVER", human_present=True, synthetic=False):
    return {
        "tier": tier,
        "sensor_type": "THERMAL",
        "flags": ["SUSPECT_SYNTHETIC"] if synthetic else [],
        "frame_start_ts_ns": NOW_NS,
        "frame_end_ts_ns":   NOW_NS + DUR_NS,
        "duration_ms":       5000.0,
        "human_presence": {"human_present": human_present, "confidence": 0.9},
    }


def lidar_cert(tier="SILVER", synthetic=False):
    return {
        "tier": tier,
        "sensor_type": "LIDAR",
        "flags": ["SUSPECT_SYNTHETIC"] if synthetic else [],
        "frame_start_ts_ns": NOW_NS,
        "frame_end_ts_ns":   NOW_NS + DUR_NS,
        "duration_ms":       5000.0,
    }


# ── IMPORT & CONSTANTS ─────────────────────────────────────────────────────────

class TestFusionImport:

    def test_fusion_module_imports(self):
        """FusionCertifier must import without error."""
        assert FusionCertifier is not None

    def test_min_streams_constant(self):
        """FUSION_MIN_STREAMS must be >= 2."""
        assert FUSION_MIN_STREAMS >= 2

    def test_sensor_weights_defined(self):
        """All 5 sensor types must have weights."""
        for s in ("IMU", "EMG", "PPG", "THERMAL", "LIDAR"):
            assert s in SENSOR_WEIGHTS, f"{s} missing from SENSOR_WEIGHTS"

    def test_biological_sensors_weighted_higher(self):
        """EMG and PPG are biological — must outweigh LiDAR."""
        assert SENSOR_WEIGHTS["EMG"] > SENSOR_WEIGHTS["LIDAR"]
        assert SENSOR_WEIGHTS["PPG"] > SENSOR_WEIGHTS["LIDAR"]


# ── INSTANTIATION ──────────────────────────────────────────────────────────────

class TestFusionInstantiation:

    def test_basic_instantiation(self):
        fc = FusionCertifier()
        assert fc is not None

    def test_device_id_stored(self):
        fc = FusionCertifier(device_id="glove_v2")
        assert fc.device_id == "glove_v2"

    def test_session_id_stored(self):
        fc = FusionCertifier(session_id="session_001")
        assert fc.session_id == "session_001"

    def test_reset_clears_streams(self):
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_emg_cert(emg_cert())
        fc.reset()
        result = fc.certify()
        assert result["tier"] == "REJECTED"
        assert "INSUFFICIENT_STREAMS" in result["flags"]


# ── INSUFFICIENT STREAMS ───────────────────────────────────────────────────────

class TestInsufficientStreams:

    def test_zero_streams_rejected(self):
        fc = FusionCertifier()
        result = fc.certify()
        assert result["tier"] == "REJECTED"
        assert result["human_in_loop_score"] == 0
        assert "INSUFFICIENT_STREAMS" in result["flags"]

    def test_one_stream_rejected(self):
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        result = fc.certify()
        assert result["tier"] == "REJECTED"
        assert "INSUFFICIENT_STREAMS" in result["flags"]

    def test_two_streams_minimum_accepted(self):
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_emg_cert(emg_cert())
        result = fc.certify()
        assert "INSUFFICIENT_STREAMS" not in result["flags"]
        assert result["n_streams"] == 2


# ── RESULT STRUCTURE ───────────────────────────────────────────────────────────

class TestResultStructure:

    def _two_stream_result(self):
        fc = FusionCertifier(device_id="test_dev", session_id="s001")
        fc.add_imu_cert(imu_cert())
        fc.add_ppg_cert(ppg_cert())
        return fc.certify()

    def test_required_keys_present(self):
        r = self._two_stream_result()
        for key in ("status", "tier", "sensor_type", "human_in_loop_score",
                    "n_streams", "streams", "coherence_checks", "flags",
                    "notes", "device_id", "tool", "issued_at_ns"):
            assert key in r, f"Missing key: {key}"

    def test_sensor_type_is_fusion(self):
        assert self._two_stream_result()["sensor_type"] == "FUSION"

    def test_tool_identifier(self):
        assert "fusion" in self._two_stream_result()["tool"].lower()

    def test_tier_is_valid(self):
        r = self._two_stream_result()
        assert r["tier"] in ("GOLD", "SILVER", "BRONZE", "REJECTED")

    def test_status_matches_tier(self):
        r = self._two_stream_result()
        if r["tier"] == "REJECTED":
            assert r["status"] == "FAIL"
        else:
            assert r["status"] == "PASS"

    def test_score_in_range(self):
        r = self._two_stream_result()
        assert 0 <= r["human_in_loop_score"] <= 100

    def test_n_streams_correct(self):
        assert self._two_stream_result()["n_streams"] == 2

    def test_streams_list_populated(self):
        r = self._two_stream_result()
        assert isinstance(r["streams"], list)
        assert len(r["streams"]) == 2

    def test_coherence_checks_dict(self):
        r = self._two_stream_result()
        assert isinstance(r["coherence_checks"], dict)

    def test_notes_has_pair_counts(self):
        r = self._two_stream_result()
        notes = r["notes"]
        assert "total_pairs_checked" in notes
        assert "coherent_pairs" in notes
        assert "failed_pairs" in notes

    def test_device_id_in_result(self):
        assert self._two_stream_result()["device_id"] == "test_dev"

    def test_issued_at_ns_is_recent(self):
        r = self._two_stream_result()
        assert r["issued_at_ns"] > NOW_NS


# ── SYNTHETIC DETECTION ────────────────────────────────────────────────────────

class TestSyntheticDetection:

    def test_synthetic_imu_rejected(self):
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert(synthetic=True))
        fc.add_emg_cert(emg_cert())
        result = fc.certify()
        assert result["tier"] == "REJECTED"
        assert result["human_in_loop_score"] == 0

    def test_synthetic_emg_rejected(self):
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_emg_cert(emg_cert(synthetic=True))
        result = fc.certify()
        assert result["tier"] == "REJECTED"

    def test_synthetic_ppg_rejected(self):
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_ppg_cert(ppg_cert(synthetic=True))
        result = fc.certify()
        assert result["tier"] == "REJECTED"

    def test_all_synthetic_rejected(self):
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert(synthetic=True))
        fc.add_emg_cert(emg_cert(synthetic=True))
        fc.add_ppg_cert(ppg_cert(synthetic=True))
        result = fc.certify()
        assert result["tier"] == "REJECTED"
        assert result["human_in_loop_score"] == 0

    def test_synthetic_flag_in_result(self):
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert(synthetic=True))
        fc.add_emg_cert(emg_cert())
        result = fc.certify()
        assert any("SYNTHETIC" in f for f in result["flags"])


# ── IMU + EMG COHERENCE ────────────────────────────────────────────────────────

class TestIMUEMGCoherence:

    def test_motion_and_muscle_pass(self):
        """Active IMU + active EMG bursts = coherent."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert(tier="SILVER", cv=0.15))
        fc.add_emg_cert(emg_cert(tier="SILVER", burst_frac=0.20))
        result = fc.certify()
        assert result["human_in_loop_score"] > 0
        assert result["tier"] != "REJECTED"

    def test_rejected_imu_reduces_score(self):
        """REJECTED IMU should lower fusion score."""
        fc_good = FusionCertifier()
        fc_good.add_imu_cert(imu_cert(tier="SILVER"))
        fc_good.add_emg_cert(emg_cert(tier="SILVER"))
        good = fc_good.certify()

        fc_bad = FusionCertifier()
        fc_bad.add_imu_cert(imu_cert(tier="REJECTED", cv=0.0))
        fc_bad.add_emg_cert(emg_cert(tier="SILVER"))
        bad = fc_bad.certify()

        assert good["human_in_loop_score"] >= bad["human_in_loop_score"]

    def test_zero_burst_frac_reduces_confidence(self):
        """EMG with no bursts (idle muscle) should score lower."""
        fc_active = FusionCertifier()
        fc_active.add_imu_cert(imu_cert())
        fc_active.add_emg_cert(emg_cert(burst_frac=0.25))
        active = fc_active.certify()

        fc_idle = FusionCertifier()
        fc_idle.add_imu_cert(imu_cert())
        fc_idle.add_emg_cert(emg_cert(burst_frac=0.0))
        idle = fc_idle.certify()

        assert active["human_in_loop_score"] >= idle["human_in_loop_score"]


# ── IMU + PPG COHERENCE ────────────────────────────────────────────────────────

class TestIMUPPGCoherence:

    def test_motion_and_pulse_pass(self):
        """Active IMU + valid pulse = strong human proof."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert(tier="SILVER", cv=0.15))
        fc.add_ppg_cert(ppg_cert(tier="SILVER", hr_bpm=70, has_pulse=True))
        result = fc.certify()
        assert result["human_in_loop_score"] > 0
        assert result["tier"] != "REJECTED"

    def test_no_pulse_fails_coherence(self):
        """PPG with no pulse detected should fail IMU_PPG coherence."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_ppg_cert(ppg_cert(has_pulse=False, hr_bpm=None))
        result = fc.certify()
        # No pulse = coherence fail, score should be lower
        assert result["human_in_loop_score"] < 80

    def test_hr_out_of_range_fails(self):
        """HR outside 30-220 BPM should fail coherence."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_ppg_cert(ppg_cert(hr_bpm=300, has_pulse=True))
        result = fc.certify()
        checks = result["coherence_checks"]
        imu_ppg = next((v for k, v in checks.items() if "ppg" in k.lower()), None)
        if imu_ppg:
            assert imu_ppg.get("fail") == "HR_OUT_OF_RANGE" or \
                   result["human_in_loop_score"] < 80


# ── EMG + PPG COHERENCE ────────────────────────────────────────────────────────

class TestEMGPPGCoherence:

    def test_dual_biological_confirmed(self):
        """EMG bursts + PPG pulse + HRV = strongest human proof."""
        fc = FusionCertifier()
        fc.add_emg_cert(emg_cert(tier="GOLD", burst_frac=0.30))
        fc.add_ppg_cert(ppg_cert(tier="GOLD", hr_bpm=65, hrv_ms=30.0, has_pulse=True))
        result = fc.certify()
        assert result["human_in_loop_score"] > 50
        assert result["tier"] in ("GOLD", "SILVER", "BRONZE")

    def test_low_hrv_reduces_confidence(self):
        """Very low HRV suggests synthetic PPG — should reduce confidence."""
        fc_hrv = FusionCertifier()
        fc_hrv.add_emg_cert(emg_cert(burst_frac=0.20))
        fc_hrv.add_ppg_cert(ppg_cert(hrv_ms=25.0, has_pulse=True))
        good = fc_hrv.certify()

        fc_no_hrv = FusionCertifier()
        fc_no_hrv.add_emg_cert(emg_cert(burst_frac=0.20))
        fc_no_hrv.add_ppg_cert(ppg_cert(hrv_ms=0.0001, has_pulse=True))
        bad = fc_no_hrv.certify()

        assert good["human_in_loop_score"] >= bad["human_in_loop_score"]


# ── THREE STREAM FUSION ────────────────────────────────────────────────────────

class TestThreeStreamFusion:

    def test_imu_emg_ppg_fusion(self):
        """IMU + EMG + PPG — the core human motion combo."""
        fc = FusionCertifier(device_id="prosthetic_v1")
        fc.add_imu_cert(imu_cert(tier="SILVER", cv=0.15))
        fc.add_emg_cert(emg_cert(tier="SILVER", burst_frac=0.20))
        fc.add_ppg_cert(ppg_cert(tier="SILVER", hr_bpm=72, hrv_ms=20.0))
        result = fc.certify()
        assert result["n_streams"] == 3
        assert result["notes"]["total_pairs_checked"] == 3
        assert result["human_in_loop_score"] > 0

    def test_imu_emg_thermal_fusion(self):
        """IMU + EMG + Thermal — robot arm with heat signature."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert(tier="SILVER"))
        fc.add_emg_cert(emg_cert(tier="SILVER", burst_frac=0.15))
        fc.add_thermal_cert(thermal_cert(tier="SILVER", human_present=True))
        result = fc.certify()
        assert result["n_streams"] == 3
        assert result["tier"] in ("GOLD", "SILVER", "BRONZE", "REJECTED")

    def test_three_streams_more_pairs(self):
        """3 streams = 3 pairs checked."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_emg_cert(emg_cert())
        fc.add_ppg_cert(ppg_cert())
        result = fc.certify()
        assert result["notes"]["total_pairs_checked"] == 3

    def test_three_stream_scores_higher_than_two(self):
        """More valid streams should generally score higher."""
        fc2 = FusionCertifier()
        fc2.add_imu_cert(imu_cert(tier="SILVER", cv=0.15))
        fc2.add_emg_cert(emg_cert(tier="SILVER", burst_frac=0.20))
        score2 = fc2.certify()["human_in_loop_score"]

        fc3 = FusionCertifier()
        fc3.add_imu_cert(imu_cert(tier="SILVER", cv=0.15))
        fc3.add_emg_cert(emg_cert(tier="SILVER", burst_frac=0.20))
        fc3.add_ppg_cert(ppg_cert(tier="SILVER", hr_bpm=70, hrv_ms=20.0))
        score3 = fc3.certify()["human_in_loop_score"]

        assert score3 >= score2, f"3 streams ({score3}) should score >= 2 streams ({score2})"


# ── FIVE STREAM FUSION ─────────────────────────────────────────────────────────

class TestFiveStreamFusion:

    def test_all_five_sensors_gold_path(self):
        """All 5 sensors at GOLD = maximum human proof."""
        fc = FusionCertifier(device_id="full_suit", session_id="demo_001")
        fc.add_imu_cert(imu_cert(tier="GOLD", cv=0.20))
        fc.add_emg_cert(emg_cert(tier="GOLD", burst_frac=0.35))
        fc.add_ppg_cert(ppg_cert(tier="GOLD", hr_bpm=68, hrv_ms=35.0))
        fc.add_thermal_cert(thermal_cert(tier="GOLD", human_present=True))
        fc.add_lidar_cert(lidar_cert(tier="GOLD"))
        result = fc.certify()
        assert result["n_streams"] == 5
        assert result["notes"]["total_pairs_checked"] == 10  # C(5,2)
        assert result["human_in_loop_score"] > 50
        assert result["tier"] in ("GOLD", "SILVER")

    def test_five_streams_ten_pairs(self):
        """5 sensors = C(5,2) = 10 coherence pairs."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_emg_cert(emg_cert())
        fc.add_ppg_cert(ppg_cert())
        fc.add_thermal_cert(thermal_cert())
        fc.add_lidar_cert(lidar_cert())
        result = fc.certify()
        assert result["notes"]["total_pairs_checked"] == 10

    def test_one_synthetic_in_five_rejects_all(self):
        """One synthetic stream should reject the entire fusion."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert(tier="GOLD", synthetic=True))  # <-- bad
        fc.add_emg_cert(emg_cert(tier="GOLD"))
        fc.add_ppg_cert(ppg_cert(tier="GOLD"))
        fc.add_thermal_cert(thermal_cert(tier="GOLD"))
        fc.add_lidar_cert(lidar_cert(tier="GOLD"))
        result = fc.certify()
        assert result["tier"] == "REJECTED"
        assert result["human_in_loop_score"] == 0


# ── TIER PROGRESSION ──────────────────────────────────────────────────────────

class TestTierProgression:

    def test_gold_tier_requires_multiple_streams(self):
        """GOLD requires strong multi-stream coherence — not achievable with 1 stream."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        result = fc.certify()
        assert result["tier"] != "GOLD"  # 1 stream can't reach GOLD

    def test_all_rejected_streams_rejected(self):
        """Two REJECTED streams = fusion REJECTED."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert(tier="REJECTED", cv=0.0))
        fc.add_emg_cert(emg_cert(tier="REJECTED", burst_frac=0.0))
        result = fc.certify()
        assert result["tier"] == "REJECTED"

    def test_gold_streams_score_higher_than_bronze(self):
        """GOLD-tier inputs should produce higher score than BRONZE-tier inputs."""
        fc_gold = FusionCertifier()
        fc_gold.add_imu_cert(imu_cert(tier="GOLD", cv=0.20))
        fc_gold.add_ppg_cert(ppg_cert(tier="GOLD", hr_bpm=68, hrv_ms=35.0))
        gold_score = fc_gold.certify()["human_in_loop_score"]

        fc_bronze = FusionCertifier()
        fc_bronze.add_imu_cert(imu_cert(tier="BRONZE", cv=0.05))
        fc_bronze.add_ppg_cert(ppg_cert(tier="BRONZE", hr_bpm=68, hrv_ms=6.0))
        bronze_score = fc_bronze.certify()["human_in_loop_score"]

        assert gold_score >= bronze_score, \
            f"GOLD inputs ({gold_score}) should score >= BRONZE inputs ({bronze_score})"


# ── PERFORMANCE & STABILITY ────────────────────────────────────────────────────

class TestFusionPerformance:

    def test_certify_is_fast(self):
        """100 fusion certifications should complete under 1 second."""
        fc = FusionCertifier()
        fc.add_imu_cert(imu_cert())
        fc.add_emg_cert(emg_cert())
        fc.add_ppg_cert(ppg_cert())

        import time as t
        start = t.time()
        for _ in range(100):
            fc.certify()
        elapsed = t.time() - start
        assert elapsed < 1.0, f"100 certify() calls took {elapsed:.2f}s"

    def test_certify_is_deterministic(self):
        """Same inputs must always produce same score and tier."""
        fc = FusionCertifier(device_id="test")
        fc.add_imu_cert(imu_cert())
        fc.add_emg_cert(emg_cert())
        fc.add_ppg_cert(ppg_cert())
        r1 = fc.certify()
        r2 = fc.certify()
        assert r1["tier"] == r2["tier"]
        assert r1["human_in_loop_score"] == r2["human_in_loop_score"]

    def test_add_stream_chaining(self):
        """add_*_cert methods should support method chaining."""
        result = (FusionCertifier()
                  .add_imu_cert(imu_cert())
                  .add_emg_cert(emg_cert())
                  .certify())
        assert "tier" in result

    def test_extra_kwargs_dont_crash(self):
        """FusionCertifier should accept unknown kwargs gracefully."""
        fc = FusionCertifier(device_id="x", session_id="y", future_param="z")
        fc.add_imu_cert(imu_cert())
        fc.add_ppg_cert(ppg_cert())
        result = fc.certify()
        assert "tier" in result
