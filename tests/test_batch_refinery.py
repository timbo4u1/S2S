"""Regression tests for batch refinery timestamp and missing-modality handling."""
import s2s_standard_v1_3.batch_refinery as refinery


class RecordingPhysicsEngine:
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def certify(self, imu_raw, segment):
        self.calls.append((imu_raw, segment))
        return {
            "tier": "BRONZE",
            "score": 45,
            "assessment_status": "PARTIAL",
            "tier_scope": "ASSESSED_CHECKS_ONLY",
            "laws_passed": ["jerk_bounds"],
            "laws_failed": [],
            "laws_not_assessed": ["rigid_body_kinematics"],
            "n_laws_assessed": 1,
            "n_laws_not_assessed": 1,
        }


def test_refinery_timestamps_continue_across_windows_and_dont_reset(monkeypatch):
    hz = 200
    acc = [[0.0, 0.0, 9.81] for _ in range(refinery.WIN * 2)]
    monkeypatch.setattr(refinery, "load_file", lambda *args, **kwargs: (acc, hz))
    monkeypatch.setattr(refinery, "PhysicsEngine", RecordingPhysicsEngine)
    RecordingPhysicsEngine.instances.clear()

    results = refinery.certify_file_windows("sample.csv")

    assert len(results) == 2
    calls = RecordingPhysicsEngine.instances[0].calls
    assert len(calls) == 2
    first, second = calls[0][0], calls[1][0]

    assert first["timestamps_ns"] == [
        int(i * 1e9 / hz) for i in range(refinery.WIN)
    ]
    assert second["timestamps_ns"] == [
        int((refinery.WIN + i) * 1e9 / hz)
        for i in range(refinery.WIN)
    ]
    assert second["timestamps_ns"][0] > first["timestamps_ns"][-1]
    assert "gyro" not in first
    assert "gyro" not in second


def test_refinery_report_exposes_partial_assessment_scope(monkeypatch):
    hz = 200
    acc = [[0.0, 0.0, 9.81] for _ in range(refinery.WIN)]
    monkeypatch.setattr(refinery, "load_file", lambda *args, **kwargs: (acc, hz))
    monkeypatch.setattr(refinery, "PhysicsEngine", RecordingPhysicsEngine)
    RecordingPhysicsEngine.instances.clear()

    result = refinery.certify_file_windows("sample.csv")[0]

    assert result["assessment_status"] == "PARTIAL"
    assert result["tier_scope"] == "ASSESSED_CHECKS_ONLY"
    assert result["laws_not_assessed"] == "rigid_body_kinematics"
    assert result["n_laws_assessed"] == 1
    assert result["n_laws_not_assessed"] == 1
