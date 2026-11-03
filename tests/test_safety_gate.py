import pytest
import s2s_standard_v1_3.s2s_safety_gate as safety_gate


class RecordingPhysicsEngine:
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def certify(self, imu_raw, segment):
        self.calls.append((dict(imu_raw), segment))
        return {
            "tier": "SILVER",
            "physical_law_score": 80,
            "flags": [],
        }


def make_gate(monkeypatch):
    RecordingPhysicsEngine.instances.clear()
    monkeypatch.setattr(
        safety_gate, "PhysicsEngine", RecordingPhysicsEngine
    )
    return safety_gate.RealTimeSafetyGate(
        window_size=2,
        step_size=1,
    )


def evaluate_two_frames(gate, gyro_a, gyro_b):
    gate.push(0, [0.0, 0.0, 9.81], gyro_a)
    return gate.push(10_000_000, [0.0, 0.0, 9.81], gyro_b)


def recorded_imu():
    engine = RecordingPhysicsEngine.instances[-1]
    assert len(engine.calls) == 1
    return engine.calls[0][0]


def test_absent_gyro_is_not_manufactured(monkeypatch):
    gate = make_gate(monkeypatch)

    evaluate_two_frames(gate, None, None)

    assert "gyro" not in recorded_imu()


def test_real_zero_gyro_is_preserved(monkeypatch):
    gate = make_gate(monkeypatch)
    zero = [0.0, 0.0, 0.0]

    evaluate_two_frames(gate, zero, zero)

    imu_raw = recorded_imu()
    assert "gyro" in imu_raw
    assert imu_raw["gyro"] == [zero, zero]


def test_partial_gyro_window_does_not_fabricate_samples(monkeypatch):
    gate = make_gate(monkeypatch)

    evaluate_two_frames(gate, [0.1, 0.2, 0.3], None)

    assert "gyro" not in recorded_imu()



@pytest.mark.parametrize("invalid", [
    [0.1, 0.2],
    [0.1, 0.2, 0.3, 0.4],
    [0.1, float("nan"), 0.3],
    [0.1, float("inf"), 0.3],
    ["0.1", 0.2, 0.3],
    [True, 0.2, 0.3],
    [10**400, 0.2, 0.3],
])
def test_invalid_gyro_vectors_are_not_passed_to_engine(monkeypatch, invalid):
    gate = make_gate(monkeypatch)

    evaluate_two_frames(gate, invalid, [0.1, 0.2, 0.3])

    imu_raw = recorded_imu()
    assert "gyro" not in imu_raw
    assert gate._last_cert["_gyro_availability"] == "partial"
    assert gate._last_cert["_gyro_samples_present"] == 1
