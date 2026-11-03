"""Regression tests for LeRobot sensor mapping and frame alignment."""

from s2s_standard_v1_3.adapters import lerobot


class RecordingPhysicsEngine:
    def __init__(self):
        self.calls = []
        self.source_types = []

    def certify(self, imu_raw, segment, source_type=None):
        self.calls.append((imu_raw, segment))
        self.source_types.append(source_type)
        return {
            "tier": "SILVER",
            "physical_law_score": 80,
            "source_type": source_type,
        }


def install_engine(monkeypatch):
    engine = RecordingPhysicsEngine()
    monkeypatch.setattr(lerobot, "_engine", engine)
    return engine


def make_frames(n=5):
    return [
        {
            "imu_accel": [0.1 * i, 0.2 * i, 9.81],
            "imu_gyro": [0.01 * i, 0.02 * i, 0.03 * i],
        }
        for i in range(n)
    ]


def test_observation_state_is_not_implicitly_treated_as_acceleration(monkeypatch):
    engine = install_engine(monkeypatch)

    frames = [
        {"observation.state": [0.1, 0.2, 0.3]}
        for _ in range(5)
    ]

    result = lerobot.certify_lerobot_episode(
        frames,
        window=4,
        step=4,
    )

    assert "error" in result
    assert "accel_keys" in result["error"]
    assert engine.calls == []


def test_absent_gyro_is_not_manufactured(monkeypatch):
    engine = install_engine(monkeypatch)
    frames = make_frames()

    result = lerobot.certify_lerobot_episode(
        frames,
        accel_keys=("imu_accel",),
        window=4,
        step=4,
    )

    assert result["total_windows"] == 1
    imu_raw, _ = engine.calls[0]

    assert "gyro" not in imu_raw
    assert result["windows"][0]["_gyro_availability"] == "absent"
    assert engine.source_types == ["ROBOT_SENSOR_MEASURED"]
    assert result["source_type"] == "ROBOT_SENSOR_MEASURED"
    assert result["windows"][0]["source_type"] == "ROBOT_SENSOR_MEASURED"


def test_partial_gyro_window_is_omitted_without_padding(monkeypatch):
    engine = install_engine(monkeypatch)
    frames = make_frames()
    frames[2]["imu_gyro"] = None

    result = lerobot.certify_lerobot_episode(
        frames,
        accel_keys=("imu_accel",),
        gyro_keys=("imu_gyro",),
        window=4,
        step=4,
    )

    assert result["total_windows"] == 1
    imu_raw, _ = engine.calls[0]

    assert "gyro" not in imu_raw
    assert len(imu_raw["accel"]) == 4

    cert = result["windows"][0]
    assert cert["_gyro_availability"] == "partial"
    assert cert["_gyro_samples_present"] == 3
    assert cert["_gyro_samples_expected"] == 4


def test_real_zero_gyro_is_preserved(monkeypatch):
    engine = install_engine(monkeypatch)
    frames = make_frames()

    for frame in frames:
        frame["imu_gyro"] = [0.0, 0.0, 0.0]

    result = lerobot.certify_lerobot_episode(
        frames,
        accel_keys=("imu_accel",),
        gyro_keys=("imu_gyro",),
        window=4,
        step=4,
    )

    imu_raw, _ = engine.calls[0]

    assert imu_raw["gyro"] == [[0.0, 0.0, 0.0]] * 4
    assert result["windows"][0]["_gyro_availability"] == "complete"


def test_missing_acceleration_frame_does_not_compress_episode(monkeypatch):
    engine = install_engine(monkeypatch)
    frames = make_frames(n=8)

    frames[1]["imu_accel"] = None

    result = lerobot.certify_lerobot_episode(
        frames,
        accel_keys=("imu_accel",),
        window=4,
        step=4,
    )

    # Original window 0-3 is skipped. Original window 4-7 is retained.
    assert result["total_windows"] == 1
    assert len(engine.calls) == 1

    imu_raw, _ = engine.calls[0]
    expected = [frame["imu_accel"] for frame in frames[4:8]]

    assert imu_raw["accel"] == expected
    assert result["windows"][0]["_window_start_frame"] == 4


def test_episode_with_exactly_one_window_is_processed(monkeypatch):
    engine = install_engine(monkeypatch)
    frames = make_frames(n=4)

    result = lerobot.certify_lerobot_episode(
        frames,
        accel_keys=("imu_accel",),
        window=4,
        step=4,
    )

    assert result["total_windows"] == 1
    assert len(engine.calls) == 1


def test_latency_reports_elapsed_certification_time(monkeypatch):
    engine = install_engine(monkeypatch)
    frames = make_frames(n=4)

    ticks = iter([10.0, 10.0125])
    monkeypatch.setattr(
        lerobot.time, "perf_counter", lambda: next(ticks)
    )

    result = lerobot.certify_lerobot_episode(
        frames,
        accel_keys=("imu_accel",),
        window=4,
        step=4,
    )

    assert len(engine.calls) == 1
    assert result["windows"][0]["_latency_ms"] == 12.5
