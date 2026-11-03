"""
Tests for VLASafetyWrapper — real-time physics gate for VLA models.
"""
import pytest
import math


def test_vla_wrapper_imports():
    """VLASafetyWrapper must import without error."""
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper
    wrapper = VLASafetyWrapper(hz=10.0, segment='forearm')
    assert wrapper is not None


def test_vla_wrapper_initializes_safe():
    """Wrapper starts in SAFE state."""
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper
    wrapper = VLASafetyWrapper(hz=10.0)
    assert wrapper.current_state == VLASafetyWrapper.SAFE


def test_vla_wrapper_buffering():
    """First steps return PENDING while buffering."""
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper
    wrapper = VLASafetyWrapper(hz=10.0, window_size=8)
    result = wrapper.check_position([0.1, 0.0, 1.0])
    assert result['tier'] == 'PENDING'
    assert result['safe'] == True
    assert result['action'] == 'EXECUTE'
    assert result['source_type'] == 'ROBOT_COMMAND_DERIVED'


def test_vla_wrapper_certifies_after_window():
    """After window_size steps, returns real certification."""
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper
    wrapper = VLASafetyWrapper(hz=10.0, window_size=8)
    
    # Smooth slow motion — should pass
    for i in range(15):
        t = i * 0.1
        result = wrapper.check_position([
            0.1 * math.sin(t),
            0.05 * math.cos(t),
            1.0 + 0.02 * math.sin(t * 2)
        ])
    
    assert result['tier'] in ('GOLD', 'SILVER', 'BRONZE', 'REJECTED')
    assert result['score'] >= 0
    assert result['action'] in ('EXECUTE', 'EXECUTE_WITH_CAUTION', 'HOLD')


def test_vla_wrapper_reset():
    """Reset clears all state."""
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper
    wrapper = VLASafetyWrapper(hz=10.0, window_size=8)
    
    for i in range(10):
        wrapper.check_position([float(i)*0.01, 0.0, 1.0])
    
    wrapper.reset()
    assert wrapper.current_state == VLASafetyWrapper.SAFE
    assert wrapper._step_count == 0
    assert len(wrapper._pos_buffer) == 0


def test_vla_wrapper_check_acceleration():
    """Direct acceleration input works."""
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper
    wrapper = VLASafetyWrapper(hz=100.0, window_size=10)
    
    for i in range(15):
        result = wrapper.check_acceleration([
            0.1, 0.0, 9.81
        ])
    
    assert result['tier'] in ('GOLD', 'SILVER', 'BRONZE', 'REJECTED', 'PENDING')
    assert 'safe' in result
    assert 'action' in result


def test_vla_wrapper_states():
    """State constants are defined correctly."""
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper
    assert VLASafetyWrapper.SAFE == 'SAFE'
    assert VLASafetyWrapper.DEGRADED == 'DEGRADED'
    assert VLASafetyWrapper.UNSAFE == 'UNSAFE'


class RecordingCommandEngine:
    def __init__(self):
        self.calls = []

    def certify(self, imu_raw, segment, source_type=None):
        self.calls.append((imu_raw, segment, source_type))
        return {
            "tier": "SILVER",
            "physical_law_score": 80,
            "laws_passed": ["jerk_bounds"],
            "laws_failed": [],
            "source_type": source_type,
        }


def install_command_engine(wrapper):
    engine = RecordingCommandEngine()
    wrapper._engine = engine
    wrapper._audit_report = lambda result: {
        "issues": [],
        "verdict": "MONITORED",
        "recommendation": "",
    }
    return engine


def test_direct_acceleration_is_command_derived_without_fake_gyro():
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper

    wrapper = VLASafetyWrapper(hz=10.0, window_size=2)
    engine = install_command_engine(wrapper)

    wrapper.check_acceleration([0.1, 0.0, 9.81])
    decision = wrapper.check_acceleration([0.2, 0.0, 9.81])

    assert len(engine.calls) == 1
    imu_raw, segment, source_type = engine.calls[0]

    assert "gyro" not in imu_raw
    assert source_type == "ROBOT_COMMAND_DERIVED"
    assert decision["source_type"] == "ROBOT_COMMAND_DERIVED"


def test_position_derived_acceleration_has_command_provenance():
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper

    wrapper = VLASafetyWrapper(hz=10.0, window_size=1)
    engine = install_command_engine(wrapper)

    wrapper.check_position([0.0, 0.0, 0.0])
    wrapper.check_position([0.1, 0.0, 0.0])
    decision = wrapper.check_position([0.25, 0.0, 0.0])

    assert len(engine.calls) == 1
    imu_raw, segment, source_type = engine.calls[0]

    assert "gyro" not in imu_raw
    assert source_type == "ROBOT_COMMAND_DERIVED"
    assert decision["source_type"] == "ROBOT_COMMAND_DERIVED"


def test_physics_engine_preserves_biological_default_source_type():
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

    result = PhysicsEngine().certify()

    assert result["source_type"] == "HIL_BIOLOGICAL"


def test_physics_engine_accepts_explicit_command_source_type():
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

    result = PhysicsEngine().certify(
        source_type="ROBOT_COMMAND_DERIVED"
    )

    assert result["source_type"] == "ROBOT_COMMAND_DERIVED"
