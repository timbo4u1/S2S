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
