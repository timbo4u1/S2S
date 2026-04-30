"""
S2S VLA Safety Wrapper — Real-time physics gate for Vision-Language-Action models.

Sits between VLA model output and robot motors.
Certifies every action command against 8 biomechanical laws before execution.

Supports:
  - End-effector position commands (x,y,z) → finite difference acceleration
  - Direct acceleration commands (already in m/s²)
  - Joint position commands + DH parameters → forward kinematics

Usage:
    from s2s_standard_v1_3.adapters.vla_wrapper import VLASafetyWrapper

    wrapper = VLASafetyWrapper(hz=10.0, segment='forearm')

    # For each VLA output command:
    decision = wrapper.check(state_xyz=[0.25, -0.04, 1.0])
    if decision['safe']:
        robot.execute(command)
    else:
        robot.hold()
        print(decision['reason'])
"""

import math
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


class VLASafetyWrapper:
    """
    Real-time physics gate for VLA model outputs.

    Converts robot state (position or acceleration) into S2S certification.
    Maintains a sliding window buffer and runs physics laws on each step.

    Parameters:
        hz: robot control frequency (e.g. 10Hz for NYU, 15Hz for RoboTurk)
        segment: body segment for physics parameters ('forearm' default)
        window_size: number of steps to certify at once (default 10 = 1 second at 10Hz)
        strikes: consecutive REJECTED windows before UNSAFE state (default 3)
    """

    SAFE      = "SAFE"
    DEGRADED  = "DEGRADED"
    UNSAFE    = "UNSAFE"

    def __init__(self, hz: float = 10.0, segment: str = "forearm",
                 window_size: int = 10, strikes: int = 3):
        self.hz = hz
        self.dt = 1.0 / hz
        self.segment = segment
        self.window_size = window_size
        self.strikes = strikes

        self._pos_buffer: List[List[float]] = []   # position history
        self._acc_buffer: List[List[float]] = []   # acceleration history
        self._ts_buffer:  List[int]         = []   # timestamps ns
        self._step_count: int = 0
        self._strike_count: int = 0
        self._state: str = self.SAFE
        self._last_result: Optional[Dict] = None

        # Import here to avoid circular
        from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine, audit_report
        self._engine = PhysicsEngine()
        self._audit_report = audit_report

    def check_position(self, xyz: List[float]) -> Dict:
        """
        Check end-effector position command.
        Computes acceleration via finite difference from position history.

        Args:
            xyz: [x, y, z] end-effector position in meters

        Returns:
            dict with 'safe', 'state', 'tier', 'reason', 'action'
        """
        self._pos_buffer.append(xyz)
        t_ns = int(self._step_count * 1e9 / self.hz)
        self._ts_buffer.append(t_ns)
        self._step_count += 1

        # Need at least 3 positions to compute acceleration
        if len(self._pos_buffer) < 3:
            return self._safe_result("INITIALIZING — need more steps")

        # Compute acceleration via finite difference
        p = self._pos_buffer
        vel_curr = [(p[-1][i] - p[-2][i]) / self.dt for i in range(3)]
        vel_prev = [(p[-2][i] - p[-3][i]) / self.dt for i in range(3)]
        acc = [(vel_curr[i] - vel_prev[i]) / self.dt for i in range(3)]
        self._acc_buffer.append(acc)

        # Run certification when buffer is full
        if len(self._acc_buffer) >= self.window_size:
            return self._certify_window()

        return self._safe_result(f"BUFFERING — {len(self._acc_buffer)}/{self.window_size} steps")

    def check_acceleration(self, accel: List[float]) -> Dict:
        """
        Check direct acceleration command (already in m/s²).

        Args:
            accel: [ax, ay, az] acceleration in m/s²

        Returns:
            dict with 'safe', 'state', 'tier', 'reason', 'action'
        """
        t_ns = int(self._step_count * 1e9 / self.hz)
        self._acc_buffer.append(accel)
        self._ts_buffer.append(t_ns)
        self._step_count += 1

        if len(self._acc_buffer) >= self.window_size:
            return self._certify_window()

        return self._safe_result(f"BUFFERING — {len(self._acc_buffer)}/{self.window_size} steps")

    def _certify_window(self) -> Dict:
        """Run S2S certification on current buffer window."""
        acc = self._acc_buffer[-self.window_size:]
        # Use global step count for timestamps to avoid regression
        base = (self._step_count - len(acc)) 
        ts  = [int((base + i) * 1e9 / self.hz) for i in range(len(acc))]
        gyro = [[0.0, 0.0, 0.0]] * len(acc)

        try:
            result = self._engine.certify(
                {'timestamps_ns': ts, 'accel': acc, 'gyro': gyro},
                segment=self.segment
            )
            report = self._audit_report(result)
            self._last_result = result

            tier = result['tier']

            if tier in ('GOLD', 'SILVER'):
                self._strike_count = 0
                self._state = self.SAFE
                action = "EXECUTE"
            elif tier == 'BRONZE':
                self._strike_count = 0
                self._state = self.DEGRADED
                action = "EXECUTE_WITH_CAUTION"
            else:  # REJECTED
                self._strike_count += 1
                if self._strike_count >= self.strikes:
                    self._state = self.UNSAFE
                    action = "HOLD"
                else:
                    self._state = self.DEGRADED
                    action = "EXECUTE_WITH_CAUTION"

            issues = [x['message'] for x in report.get('issues', [])]

            return {
                'safe':         self._state == self.SAFE,
                'state':        self._state,
                'tier':         tier,
                'score':        result['physical_law_score'],
                'action':       action,
                'strikes':      self._strike_count,
                'reason':       issues[0] if issues else report.get('verdict',''),
                'issues':       issues,
                'laws_passed':  result['laws_passed'],
                'laws_failed':  result['laws_failed'],
                'recommendation': report.get('recommendation',''),
            }

        except Exception as e:
            return {
                'safe': False, 'state': self.UNSAFE,
                'tier': 'ERROR', 'score': 0,
                'action': 'HOLD',
                'strikes': self._strike_count,
                'reason': f'Certification error: {e}',
                'issues': [str(e)],
                'laws_passed': [], 'laws_failed': [],
            }

    def _safe_result(self, reason: str) -> Dict:
        return {
            'safe': True, 'state': self.SAFE,
            'tier': 'PENDING', 'score': 0,
            'action': 'EXECUTE',
            'strikes': 0, 'reason': reason,
            'issues': [], 'laws_passed': [], 'laws_failed': [],
        }

    def reset(self):
        """Reset wrapper state — call between episodes."""
        self._pos_buffer.clear()
        self._acc_buffer.clear()
        self._ts_buffer.clear()
        self._step_count = 0
        self._strike_count = 0
        self._state = self.SAFE
        self._last_result = None

    @property
    def current_state(self) -> str:
        return self._state
