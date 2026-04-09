"""
s2s_safety_gate.py — Real-Time Safety Gate for S2S

Wraps PhysicsEngine with sliding window and three-strike logic.
Returns is_safe boolean + reason string on every window.

Usage:
    from s2s_standard_v1_3.s2s_safety_gate import RealTimeSafetyGate

    gate = RealTimeSafetyGate(segment="forearm")

    for ts_ns, accel, gyro in sensor_stream:
        is_safe, reason, cert = gate.push(ts_ns, accel, gyro)
        if not is_safe:
            print(f"UNSAFE: {reason}")
            # halt robot / alert operator / log event

Latency: 1.44ms per window at 2000Hz (verified on NinaPro DB5)
Strikes:  3 consecutive REJECTED windows required before is_safe=False
          prevents false triggers from single noise samples
"""
from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import time

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine


class RealTimeSafetyGate:
    """
    Real-time physics-based safety monitor for IMU sensor streams.

    NOT a hardware safety system. Does not interface with hardware directly.
    Use as a software monitoring layer that flags physically suspicious data.

    Architecture:
        push() one frame at a time
        Returns (is_safe, reason, cert) when window is full
        Three consecutive REJECTED windows → is_safe = False
        One PASS window resets the strike counter

    Positioning:
        S2S monitors sensor data quality in real-time and flags
        physically impossible readings before they reach your
        control system or training pipeline.
    """

    def __init__(
        self,
        segment: str = "forearm",
        window_size: int = 500,
        step_size: int = 100,
        strikes_required: int = 3,
        device_id: str = "unknown",
    ):
        """
        Args:
            segment:         body segment — forearm, hand, finger, upper_arm, walking
            window_size:     samples per evaluation window (500 = 250ms at 2000Hz)
            step_size:       evaluate every N new frames (100 = every 50ms at 2000Hz)
            strikes_required: consecutive rejections before is_safe=False
            device_id:       identifier logged in every certificate
        """
        self.segment         = segment
        self.window_size     = window_size
        self.step_size       = step_size
        self.strikes_required = strikes_required
        self.device_id       = device_id

        self._engine         = PhysicsEngine()
        self._ts_buf         = deque(maxlen=window_size)
        self._acc_buf        = deque(maxlen=window_size)
        self._gyro_buf       = deque(maxlen=window_size)
        self._frames_since   = 0
        self._strikes        = 0
        self._total_certs    = 0
        self._total_frames   = 0
        self._last_cert      = None
        self._session_start  = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        ts_ns: int,
        accel: List[float],
        gyro: Optional[List[float]] = None,
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Push one sensor frame.

        Args:
            ts_ns: timestamp in nanoseconds
            accel: [x, y, z] in m/s²
            gyro:  [x, y, z] in rad/s (optional — pass None if no gyro)

        Returns:
            (is_safe, reason, cert)
            is_safe: True = data is physically plausible
                     False = 3+ consecutive REJECTED windows
            reason:  human-readable explanation
            cert:    full certification dict (None if window not yet full)
        """
        self._ts_buf.append(ts_ns)
        self._acc_buf.append(accel)
        self._gyro_buf.append(gyro or [0.0, 0.0, 0.0])
        self._frames_since += 1
        self._total_frames += 1

        # Not enough data yet
        if len(self._ts_buf) < self.window_size:
            remaining = self.window_size - len(self._ts_buf)
            return True, f"BUFFERING ({remaining} frames to first evaluation)", None

        # Not time to evaluate yet
        if self._frames_since < self.step_size:
            if self._last_cert is not None:
                return self._is_safe_result()
            return True, "BUFFERING", None

        # Evaluate window
        self._frames_since = 0
        imu_raw = {
            "timestamps_ns": list(self._ts_buf),
            "accel":         list(self._acc_buf),
            "gyro":          list(self._gyro_buf),
        }

        try:
            t0   = time.perf_counter()
            cert = self._engine.certify(imu_raw, segment=self.segment)
            cert["_latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            cert["_device_id"]  = self.device_id
        except Exception as e:
            cert = {"tier": "REJECTED", "physical_law_score": 0,
                    "_error": str(e), "_latency_ms": 0}

        self._total_certs += 1
        self._last_cert = cert

        # Strike logic
        if cert["tier"] == "REJECTED":
            self._strikes += 1
        else:
            self._strikes = 0  # reset on any passing window

        return self._is_safe_result()

    def reset(self) -> None:
        """Clear buffers and strike counter. Call after sensor reconnect."""
        self._ts_buf.clear()
        self._acc_buf.clear()
        self._gyro_buf.clear()
        self._frames_since = 0
        self._strikes      = 0
        self._last_cert    = None

    def status(self) -> Dict[str, Any]:
        """Return current gate status summary."""
        return {
            "is_safe":        self._strikes < self.strikes_required,
            "strikes":        self._strikes,
            "strikes_required": self.strikes_required,
            "total_frames":   self._total_frames,
            "total_certs":    self._total_certs,
            "segment":        self.segment,
            "device_id":      self.device_id,
            "uptime_s":       round(time.time() - self._session_start, 1),
            "last_tier":      self._last_cert["tier"] if self._last_cert else None,
            "last_score":     self._last_cert.get("physical_law_score") if self._last_cert else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_safe_result(self) -> Tuple[bool, str, Optional[Dict]]:
        cert     = self._last_cert
        tier     = cert["tier"] if cert else "UNKNOWN"
        score    = cert.get("physical_law_score", 0) if cert else 0
        flags    = cert.get("flags", []) if cert else []
        is_safe  = self._strikes < self.strikes_required

        if not is_safe:
            reason = (
                f"UNSAFE: {self._strikes} consecutive REJECTED windows "
                f"| last score={score} | flags={flags}"
            )
        elif tier == "REJECTED":
            reason = (
                f"WARNING: strike {self._strikes}/{self.strikes_required} "
                f"| score={score} | flags={flags}"
            )
        elif tier == "BRONZE":
            reason = f"DEGRADED: score={score} (below SILVER threshold)"
        else:
            reason = f"SAFE: {tier} score={score}"

        return is_safe, reason, cert
