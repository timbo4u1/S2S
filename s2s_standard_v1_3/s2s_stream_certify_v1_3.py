#!/usr/bin/env python3
"""
s2s_stream_certify_v1_3.py — S2S Real-Time Streaming Certifier

Upgrades over v1.2 (file-based):
  - StreamCertifier class: push one frame at a time, get certificates on the fly
  - Sliding window with configurable size and step
  - Per-window GOLD/SILVER/BRONZE/REJECTED decision (same physics as v1.2)
  - Human tremor check, synthetic detection, variance floor — all live
  - JSON-lines output mode (one cert JSON per line) for piping to any consumer
  - TCP server mode: listen on a port, accept JSON-line frames, emit certs
  - stdin mode: read JSON-line frames from stdin (for local pipe from device SDK)

Frame input format (JSON line):
  {"ts_ns": 1234567890, "channels": {"accel_x": 0.1, "accel_y": -0.2, ...}}

Certificate output format (JSON line, same schema as v1.2 file cert):
  {"status": "PASS", "tier": "GOLD", "window": 256, "frame_end_ts_ns": ..., ...}

Usage examples:
  # stdin pipe mode (device SDK pipes frames as JSON lines)
  python3 -m s2s_standard_v1_3.s2s_stream_certify_v1_3 --mode stdin

  # TCP server mode (device connects and sends frames)
  python3 -m s2s_standard_v1_3.s2s_stream_certify_v1_3 --mode tcp --port 9876

  # Python API
  from s2s_standard_v1_3.s2s_stream_certify_v1_3 import StreamCertifier
  sc = StreamCertifier(sensor_names=["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"])
  result = sc.push_frame(ts_ns=time.time_ns(), values=[0.1, -0.2, 9.8, 0.01, -0.01, 0.0])
  if result:
      print(result)  # dict cert, emitted every STREAM_STEP frames
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants import (package or local fallback)
# ---------------------------------------------------------------------------
try:
    from .constants import (
        HUMAN_TREMOR_LOW_HZ, HUMAN_TREMOR_HIGH_HZ,
        MIN_HUMAN_TREMOR_ENERGY_FRACTION, MIN_HUMAN_SAMPLING_HZ,
        STREAM_WINDOW_DEFAULT, STREAM_WINDOW_MIN,
        STREAM_STEP_DEFAULT, STREAM_MAX_FRAME_AGE_NS,
        V1_2_SAMPLE_COUNT_GOLD, V1_2_SAMPLE_COUNT_SILVER,
        V1_2_CV_GOLD, V1_2_CV_SILVER, V1_2_CV_BRONZE,
        V1_2_REMOVED_PCT_REJECT, TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
    )
except Exception:
    from constants import (
        HUMAN_TREMOR_LOW_HZ, HUMAN_TREMOR_HIGH_HZ,
        MIN_HUMAN_TREMOR_ENERGY_FRACTION, MIN_HUMAN_SAMPLING_HZ,
        STREAM_WINDOW_DEFAULT, STREAM_WINDOW_MIN,
        STREAM_STEP_DEFAULT, STREAM_MAX_FRAME_AGE_NS,
        V1_2_SAMPLE_COUNT_GOLD, V1_2_SAMPLE_COUNT_SILVER,
        V1_2_CV_GOLD, V1_2_CV_SILVER, V1_2_CV_BRONZE,
        V1_2_REMOVED_PCT_REJECT, TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
    )

# ---------------------------------------------------------------------------
# Shared physics helpers (same logic as v1.2 certifier, no file I/O)
# ---------------------------------------------------------------------------

def _compute_jitter(timestamps: List[int]) -> Dict[str, Optional[float]]:
    if len(timestamps) < 2:
        return {"mean_delta_ns": None, "rms_jitter_ns": None, "cv": None,
                "p2p_ns": None, "c2c_ns": None}
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    mu    = statistics.mean(deltas)
    sigma = statistics.pstdev(deltas)
    p2p   = max(deltas) - min(deltas)
    c2c   = max(abs(b - a) for a, b in zip(deltas, deltas[1:])) if len(deltas) >= 2 else 0
    cv    = (sigma / mu) if mu > 0 else None
    return {"mean_delta_ns": mu, "rms_jitter_ns": sigma, "cv": cv,
            "p2p_ns": p2p, "c2c_ns": c2c}


def _variance(channel: List[float]) -> float:
    finite = [v for v in channel if math.isfinite(v)]
    if not finite:
        return 0.0
    mu = statistics.mean(finite)
    return sum((x - mu) ** 2 for x in finite) / len(finite)


def _freq_projection(channel: List[float], timestamps_ns: List[int], freq_hz: float) -> float:
    """
    Estimate fraction of signal energy at freq_hz using sin/cos projection.

    Normalization: raw projection energy scales as N² for a pure sine, but as N
    for white noise. Dividing by (N/2) corrects this so that:
      - pure sine at freq_hz  → fraction ≈ 1.0
      - white noise           → fraction ≈ 2/N  (near zero)
    Returns value in [0, 1].
    """
    n = min(len(channel), len(timestamps_ns))
    if n < 4:
        return 0.0
    t = [timestamps_ns[i] * 1e-9 for i in range(n)]
    s = [channel[i] for i in range(n)]
    mu = sum(s) / n
    s = [v - mu for v in s]
    sd = sum(s[i] * math.sin(2 * math.pi * freq_hz * t[i]) for i in range(n))
    cd = sum(s[i] * math.cos(2 * math.pi * freq_hz * t[i]) for i in range(n))
    total = sum(v * v for v in s) or 1.0
    # Correct normalization: divide raw projection by (n/2) * total_energy
    return max(0.0, min(1.0, (sd * sd + cd * cd) / (total * n / 2)))


def _band_energy(channel: List[float], timestamps_ns: List[int],
                 f_lo: float, f_hi: float, steps: int = 6) -> float:
    n = min(len(channel), len(timestamps_ns))
    if n < 8:
        return 0.0
    total = sum(
        _freq_projection(channel, timestamps_ns, f_lo + (f_hi - f_lo) * i / steps)
        for i in range(steps + 1)
    )
    return max(0.0, min(1.0, total / (steps + 1)))


def _decide_tier(count: int, cv: Optional[float], sample_gold: int = V1_2_SAMPLE_COUNT_GOLD) -> str:
    if cv is None:
        return TIER_REJECT
    if count >= sample_gold and cv < V1_2_CV_GOLD:
        return TIER_GOLD
    if count >= V1_2_SAMPLE_COUNT_SILVER and cv < V1_2_CV_SILVER:
        return TIER_SILVER
    if count >= 100 and cv < V1_2_CV_BRONZE:
        return TIER_BRONZE
    return TIER_REJECT


# ---------------------------------------------------------------------------
# StreamCertifier — the core class
# ---------------------------------------------------------------------------

class StreamCertifier:
    """
    Real-time sliding-window certifier.

    Push one frame at a time via push_frame(). Every `step` frames a
    certificate dict is returned (or None if window not yet full).

    Thread safety: NOT thread-safe by default. Wrap in a lock if feeding
    from multiple threads.
    """

    def __init__(
        self,
        sensor_names: List[str],
        window: int = STREAM_WINDOW_DEFAULT,
        step: int = STREAM_STEP_DEFAULT,
        min_human_tremor_energy: float = MIN_HUMAN_TREMOR_ENERGY_FRACTION,
        min_sampling_hz: float = MIN_HUMAN_SAMPLING_HZ,
        synthetic_cv_threshold: float = 1e-8,
        variance_floor: float = 1e-9,
        hum_energy_threshold: float = 0.7,
        sample_gold: Optional[int] = None,  # defaults to window size for streaming
        max_frame_age_ns: int = STREAM_MAX_FRAME_AGE_NS,
        device_id: Optional[str] = None,
    ):
        if window < STREAM_WINDOW_MIN:
            raise ValueError(f"window must be >= {STREAM_WINDOW_MIN}")
        if step < 1 or step > window:
            raise ValueError("step must be 1 <= step <= window")

        self.sensor_names           = sensor_names
        self.n_channels             = len(sensor_names)
        self.window                 = window
        self.step                   = step
        self.min_human_tremor_energy = min_human_tremor_energy
        self.min_sampling_hz        = min_sampling_hz
        self.synthetic_cv_threshold = synthetic_cv_threshold
        self.variance_floor         = variance_floor
        self.hum_energy_threshold   = hum_energy_threshold
        # For streaming, gold threshold = window size (each cert covers one window)
        self.sample_gold            = sample_gold if sample_gold is not None else window
        self.max_frame_age_ns       = max_frame_age_ns
        self.device_id              = device_id or "unknown"

        # Internal buffers — deques for O(1) append/popleft
        self._ts_buf:  Deque[int]         = deque(maxlen=window)
        self._ch_bufs: List[Deque[float]] = [deque(maxlen=window) for _ in range(self.n_channels)]

        # Counter: how many frames since last certificate was emitted
        self._frames_since_cert: int = 0
        # Total frames pushed in this session
        self._total_frames: int = 0
        # Certificates emitted count
        self._certs_emitted: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_frame(self, ts_ns: int, values: List[float]) -> Optional[Dict[str, Any]]:
        """
        Push one sensor frame.

        Args:
            ts_ns:  timestamp in nanoseconds (monotonic, device clock)
            values: list of floats, one per channel (must match sensor_names length)

        Returns:
            A certificate dict if this frame triggered a window evaluation,
            None otherwise.
        """
        if len(values) != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channel values, got {len(values)}"
            )

        # Evict stale frames if device clock jumped or restarted
        if self._ts_buf and ts_ns - self._ts_buf[-1] > self.max_frame_age_ns:
            self._flush_buffers()

        # Append new frame
        self._ts_buf.append(ts_ns)
        for i, v in enumerate(values):
            self._ch_bufs[i].append(v)

        self._total_frames += 1
        self._frames_since_cert += 1

        # Emit certificate when window is full AND step threshold reached
        if (len(self._ts_buf) >= self.window and
                self._frames_since_cert >= self.step):
            self._frames_since_cert = 0
            cert = self._evaluate_window()
            self._certs_emitted += 1
            return cert

        return None

    def push_frame_dict(self, frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convenience: push a frame from a dict with keys 'ts_ns' and 'channels'.
        E.g. {"ts_ns": 1234567890, "channels": {"accel_x": 0.1, ...}}
        """
        ts_ns  = int(frame["ts_ns"])
        ch_map = frame["channels"]
        values = [float(ch_map.get(name, float("nan"))) for name in self.sensor_names]
        return self.push_frame(ts_ns, values)

    @property
    def buffer_fill(self) -> int:
        """How many frames are currently in the window buffer."""
        return len(self._ts_buf)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_frames": self._total_frames,
            "certs_emitted": self._certs_emitted,
            "buffer_fill":   self.buffer_fill,
            "window":        self.window,
            "step":          self.step,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_buffers(self) -> None:
        self._ts_buf.clear()
        for buf in self._ch_bufs:
            buf.clear()
        self._frames_since_cert = 0

    def _evaluate_window(self) -> Dict[str, Any]:
        """Run all physics checks on current window and return a certificate."""
        timestamps = list(self._ts_buf)
        channels   = [list(buf) for buf in self._ch_bufs]

        # --- Jitter metrics ---
        jm  = _compute_jitter(timestamps)
        cv  = jm.get("cv")
        mu_ns = jm.get("mean_delta_ns")
        sampling_hz = (1e9 / mu_ns) if (mu_ns and mu_ns > 0) else None
        human_check_ok = (
            sampling_hz is not None and sampling_hz >= self.min_sampling_hz
        )

        flags: List[str] = []
        notes: Dict[str, Any] = {}

        # --- Synthetic detection ---
        if cv is not None and cv < self.synthetic_cv_threshold:
            flags.append("SUSPECT_SYNTHETIC")
            notes["suspicion_reason"] = (
                f"cv={cv:.2e} < threshold={self.synthetic_cv_threshold:.2e}"
            )

        # --- Per-channel physics ---
        per_ch: Dict[str, Any] = {}
        all_below_var = True
        all_hum       = True
        eligible_tremor = 0
        low_tremor      = 0

        for idx, ch in enumerate(channels):
            name   = self.sensor_names[idx] if idx < len(self.sensor_names) else f"ch_{idx}"
            var    = _variance(ch)
            var_ok = var >= self.variance_floor

            e50 = _freq_projection(ch, timestamps, 50.0)
            e60 = _freq_projection(ch, timestamps, 60.0)
            hum_frac = max(e50, e60)

            tremor_frac   = 0.0
            tremor_checked = False
            if human_check_ok:
                tremor_checked = True
                tremor_frac    = _band_energy(ch, timestamps,
                                              HUMAN_TREMOR_LOW_HZ,
                                              HUMAN_TREMOR_HIGH_HZ)
                eligible_tremor += 1
                if tremor_frac < self.min_human_tremor_energy:
                    low_tremor += 1

            ch_flags: List[str] = []
            if not var_ok:
                ch_flags.append("VARIANCE_BELOW_FLOOR")
            if hum_frac >= self.hum_energy_threshold:
                ch_flags.append("STRONG_50_60HZ_HUM")
            if tremor_checked and tremor_frac < self.min_human_tremor_energy:
                ch_flags.append("LOW_HUMAN_TREMOR")

            per_ch[name] = {
                "variance":           var,
                "variance_ok":        var_ok,
                "dominant_50hz_frac": e50,
                "dominant_60hz_frac": e60,
                "human_tremor_frac":  tremor_frac,
                "human_tremor_checked": tremor_checked,
                "channel_flags":      ch_flags,
            }

            if var_ok:
                all_below_var = False
            if hum_frac < self.hum_energy_threshold:
                all_hum = False

        # --- Humanity flag ---
        if eligible_tremor > 0 and low_tremor == eligible_tremor:
            flags.append("NON_HUMAN_INTERACTION")
            notes["humanity_reason"] = "No 8-12Hz tremor detected in eligible channels"

        # --- Tier decision ---
        tier = _decide_tier(len(timestamps), cv, self.sample_gold)

        if all_below_var:
            flags.append("ALL_CHANNELS_VARIANCE_BELOW_FLOOR")
            tier = TIER_REJECT
        if all_hum and channels:
            flags.append("ALL_CHANNELS_STRONG_50_60HZ_HUM")
            tier = TIER_REJECT

        # --- Sensor stats ---
        sensor_stats: Dict[str, Any] = {}
        for idx, ch in enumerate(channels):
            name   = self.sensor_names[idx] if idx < len(self.sensor_names) else f"ch_{idx}"
            finite = [v for v in ch if math.isfinite(v)]
            sensor_stats[name] = {
                "count":     len(ch),
                "count_nan": len(ch) - len(finite),
                "mean":      statistics.mean(finite) if finite else None,
                "peak":      max(abs(v) for v in finite) if finite else None,
            }

        return {
            "status":          "PASS" if tier != TIER_REJECT else "FAIL",
            "tier":            tier,
            "window":          self.window,
            "step":            self.step,
            "frame_start_ts_ns": timestamps[0],
            "frame_end_ts_ns":   timestamps[-1],
            "duration_ms":     (timestamps[-1] - timestamps[0]) / 1e6,
            "sampling_hz":     round(sampling_hz, 2) if sampling_hz else None,
            "metrics":         jm,
            "sensor_stats":    sensor_stats,
            "physics":         per_ch,
            "flags":           flags,
            "notes":           notes,
            "device_id":       self.device_id,
            "cert_index":      self._certs_emitted,
            "total_frames_seen": self._total_frames,
            "tool":            "s2s_stream_certify_v1_3",
            "issued_at_ns":    time.time_ns(),
        }


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _make_certifier_from_args(args: argparse.Namespace) -> StreamCertifier:
    names = args.channels.split(",") if args.channels else [
        "accel_x", "accel_y", "accel_z",
        "gyro_x",  "gyro_y",  "gyro_z",
    ]
    return StreamCertifier(
        sensor_names             = [n.strip() for n in names],
        window                   = args.window,
        step                     = args.step,
        synthetic_cv_threshold   = args.synthetic_cv_threshold,
        variance_floor           = args.variance_floor,
        hum_energy_threshold     = args.hum_energy_threshold,
        min_human_tremor_energy  = args.min_human_tremor_energy,
        min_sampling_hz          = args.min_sampling_hz,
        device_id                = args.device_id,
    )


def run_stdin_mode(certifier: StreamCertifier, quiet: bool = False) -> None:
    """
    Read JSON-line frames from stdin, emit JSON-line certificates to stdout.
    Frame format: {"ts_ns": int, "channels": {name: float, ...}}
    """
    if not quiet:
        print(json.dumps({
            "event": "stream_start",
            "sensor_names": certifier.sensor_names,
            "window": certifier.window,
            "step": certifier.step,
        }), flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"event": "parse_error", "error": str(e)}), flush=True)
            continue

        try:
            cert = certifier.push_frame_dict(frame)
        except Exception as e:
            print(json.dumps({"event": "frame_error", "error": str(e)}), flush=True)
            continue

        if cert is not None:
            print(json.dumps(cert), flush=True)

    if not quiet:
        print(json.dumps({"event": "stream_end", "stats": certifier.stats}), flush=True)


def run_tcp_mode(certifier: StreamCertifier, host: str, port: int, quiet: bool = False) -> None:
    """
    TCP server: accept one connection, read JSON-line frames, emit JSON-line certs.
    Restarts listening after disconnect.
    """
    import socket

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)

    if not quiet:
        print(json.dumps({"event": "tcp_listen", "host": host, "port": port}), flush=True)

    while True:
        conn, addr = srv.accept()
        if not quiet:
            print(json.dumps({"event": "tcp_connect", "addr": str(addr)}), flush=True)

        buf = ""
        with conn:
            conn_file = conn.makefile("r", encoding="utf-8", errors="replace")
            for raw_line in conn_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    frame = json.loads(line)
                    cert  = certifier.push_frame_dict(frame)
                    if cert is not None:
                        msg = json.dumps(cert) + "\n"
                        conn.sendall(msg.encode("utf-8"))
                        if not quiet:
                            print(json.dumps(cert), flush=True)
                except Exception as e:
                    err_msg = json.dumps({"event": "error", "error": str(e)}) + "\n"
                    conn.sendall(err_msg.encode("utf-8"))

        if not quiet:
            print(json.dumps({"event": "tcp_disconnect", "stats": certifier.stats}), flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S2S v1.3 — Real-Time Streaming Certifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["stdin", "tcp"], default="stdin",
                   help="Input mode: read from stdin or listen on TCP port")
    p.add_argument("--port", type=int, default=9876,
                   help="TCP port (only used with --mode tcp)")
    p.add_argument("--host", default="0.0.0.0",
                   help="TCP bind host (only used with --mode tcp)")
    p.add_argument("--channels", default=None,
                   help="Comma-separated channel names. Default: 6-DOF IMU")
    p.add_argument("--window", type=int, default=STREAM_WINDOW_DEFAULT,
                   help="Sliding window size in samples")
    p.add_argument("--step", type=int, default=STREAM_STEP_DEFAULT,
                   help="Emit a certificate every N new samples")
    p.add_argument("--device-id", default="unknown",
                   help="Device identifier embedded in certificates")
    p.add_argument("--synthetic-cv-threshold", type=float, default=1e-8)
    p.add_argument("--variance-floor", type=float, default=1e-9)
    p.add_argument("--hum-energy-threshold", type=float, default=0.7)
    p.add_argument("--min-human-tremor-energy", type=float,
                   default=MIN_HUMAN_TREMOR_ENERGY_FRACTION)
    p.add_argument("--min-sampling-hz", type=float, default=MIN_HUMAN_SAMPLING_HZ)
    p.add_argument("--quiet", action="store_true",
                   help="Suppress status/event messages, emit only certificates")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    certifier = _make_certifier_from_args(args)

    if args.mode == "stdin":
        run_stdin_mode(certifier, quiet=args.quiet)
    elif args.mode == "tcp":
        run_tcp_mode(certifier, host=args.host, port=args.port, quiet=args.quiet)


if __name__ == "__main__":
    main()
