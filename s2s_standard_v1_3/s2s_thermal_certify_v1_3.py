#!/usr/bin/env python3
"""
s2s_thermal_certify_v1_3.py — S2S Thermal Camera Certifier

Thermal cameras (e.g. FLIR Lepton, MLX90640) record 2D temperature maps.
Real thermal data from a human-in-the-loop interaction has:
  - Body heat signature: 30–37°C region in the frame
  - Frame-to-frame temperature drift (real sensors have noise + drift)
  - Spatial temperature gradient (hot body parts vs cold background)
  - No perfectly static frames — breathing, blood flow, movement all cause changes

Synthetic thermal data:
  - Perfect flat temperatures (no noise)
  - Zero frame-to-frame change
  - No body-temperature region (just room temp 18–26°C everywhere)
  - Perfectly uniform spatial distribution

Certificate tiers:
  GOLD   — body heat confirmed, real noise, spatial gradient, frame motion
  SILVER — valid thermal data, moderate confidence in human presence
  BRONZE — valid signal, insufficient body heat or weak gradient
  REJECTED — flat/static/synthetic, sensor fault, all room-temp (no person)

Usage:
  from s2s_standard_v1_3.s2s_thermal_certify_v1_3 import ThermalStreamCertifier
  tc = ThermalStreamCertifier(frame_width=32, frame_height=24, device_id='lepton_01')
  cert = tc.push_frame(ts_ns, flat_temperature_list)  # 32*24 = 768 floats in °C
"""
from __future__ import annotations

import json
import math
import statistics
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import argparse, struct, zlib
from pathlib import Path

try:
    from .constants import (
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        MAGIC, TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA,
        TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
    )
except Exception:
    from constants import (
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        MAGIC, TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA,
        TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
    )

# ---------------------------------------------------------------------------
# Thermal constants
# ---------------------------------------------------------------------------
THERMAL_BODY_TEMP_LO    = 28.0   # minimum skin surface temperature (°C)
THERMAL_BODY_TEMP_HI    = 38.5   # maximum realistic skin temperature (°C)
THERMAL_ROOM_TEMP_LO    = 15.0   # cold room
THERMAL_ROOM_TEMP_HI    = 30.0   # warm room
THERMAL_MIN_BODY_FRAC   = 0.05   # at least 5% of pixels in body-temp range = human present
THERMAL_GOLD_BODY_FRAC  = 0.15   # 15%+ body pixels = confident human present
THERMAL_NOISE_FLOOR     = 0.05   # real sensors have ≥ 0.05°C pixel noise
THERMAL_GRADIENT_GOLD   = 5.0    # spatial temp range > 5°C = meaningful scene
THERMAL_GRADIENT_SILVER = 2.0
THERMAL_DELTA_GOLD      = 0.05   # frame-to-frame mean change > 0.05°C = real motion/drift
THERMAL_DELTA_SYNTHETIC = 1e-6   # frame-to-frame change < this = static (synthetic)
THERMAL_SATURATION_HI   = 100.0  # above = sensor fault
THERMAL_SATURATION_LO   = -20.0  # below = sensor fault

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frame(flat: List[float], width: int, height: int) -> List[List[float]]:
    """Reshape flat list into 2D grid."""
    n = width * height
    flat = flat[:n]
    return [[flat[r * width + c] for c in range(width)] for r in range(height)]


def _frame_stats(flat: List[float]) -> Dict[str, float]:
    finite = [v for v in flat if math.isfinite(v)]
    if not finite:
        return {"min": 0, "max": 0, "mean": 0, "variance": 0, "range": 0}
    mu  = statistics.mean(finite)
    var = sum((x - mu)**2 for x in finite) / len(finite)
    return {
        "min":      round(min(finite), 3),
        "max":      round(max(finite), 3),
        "mean":     round(mu, 3),
        "variance": round(var, 4),
        "range":    round(max(finite) - min(finite), 3),
        "std":      round(math.sqrt(var), 4),
    }


def _body_heat_fraction(flat: List[float]) -> float:
    """Fraction of pixels in human body-temperature range."""
    finite = [v for v in flat if math.isfinite(v)]
    if not finite:
        return 0.0
    body = sum(1 for v in finite if THERMAL_BODY_TEMP_LO <= v <= THERMAL_BODY_TEMP_HI)
    return body / len(finite)


def _spatial_gradient(flat: List[float], width: int, height: int) -> float:
    """
    Estimate spatial temperature gradient using central differences.
    Returns mean absolute gradient magnitude (°C/pixel).
    """
    if len(flat) < width * 2:
        return 0.0
    grads: List[float] = []
    for r in range(height):
        for c in range(width):
            idx = r * width + c
            if not math.isfinite(flat[idx]):
                continue
            # horizontal gradient
            if c + 1 < width and math.isfinite(flat[idx + 1]):
                grads.append(abs(flat[idx + 1] - flat[idx]))
            # vertical gradient
            if r + 1 < height and math.isfinite(flat[idx + width]):
                grads.append(abs(flat[idx + width] - flat[idx]))
    return statistics.mean(grads) if grads else 0.0


def _frame_to_frame_delta(frames: List[List[float]]) -> List[float]:
    """Mean absolute pixel-wise temperature change between consecutive frames."""
    deltas = []
    for fa, fb in zip(frames, frames[1:]):
        n = min(len(fa), len(fb))
        diffs = [abs(fb[i] - fa[i]) for i in range(n)
                 if math.isfinite(fa[i]) and math.isfinite(fb[i])]
        if diffs:
            deltas.append(statistics.mean(diffs))
    return deltas


def _compute_jitter(timestamps: List[int]) -> Dict[str, Optional[float]]:
    if len(timestamps) < 2:
        return {"mean_delta_ns": None, "rms_jitter_ns": None, "cv": None}
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    mu = statistics.mean(deltas)
    sigma = statistics.pstdev(deltas)
    cv = sigma / mu if mu > 0 else None
    return {"mean_delta_ns": mu, "rms_jitter_ns": sigma, "cv": cv}


# ---------------------------------------------------------------------------
# Core certification function
# ---------------------------------------------------------------------------

def certify_thermal_frames(
    frames_flat: List[List[float]],
    timestamps_ns: List[int],
    width: int,
    height: int,
    device_id: str = "unknown",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Certify a sequence of thermal camera frames.
    Each frame is a flat list of width*height temperature values in °C.
    """
    flags: List[str] = []
    notes: Dict[str, Any] = {}

    if not frames_flat:
        return {
            "status": "FAIL", "tier": TIER_REJECT,
            "sensor_type": "THERMAL",
            "flags": ["NO_FRAMES"],
            "tool": "s2s_thermal_certify_v1_3",
            "issued_at_ns": time.time_ns(),
        }

    # Use latest frame for spatial analysis
    latest = frames_flat[-1]

    # Sensor fault check (out-of-range temperatures)
    fault_pixels = sum(1 for v in latest if math.isfinite(v) and
                      (v > THERMAL_SATURATION_HI or v < THERMAL_SATURATION_LO))
    fault_frac   = fault_pixels / max(len(latest), 1)
    if fault_frac > 0.1:
        flags.append("SENSOR_FAULT_OUT_OF_RANGE")
        notes["fault_fraction"] = round(fault_frac, 4)

    # Frame stats
    stats = _frame_stats(latest)

    # Spatial gradient (how much temperature varies across the scene)
    gradient_mean = _spatial_gradient(latest, width, height)
    spatial_range = stats.get("range", 0)

    if spatial_range < 0.5:
        flags.append("SPATIALLY_UNIFORM_POSSIBLY_COVERED")

    # Body heat detection
    body_frac = _body_heat_fraction(latest)
    human_present = body_frac >= THERMAL_MIN_BODY_FRAC
    confident_human = body_frac >= THERMAL_GOLD_BODY_FRAC

    if not human_present:
        flags.append("NO_BODY_TEMPERATURE_DETECTED")
        notes["body_frac"] = round(body_frac, 4)
        notes["body_temp_range"] = f"{THERMAL_BODY_TEMP_LO}–{THERMAL_BODY_TEMP_HI}°C"

    # Pixel noise (spatial std should be ≥ THERMAL_NOISE_FLOOR for real sensors)
    pixel_std = stats.get("std", 0)
    if pixel_std < THERMAL_NOISE_FLOOR:
        flags.append("SUSPICIOUSLY_LOW_NOISE")

    # Frame-to-frame temporal analysis
    frame_deltas = _frame_to_frame_delta(frames_flat)
    mean_temporal_delta = statistics.mean(frame_deltas) if frame_deltas else 0.0

    if mean_temporal_delta < THERMAL_DELTA_SYNTHETIC and len(frames_flat) > 2:
        flags.append("SUSPECT_SYNTHETIC")
        notes["static_reason"] = "Zero frame-to-frame temperature change"

    # Timing jitter
    jm = _compute_jitter(timestamps_ns)
    cv = jm.get("cv")
    if cv is not None and cv < 1e-8:
        if "SUSPECT_SYNTHETIC" not in flags:
            flags.append("SUSPECT_SYNTHETIC")
        notes["timing_reason"] = "Perfect timing (cv≈0)"

    # Tier decision
    if "SUSPECT_SYNTHETIC" in flags or "SENSOR_FAULT_OUT_OF_RANGE" in flags:
        tier = TIER_REJECT
    elif (confident_human and
          spatial_range >= THERMAL_GRADIENT_GOLD and
          mean_temporal_delta >= THERMAL_DELTA_GOLD and
          not flags):
        tier = TIER_GOLD
    elif (human_present and spatial_range >= THERMAL_GRADIENT_SILVER):
        tier = TIER_SILVER
    elif spatial_range >= 0.5 and pixel_std >= THERMAL_NOISE_FLOOR:
        tier = TIER_BRONZE
    else:
        tier = TIER_REJECT

    return {
        "status":            "PASS" if tier != TIER_REJECT else "FAIL",
        "tier":              tier,
        "sensor_type":       "THERMAL",
        "resolution":        f"{width}x{height}",
        "n_frames":          len(frames_flat),
        "frame_start_ts_ns": timestamps_ns[0] if timestamps_ns else None,
        "frame_end_ts_ns":   timestamps_ns[-1] if timestamps_ns else None,
        "duration_ms":       (timestamps_ns[-1] - timestamps_ns[0]) / 1e6
                             if len(timestamps_ns) > 1 else 0,
        "latest_frame_stats": stats,
        "human_presence": {
            "body_heat_fraction":   round(body_frac, 4),
            "human_present":        human_present,
            "confident_human":      confident_human,
            "body_temp_range_C":    [THERMAL_BODY_TEMP_LO, THERMAL_BODY_TEMP_HI],
        },
        "spatial_analysis": {
            "gradient_mean_C_per_px": round(gradient_mean, 4),
            "spatial_range_C":        round(spatial_range, 3),
            "pixel_noise_std_C":      round(pixel_std, 4),
        },
        "temporal_analysis": {
            "mean_frame_delta_C":   round(mean_temporal_delta, 6),
            "n_frame_pairs":        len(frame_deltas),
            "motion_confirmed":     mean_temporal_delta >= THERMAL_DELTA_GOLD,
        },
        "timing_metrics":    jm,
        "flags":             flags,
        "notes":             notes,
        "device_id":         device_id,
        "session_id":        session_id,
        "tool":              "s2s_thermal_certify_v1_3",
        "issued_at_ns":      time.time_ns(),
    }


# ---------------------------------------------------------------------------
# ThermalStreamCertifier
# ---------------------------------------------------------------------------

class ThermalStreamCertifier:
    """
    Real-time thermal certifier.
    push_frame(ts_ns, flat_pixels) where flat_pixels is a list of
    width*height temperature values in °C.
    """

    def __init__(
        self,
        frame_width: int  = 32,
        frame_height: int = 24,
        window: int       = 30,   # 30 frames at typical 8–30 fps
        step: int         = 5,
        device_id: str    = "unknown",
        session_id: Optional[str] = None,
    ):
        self.width      = frame_width
        self.height     = frame_height
        self.n_pixels   = frame_width * frame_height
        self.window     = window
        self.step       = step
        self.device_id  = device_id
        self.session_id = session_id

        self._ts_buf:     Deque[int]         = deque(maxlen=window)
        self._frame_buf:  Deque[List[float]] = deque(maxlen=window)
        self._frames_since_cert = 0
        self._total_frames      = 0
        self._certs_emitted     = 0

    def push_frame(self, ts_ns: int, pixels: List[float]) -> Optional[Dict[str, Any]]:
        """Push one thermal frame (flat list of width*height temperature values in °C)."""
        if len(pixels) != self.n_pixels:
            raise ValueError(
                f"Expected {self.n_pixels} pixels ({self.width}x{self.height}), "
                f"got {len(pixels)}"
            )
        self._ts_buf.append(ts_ns)
        self._frame_buf.append(list(pixels))
        self._total_frames      += 1
        self._frames_since_cert += 1

        if len(self._ts_buf) >= self.window and self._frames_since_cert >= self.step:
            self._frames_since_cert = 0
            cert = certify_thermal_frames(
                frames_flat   = list(self._frame_buf),
                timestamps_ns = list(self._ts_buf),
                width         = self.width,
                height        = self.height,
                device_id     = self.device_id,
                session_id    = self.session_id,
            )
            cert["cert_index"]        = self._certs_emitted
            cert["total_frames_seen"] = self._total_frames
            self._certs_emitted += 1
            return cert

        return None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "resolution":    f"{self.width}x{self.height}",
            "total_frames":  self._total_frames,
            "certs_emitted": self._certs_emitted,
            "buffer_fill":   len(self._ts_buf),
            "window":        self.window,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2S v1.3 Thermal Certifier",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out-json-dir", default="library/thermal_certs")
    p.add_argument("--width",  type=int, default=32)
    p.add_argument("--height", type=int, default=24)
    p.add_argument("--device-id", default="unknown")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


class ParseError(Exception):
    pass


def _read_s2s_raw(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER_TOTAL_LEN:
        raise ParseError("FILE_TOO_SMALL")
    header_core = data[0:HEADER_CORE_LEN]
    if (zlib.crc32(header_core) & 0xFFFFFFFF) != int.from_bytes(data[HEADER_CORE_LEN:HEADER_CORE_LEN+4], "little"):
        raise ParseError("HEADER_CRC_MISMATCH")
    magic, _, _, payload_len = struct.unpack(HEADER_CORE_FMT, header_core)
    if magic != MAGIC:
        raise ParseError("INVALID_MAGIC")
    payload = data[HEADER_TOTAL_LEN:HEADER_TOTAL_LEN + payload_len]
    pos = 0; meta = {}; timestamps = []; channels = []
    while pos + 6 <= len(payload):
        t = int.from_bytes(payload[pos:pos+2], "little")
        l = int.from_bytes(payload[pos+2:pos+6], "little")
        val = payload[pos+6:pos+6+l]
        if t == int(TLV_META_JSON):
            try: meta = json.loads(val.decode("utf-8"))
            except: pass
        elif t == int(TLV_TIMESTAMPS_NS):
            timestamps = [struct.unpack("<Q", val[i*8:(i+1)*8])[0] for i in range(len(val)//8)]
        elif t == int(TLV_SENSOR_DATA):
            channels.append([struct.unpack("<d", val[i*8:(i+1)*8])[0] for i in range(len(val)//8)])
        pos += 6 + l
    return meta, timestamps, channels


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_json_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import glob as _glob
    paths: List[Path] = []
    for token in args.inputs:
        p = Path(token)
        if p.is_file(): paths.append(p)
        elif p.is_dir(): paths.extend(sorted(p.rglob("*.s2s")))
        else: paths.extend(sorted(Path(x) for x in _glob.glob(token, recursive=True)))

    if not args.quiet:
        print(f"S2S Thermal Certifier v1.3 — {len(paths)} file(s)")

    for path in paths:
        try:
            meta, timestamps, channels = _read_s2s_raw(path)
        except ParseError as e:
            (out_dir / (path.name + ".thermal.json")).write_text(
                json.dumps({"status": "FAIL", "reason": str(e)}, indent=2))
            continue

        # Treat all channels as a sequence of frames
        # Each channel = one frame (flat pixel list)
        frames_flat = channels if channels else []
        result = certify_thermal_frames(
            frames_flat=frames_flat, timestamps_ns=timestamps,
            width=args.width, height=args.height, device_id=args.device_id,
        )
        out_path = out_dir / (path.name + ".thermal.json")
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"[THERMAL] {path.name} → {result['tier']} flags={result['flags']}")

    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    main()
