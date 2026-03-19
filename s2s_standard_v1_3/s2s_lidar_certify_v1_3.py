#!/usr/bin/env python3
"""
s2s_lidar_certify_v1_3.py — S2S LiDAR Certifier

Handles two LiDAR modes:
  1. Scalar (1D) — single distance measurements over time (e.g. time-of-flight rangefinder)
  2. Point cloud (3D) — XYZ point arrays per frame (e.g. rotating LiDAR, structured light)

What makes LiDAR data "real" vs synthetic:
  - Real LiDAR has measurement noise (surface texture, air, lens aberration)
  - Real scenes have structural variance — walls, floors, objects at different depths
  - Synthetic data has perfect geometric surfaces with zero noise
  - Real scans show frame-to-frame micro-motion (hand tremor, breathing, vibration)
  - Synthetic scans are perfectly still between frames

Certificate tiers:
  GOLD   — high variance, confirmed scene structure, real measurement noise, motion present
  SILVER — valid signal, moderate noise, acceptable structure
  BRONZE — usable but low variance or minimal motion
  REJECTED — flat signal (sensor covered), synthetic geometry, or all-zero / saturated

Usage:
  # Streaming 1D:
  from s2s_standard_v1_3.s2s_lidar_certify_v1_3 import LiDARStreamCertifier
  lc = LiDARStreamCertifier(mode='scalar', device_id='tof_sensor_01')
  cert = lc.push_frame(ts_ns, [distance_m])

  # Streaming 3D point cloud:
  lc3d = LiDARStreamCertifier(mode='pointcloud', n_points_per_frame=360)
  cert = lc3d.push_frame(ts_ns, flat_xyz_list)  # [x0,y0,z0, x1,y1,z1, ...]

  # Batch file:
  python3 -m s2s_standard_v1_3.s2s_lidar_certify_v1_3 scan.s2s --out-json-dir certs/
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import time
import zlib
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from .constants import (
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        MAGIC, TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA,
        TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
        STREAM_WINDOW_DEFAULT, STREAM_STEP_DEFAULT,
    )
except Exception:
    from constants import (
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        MAGIC, TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA,
        TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
        STREAM_WINDOW_DEFAULT, STREAM_STEP_DEFAULT,
    )

# ---------------------------------------------------------------------------
# LiDAR-specific constants
# ---------------------------------------------------------------------------
LIDAR_SCALAR_MIN_HZ          = 10.0    # minimum sampling rate for 1D LiDAR
LIDAR_NOISE_FLOOR_M          = 0.0005  # realistic noise floor (0.5mm)
LIDAR_VAR_GOLD               = 0.01    # variance > 0.01m² = meaningful scene depth variance
LIDAR_VAR_SILVER             = 0.001
LIDAR_VAR_BRONZE             = 1e-4
LIDAR_SATURATION_M           = 40.0    # typical max range; above = invalid
LIDAR_FRAME_DELTA_GOLD       = 0.0005  # frame-to-frame RMS change > 0.5mm = real motion
LIDAR_SYNTHETIC_DELTA_THRESH = 1e-9    # frame-to-frame delta < this = perfectly static (synthetic)
LIDAR_POINTCLOUD_MIN_POINTS  = 16      # minimum valid points per frame

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _variance(data: List[float]) -> float:
    finite = [v for v in data if math.isfinite(v) and v > 0]
    if len(finite) < 2:
        return 0.0
    mu = statistics.mean(finite)
    return sum((x - mu) ** 2 for x in finite) / len(finite)


def _rms(data: List[float]) -> float:
    finite = [v for v in data if math.isfinite(v)]
    if not finite:
        return 0.0
    return math.sqrt(sum(v * v for v in finite) / len(finite))


def _compute_jitter(timestamps: List[int]) -> Dict[str, Optional[float]]:
    if len(timestamps) < 2:
        return {"mean_delta_ns": None, "rms_jitter_ns": None, "cv": None}
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    mu = statistics.mean(deltas)
    sigma = statistics.pstdev(deltas)
    cv = sigma / mu if mu > 0 else None
    return {"mean_delta_ns": mu, "rms_jitter_ns": sigma, "cv": cv}


def _frame_to_frame_rms_delta(frames: List[List[float]]) -> float:
    """
    Compute RMS of per-point differences between consecutive frames.
    Detects micro-motion between scans — real devices always have some.
    """
    if len(frames) < 2:
        return 0.0
    deltas: List[float] = []
    for fa, fb in zip(frames, frames[1:]):
        n = min(len(fa), len(fb))
        for i in range(n):
            if math.isfinite(fa[i]) and math.isfinite(fb[i]):
                deltas.append(fb[i] - fa[i])
    if not deltas:
        return 0.0
    return math.sqrt(sum(d * d for d in deltas) / len(deltas))


# ---------------------------------------------------------------------------
# 1D Scalar LiDAR analysis
# ---------------------------------------------------------------------------

def analyze_scalar_lidar(
    distances: List[float],
    timestamps_ns: List[int],
    device_id: str = "unknown",
) -> Dict[str, Any]:
    """Certify a window of 1D distance measurements."""
    flags: List[str] = []
    notes: Dict[str, Any] = {}

    finite = [v for v in distances if math.isfinite(v) and 0 < v < LIDAR_SATURATION_M]
    nan_count = len(distances) - len(finite)
    nan_frac  = nan_count / max(len(distances), 1)

    if nan_frac > 0.5:
        flags.append("EXCESSIVE_INVALID_READINGS")

    # Saturation check
    sat_count = sum(1 for v in distances if math.isfinite(v) and v >= LIDAR_SATURATION_M)
    sat_frac  = sat_count / max(len(distances), 1)
    if sat_frac > 0.3:
        flags.append("SIGNAL_SATURATED")

    # Variance (scene depth richness)
    var = _variance(finite)

    # Frame-to-frame motion (micro-motion detection)
    # For 1D, each sample is a "frame"
    deltas = [abs(b - a) for a, b in zip(finite, finite[1:]) if math.isfinite(a) and math.isfinite(b)]
    mean_delta = statistics.mean(deltas) if deltas else 0.0
    delta_rms  = math.sqrt(sum(d * d for d in deltas) / len(deltas)) if deltas else 0.0

    # Synthetic detection: perfectly static signal
    if delta_rms < LIDAR_SYNTHETIC_DELTA_THRESH:
        flags.append("SUSPECT_SYNTHETIC")
        notes["synthetic_reason"] = "Zero frame-to-frame variation (perfectly static)"

    # Noise floor check
    if var < LIDAR_VAR_BRONZE and not flags:
        flags.append("LOW_VARIANCE_POSSIBLY_COVERED")

    # Jitter
    jm = _compute_jitter(timestamps_ns)
    cv = jm.get("cv")
    if cv is not None and cv < 1e-8:
        if "SUSPECT_SYNTHETIC" not in flags:
            flags.append("SUSPECT_SYNTHETIC")
        notes["timing_reason"] = "Perfect timing (cv≈0)"

    # Tier decision
    if "SUSPECT_SYNTHETIC" in flags or "EXCESSIVE_INVALID_READINGS" in flags:
        tier = TIER_REJECT
    elif var >= LIDAR_VAR_GOLD and delta_rms >= LIDAR_FRAME_DELTA_GOLD:
        tier = TIER_GOLD
    elif var >= LIDAR_VAR_SILVER:
        tier = TIER_SILVER
    elif var >= LIDAR_VAR_BRONZE:
        tier = TIER_BRONZE
    else:
        tier = TIER_REJECT

    finite_vals = finite if finite else [0.0]
    return {
        "status":           "PASS" if tier != TIER_REJECT else "FAIL",
        "tier":             tier,
        "sensor_type":      "LIDAR_SCALAR",
        "mode":             "scalar",
        "n_samples":        len(distances),
        "n_valid":          len(finite),
        "nan_fraction":     round(nan_frac, 4),
        "saturation_fraction": round(sat_frac, 4),
        "distance_stats": {
            "min_m":    round(min(finite_vals), 4),
            "max_m":    round(max(finite_vals), 4),
            "mean_m":   round(statistics.mean(finite_vals), 4),
            "variance": round(var, 6),
        },
        "motion": {
            "mean_delta_m":   round(mean_delta, 6),
            "delta_rms_m":    round(delta_rms, 6),
            "micro_motion_confirmed": delta_rms >= LIDAR_FRAME_DELTA_GOLD,
        },
        "timing_metrics":   jm,
        "flags":            flags,
        "notes":            notes,
        "device_id":        device_id,
        "tool":             "s2s_lidar_certify_v1_3",
        "issued_at_ns":     time.time_ns(),
    }


# ---------------------------------------------------------------------------
# 3D Point Cloud LiDAR analysis
# ---------------------------------------------------------------------------

def _pointcloud_stats(points_xyz: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    """
    Compute structural statistics of a 3D point cloud.
    Returns density, axis variances, bounding box, planarity estimate.
    """
    if not points_xyz:
        return {"n_points": 0}

    xs = [p[0] for p in points_xyz]
    ys = [p[1] for p in points_xyz]
    zs = [p[2] for p in points_xyz]

    var_x = _variance(xs)
    var_y = _variance(ys)
    var_z = _variance(zs)

    # Planarity: if one axis has near-zero variance, scene is a flat plane
    variances = sorted([var_x, var_y, var_z])
    planarity = 1.0 - (variances[0] / (variances[2] + 1e-9))

    # Distance from origin distribution
    distances = [math.sqrt(x*x + y*y + z*z) for x, y, z in points_xyz]
    dist_var  = _variance(distances)

    return {
        "n_points":    len(points_xyz),
        "var_x":       round(var_x, 6),
        "var_y":       round(var_y, 6),
        "var_z":       round(var_z, 6),
        "total_var":   round(var_x + var_y + var_z, 6),
        "planarity":   round(planarity, 4),
        "dist_var":    round(dist_var, 6),
        "bbox": {
            "x": [round(min(xs), 3), round(max(xs), 3)],
            "y": [round(min(ys), 3), round(max(ys), 3)],
            "z": [round(min(zs), 3), round(max(zs), 3)],
        },
    }


def analyze_pointcloud_lidar(
    frames_xyz: List[List[float]],   # each frame: flat list [x0,y0,z0, x1,y1,z1,...]
    timestamps_ns: List[int],
    device_id: str = "unknown",
    min_points: int = LIDAR_POINTCLOUD_MIN_POINTS,
) -> Dict[str, Any]:
    """Certify a sequence of 3D point cloud frames."""
    flags: List[str] = []
    notes: Dict[str, Any] = {}

    # Parse flat XYZ lists into tuples
    def _parse_frame(flat: List[float]) -> List[Tuple[float, float, float]]:
        pts = []
        for i in range(0, len(flat) - 2, 3):
            x, y, z = flat[i], flat[i+1], flat[i+2]
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                pts.append((x, y, z))
        return pts

    parsed_frames = [_parse_frame(f) for f in frames_xyz]
    valid_frames  = [f for f in parsed_frames if len(f) >= min_points]

    if not valid_frames:
        return {
            "status": "FAIL", "tier": TIER_REJECT,
            "sensor_type": "LIDAR_POINTCLOUD",
            "flags": ["NO_VALID_FRAMES"],
            "tool": "s2s_lidar_certify_v1_3",
            "issued_at_ns": time.time_ns(),
        }

    # Per-frame stats on latest frame
    latest_stats = _pointcloud_stats(valid_frames[-1])

    # Frame-to-frame motion: compare centroid movement
    centroids: List[Tuple[float, float, float]] = []
    for frame in valid_frames:
        cx = statistics.mean(p[0] for p in frame)
        cy = statistics.mean(p[1] for p in frame)
        cz = statistics.mean(p[2] for p in frame)
        centroids.append((cx, cy, cz))

    centroid_deltas = []
    for ca, cb in zip(centroids, centroids[1:]):
        d = math.sqrt(sum((b - a)**2 for a, b in zip(ca, cb)))
        centroid_deltas.append(d)

    centroid_motion_rms = _rms(centroid_deltas) if centroid_deltas else 0.0

    # Point-level frame-to-frame delta (first N points that exist in both frames)
    flat_deltas_rms = _frame_to_frame_rms_delta(
        [[math.sqrt(p[0]**2+p[1]**2+p[2]**2) for p in f] for f in valid_frames]
    )

    # Synthetic detection
    jm = _compute_jitter(timestamps_ns)
    cv = jm.get("cv")
    if cv is not None and cv < 1e-8:
        flags.append("SUSPECT_SYNTHETIC")
        notes["timing_reason"] = "Perfect timing (cv≈0)"

    if flat_deltas_rms < LIDAR_SYNTHETIC_DELTA_THRESH and len(valid_frames) > 2:
        if "SUSPECT_SYNTHETIC" not in flags:
            flags.append("SUSPECT_SYNTHETIC")
        notes["motion_reason"] = "Zero frame-to-frame variation in point distances"

    # Structure check — if planarity > 0.99, scene is suspiciously perfect flat plane
    if latest_stats.get("planarity", 0) > 0.99:
        flags.append("PERFECTLY_PLANAR_SCENE")
        notes["planarity_note"] = "Scene appears perfectly flat — may be synthetic"

    # Tier decision based on structural variance + motion
    total_var = latest_stats.get("total_var", 0)
    if "SUSPECT_SYNTHETIC" in flags:
        tier = TIER_REJECT
    elif total_var >= LIDAR_VAR_GOLD and centroid_motion_rms >= LIDAR_FRAME_DELTA_GOLD:
        tier = TIER_GOLD
    elif total_var >= LIDAR_VAR_SILVER:
        tier = TIER_SILVER
    elif total_var >= LIDAR_VAR_BRONZE:
        tier = TIER_BRONZE
    else:
        tier = TIER_REJECT
        flags.append("INSUFFICIENT_SCENE_VARIANCE")

    return {
        "status":            "PASS" if tier != TIER_REJECT else "FAIL",
        "tier":              tier,
        "sensor_type":       "LIDAR_POINTCLOUD",
        "mode":              "pointcloud",
        "n_frames":          len(frames_xyz),
        "n_valid_frames":    len(valid_frames),
        "latest_frame_stats": latest_stats,
        "motion": {
            "centroid_motion_rms_m":  round(centroid_motion_rms, 6),
            "point_delta_rms_m":      round(flat_deltas_rms, 6),
            "micro_motion_confirmed": centroid_motion_rms >= LIDAR_FRAME_DELTA_GOLD,
        },
        "timing_metrics":    jm,
        "flags":             flags,
        "notes":             notes,
        "device_id":         device_id,
        "tool":              "s2s_lidar_certify_v1_3",
        "issued_at_ns":      time.time_ns(),
    }


# ---------------------------------------------------------------------------
# LiDARStreamCertifier
# ---------------------------------------------------------------------------

class LiDARStreamCertifier:
    """
    Real-time LiDAR certifier for both scalar and point cloud modes.

    Scalar mode:  push_frame(ts_ns, [distance_m])
    Pointcloud:   push_frame(ts_ns, [x0,y0,z0, x1,y1,z1, ...])
    """

    def __init__(
        self,
        mode: str = "scalar",         # "scalar" or "pointcloud"
        n_points_per_frame: int = 1,   # for pointcloud: expected XYZ points per frame
        window: int = 128,
        step: int = 16,
        device_id: str = "unknown",
    ):
        if mode not in ("scalar", "pointcloud"):
            raise ValueError("mode must be 'scalar' or 'pointcloud'")
        self.mode               = mode
        self.n_points_per_frame = n_points_per_frame
        self.window             = window
        self.step               = step
        self.device_id          = device_id

        self._ts_buf:     Deque[int]         = deque(maxlen=window)
        self._frame_buf:  Deque[List[float]] = deque(maxlen=window)
        self._frames_since_cert = 0
        self._total_frames      = 0
        self._certs_emitted     = 0

    def push_frame(self, ts_ns: int, values: List[float]) -> Optional[Dict[str, Any]]:
        self._ts_buf.append(ts_ns)
        self._frame_buf.append(list(values))
        self._total_frames      += 1
        self._frames_since_cert += 1

        if len(self._ts_buf) >= self.window and self._frames_since_cert >= self.step:
            self._frames_since_cert = 0
            timestamps = list(self._ts_buf)
            frames     = list(self._frame_buf)

            if self.mode == "scalar":
                distances = [f[0] if f else float("nan") for f in frames]
                cert = analyze_scalar_lidar(distances, timestamps, self.device_id)
            else:
                cert = analyze_pointcloud_lidar(frames, timestamps, self.device_id)

            cert["frame_start_ts_ns"] = timestamps[0]
            cert["frame_end_ts_ns"]   = timestamps[-1]
            cert["duration_ms"]       = (timestamps[-1] - timestamps[0]) / 1e6
            cert["cert_index"]        = self._certs_emitted
            cert["total_frames_seen"] = self._total_frames
            self._certs_emitted += 1
            return cert

        return None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "mode":          self.mode,
            "total_frames":  self._total_frames,
            "certs_emitted": self._certs_emitted,
            "buffer_fill":   len(self._ts_buf),
            "window":        self.window,
        }


# ---------------------------------------------------------------------------
# CLI (batch .s2s files)
# ---------------------------------------------------------------------------

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2S v1.3 LiDAR Certifier",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out-json-dir", default="library/lidar_certs")
    p.add_argument("--mode", choices=["scalar", "pointcloud"], default="scalar")
    p.add_argument("--device-id", default="unknown")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


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
        print(f"S2S LiDAR Certifier v1.3 — {len(paths)} file(s) → {out_dir}")

    for path in paths:
        try:
            meta, timestamps, channels = _read_s2s_raw(path)
        except ParseError as e:
            (out_dir / (path.name + ".lidar.json")).write_text(json.dumps({"status":"FAIL","reason":str(e)}, indent=2))
            continue

        if args.mode == "scalar":
            distances = channels[0] if channels else []
            result = analyze_scalar_lidar(distances, timestamps, device_id=args.device_id)
        else:
            # Each channel is one coordinate axis; interleave as XYZ frames
            frames_xyz = []
            if len(channels) >= 3:
                n = min(len(channels[0]), len(channels[1]), len(channels[2]))
                for i in range(n):
                    frames_xyz.append([channels[0][i], channels[1][i], channels[2][i]])
            result = analyze_pointcloud_lidar([frames_xyz], timestamps, device_id=args.device_id)

        out_path = out_dir / (path.name + ".lidar.json")
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"[LIDAR] {path.name} → {result['tier']} flags={result['flags']}")

    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    main()
