"""
S2S adapter for LeRobot datasets.

Requires explicit acceleration-column mapping. Joint positions and actions
are not treated as measured accelerometer data.

Missing sensor frames retain their original positions. Incomplete
acceleration windows are skipped; incomplete gyro windows are assessed
without a gyro stream.
"""
from __future__ import annotations

from numbers import Real
from typing import Dict, List, Optional, Any
import math
import time

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

_engine = PhysicsEngine()
SOURCE_TYPE = "ROBOT_SENSOR_MEASURED"


def _normalize_keys(keys):
    if keys is None:
        return None
    if isinstance(keys, str):
        return (keys,)
    return tuple(keys)


def _extract_vector(frame: Dict, keys) -> Optional[List[float]]:
    """
    Extract exactly three finite numeric components from the requested keys.

    Supports one vector-valued key or multiple scalar axis keys.
    Returns None for missing, malformed, or incomplete data.
    """
    if not isinstance(frame, dict) or not keys:
        return None

    values = []

    for key in keys:
        if key not in frame:
            return None

        value = frame[key]
        if value is None:
            return None

        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, (str, bytes, bool)):
            return None

        if isinstance(value, (list, tuple)):
            components = list(value)
        else:
            components = [value]

        values.extend(components)

    if len(values) != 3:
        return None

    vector = []
    for component in values:
        if isinstance(component, (str, bytes, bool)):
            return None
        try:
            number = float(component)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(number):
            return None

        vector.append(number)

    return vector


def _empty_result(error: str) -> Dict[str, Any]:
    return {
        "error": error,
        "source_type": SOURCE_TYPE,
        "total_windows": 0,
        "certified": 0,
        "rejected": 0,
        "pass_rate": 0.0,
        "summary_tier": "N/A",
        "windows": [],
        "windows_skipped_invalid_accel": 0,
        "gyro_window_availability": {
            "complete": 0,
            "partial": 0,
            "absent": 0,
        },
    }


def certify_lerobot_episode(
    frames: List[Dict],
    accel_keys: Optional[tuple] = None,
    gyro_keys: Optional[tuple] = None,
    segment: str = "forearm",
    hz: float = 30.0,
    window: int = 256,
    step: int = 128,
) -> Dict[str, Any]:
    """
    Certify measured acceleration data in one LeRobot episode.

    Args:
        frames:     Original episode frames.
        accel_keys: Explicit keys containing measured acceleration.
                    Required. Their combined values must contain 3 axes.
        gyro_keys:  Optional keys containing measured gyroscope data.
                    Their combined values must contain 3 axes.
        segment:    Body segment.
        hz:         Sampling rate in Hz.
        window:     Samples per certification window.
        step:       Original frame-index step between windows.

    Missing acceleration frames are not deleted or compressed. Any
    window containing an invalid acceleration frame is skipped.

    Gyro is included only when every frame in a window has a complete,
    finite three-axis gyro vector. Real zero-valued measurements are valid.
    """
    accel_keys = _normalize_keys(accel_keys)
    gyro_keys = _normalize_keys(gyro_keys)

    if not accel_keys:
        return _empty_result(
            "No explicit acceleration mapping supplied. "
            "Pass accel_keys containing measured accelerometer data; "
            "observation.state and action fields are not assumed to be IMU."
        )

    if not math.isfinite(float(hz)) or hz <= 0:
        return _empty_result("hz must be a positive finite sampling rate")

    if window <= 0 or step <= 0:
        return _empty_result("window and step must both be positive")

    n_frames = len(frames)

    if n_frames < window:
        return _empty_result(
            "Not enough frames: {} < window {}".format(n_frames, window)
        )

    # One entry per original frame. Invalid measurements remain None;
    # no frame is removed, so all modalities retain their original indices.
    accel_data = [
        _extract_vector(frame, accel_keys)
        for frame in frames
    ]

    if gyro_keys:
        gyro_data = [
            _extract_vector(frame, gyro_keys)
            for frame in frames
        ]
    else:
        gyro_data = [None] * n_frames

    invalid_accel_frames = sum(
        sample is None for sample in accel_data
    )

    dt_ns = int(round(1e9 / float(hz)))
    results = []
    skipped_invalid_accel = 0

    gyro_window_counts = {
        "complete": 0,
        "partial": 0,
        "absent": 0,
        "invalid": 0,
    }

    # +1 ensures an episode with exactly one window is processed.
    for start in range(0, n_frames - window + 1, step):
        end = start + window

        acc_win = accel_data[start:end]

        # Never compress the frame sequence by removing invalid rows.
        if any(sample is None for sample in acc_win):
            skipped_invalid_accel += 1
            continue

        gyro_win = gyro_data[start:end]
        gyro_present = sum(
            sample is not None for sample in gyro_win
        )

        imu_raw = {
            "timestamps_ns": [
                (start + i) * dt_ns
                for i in range(window)
            ],
            "accel": acc_win,
        }

        if gyro_present == window:
            gyro_availability = "complete"
            imu_raw["gyro"] = gyro_win
        elif gyro_present > 0:
            gyro_availability = "partial"
        elif gyro_keys:
            # Gyro keys were supplied but all extracted vectors are invalid.
            # This is distinct from 'absent' (no gyro keys at all).
            gyro_availability = "invalid"
        else:
            gyro_availability = "absent"

        t0 = time.perf_counter()
        cert = dict(
            _engine.certify(
                imu_raw,
                segment=segment,
                source_type=SOURCE_TYPE,
            )
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        cert["_gyro_availability"] = gyro_availability
        cert["_gyro_samples_present"] = gyro_present
        cert["_gyro_samples_expected"] = window
        cert["_accel_frames_present"] = window
        cert["_accel_frames_expected"] = window
        cert["_window_start_frame"] = start
        cert["_latency_ms"] = round(latency_ms, 3)

        gyro_window_counts[gyro_availability] += 1
        results.append(cert)

    tiers = [result["tier"] for result in results]

    certified = sum(
        1 for tier in tiers if tier != "REJECTED"
    )
    rejected = len(results) - certified
    pass_rate = certified / len(results) if results else 0.0

    if not tiers:
        summary = "N/A"
    elif tiers.count("GOLD") > len(tiers) * 0.5:
        summary = "GOLD"
    elif tiers.count("REJECTED") > len(tiers) * 0.3:
        summary = "REJECTED"
    elif pass_rate > 0.7:
        summary = "SILVER"
    else:
        summary = "BRONZE"

    return {
        "source_type": SOURCE_TYPE,
        "total_windows": len(results),
        "certified": certified,
        "rejected": rejected,
        "pass_rate": round(pass_rate, 3),
        "summary_tier": summary,
        "segment": segment,
        "hz": hz,
        "windows": results,
        "invalid_accel_frames": invalid_accel_frames,
        "windows_skipped_invalid_accel": skipped_invalid_accel,
        "gyro_window_availability": gyro_window_counts,
    }


def certify_lerobot_dataframe(
    df,
    accel_cols: Optional[List[str]] = None,
    gyro_cols: Optional[List[str]] = None,
    segment: str = "forearm",
    hz: float = 30.0,
    window: int = 256,
    step: int = 128,
) -> Dict[str, Any]:
    """
    Certify a LeRobot episode from a pandas DataFrame.

    Only explicitly named acceleration columns are considered for
    automatic discovery. Generic state, action, and gyro columns are
    not treated as acceleration.
    """
    if accel_cols is None:
        candidates = [
            col for col in df.columns
            if any(
                token in str(col).lower()
                for token in ("accelerometer", "acceleration", "accel")
            )
        ]

        if not candidates:
            return _empty_result(
                "No explicit acceleration columns found. "
                "Pass accel_cols for measured acceleration data. "
                "observation.state, action, and gyro columns are not "
                "assumed to contain accelerometer measurements. "
                "Available columns: {}".format(list(df.columns))
            )

        accel_cols = candidates

    accel_cols = _normalize_keys(accel_cols)
    gyro_cols = _normalize_keys(gyro_cols)

    frames = []

    for _, row in df.iterrows():
        frame = {}

        for col in accel_cols:
            frame[col] = row[col]

        if gyro_cols:
            for col in gyro_cols:
                frame[col] = row[col]

        frames.append(frame)

    return certify_lerobot_episode(
        frames,
        accel_keys=accel_cols,
        gyro_keys=gyro_cols,
        segment=segment,
        hz=hz,
        window=window,
        step=step,
    )
