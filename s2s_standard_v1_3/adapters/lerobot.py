"""
s2s_standard_v1_3/adapters/lerobot.py

S2S adapter for LeRobot datasets (Hugging Face).
Certifies IMU data in LeRobot episodes against 7 biomechanical laws.

Usage:
    from s2s_standard_v1_3.adapters.lerobot import certify_lerobot_episode

    # Works with any LeRobot dataset that has IMU/acceleration columns
    results = certify_lerobot_episode(
        episode_data,          # dict with 'timestamp' and acceleration keys
        accel_keys=('observation.state',),
        segment='forearm',
        hz=30.0,
    )

LeRobot dataset format:
    Datasets are stored as Parquet files on Hugging Face.
    Each episode is a sequence of frames with observations and actions.
    S2S treats each sliding window of frames as one certification window.

Example with huggingface_hub:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    parquet = hf_hub_download(
        repo_id='lerobot/pusht',
        filename='data/train/episode_000000.parquet',
        repo_type='dataset'
    )
    df = pd.read_parquet(parquet)
    results = certify_lerobot_dataframe(df, hz=10.0, segment='forearm')
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
import time

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

_engine = PhysicsEngine()


def certify_lerobot_episode(
    frames: List[Dict],
    accel_keys: tuple = ('observation.state',),
    gyro_keys: Optional[tuple] = None,
    segment: str = 'forearm',
    hz: float = 30.0,
    window: int = 256,
    step: int = 128,
) -> Dict[str, Any]:
    """
    Certify one LeRobot episode against S2S biomechanical laws.

    Args:
        frames:     list of frame dicts from a LeRobot episode
        accel_keys: keys in each frame containing acceleration data
        gyro_keys:  keys containing gyroscope data (optional)
        segment:    body segment — forearm, hand, upper_arm, walking
        hz:         sampling rate of the episode
        window:     certification window size in samples
        step:       step between windows

    Returns:
        dict with:
            total_windows:   int
            certified:       int  (GOLD/SILVER/BRONZE)
            rejected:        int
            pass_rate:       float
            windows:         list of per-window results
            summary_tier:    str  overall episode tier
    """
    import random
    r = random.Random(42)

    # Extract acceleration values from frames
    accel_data = []
    for frame in frames:
        vals = []
        for key in accel_keys:
            v = frame.get(key, [])
            if isinstance(v, (list, tuple)):
                vals.extend(v[:3])
            elif isinstance(v, (int, float)):
                vals.append(float(v))
        if len(vals) >= 3:
            accel_data.append(vals[:3])
        elif len(vals) == 1:
            accel_data.append([vals[0], 0.0, 0.0])

    if len(accel_data) < window:
        return {
            'error': f'Not enough frames: {len(accel_data)} < window {window}',
            'total_windows': 0, 'certified': 0, 'rejected': 0,
            'pass_rate': 0.0, 'windows': [],
        }

    # Extract gyro if available
    gyro_data = []
    if gyro_keys:
        for frame in frames:
            vals = []
            for key in gyro_keys:
                v = frame.get(key, [])
                if isinstance(v, (list, tuple)):
                    vals.extend(v[:3])
            if len(vals) >= 3:
                gyro_data.append(vals[:3])
            else:
                gyro_data.append([0.0, 0.0, 0.0])
    else:
        gyro_data = [[0.0, 0.0, 0.0]] * len(accel_data)

    # Slide window through episode
    results = []
    dt_ns = int(1e9 / hz)
    for start in range(0, len(accel_data) - window, step):
        acc_win  = accel_data[start:start + window]
        gyro_win = gyro_data[start:start + window]
        ts = [int(start * dt_ns + i * dt_ns + r.gauss(0, dt_ns * 0.01))
              for i in range(window)]

        t0   = time.perf_counter()
        cert = _engine.certify(
            {'timestamps_ns': ts, 'accel': acc_win, 'gyro': gyro_win},
            segment=segment
        )
        cert['_latency_ms'] = round((time.perf_counter() - t0) * 1000, 3)
        cert['_window_start_frame'] = start
        results.append(cert)

    certified = sum(1 for r in results if r['tier'] != 'REJECTED')
    rejected  = len(results) - certified
    pass_rate = certified / len(results) if results else 0.0

    # Overall episode tier
    tiers = [r['tier'] for r in results]
    if tiers.count('GOLD') > len(tiers) * 0.5:
        summary = 'GOLD'
    elif tiers.count('REJECTED') > len(tiers) * 0.3:
        summary = 'REJECTED'
    elif certified / len(tiers) > 0.7:
        summary = 'SILVER'
    else:
        summary = 'BRONZE'

    return {
        'total_windows': len(results),
        'certified':     certified,
        'rejected':      rejected,
        'pass_rate':     round(pass_rate, 3),
        'summary_tier':  summary,
        'segment':       segment,
        'hz':            hz,
        'windows':       results,
    }


def certify_lerobot_dataframe(
    df,
    accel_cols: Optional[List[str]] = None,
    gyro_cols:  Optional[List[str]] = None,
    segment: str = 'forearm',
    hz: float = 30.0,
    window: int = 256,
    step: int = 128,
) -> Dict[str, Any]:
    """
    Certify a LeRobot episode from a pandas DataFrame (Parquet format).

    Auto-detects acceleration columns if accel_cols not provided.
    """
    if accel_cols is None:
        # Auto-detect: look for columns with IMU-specific names
        # NOTE: Standard LeRobot datasets (PushT, ALOHA, etc.) use
        # observation.state for joint positions/pixel coords — NOT IMU data.
        # S2S requires real acceleration data in m/s².
        # Pass accel_cols explicitly for datasets with IMU streams.
        candidates = [c for c in df.columns
                      if any(k in c.lower() for k in
                             ('accel', 'imu', 'acceleration', 'gyro', 'inertial'))]
        if not candidates:
            return {
                'error': (
                    'No IMU/acceleration columns found. '
                    'Standard LeRobot datasets (PushT, ALOHA) use joint positions, '
                    'not raw IMU. Pass accel_cols explicitly for datasets with IMU. '
                    f'Available columns: {list(df.columns)}'
                ),
                'total_windows': 0, 'certified': 0, 'rejected': 0,
                'pass_rate': 0.0, 'windows': [], 'summary_tier': 'N/A',
            }
        accel_cols = candidates[:1]

    frames = []
    for _, row in df.iterrows():
        frame = {}
        for col in accel_cols:
            frame[col] = row[col] if hasattr(row[col], '__len__') else [float(row[col])]
        if gyro_cols:
            for col in gyro_cols:
                frame[col] = row[col] if hasattr(row[col], '__len__') else [float(row[col])]
        frames.append(frame)

    return certify_lerobot_episode(
        frames,
        accel_keys=tuple(accel_cols),
        gyro_keys=tuple(gyro_cols) if gyro_cols else None,
        segment=segment, hz=hz, window=window, step=step,
    )
