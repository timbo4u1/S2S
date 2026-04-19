"""
Smart column detector for any sensor dataset.
No column names needed — detects sensor type from data statistics.
"""
import numpy as np
from typing import Dict, List, Tuple

def detect_columns(data: np.ndarray, col_names: List[str] = None) -> Dict:
    """
    Detect sensor types from data statistics.
    Works even with no column names.
    
    Returns:
        {
          'accel': [col_indices],
          'gyro':  [col_indices], 
          'emg':   [col_indices],
          'ppg':   [col_indices],
          'timestamp': col_index or None,
          'confidence': 0-1
        }
    """
    result = {'accel':[], 'gyro':[], 'emg':[], 'ppg':[], 
              'timestamp': None, 'unknown': [], 'confidence': 0.0}
    
    n_cols = data.shape[1]
    votes = []
    
    for i in range(n_cols):
        col = data[:, i]
        # Remove NaN
        col = col[~np.isnan(col)]
        if len(col) < 10:
            continue
            
        median_abs = float(np.median(np.abs(col)))
        std = float(np.std(col))
        mean = float(np.mean(col))
        max_val = float(np.max(np.abs(col)))
        
        # Timestamp: monotonically increasing, huge values
        diffs = np.diff(col)
        is_monotonic = np.all(diffs >= 0)
        if is_monotonic and (median_abs > 100 or (median_abs > 0 and np.all(diffs >= 0) and np.all(diffs < 1e8))):
            result['timestamp'] = i
            votes.append((i, 'timestamp', 0.95))
            continue
        
        # Accelerometer: gravity ~9.81 m/s²
        # or milli-g: ~1000 (9810 milli-g), divide by 1000
        if 7.0 <= median_abs <= 12.0 and std < 5.0:
            result['accel'].append(i)
            votes.append((i, 'accel_ms2', 0.9))
        elif 800 <= median_abs <= 1200 and std < 500:
            result['accel'].append(i)
            votes.append((i, 'accel_millig', 0.85))
        
        # Gyroscope: near zero at rest, moderate range
        elif median_abs < 1.0 and std < 2.0 and max_val < 20.0:
            result['gyro'].append(i)
            votes.append((i, 'gyro_rads', 0.8))
        elif median_abs < 100 and std < 500 and max_val < 5000:
            result['gyro'].append(i)
            votes.append((i, 'gyro_scaled', 0.7))
        
        # EMG: zero-mean, high variance, spiky
        elif abs(mean) < 0.1 * std and std > 0.01 and max_val > 3 * std:
            result['emg'].append(i)
            votes.append((i, 'emg', 0.75))
        
        # PPG: positive values, periodic
        elif mean > 0 and std / mean < 0.3 and max_val < 10000:
            result['ppg'].append(i)
            votes.append((i, 'ppg', 0.6))
        
        else:
            result['unknown'].append(i)
    
    # Confidence: how many columns were classified
    classified = len(votes)
    total = n_cols
    result['confidence'] = round(classified / total, 2) if total > 0 else 0
    result['_votes'] = votes
    
    return result


def detect_from_file(filepath: str, delimiter: str = None) -> Dict:
    """Auto-detect sensor columns from any CSV or space-delimited file."""
    import os
    ext = os.path.splitext(filepath)[1].lower()
    
    if delimiter is None:
        delimiter = ',' if ext == '.csv' else ' '
    
    # Read first 500 rows for speed
    with open(filepath) as _f:
        _first = _f.readline().strip()
    _has_hdr = any(c.isalpha() for c in _first.replace('nan','').replace('NaN',''))
    data = np.genfromtxt(filepath, delimiter=delimiter, max_rows=500,
                         skip_header=1 if _has_hdr else 0)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    result = detect_columns(data)
    result['file'] = filepath
    result['shape'] = data.shape
    return result


def certify_file(filepath: str, segment: str = "forearm",
                 delimiter: str = None, max_windows: int = 100) -> dict:
    """
    Certify any sensor file with zero configuration.
    Auto-detects accel, gyro, EMG columns from data statistics.
    
    Args:
        filepath: path to CSV or space-delimited sensor file
        segment:  body segment (forearm, upper_arm, hand, walking)
        delimiter: auto-detected if None
        max_windows: maximum windows to certify
        
    Returns:
        dict with tier, score, pass_rate, law_details, detected_columns
    
    Example:
        from s2s_standard_v1_3.adapters.column_detect import certify_file
        result = certify_file("my_sensor_data.csv", segment="forearm")
        print(result["tier"], result["pass_rate"])
    """
    import sys
    sys.path.insert(0, ".")
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
    import random

    detection = detect_from_file(filepath, delimiter)
    _delim = delimiter if delimiter else detection.get("_delim", " ")
    # Detect if first row is a header (contains non-numeric text)
    with open(filepath) as _f:
        _first = _f.readline().strip()
    _has_header = any(c.isalpha() for c in _first.replace('nan','').replace('NaN',''))
    skip = 1 if _has_header else 0

    # Streaming chunk reader — works on files of any size, low RAM usage
    # Reads chunk_size rows at a time instead of loading entire file
    chunk_size = max_windows * window  # only read what we need
    data = np.genfromtxt(filepath,
                         delimiter=_delim,
                         skip_header=skip,
                         max_rows=chunk_size)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    acc_cols  = detection["accel"][:3]
    gyro_cols = detection["gyro"][:3]

    # Fallback: if fewer than 3 accel cols detected, try consecutive columns
    # near detected ones (handles near-zero X/Y axes misidentified as gyro)
    if len(acc_cols) < 3 and acc_cols:
        first = acc_cols[0]
        candidates = [first, first+1, first+2]
        if all(c < data.shape[1] for c in candidates):
            acc_cols = candidates

    if not acc_cols:
        return {"error": "No accelerometer columns detected",
                "detected": detection, "tier": "N/A"}

    acc_raw = data[:, acc_cols]
    if gyro_cols:
        gyro_raw = data[:, gyro_cols]
        valid = (~np.isnan(acc_raw).any(axis=1) &
                 ~np.isnan(gyro_raw).any(axis=1))
        # Auto-detect accel unit from median absolute value
        acc_med = float(np.nanmedian(np.abs(acc_raw[valid])))
        if 5.0 <= acc_med <= 15.0:
            acc = acc_raw[valid]              # already m/s2
        elif 500 <= acc_med <= 1500:
            acc = acc_raw[valid] * 0.00981    # milli-g → m/s2
        elif 0.5 <= acc_med <= 1.5:
            acc = acc_raw[valid] * 9.81       # g → m/s2
        else:
            acc = acc_raw[valid] * 0.00981    # assume milli-g

        # Auto-detect gyro unit
        gyro_med = float(np.nanmedian(np.abs(gyro_raw[valid])))
        if gyro_med < 20.0:
            gyro = gyro_raw[valid]            # already rad/s
        else:
            gyro = gyro_raw[valid] / 1000.0   # scaled → rad/s
    else:
        valid = ~np.isnan(acc_raw).any(axis=1)
        acc_med = float(np.nanmedian(np.abs(acc_raw[valid])))
        if 5.0 <= acc_med <= 15.0:
            acc = acc_raw[valid]
        elif 500 <= acc_med <= 1500:
            acc = acc_raw[valid] * 0.00981
        elif 0.5 <= acc_med <= 1.5:
            acc = acc_raw[valid] * 9.81
        else:
            acc = acc_raw[valid] * 0.00981
        gyro = np.zeros_like(acc)

    engine = PhysicsEngine()
    r = random.Random(42)
    window = 256
    step   = 256

    # Auto-detect Hz from timestamp column
    hz = 30.0  # default
    # Use lowest-index timestamp column (most likely the primary clock)
    ts_candidates = [v[0] for v in detection.get("_votes",[]) if "timestamp" in v[1]]
    ts_col = min(ts_candidates) if ts_candidates else None
    if ts_col is not None:
        try:
            ts_vals = data[:50, ts_col]
            ts_vals = ts_vals[~np.isnan(ts_vals)]
            if len(ts_vals) > 2:
                diffs = np.diff(ts_vals)
                med_diff = float(np.median(diffs[diffs > 0]))
                if med_diff > 1e6:        # nanoseconds
                    hz = round(1e9 / med_diff, 1)
                elif med_diff > 1000:     # microseconds
                    hz = round(1e6 / med_diff, 1)
                elif med_diff > 1:        # milliseconds
                    hz = round(1e3 / med_diff, 1)
                elif 0 < med_diff <= 1:   # seconds
                    hz = round(1.0 / med_diff, 1)
                hz = max(1.0, min(hz, 10000.0))  # sanity clamp
        except Exception:
            hz = 30.0

    tiers  = []
    scores = []
    for start in range(0, min(len(acc) - window, max_windows * step), step):
        chunk_a = acc[start:start+window].tolist()
        chunk_g = gyro[start:start+window].tolist()
        ts = [int(i * 1e9 / hz) for i in range(window)]
        cert = engine.certify(
            {"timestamps_ns": ts, "accel": chunk_a, "gyro": chunk_g},
            segment=segment
        )
        tiers.append(cert["tier"])
        scores.append(cert["physical_law_score"])

    if not tiers:
        return {"error": "No windows processed", "tier": "N/A"}

    from collections import Counter
    counts = Counter(tiers)
    certified = counts["GOLD"] + counts["SILVER"] + counts["BRONZE"]
    pass_rate = certified / len(tiers)

    # Overall tier by majority
    if counts["GOLD"] > len(tiers) * 0.5:
        summary = "GOLD"
    elif counts["REJECTED"] > len(tiers) * 0.3:
        summary = "REJECTED"
    elif pass_rate > 0.7:
        summary = "SILVER"
    else:
        summary = "BRONZE"

    return {
        "tier":             summary,
        "pass_rate":        round(pass_rate, 3),
        "mean_score":       round(sum(scores)/len(scores), 1),
        "total_windows":    len(tiers),
        "certified":        certified,
        "rejected":         counts["REJECTED"],
        "tier_counts":      dict(counts),
        "detected_columns": {
            "accel":      acc_cols,
            "gyro":       gyro_cols,
            "confidence": detection["confidence"]
        },
        "detected_hz":      hz,
        "segment":          segment,
    }
