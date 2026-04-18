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
    data = np.genfromtxt(filepath, delimiter=delimiter, max_rows=500)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    result = detect_columns(data)
    result['file'] = filepath
    result['shape'] = data.shape
    return result
