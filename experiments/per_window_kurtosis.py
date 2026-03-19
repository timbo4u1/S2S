#!/usr/bin/env python3
"""
Analyze kurtosis distribution of rate-normalized jerk for NinaPro subjects s3 and s5
"""

import json
import numpy as np
import os
from scipy.stats import kurtosis
import statistics

def load_ninapro_subject(subject_dir):
    """Load NinaPro subject data"""
    import scipy.io
    import glob
    
    # Find the relevant files
    mat_files = glob.glob(os.path.join(subject_dir, "*.mat"))
    
    if not mat_files:
        print(f"  No .mat files found in {subject_dir}")
        return None, None, None
    
    # Try to load the main data file
    data = None
    
    for mat_file in mat_files:
        try:
            mat_data = scipy.io.loadmat(mat_file)
            # Look for common NinaPro variable names
            for key in ['stimulus', 'emg', 'acc', 'restimulus']:
                if key in mat_data:
                    if data is None:
                        data = {}
                    data[key] = mat_data[key]
            if data is not None:
                break
        except Exception as e:
            print(f"  Error loading {mat_file}: {e}")
            continue
    
    if data is None:
        print(f"  Could not load any data from {subject_dir}")
        return None, None, None
    
    emg = data.get('emg', [])
    accel = data.get('acc', [])
    stimulus = data.get('restimulus', data.get('stimulus', []))
    
    return emg, accel, stimulus

def calculate_rate_normalized_jerk(accel, sample_rate=2000):
    """Calculate rate-normalized jerk from accelerometer data"""
    # Use only z-axis (gravity direction) for consistency with physics engine
    jerk_z = accel[:, 2]  # Z-axis
    
    # Apply smoothing (7-sample window as in physics engine)
    window_size = 7
    kernel = np.ones(2 * window_size + 1) / (2 * window_size + 1)
    smoothed = np.convolve(np.pad(jerk_z, window_size, mode='edge'), kernel, mode='valid')
    
    # Remove gravity (median)
    gravity = np.median(smoothed)
    gravity_removed = smoothed - gravity
    
    # Calculate jerk (3rd derivative)
    dt = 1.0 / sample_rate
    vel = np.gradient(gravity_removed, dt)
    jerk = np.gradient(vel, dt)
    
    # Apply rate normalization: divide by (sample_rate/50)^3
    rate_normalization_factor = (sample_rate / 50.0) ** 3
    normalized_jerk = jerk / rate_normalization_factor
    
    return normalized_jerk

def analyze_windows(signal, window_size=256, step_size=128):
    """Analyze kurtosis in sliding windows"""
    n_windows = (len(signal) - window_size) // step_size + 1
    kurtosis_values = []
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window_data = signal[start:end]
        
        # Calculate kurtosis using scipy (Fisher's definition, normal=0)
        if len(window_data) > 3:  # Need at least 4 points for kurtosis
            kurt = kurtosis(window_data, fisher=True)
            kurtosis_values.append(kurt)
    
    return kurtosis_values

def main():
    subjects = ['s3', 's5']
    base_dir = '/Users/timbo/ninapro_db5'
    
    for subject in subjects:
        print(f"\n=== Subject {subject.upper()} ===")
        
        # Load data
        subject_dir = os.path.join(base_dir, subject)
        emg, accel, stimulus = load_ninapro_subject(subject_dir)
        
        if accel is None:
            print(f"Could not load data for {subject}")
            continue
        
        print(f"Data shape: {accel.shape}")
        
        # Calculate rate-normalized jerk
        normalized_jerk = calculate_rate_normalized_jerk(accel)
        print(f"Normalized jerk shape: {len(normalized_jerk)} samples")
        print(f"Rate normalization factor: {(2000/50)**3:.1f}")
        
        # Analyze windows
        kurtosis_values = analyze_windows(normalized_jerk)
        print(f"Total windows analyzed: {len(kurtosis_values)}")
        
        if len(kurtosis_values) == 0:
            continue
        
        # Calculate statistics
        kurt_array = np.array(kurtosis_values)
        
        # Count windows below different thresholds
        below_1 = np.sum(kurt_array < 1.0)
        below_3 = np.sum(kurt_array < 3.0)
        below_5 = np.sum(kurt_array < 5.0)
        below_10 = np.sum(kurt_array < 10.0)
        total_windows = len(kurt_array)
        
        # Calculate percentages
        pct_1 = (below_1 / total_windows) * 100
        pct_3 = (below_3 / total_windows) * 100
        pct_5 = (below_5 / total_windows) * 100
        pct_10 = (below_10 / total_windows) * 100
        
        # Print results
        print(f"Kurtosis statistics:")
        print(f"  Mean: {np.mean(kurt_array):.3f}")
        print(f"  Std:  {np.std(kurt_array):.3f}")
        print(f"  Min:  {np.min(kurt_array):.3f}")
        print(f"  Max:  {np.max(kurt_array):.3f}")
        print(f"\nWindows with kurtosis below thresholds:")
        print(f"  < 1.0:  {below_1}/{total_windows} ({pct_1:.1f}%)")
        print(f"  < 3.0:  {below_3}/{total_windows} ({pct_3:.1f}%)")
        print(f"  < 5.0:  {below_5}/{total_windows} ({pct_5:.1f}%)")
        print(f"  < 10.0: {below_10}/{total_windows} ({pct_10:.1f}%)")

if __name__ == "__main__":
    main()
