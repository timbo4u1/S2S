#!/usr/bin/env python3
"""
Extract SILVER and BRONZE windows from PTT-PPG dataset for CNN training
Saves windows to training_X.npy and labels to training_y.npy (1=SILVER, 0=BRONZE)
"""

import os
import sys
import numpy as np
import wfdb
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

def load_ptt_ppg_subject(data_dir, subject_id):
    """Load one subject from PTT-PPG dataset using wfdb"""
    subject_data = {}
    
    for activity in ['walk', 'sit', 'run']:
        try:
            # Use wfdb to read the record
            record = wfdb.rdrecord(os.path.join(data_dir, f's{subject_id}_{activity}'))
            
            # Find accel channels
            accel_indices = []
            for i, sig_name in enumerate(record.sig_name):
                if sig_name in ['a_x', 'a_y', 'a_z']:
                    accel_indices.append(i)
            
            if len(accel_indices) == 3:
                # Extract accel data and convert to m/s² (record is already in proper units)
                accel_data = record.p_signal[:, accel_indices].astype(np.float32)
                subject_data[activity] = accel_data
                subject_data[f"{activity}_hz"] = record.fs
                print(f"    Loaded {activity}: {accel_data.shape} @ {record.fs}Hz")
            
        except Exception as e:
            print(f"    Warning: Could not load s{subject_id}_{activity}: {e}")
    
    return subject_data

def extract_ambiguous_windows(data_dir, output_dir="data"):
    """Extract SILVER and BRONZE windows for CNN training"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize
    engine = PhysicsEngine()
    silver_windows = []
    bronze_windows = []
    silver_scores = []
    bronze_scores = []
    
    print("Extracting SILVER and BRONZE windows from PTT-PPG...")
    
    for subject in range(1, 23):  # Subjects 1-22
        print(f"\nSubject {subject}:")
        subject_data = load_ptt_ppg_subject(data_dir, subject)
        
        if not subject_data:
            continue
            
        for activity in ['walk', 'sit', 'run']:
            if activity not in subject_data:
                continue
                
            data = subject_data[activity]
            sample_rate = subject_data[f"{activity}_hz"]
            
            if len(data) < 256:
                continue
            
            # Process windows
            n_windows = len(data) // 256
            for i in range(n_windows):
                start = i * 256
                end = start + 256
                window_data = data[start:end]
                
                # Certify window
                imu_raw = {
                    "accel": window_data.tolist(),
                    "gyro": [[0, 0, 0]] * 256,
                    "timestamps_ns": [int(1e9/sample_rate * j) for j in range(256)],
                    "sample_rate_hz": sample_rate
                }
                
                result = engine.certify(imu_raw, segment="walking")
                tier = result['tier']
                score = result['physical_law_score']
                
                # Collect SILVER and BRONZE windows
                if tier == 'SILVER':
                    silver_windows.append(window_data)
                    silver_scores.append(score)
                elif tier == 'BRONZE':
                    bronze_windows.append(window_data)
                    bronze_scores.append(score)
            
            print(f"  {activity}: {len(silver_windows)} SILVER, {len(bronze_windows)} BRONZE")
    
    print(f"\n=== EXTRACTION RESULTS ===")
    print(f"SILVER windows: {len(silver_windows)}")
    print(f"BRONZE windows: {len(bronze_windows)}")
    
    if silver_scores:
        print(f"SILVER score range: [{min(silver_scores)}, {max(silver_scores)}]")
    if bronze_scores:
        print(f"BRONZE score range: [{min(bronze_scores)}, {max(bronze_scores)}]")
    
    # Check if we have any SILVER windows
    if len(silver_windows) == 0:
        print("\n⚠️  NO SILVER WINDOWS FOUND!")
        print("All windows are BRONZE. This means the scoring thresholds need adjustment.")
        print("Current thresholds from physics engine:")
        print("  SILVER: score >= 55")
        print("  BRONZE: score >= 35")
        print(f"  Max BRONZE score: {max(bronze_scores) if bronze_scores else 0}")
        print("\nRecommendation: Lower SILVER threshold to capture high-quality BRONZE windows")
        return False
    
    # Convert to numpy arrays
    X_silver = np.array(silver_windows)
    X_bronze = np.array(bronze_windows)
    
    # Create labels (1=SILVER, 0=BRONZE)
    y_silver = np.ones(len(X_silver))
    y_bronze = np.zeros(len(X_bronze))
    
    # Combine datasets
    X = np.concatenate([X_silver, X_bronze], axis=0)
    y = np.concatenate([y_silver, y_bronze], axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Save to files
    np.save(os.path.join(output_dir, 'training_X.npy'), X)
    np.save(os.path.join(output_dir, 'training_y.npy'), y)
    
    print(f"\n✅ Training data saved:")
    print(f"  {output_dir}/training_X.npy: {X.shape} (windows)")
    print(f"  {output_dir}/training_y.npy: {y.shape} (labels)")
    print(f"  SILVER samples: {len(y_silver)} ({len(y_silver)/len(y)*100:.1f}%)")
    print(f"  BRONZE samples: {len(y_bronze)} ({len(y_bronze)/len(y)*100:.1f}%)")
    
    return True

def main():
    data_dir = "/Users/timbo/physionet.org/files/pulse-transit-time-ppg/1.1.0"
    success = extract_ambiguous_windows(data_dir)
    
    if not success:
        print("\n💡 SUGGESTION: Adjust SILVER threshold in physics engine to get training data")
        print("Or use top X% of BRONZE windows as 'high-quality' for training")

if __name__ == "__main__":
    main()
