#!/usr/bin/env python3
"""
Extract training data with activity labels from PTT-PPG dataset
Combines raw IMU data with physics features for activity classification
Labels: walk=0, sit=1, run=2
"""

import os
import sys
import numpy as np
import wfdb
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

def extract_physics_features(result):
    """Extract physics features from certification result"""
    features = []
    
    # Basic scores
    features.append(result['physical_law_score'])
    features.append(1 if result['tier'] == 'GOLD' else 0)
    features.append(1 if result['tier'] == 'SILVER' else 0)
    features.append(1 if result['tier'] == 'BRONZE' else 0)
    features.append(1 if result['tier'] == 'REJECTED' else 0)
    
    # Law-specific features
    law_details = result['law_details']
    
    # Resonance frequency
    if 'resonance_frequency' in law_details:
        rf = law_details['resonance_frequency']
        features.append(rf.get('measured_peak_hz', 0))
        features.append(rf.get('measured_peak_energy', 0))
        features.append(1 if rf.get('in_expected_range', False) else 0)
        features.append(rf.get('confidence', 0))
    else:
        features.extend([0, 0, 0, 0])  # Missing law
    
    # Rigid body kinematics
    if 'rigid_body_kinematics' in law_details:
        rb = law_details['rigid_body_kinematics']
        features.append(rb.get('rms_measured_ms2', 0))
        features.append(rb.get('rms_predicted_ms2', 0))
        features.append(rb.get('gyro_accel_scale_ratio', 0))
        features.append(rb.get('confidence', 0))
    else:
        features.extend([0, 0, 0, 0])  # Missing law
    
    # Jerk bounds (if not skipped)
    if 'jerk_bounds' in law_details:
        jb = law_details['jerk_bounds']
        if 'skip' in jb:
            features.extend([0, 0, 0, 0])  # Skipped
        else:
            features.append(jb.get('peak_jerk_ms3', 0))
            features.append(jb.get('rms_jerk_ms3', 0))
            features.append(jb.get('p95_jerk_ms3', 0))
            features.append(jb.get('confidence', 0))
    else:
        features.extend([0, 0, 0, 0])  # Missing law
    
    # IMU consistency
    if 'imu_internal_consistency' in law_details:
        ic = law_details['imu_internal_consistency']
        features.append(ic.get('pearson_r_var_coupling', 0))
        features.append(ic.get('confidence', 0))
    else:
        features.extend([0, 0])  # Missing law
    
    return np.array(features, dtype=np.float32)

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

def extract_activity_training_data(data_dir, output_dir="data"):
    """Extract training data with activity labels and physics features"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize
    engine = PhysicsEngine()
    
    # Activity mapping: walk=0, sit=1, run=2
    activity_map = {'walk': 0, 'sit': 1, 'run': 2}
    
    # Storage
    all_windows = []
    all_physics_features = []
    all_labels = []
    all_subjects = []
    
    print("Extracting activity-labeled training data from PTT-PPG...")
    
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
            activity_windows = 0
            
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
                
                # Extract physics features
                physics_features = extract_physics_features(result)
                
                # Store data
                all_windows.append(window_data)
                all_physics_features.append(physics_features)
                all_labels.append(activity_map[activity])
                all_subjects.append(subject)
                
                activity_windows += 1
            
            print(f"  {activity}: {activity_windows} windows")
    
    # Convert to numpy arrays
    X_raw = np.array(all_windows)  # Shape: (n_samples, 256, 3)
    X_physics = np.array(all_physics_features)  # Shape: (n_samples, n_features)
    y = np.array(all_labels)  # Shape: (n_samples,)
    subjects = np.array(all_subjects)
    
    print(f"\n=== EXTRACTION RESULTS ===")
    print(f"Total windows: {len(X_raw)}")
    print(f"Raw IMU shape: {X_raw.shape}")
    print(f"Physics features shape: {X_physics.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Physics features per window: {X_physics.shape[1]}")
    
    # Activity distribution
    for activity, label in activity_map.items():
        count = np.sum(y == label)
        print(f"{activity}: {count} windows ({count/len(y)*100:.1f}%)")
    
    # Save datasets
    np.save(os.path.join(output_dir, 'training_X_raw.npy'), X_raw)
    np.save(os.path.join(output_dir, 'training_X_physics.npy'), X_physics)
    np.save(os.path.join(output_dir, 'training_y.npy'), y)
    np.save(os.path.join(output_dir, 'training_subjects.npy'), subjects)
    
    # Save metadata
    metadata = {
        'n_samples': len(X_raw),
        'raw_shape': X_raw.shape,
        'physics_features': X_physics.shape[1],
        'activities': ['walk', 'sit', 'run'],
        'activity_labels': activity_map,
        'sample_rate_hz': 500,
        'window_size_samples': 256,
        'physics_feature_names': [
            'physical_law_score', 'is_gold', 'is_silver', 'is_bronze', 'is_rejected',
            'resonance_peak_hz', 'resonance_peak_energy', 'resonance_in_range', 'resonance_confidence',
            'rigid_rms_measured', 'rigid_rms_predicted', 'rigid_gyro_accel_ratio', 'rigid_confidence',
            'jerk_peak', 'jerk_rms', 'jerk_p95', 'jerk_confidence',
            'imu_r_coupling', 'imu_confidence'
        ]
    }
    
    np.save(os.path.join(output_dir, 'training_metadata.npy'), metadata)
    
    print(f"\n✅ Training data saved to {output_dir}/:")
    print(f"  training_X_raw.npy: {X_raw.shape} (raw IMU windows)")
    print(f"  training_X_physics.npy: {X_physics.shape} (physics features)")
    print(f"  training_y.npy: {y.shape} (activity labels)")
    print(f"  training_subjects.npy: {subjects.shape} (subject IDs)")
    print(f"  training_metadata.npy: metadata")
    
    return X_raw, X_physics, y, subjects, metadata

def main():
    data_dir = "/Users/timbo/physionet.org/files/pulse-transit-time-ppg/1.1.0"
    X_raw, X_physics, y, subjects, metadata = extract_activity_training_data(data_dir)
    
    print(f"\n🎯 Next steps:")
    print(f"1. Train CNN on raw IMU: X_raw → y")
    print(f"2. Train classifier on physics features: X_physics → y") 
    print(f"3. Compare performance to prove physics features carry signal")
    print(f"4. Build hybrid model combining both")

if __name__ == "__main__":
    main()
