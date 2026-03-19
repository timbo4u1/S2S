#!/usr/bin/env python3
"""
Test physics filtering on NinaPro DB5 gesture recognition
Compare high-quality subjects vs all subjects using simple classifier
"""

import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import scipy.io

sys.path.insert(0, os.path.dirname(__file__))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

def load_ninapro_subject(subject_dir):
    """Load NinaPro DB5 subject data with gesture labels"""
    print(f"Loading {subject_dir}...")
    
    # Find main data file
    import glob
    mat_files = glob.glob(os.path.join(subject_dir, "*.mat"))
    data = None
    
    for mat_file in mat_files:
        try:
            mat_data = scipy.io.loadmat(mat_file)
            for key in ['stimulus', 'emg', 'acc', 'restimulus']:
                if key in mat_data:
                    data = mat_data
                    break
            if data is not None:
                break
        except Exception as e:
            print(f"  Error loading {mat_file}: {e}")
            continue
    
    if data is None:
        return None
    
    # Extract EMG and accelerometer data
    emg_data = None
    acc_data = None
    stimulus = None
    
    if 'emg' in data:
        emg_data = data['emg']
    elif 'EMG' in data:
        emg_data = data['EMG']
    
    if 'acc' in data:
        acc_data = data['acc']
    elif 'ACC' in data:
        acc_data = data['ACC']
    elif 'accxyz' in data:
        acc_data = data['accxyz']
    
    if 'stimulus' in data:
        stimulus = data['stimulus'].flatten()
    elif 'restimulus' in data:
        stimulus = data['restimulus'].flatten()
    
    if emg_data is None or acc_data is None or stimulus is None:
        print(f"  Missing required data fields")
        return None
    
    print(f"  EMG: {emg_data.shape}, ACC: {acc_data.shape}, Stimulus: {stimulus.shape}")
    
    return {
        'emg': emg_data,
        'acc': acc_data,
        'stimulus': stimulus,
        'subject_dir': subject_dir
    }

def create_windows_with_labels(data, window_size=256, step_size=128):
    """Create windows with gesture labels"""
    acc = data['acc']
    emg = data['emg']
    stimulus = data['stimulus']
    
    n_samples = acc.shape[0]
    windows = []
    labels = []
    
    for i in range(0, n_samples - window_size + 1, step_size):
        window_acc = acc[i:i+window_size]
        window_emg = emg[i:i+window_size]
        window_stimulus = stimulus[i:i+window_size]
        
        # Determine dominant gesture in window
        if len(window_stimulus) == 0:
            continue
            
        # Count non-zero stimulus (active gestures)
        active_count = np.sum(window_stimulus > 0)
        # Only keep windows with at least 50% active gesture data
        if active_count < window_size * 0.5:
            continue
        
        # Use most common gesture as window label
        unique, counts = np.unique(window_stimulus[window_stimulus > 0], return_counts=True)
        if len(unique) > 0:
            dominant_idx = np.argmax(counts)
            gesture_label = unique[dominant_idx]
        else:
            continue
        
        # Combine EMG and ACC features
        window_features = np.concatenate([
            window_emg.flatten(),  # 16*256 = 4096 EMG features
            window_acc.flatten()   # 3*256 = 768 ACC features
        ])
        
        windows.append(window_features)
        labels.append(gesture_label)
    
    return np.array(windows), np.array(labels)

def certify_and_filter_windows(data, sample_rate=2000, segment='forearm', 
                           quality_threshold=0.1):  # 10% rejection threshold
    """Certify windows and filter by quality"""
    engine = PhysicsEngine()
    
    windows, labels = create_windows_with_labels(data)
    print(f"Created {len(windows)} windows with {len(np.unique(labels))} unique gestures")
    
    if len(windows) == 0:
        return [], [], []
    
    # Certify each window
    quality_scores = []
    for i, (window, label) in enumerate(zip(windows, labels)):
        # Extract just the ACC part for physics certification
        window_acc = window[4096:].reshape(256, 3)  # Last 768 values are ACC
        
        # Create timestamps (nanoseconds)
        dt_ns = int(1e9 / sample_rate)
        timestamps_ns = [int(dt_ns * j) for j in range(len(window_acc))]
        
        # Create IMU data structure
        imu_raw = {
            "accel": window_acc.tolist(),
            "gyro": [[0.0, 0.0, 0.0]] * len(window_acc),  # No gyro data
            "timestamps_ns": timestamps_ns,
            "sample_rate_hz": sample_rate
        }
        
        # Certify with physics engine
        try:
            result = engine.certify(imu_raw, segment=segment)
            # Use physics score as quality metric
            quality_scores.append(result['physical_law_score'] / 100.0)  # Normalize to 0-1
            
        except Exception as e:
            print(f"  Error certifying window {i}: {e}")
            quality_scores.append(0.0)  # Low quality for errors
    
    quality_scores = np.array(quality_scores)
    
    # Filter by quality threshold
    high_quality_mask = quality_scores >= quality_threshold
    low_quality_mask = quality_scores < quality_threshold
    
    print(f"High-quality windows: {np.sum(high_quality_mask)}/{len(windows)} ({np.mean(high_quality_mask)*100:.1f}%)")
    print(f"Low-quality windows:  {np.sum(low_quality_mask)}/{len(windows)} ({np.mean(low_quality_mask)*100:.1f}%)")
    
    return (windows[high_quality_mask], labels[high_quality_mask], 
            windows[low_quality_mask], labels[low_quality_mask],
            quality_scores)

def train_and_evaluate(X, y, subjects, model_name, n_splits=3):
    """Train Random Forest with subject-wise cross-validation"""
    print(f"\n=== Training {model_name} ===")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Use fewer splits if we have fewer subjects
    n_subjects = len(np.unique(subjects))
    n_splits = min(n_splits, n_subjects)
    
    group_kfold = GroupKFold(n_splits=n_splits)
    accuracies = []
    all_predictions = []
    all_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, subjects)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        # Evaluate
        val_acc = rf.score(X_val_scaled, y_val)
        accuracies.append(val_acc)
        
        # Predictions for analysis
        y_pred = rf.predict(X_val_scaled)
        all_predictions.extend(y_pred)
        all_labels.extend(y_val)
        
        print(f"  Val accuracy: {val_acc:.4f}")
    
    # Overall results
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{model_name} Results:")
    print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Best fold: {max(accuracies):.4f}")
    print(f"  Worst fold: {min(accuracies):.4f}")
    
    # Classification report
    unique_labels = np.unique(np.concatenate([all_labels, all_predictions]))
    target_names = [f"Gesture {i}" for i in unique_labels]
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                          target_names=target_names, digits=4, zero_division=0))
    
    return mean_acc, accuracies

def main():
    """Compare high-quality vs all subjects"""
    # Load results to identify high-quality subjects
    with open('experiments/results_ninapro_physics.json', 'r') as f:
        results = json.load(f)
    
    # Identify high-quality subjects (≤10% rejection rate)
    high_quality_subjects = []
    low_quality_subjects = []
    
    for subject_id, rate in results['summary']['subject_rejection_rates'].items():
        if rate <= 0.10:  # ≤10% rejection
            high_quality_subjects.append(subject_id)
        else:
            low_quality_subjects.append(subject_id)
    
    print(f"High-quality subjects (≤10% rejection): {high_quality_subjects}")
    print(f"Low-quality subjects (>10% rejection): {low_quality_subjects}")
    
    # Load and process high-quality subjects
    print(f"\n{'='*60}")
    print("PROCESSING HIGH-QUALITY SUBJECTS")
    print(f"{'='*60}")
    
    all_X_hq, all_y_hq, all_subjects_hq = [], [], []
    for subject_id in high_quality_subjects:
        subject_dir = f"/Users/timbo/ninapro_db5/{subject_id}"
        data = load_ninapro_subject(subject_dir)
        if data is None:
            continue
        
        X_hq, y_hq, _, _, _ = certify_and_filter_windows(data)
        if len(X_hq) == 0:
            continue
        
        # Create subject IDs array
        subject_ids = np.full(len(X_hq), subject_id)
        all_X_hq.append(X_hq)
        all_y_hq.append(y_hq)
        all_subjects_hq.append(subject_ids)
    
    if len(all_X_hq) == 0:
        print("No high-quality data found")
        return
    
    # Combine all high-quality data
    X_hq_combined = np.concatenate(all_X_hq)
    y_hq_combined = np.concatenate(all_y_hq)
    subjects_hq_combined = np.concatenate(all_subjects_hq)
    
    print(f"\nHigh-quality dataset: {len(X_hq_combined)} samples, {X_hq_combined.shape[1]} features")
    
    # Train on high-quality only
    hq_acc, hq_folds = train_and_evaluate(X_hq_combined, y_hq_combined, 
                                         subjects_hq_combined, 
                                         "High-Quality Only")
    
    # Now process all subjects for comparison
    print(f"\n{'='*60}")
    print("PROCESSING ALL SUBJECTS")
    print(f"{'='*60}")
    
    all_X_all, all_y_all, all_subjects_all = [], [], []
    for subject_id in ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']:
        subject_dir = f"/Users/timbo/ninapro_db5/{subject_id}"
        data = load_ninapro_subject(subject_dir)
        if data is None:
            continue
        
        X_all, y_all, _, _, _ = certify_and_filter_windows(data, quality_threshold=0.0)  # No filtering
        if len(X_all) == 0:
            continue
        
        # Create subject IDs array
        subject_ids = np.full(len(X_all), subject_id)
        all_X_all.append(X_all)
        all_y_all.append(y_all)
        all_subjects_all.append(subject_ids)
    
    if len(all_X_all) == 0:
        print("No data found")
        return
    
    # Combine all data
    X_all_combined = np.concatenate(all_X_all)
    y_all_combined = np.concatenate(all_y_all)
    subjects_all_combined = np.concatenate(all_subjects_all)
    
    print(f"\nAll subjects dataset: {len(X_all_combined)} samples, {X_all_combined.shape[1]} features")
    
    # Train on all subjects
    all_acc, all_folds = train_and_evaluate(X_all_combined, y_all_combined,
                                         subjects_all_combined,
                                         "All Subjects")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON: PHYSICS FILTERING EFFECT")
    print(f"{'='*60}")
    
    improvement = hq_acc - all_acc
    print(f"High-quality only:   {hq_acc:.4f} ± {np.std(hq_folds):.4f}")
    print(f"All subjects:       {all_acc:.4f} ± {np.std(all_folds):.4f}")
    print(f"Improvement:         {improvement:+.4f} ({improvement/all_acc*100:+.1f}%)")
    
    if improvement > 0.01:  # 1% improvement threshold
        print(f"\n🎉 SUCCESS: Physics filtering improves gesture recognition!")
        print(f"   Quality filtering adds {improvement*100:.1f}% accuracy")
    elif improvement > -0.01:
        print(f"\n❌ Physics filtering hurts performance")
    else:
        print(f"\n~ Neutral effect: No significant difference")
    
    print(f"\nKey insight: Physics-based quality control {'helps' if improvement > 0 else 'hurts'} gesture recognition")
    
    # Save comparison results
    comparison_results = {
        'experiment': 'ninapro_physics_filtering_test',
        'timestamp': '2026-03-09T19:00:00',
        'dataset': 'NinaPro DB5',
        'high_quality_subjects': high_quality_subjects,
        'low_quality_subjects': low_quality_subjects,
        'high_quality_accuracy': float(hq_acc),
        'high_quality_std': float(np.std(hq_folds)),
        'all_subjects_accuracy': float(all_acc),
        'all_subjects_std': float(np.std(all_folds)),
        'improvement': float(improvement),
        'improvement_percent': float(improvement / all_acc * 100),
        'high_quality_samples': len(X_hq_combined),
        'all_subjects_samples': len(X_all_combined),
        'features': X_hq_combined.shape[1]
    }
    
    with open('experiments/results_ninapro_filtering_test.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nResults saved to experiments/results_ninapro_filtering_test.json")

if __name__ == "__main__":
    main()
