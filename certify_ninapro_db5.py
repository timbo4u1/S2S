#!/usr/bin/env python3
"""
Certify NinaPro DB5 dataset with fixed physics engine
10 subjects, 2000Hz, 16-channel EMG + 3-axis accelerometer
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict, Counter
import glob

sys.path.insert(0, os.path.dirname(__file__))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

def load_ninapro_subject(subject_dir):
    """Load NinaPro DB5 subject data"""
    print(f"Loading {subject_dir}...")
    
    # Find the relevant files
    mat_files = glob.glob(os.path.join(subject_dir, "*.mat"))
    if not mat_files:
        print(f"  No .mat files found in {subject_dir}")
        return None
    
    # Try to load the main data file
    import scipy.io
    data = None
    
    for mat_file in mat_files:
        try:
            mat_data = scipy.io.loadmat(mat_file)
            # Look for common NinaPro variable names
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
        print(f"  No valid NinaPro data found in {subject_dir}")
        return None
    
    # Extract EMG and accelerometer data
    emg_data = None
    acc_data = None
    stimulus = None
    
    # Common NinaPro DB5 variable patterns
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
    
    if emg_data is None or acc_data is None:
        print(f"  Missing EMG or accelerometer data")
        return None
    
    print(f"  EMG shape: {emg_data.shape}")
    print(f"  ACC shape: {acc_data.shape}")
    if stimulus is not None:
        print(f"  Stimulus shape: {stimulus.shape}")
    
    return {
        'emg': emg_data,
        'acc': acc_data,
        'stimulus': stimulus,
        'subject_dir': subject_dir
    }

def create_windows(data, window_size=256, step_size=128):
    """Create windows from continuous data"""
    acc = data['acc']
    stimulus = data.get('stimulus')
    
    n_samples = acc.shape[0]
    windows = []
    
    for i in range(0, n_samples - window_size + 1, step_size):
        window_acc = acc[i:i+window_size]
        
        # Check if we have stimulus data and if window is mostly one gesture
        if stimulus is not None:
            window_stimulus = stimulus[i:i+window_size]
            if len(window_stimulus) == 0:
                continue
            
            # Count non-zero stimulus (active gestures)
            active_count = np.sum(window_stimulus > 0)
            # Only keep windows with at least 50% active gesture data
            if active_count < window_size * 0.5:
                continue
        
        windows.append(window_acc)
    
    return windows

def certify_windows(windows, sample_rate=2000, segment='forearm'):
    """Certify windows with physics engine"""
    engine = PhysicsEngine()
    
    tier_counts = Counter()
    law_failures = defaultdict(int)
    total_windows = len(windows)
    
    for i, window in enumerate(windows):
        # Create timestamps (nanoseconds)
        dt_ns = int(1e9 / sample_rate)
        timestamps_ns = [int(dt_ns * j) for j in range(len(window))]
        
        # Create IMU data structure
        imu_raw = {
            "accel": window.tolist(),
            "gyro": [[0.0, 0.0, 0.0]] * len(window),  # No gyro data in NinaPro DB5
            "timestamps_ns": timestamps_ns,
            "sample_rate_hz": sample_rate
        }
        
        # Certify with physics engine
        try:
            result = engine.certify(imu_raw, segment=segment)
            tier = result['tier']
            tier_counts[tier] += 1
            
            # Track which laws failed
            for law in result['laws_failed']:
                law_failures[law] += 1
                
        except Exception as e:
            print(f"  Error certifying window {i}: {e}")
            tier_counts['ERROR'] += 1
    
    return tier_counts, law_failures, total_windows

def process_subject(subject_dir, subject_id):
    """Process a single NinaPro subject"""
    print(f"\n=== Processing Subject {subject_id} ===")
    
    # Load data
    data = load_ninapro_subject(subject_dir)
    if data is None:
        return None
    
    # Create windows
    windows = create_windows(data)
    print(f"Created {len(windows)} windows")
    
    if len(windows) == 0:
        print("No windows created")
        return None
    
    # Certify windows
    tier_counts, law_failures, total_windows = certify_windows(windows)
    
    # Calculate rejection rate
    rejected_count = tier_counts.get('REJECTED', 0) + tier_counts.get('ERROR', 0)
    rejection_rate = rejected_count / total_windows if total_windows > 0 else 0
    
    # Prepare results
    results = {
        'subject_id': subject_id,
        'subject_dir': subject_dir,
        'total_windows': total_windows,
        'tier_counts': dict(tier_counts),
        'rejection_rate': rejection_rate,
        'law_failures': dict(law_failures),
        'top_failing_laws': sorted(law_failures.items(), key=lambda x: x[1], reverse=True)[:5]
    }
    
    # Print summary
    print(f"Total windows: {total_windows}")
    print(f"Tier distribution:")
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'REJECTED', 'ERROR']:
        count = tier_counts.get(tier, 0)
        percentage = count / total_windows * 100 if total_windows > 0 else 0
        print(f"  {tier:<10}: {count:5d} ({percentage:5.1f}%)")
    print(f"Rejection rate: {rejection_rate:.2%}")
    print(f"Top failing laws:")
    for law, count in results['top_failing_laws']:
        print(f"  {law}: {count}")
    
    return results

def main():
    """Process all NinaPro DB5 subjects"""
    ninapro_dir = os.path.expanduser("~/ninapro_db5")
    
    if not os.path.exists(ninapro_dir):
        print(f"NinaPro DB5 directory not found: {ninapro_dir}")
        return
    
    # Extract all zip files first
    print("Extracting subject zip files...")
    for item in os.listdir(ninapro_dir):
        if item.endswith('.zip'):
            zip_path = os.path.join(ninapro_dir, item)
            extract_dir = os.path.join(ninapro_dir, item.replace('.zip', ''))
            if not os.path.exists(extract_dir):
                print(f"Extracting {item}...")
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(ninapro_dir)
    
    # Find all subject directories
    subject_dirs = []
    for item in os.listdir(ninapro_dir):
        subject_path = os.path.join(ninapro_dir, item)
        if os.path.isdir(subject_path) and (item.startswith('S') or item.startswith('s')):
            subject_dirs.append((subject_path, item))
    
    subject_dirs.sort()
    print(f"Found {len(subject_dirs)} subjects: {[s[1] for s in subject_dirs]}")
    
    if len(subject_dirs) == 0:
        print("No subject directories found")
        return
    
    # Process all subjects
    all_results = {}
    summary_stats = {
        'total_subjects': len(subject_dirs),
        'total_windows': 0,
        'overall_tier_counts': Counter(),
        'overall_law_failures': defaultdict(int),
        'subject_rejection_rates': {}
    }
    
    for subject_dir, subject_id in subject_dirs:
        try:
            results = process_subject(subject_dir, subject_id)
            if results is None:
                continue
            
            all_results[subject_id] = results
            
            # Update summary statistics
            summary_stats['total_windows'] += results['total_windows']
            summary_stats['subject_rejection_rates'][subject_id] = results['rejection_rate']
            
            for tier, count in results['tier_counts'].items():
                summary_stats['overall_tier_counts'][tier] += count
            
            for law, count in results['law_failures'].items():
                summary_stats['overall_law_failures'][law] += count
                
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY - NinaPro DB5")
    print(f"{'='*60}")
    print(f"Subjects processed: {summary_stats['total_subjects']}")
    print(f"Total windows: {summary_stats['total_windows']}")
    print(f"\nOverall tier distribution:")
    total_windows = summary_stats['total_windows']
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'REJECTED', 'ERROR']:
        count = summary_stats['overall_tier_counts'].get(tier, 0)
        percentage = count / total_windows * 100 if total_windows > 0 else 0
        print(f"  {tier:<10}: {count:6d} ({percentage:5.1f}%)")
    
    if total_windows > 0:
        overall_rejection = (summary_stats['overall_tier_counts'].get('REJECTED', 0) + 
                            summary_stats['overall_tier_counts'].get('ERROR', 0)) / total_windows
        print(f"\nOverall rejection rate: {overall_rejection:.2%}")
    else:
        print(f"\nNo windows processed")
    
    print(f"\nTop failing laws across all subjects:")
    top_laws = sorted(summary_stats['overall_law_failures'].items(), 
                      key=lambda x: x[1], reverse=True)[:10]
    for law, count in top_laws:
        print(f"  {law}: {count}")
    
    print(f"\nRejection rates per subject:")
    for subject_id, rate in sorted(summary_stats['subject_rejection_rates'].items()):
        print(f"  {subject_id}: {rate:.2%}")
    
    # Save results
    output_file = 'experiments/results_ninapro_physics.json'
    
    # Convert defaultdict to dict for JSON serialization
    summary_stats['overall_law_failures'] = dict(summary_stats['overall_law_failures'])
    summary_stats['overall_tier_counts'] = dict(summary_stats['overall_tier_counts'])
    
    results_data = {
        'experiment': 'ninapro_db5_physics_certification',
        'timestamp': '2026-03-09T12:00:00',
        'dataset': 'NinaPro DB5',
        'description': '10 subjects, 2000Hz, 16-channel EMG + 3-axis accelerometer',
        'segment': 'forearm',
        'window_size': 256,
        'step_size': 128,
        'sample_rate': 2000,
        'summary': summary_stats,
        'subjects': all_results
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
