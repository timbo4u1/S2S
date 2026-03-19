#!/usr/bin/env python3
"""
Count GOLD/SILVER/BRONZE/REJECTED windows from real datasets using S2S physics engine
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
import wfdb

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
                print(f"    Loaded {activity}: {accel_data.shape} @ {record.fs}Hz")
                # Store sample rate for certification
                subject_data[f"{activity}_hz"] = record.fs
            
        except Exception as e:
            print(f"    Warning: Could not load s{subject_id}_{activity}: {e}")
    
    return subject_data

def load_ninapro_subject(data_dir, subject_id):
    """Load one subject from NinaPro DB5"""
    import scipy.io as sio
    
    subject_dir = os.path.join(data_dir, f's{subject_id}')
    if not os.path.exists(subject_dir):
        # Extract from zip if needed
        import zipfile
        zip_path = os.path.join(data_dir, f's{subject_id}.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
    
    all_emg = []
    all_acc = []
    
    for exercise in [1, 2, 3]:
        mat_path = os.path.join(subject_dir, f'S{subject_id}_E{exercise}_A1.mat')
        if os.path.exists(mat_path):
            try:
                d = sio.loadmat(mat_path)
                emg = d['emg'].astype(np.float32)
                acc = d['acc'].astype(np.float32)
                all_emg.append(emg)
                all_acc.append(acc)
            except Exception as e:
                print(f"  Warning: {mat_path} failed: {e}")
    
    if all_emg:
        return np.vstack(all_emg), np.vstack(all_acc)
    return None, None

def certify_windows(data, sample_rate, window_size=256, segment="forearm"):
    """Certify windows using S2S physics engine"""
    engine = PhysicsEngine()
    tier_counts = defaultdict(int)
    
    n_windows = len(data) // window_size
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_data = data[start:end]
        
        if len(window_data) == window_size:
            imu_raw = {
                "accel": window_data.tolist(),
                "gyro": [[0, 0, 0]] * window_size,  # No gyro data
                "timestamps_ns": [int(1e9/sample_rate * j) for j in range(window_size)],
                "sample_rate_hz": sample_rate
            }
            
            result = engine.certify(imu_raw, segment=segment)
            tier = result['tier']
            tier_counts[tier] += 1
    
    return dict(tier_counts)

def main():
    print("=== S2S Physics Engine Certification on Real Datasets ===\n")
    
    # Initialize totals
    total_tiers = defaultdict(int)
    
    # Process PTT-PPG dataset
    print("Processing PTT-PPG dataset...")
    ptt_dir = "/Users/timbo/physionet.org/files/pulse-transit-time-ppg/1.1.0"
    ptt_tiers = defaultdict(int)
    
    for subject in range(1, 23):  # Subjects 1-22
        subject_data = load_ptt_ppg_subject(ptt_dir, subject)
        if subject_data:
            print(f"  Subject {subject}: {list(subject_data.keys())}")
            for activity, data in subject_data.items():
                if activity.endswith('_hz'):
                    continue  # Skip sample rate storage
                if len(data) > 256:
                    sample_rate = subject_data[f"{activity}_hz"]
                    tiers = certify_windows(data, sample_rate=sample_rate, window_size=256, segment="walking")
                    for tier, count in tiers.items():
                        ptt_tiers[tier] += count
                        total_tiers[tier] += count
    
    print(f"\nPTT-PPG Results:")
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'REJECTED']:
        print(f"  {tier}: {ptt_tiers[tier]}")
    ptt_total = sum(ptt_tiers.values())
    print(f"  Total: {ptt_total}")
    
    # Process NinaPro DB5
    print("\nProcessing NinaPro DB5...")
    ninapro_dir = "/Users/timbo/ninapro_db5"
    ninapro_tiers = defaultdict(int)
    
    for subject in range(1, 11):  # Subjects 1-10
        emg, acc = load_ninapro_subject(ninapro_dir, subject)
        if acc is not None and len(acc) > 256:
            print(f"  Subject {subject}: {len(acc)} samples")
            tiers = certify_windows(acc, sample_rate=2000, window_size=400)  # 200ms windows at 2kHz
            for tier, count in tiers.items():
                ninapro_tiers[tier] += count
                total_tiers[tier] += count
    
    print(f"\nNinaPro DB5 Results:")
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'REJECTED']:
        print(f"  {tier}: {ninapro_tiers[tier]}")
    ninapro_total = sum(ninapro_tiers.values())
    print(f"  Total: {ninapro_total}")
    
    # Overall summary
    print(f"\n=== OVERALL SUMMARY ===")
    overall_total = sum(total_tiers.values())
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'REJECTED']:
        count = total_tiers[tier]
        pct = count / overall_total * 100 if overall_total > 0 else 0
        print(f"{tier}: {count} ({pct:.1f}%)")
    
    print(f"Total windows: {overall_total}")
    print(f"Certain cases (GOLD + REJECTED): {total_tiers['GOLD'] + total_tiers['REJECTED']} ({(total_tiers['GOLD'] + total_tiers['REJECTED'])/overall_total*100:.1f}%)")
    print(f"Ambiguous cases (SILVER + BRONZE): {total_tiers['SILVER'] + total_tiers['BRONZE']} ({(total_tiers['SILVER'] + total_tiers['BRONZE'])/overall_total*100:.1f}%)")

if __name__ == "__main__":
    main()
