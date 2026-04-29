#!/usr/bin/env python3
"""
Certify the S2S dataset with the fixed physics engine
Creates JSON files with physics certification results for Level 1 experiment
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

def load_s2s_dataset(data_dir):
    """Load S2S dataset and extract windows with physics certification"""
    print(f"Loading S2S dataset from {data_dir}...")
    
    all_windows = []
    
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith('.json'): continue
            fpath = os.path.join(root, fname)
            
            try:
                record = json.load(open(fpath))
                
                # Extract basic info
                domain = record.get('domain', '')
                if not domain:
                    continue
                    
                # Extract IMU data (assume it's in the record)
                accel = record.get('accel', [])
                if not accel or len(accel) < 20:
                    continue
                
                # Create window from IMU data
                n = len(accel)
                for i in range(0, n, 256):  # Non-overlapping windows
                    if i + 256 > n:
                        break
                        
                    window_accel = accel[i:i+256]
                    if len(window_accel) < 256:
                        continue
                    
                    # Create timestamps (assume 100Hz)
                    timestamps_ns = [int(1e9/100 * j) for j in range(256)]
                    
                    # Certify with physics engine
                    imu_raw = {
                        "accel": window_accel,
                        "gyro": [[0, 0, 0]] * 256,  # No gyro data
                        "timestamps_ns": timestamps_ns,
                        "sample_rate_hz": 100
                    }
                    
                    result = engine.certify(imu_raw, segment="forearm")
                    
                    # Create S2S format record
                    certified_record = {
                        "domain": domain,
                        "accel": window_accel,
                        "physics_score": result['physical_law_score'],
                        "physics_tier": result['tier'],
                        "physics_laws_passed": result['laws_passed'],
                        "jerk_p95_ms3": result['law_details'].get('jerk_bounds', {}).get('p95_jerk_ms3', 0),
                        "imu_coupling_r": result['law_details'].get('imu_internal_consistency', {}).get('pearson_r_var_coupling', 0),
                        "physics_laws": result['laws_checked']
                    }
                    
                    all_windows.append(certified_record)
                    
            except Exception as e:
                print(f"  Error processing {fname}: {e}")
                continue
    
    print(f"Created {len(all_windows)} certified windows")
    return all_windows

def save_s2s_format(windows, output_path):
    """Save windows in S2S JSON format"""
    print(f"Saving S2S dataset to {output_path}...")
    
    # Group by domain for analysis
    domain_windows = defaultdict(list)
    for window in windows:
        domain_windows[window['domain']].append(window)
    
    # Save each domain to separate files for easier processing
    os.makedirs(output_path, exist_ok=True)
    
    for domain, windows_list in domain_windows.items():
        if not windows_list:
            continue
            
        filename = f"s2s_{domain.lower()}_certified.json"
        filepath = os.path.join(output_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(windows_list, f, indent=2)
        
        print(f"  {domain}: {len(windows_list)} windows → {filename}")

def main():
    data_dir = "data/s2s_dataset/"
    output_dir = "data/s2s_certified/"
    
    # Load and certify dataset
    windows = load_s2s_dataset(data_dir)
    
    # Save in S2S format
    save_s2s_format(windows, output_dir)
    
    print(f"\n✅ S2S dataset certified and saved!")
    print(f"Ready for Level 1 experiment")

if __name__ == "__main__":
    main()
