#!/usr/bin/env python3
"""
Debug what PhysicsEngine actually returns for real PTT-PPG walking data
"""

import os
import sys
import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(__file__))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

def debug_ptt_ppg():
    print("=== Debugging PTT-PPG Walking Data ===")
    
    # Load subject 1 walk
    ptt_dir = "/Users/timbo/physionet.org/files/pulse-transit-time-ppg/1.1.0"
    record = wfdb.rdrecord(os.path.join(ptt_dir, 's1_walk'))
    
    # Find accel channels
    accel_indices = []
    for i, sig_name in enumerate(record.sig_name):
        if sig_name in ['a_x', 'a_y', 'a_z']:
            accel_indices.append(i)
    
    accel_data = record.p_signal[:, accel_indices].astype(np.float32)
    print(f"Loaded data shape: {accel_data.shape} @ {record.fs}Hz")
    print(f"Accel range: [{np.min(accel_data):.2f}, {np.max(accel_data):.2f}] m/s²")
    
    # Test first window
    window = accel_data[:256]
    print(f"\nFirst window stats:")
    print(f"  Mean: {np.mean(window, axis=0)}")
    print(f"  Std:  {np.std(window, axis=0)}")
    
    # Test physics engine
    engine = PhysicsEngine()
    imu_raw = {
        "accel": window.tolist(),
        "gyro": [[0, 0, 0]] * 256,
        "timestamps_ns": [int(1e9/record.fs * j) for j in range(256)],
        "sample_rate_hz": record.fs
    }
    
    result = engine.certify(imu_raw, segment="walking")
    print(f"\nPhysics Engine Result:")
    print(f"  Tier: {result['tier']}")
    print(f"  Score: {result['physical_law_score']}")
    print(f"  Laws checked: {result['laws_checked']}")
    print(f"  Laws passed: {result['laws_passed']}")
    print(f"  Laws failed: {result['laws_failed']}")
    
    print(f"\nLaw details:")
    for law, details in result['law_details'].items():
        print(f"  {law}:")
        for key, value in details.items():
            print(f"    {key}: {value}")
        print()

if __name__ == "__main__":
    debug_ptt_ppg()
