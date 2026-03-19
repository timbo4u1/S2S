#!/usr/bin/env python3
"""
Debug the jerk calculation - why is it 47,730 m/s³ for walking?
"""

import os
import sys
import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(__file__))

def debug_jerk():
    print("=== Debugging Jerk Calculation ===")
    
    # Load PTT-PPG data
    ptt_dir = "/Users/timbo/physionet.org/files/pulse-transit-time-ppg/1.1.0"
    record = wfdb.rdrecord(os.path.join(ptt_dir, 's1_walk'))
    
    # Find accel channels
    accel_indices = []
    for i, sig_name in enumerate(record.sig_name):
        if sig_name in ['a_x', 'a_y', 'a_z']:
            accel_indices.append(i)
    
    accel_data = record.p_signal[:, accel_indices].astype(np.float32)
    print(f"Data shape: {accel_data.shape} @ {record.fs}Hz")
    
    # Check first window
    window = accel_data[:256]
    dt = 1.0 / record.fs  # 0.002s for 500Hz
    
    print(f"\nRaw accel stats (first window):")
    print(f"  X: mean={np.mean(window[:,0]):.3f}, std={np.std(window[:,0]):.3f}, range=[{np.min(window[:,0]):.3f}, {np.max(window[:,0]):.3f}]")
    print(f"  Y: mean={np.mean(window[:,1]):.3f}, std={np.std(window[:,1]):.3f}, range=[{np.min(window[:,1]):.3f}, {np.max(window[:,1]):.3f}]")
    print(f"  Z: mean={np.mean(window[:,2]):.3f}, std={np.std(window[:,2]):.3f}, range=[{np.min(window[:,2]):.3f}, {np.max(window[:,2]):.3f}]")
    
    # Manual jerk calculation for X axis
    sig_raw = window[:, 0]
    
    # Smooth with w=7
    w = 7
    kernel = np.ones(2*w + 1) / (2*w + 1)
    s1 = np.convolve(np.pad(sig_raw, w, mode='edge'), kernel, mode='valid')
    
    # First derivative (velocity)
    vel = (s1[2:] - s1[:-2]) / (2 * dt)
    
    # Smooth velocity
    s2 = np.convolve(np.pad(vel, w, mode='edge'), kernel, mode='valid')
    
    # Second derivative (acceleration)
    accel_from_vel = (s2[2:] - s2[:-2]) / (2 * dt)
    
    # Smooth acceleration 
    s3 = np.convolve(np.pad(accel_from_vel, w, mode='edge'), kernel, mode='valid')
    
    # Third derivative (jerk)
    jerk = (s3[2:] - s3[:-2]) / (2 * dt)
    
    print(f"\nJerk calculation steps (X axis):")
    print(f"  Original accel range: [{np.min(sig_raw):.3f}, {np.max(sig_raw):.3f}] m/s²")
    print(f"  After smoothing (s1): [{np.min(s1):.3f}, {np.max(s1):.3f}] m/s²")
    print(f"  Velocity range: [{np.min(vel):.3f}, {np.max(vel):.3f}] m/s")
    print(f"  After smoothing (s2): [{np.min(s2):.3f}, {np.max(s2):.3f}] m/s")
    print(f"  Acceleration from vel: [{np.min(accel_from_vel):.3f}, {np.max(accel_from_vel):.3f}] m/s²")
    print(f"  After smoothing (s3): [{np.min(s3):.3f}, {np.max(s3):.3f}] m/s²")
    print(f"  Final jerk range: [{np.min(jerk):.3f}, {np.max(jerk):.3f}] m/s³")
    print(f"  P95 jerk: {np.percentile(np.abs(jerk), 95):.1f} m/s³")
    
    # Check if this makes sense for walking
    print(f"\nSanity check:")
    print(f"  Walking speed ~1.5 m/s, cadence ~2 steps/s")
    print(f"  Expected acceleration ~3 m/s², jerk ~10 m/s³")
    print(f"  Calculated jerk P95: {np.percentile(np.abs(jerk), 95):.0f} m/s³")
    print(f"  --> This is {np.percentile(np.abs(jerk), 95)/10:.0f}x higher than expected!")

if __name__ == "__main__":
    debug_jerk()
