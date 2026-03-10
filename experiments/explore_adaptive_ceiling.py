#!/usr/bin/env python3
"""
Explore Adaptive Ceiling Limits for NinaPro Subjects
==================================================
Analyzes actual observed maximums vs S2S hardcoded limits.
Shows which subjects exceed ceilings but stay above floors.
"""

import numpy as np
import scipy.io
import os
import sys
from pathlib import Path

def load_ninapro_subject(subject_path):
    """Load accelerometer data from NinaPro subject"""
    try:
        import glob
        mat_files = glob.glob(os.path.join(subject_path, "*.mat"))
        if not mat_files:
            return None, None, None
        
        # Load first .mat file
        mat = scipy.io.loadmat(mat_files[0])
        
        # Find accelerometer data
        for key in mat.keys():
            if not key.startswith('_'):
                data = mat[key]
                if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
                    accel = data[:, :3]
                    timestamps = np.arange(len(accel)) / 2000.0  # 2000Hz sampling
                    return accel, timestamps, mat_files[0]
        
        return None, None, None
        
    except Exception as e:
        print(f"Error loading {subject_path}: {e}")
        return None, None, None

def calculate_jerk(accel, sample_rate=2000):
    """Calculate jerk from acceleration data with rate normalization"""
    # Remove gravity (median subtraction)
    gravity = np.median(accel, axis=0)
    accel_no_gravity = accel - gravity
    
    # Calculate jerk (derivative of acceleration)
    dt = 1.0 / sample_rate
    jerk = np.diff(accel_no_gravity, axis=0) / dt
    
    # Rate normalization: divide by (sample_rate/50)³
    # This normalizes 2000Hz data to equivalent 50Hz scale
    rate_factor = (sample_rate / 50.0) ** 3
    normalized_jerk = jerk / rate_factor
    
    return normalized_jerk

def calculate_resonance_frequency(accel, sample_rate=2000):
    """Calculate dominant frequency from accelerometer data"""
    # Use z-axis (vertical) for resonance
    z_accel = accel[:, 2]
    
    # FFT to find dominant frequency
    fft = np.abs(np.fft.rfft(z_accel))
    freqs = np.fft.rfftfreq(len(z_accel), 1/sample_rate)
    
    # Find peak frequency (excluding DC component)
    if len(fft) > 1:
        peak_idx = np.argmax(fft[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_idx]
        peak_energy = fft[peak_idx]
    else:
        peak_freq = 0.0
        peak_energy = 0.0
    
    return peak_freq, peak_energy

def get_s2s_limits():
    """Extract hardcoded limits from s2s_physics_v1_3.py"""
    limits = {}
    
    try:
        # Add s2s_standard_v1_3 to path
        sys.path.insert(0, 's2s_standard_v1_3')
        import s2s_physics_v1_3 as physics
        
        # Get jerk limits
        limits['jerk_max_ms3'] = physics.JERK_MAX_MS3
        limits['jerk_max_walking_ms3'] = physics.JERK_MAX_WALKING_MS3
        
        # Get resonance frequency ranges from SEGMENT_PARAMS
        limits['resonance_ranges'] = physics.SEGMENT_PARAMS
        
        # Get gravity range
        limits['gravity_min_ms2'] = physics.GRAVITY_MIN_MS2
        limits['gravity_max_ms2'] = physics.GRAVITY_MAX_MS2
        
        # Get acceleration RMS ranges
        limits['accel_rms_min_ms2'] = physics.ACCEL_RMS_MIN_MS2
        limits['accel_rms_max_ms2'] = physics.ACCEL_RMS_MAX_MS2
        
        print("Successfully loaded S2S limits from s2s_physics_v1_3.py")
        
    except Exception as e:
        print(f"Error loading S2S limits: {e}")
        # Fallback hardcoded values
        limits = {
            'jerk_max_ms3': 500.0,
            'jerk_max_walking_ms3': 1000.0,
            'resonance_ranges': {
                'forearm': (10.0, 50.0, 8.0, 12.0),  # I, K, f_lo, f_hi
                'walking': (10.0, 50.0, 1.0, 3.0),
            },
            'gravity_min_ms2': 8.0,
            'gravity_max_ms2': 12.0,
            'accel_rms_min_ms2': 0.1,
            'accel_rms_max_ms2': 2.0,
        }
    
    return limits

def analyze_subject(subject_path, subject_name, limits):
    """Analyze a single NinaPro subject with rate normalization"""
    accel, timestamps, data_file = load_ninapro_subject(subject_path)
    
    if accel is None:
        return None
    
    sample_rate = 2000  # NinaPro is 2000Hz
    
    # Calculate metrics with normalization
    jerk = calculate_jerk(accel, sample_rate=sample_rate)
    resonance_freq, resonance_energy = calculate_resonance_frequency(accel, sample_rate=sample_rate)
    
    # Calculate maximums
    max_jerk = np.max(np.linalg.norm(jerk, axis=1))  # RMS jerk magnitude
    max_accel = np.max(np.linalg.norm(accel, axis=1))  # Total acceleration magnitude
    
    # Calculate RMS acceleration with rate normalization
    accel_rms = np.sqrt(np.mean(np.linalg.norm(accel, axis=1)**2))
    # Rate normalization: divide by (sample_rate/50)
    rate_factor_accel = sample_rate / 50.0
    normalized_accel_rms = accel_rms / rate_factor_accel
    
    # Get S2S limits for forearm segment
    forearm_params = limits['resonance_ranges'].get('forearm', (10.0, 50.0, 8.0, 12.0))
    resonance_floor, resonance_ceiling = forearm_params[2], forearm_params[3]
    
    # Determine status
    jerk_ceiling = limits['jerk_max_ms3']
    accel_ceiling = limits['accel_rms_max_ms2']
    
    jerk_status = "EXCEEDS" if max_jerk > jerk_ceiling else "WITHIN"
    accel_status = "EXCEEDS" if normalized_accel_rms > accel_ceiling else "WITHIN"
    resonance_status = "EXCEEDS" if resonance_freq > resonance_ceiling else "WITHIN"
    
    return {
        'subject': subject_name,
        'data_file': os.path.basename(data_file),
        'samples': len(accel),
        'sample_rate': sample_rate,
        'max_jerk_ms3': max_jerk,
        'max_accel_ms2': max_accel,
        'accel_rms_ms2': accel_rms,
        'accel_rms_normalized_ms2': normalized_accel_rms,
        'resonance_freq_hz': resonance_freq,
        'resonance_energy': resonance_energy,
        'jerk_ceiling': jerk_ceiling,
        'accel_ceiling': accel_ceiling,
        'resonance_floor': resonance_floor,
        'resonance_ceiling': resonance_ceiling,
        'jerk_status': jerk_status,
        'accel_status': accel_status,
        'resonance_status': resonance_status,
        'rate_factor_jerk': (sample_rate / 50.0) ** 3,
        'rate_factor_accel': sample_rate / 50.0,
    }

def main():
    print("=" * 80)
    print("Adaptive Ceiling Exploration - NinaPro Subjects s1-s10")
    print("=" * 80)
    
    # Load S2S limits
    print("\n1. Loading S2S hardcoded limits...")
    limits = get_s2s_limits()
    
    print(f"\nS2S Hardcoded Limits:")
    print(f"  JERK_MAX_MS3: {limits['jerk_max_ms3']} m/s³")
    print(f"  JERK_MAX_WALKING_MS3: {limits.get('jerk_max_walking_ms3', 'N/A')} m/s³")
    print(f"  ACCEL_RMS_MAX_MS2: {limits['accel_rms_max_ms2']} m/s²")
    print(f"  RESONANCE_FOREARM: {limits['resonance_ranges']['forearm'][2]}-{limits['resonance_ranges']['forearm'][3]} Hz")
    
    # Analyze NinaPro subjects
    print(f"\n2. Analyzing NinaPro subjects...")
    ninapro_path = "/Users/timbo/ninapro_db5"
    
    subjects = []
    for i in range(1, 11):
        subject_name = f"s{i}"
        subject_path = os.path.join(ninapro_path, subject_name)
        
        if os.path.exists(subject_path):
            print(f"  Analyzing {subject_name}...")
            result = analyze_subject(subject_path, subject_name, limits)
            if result:
                subjects.append(result)
        else:
            print(f"  {subject_name}: Not found")
    
    # Print comparison table
    print(f"\n" + "=" * 80)
    print("COMPARISON TABLE: Normalized Values vs S2S Limits")
    print("=" * 80)
    
    print(f"\nRate Normalization Factors:")
    print(f"  Jerk: (sample_rate/50)³ = {(2000/50)**3:.0f}x reduction")
    print(f"  Accel: (sample_rate/50) = {(2000/50):.0f}x reduction")
    
    print(f"\n{'Subject':<8} {'Norm Jerk':<10} {'Jerk Ceil':<10} {'Jerk':<7} {'Norm Accel':<10} {'Accel Ceil':<10} {'Accel':<7} {'Res Freq':<9} {'Res Range':<10} {'Res':<7}")
    print(f"{'':<8} {'(m/s³)':<10} {'(m/s³)':<10} {'':<7} {'(m/s²)':<10} {'(m/s²)':<10} {'':<7} {'(Hz)':<9} {'(Hz)':<10} {'':<7}")
    print("-" * 100)
    
    ceiling_exceeders = []
    
    for subj in subjects:
        jerk_diff = subj['max_jerk_ms3'] - subj['jerk_ceiling']
        accel_diff = subj['accel_rms_normalized_ms2'] - subj['accel_ceiling']
        resonance_diff = subj['resonance_freq_hz'] - subj['resonance_ceiling']
        
        # Check if any ceiling is exceeded
        exceeds_ceiling = (jerk_diff > 0) or (accel_diff > 0) or (resonance_diff > 0)
        
        if exceeds_ceiling:
            ceiling_exceeders.append(subj)
        
        print(f"{subj['subject']:<8} "
              f"{subj['max_jerk_ms3']:<10.1f} "
              f"{subj['jerk_ceiling']:<10.0f} "
              f"{subj['jerk_status']:<7} "
              f"{subj['accel_rms_normalized_ms2']:<10.2f} "
              f"{subj['accel_ceiling']:<10.1f} "
              f"{subj['accel_status']:<7} "
              f"{subj['resonance_freq_hz']:<9.1f} "
              f"{subj['resonance_floor']:.0f}-{subj['resonance_ceiling']:.0f} "
              f"{subj['resonance_status']:<7}")
    
    # Summary of ceiling exceeders
    print(f"\n" + "=" * 80)
    print("CEILING EXCEEDERS ANALYSIS")
    print("=" * 80)
    
    if ceiling_exceeders:
        print(f"\nSubjects exceeding S2S ceilings after normalization ({len(ceiling_exceeders)}/{len(subjects)}):")
        
        for subj in ceiling_exceeders:
            violations = []
            if subj['max_jerk_ms3'] > subj['jerk_ceiling']:
                violations.append(f"Jerk {subj['max_jerk_ms3']:.1f}>{subj['jerk_ceiling']:.0f}")
            if subj['accel_rms_normalized_ms2'] > subj['accel_ceiling']:
                violations.append(f"Accel {subj['accel_rms_normalized_ms2']:.1f}>{subj['accel_ceiling']:.1f}")
            if subj['resonance_freq_hz'] > subj['resonance_ceiling']:
                violations.append(f"Res {subj['resonance_freq_hz']:.1f}>{subj['resonance_ceiling']:.0f}")
            
            print(f"  {subj['subject']}: {', '.join(violations)}")
    else:
        print("\nNo subjects exceed S2S ceilings after normalization.")
    
    # Statistical summary
    print(f"\n" + "=" * 80)
    print("STATISTICAL SUMMARY - NORMALIZED VALUES")
    print("=" * 80)
    
    all_jerk = [s['max_jerk_ms3'] for s in subjects]
    all_accel_normalized = [s['accel_rms_normalized_ms2'] for s in subjects]
    all_resonance = [s['resonance_freq_hz'] for s in subjects]
    
    print(f"\nNormalized Jerk Statistics (m/s³):")
    print(f"  Range: {np.min(all_jerk):.1f} - {np.max(all_jerk):.1f}")
    print(f"  Mean:  {np.mean(all_jerk):.1f} ± {np.std(all_jerk):.1f}")
    print(f"  S2S Ceiling: {limits['jerk_max_ms3']:.0f}")
    print(f"  Subjects > Ceiling: {sum(1 for j in all_jerk if j > limits['jerk_max_ms3'])}")
    
    print(f"\nNormalized Acceleration RMS Statistics (m/s²):")
    print(f"  Range: {np.min(all_accel_normalized):.2f} - {np.max(all_accel_normalized):.2f}")
    print(f"  Mean:  {np.mean(all_accel_normalized):.2f} ± {np.std(all_accel_normalized):.2f}")
    print(f"  S2S Ceiling: {limits['accel_rms_max_ms2']:.1f}")
    print(f"  Subjects > Ceiling: {sum(1 for a in all_accel_normalized if a > limits['accel_rms_max_ms2'])}")
    
    print(f"\nResonance Frequency Statistics (Hz):")
    print(f"  Range: {np.min(all_resonance):.1f} - {np.max(all_resonance):.1f}")
    print(f"  Mean:  {np.mean(all_resonance):.1f} ± {np.std(all_resonance):.1f}")
    print(f"  S2S Range: {limits['resonance_ranges']['forearm'][2]:.0f}-{limits['resonance_ranges']['forearm'][3]:.0f} Hz")
    print(f"  Subjects > Ceiling: {sum(1 for r in all_resonance if r > limits['resonance_ranges']['forearm'][3])}")
    
    # Show original vs normalized comparison
    print(f"\n" + "=" * 80)
    print("NORMALIZATION IMPACT ANALYSIS")
    print("=" * 80)
    
    all_accel_original = [s['accel_rms_ms2'] for s in subjects]
    
    print(f"\nAcceleration RMS - Before vs After Normalization:")
    print(f"  Original Range: {np.min(all_accel_original):.1f} - {np.max(all_accel_original):.1f} m/s²")
    print(f"  Normalized Range: {np.min(all_accel_normalized):.2f} - {np.max(all_accel_normalized):.2f} m/s²")
    print(f"  Reduction Factor: {np.mean(all_accel_original)/np.mean(all_accel_normalized):.1f}x")
    print(f"  S2S Ceiling: {limits['accel_rms_max_ms2']:.1f} m/s²")
    
    exceeders_before = sum(1 for a in all_accel_original if a > limits['accel_rms_max_ms2'])
    exceeders_after = sum(1 for a in all_accel_normalized if a > limits['accel_rms_max_ms2'])
    
    print(f"  Exceeders Before: {exceeders_before}/10")
    print(f"  Exceeders After: {exceeders_after}/10")
    print(f"  Improvement: {exceeders_before - exceeders_after} subjects now within limits")
    
    print(f"\n" + "=" * 80)
    print("ADAPTIVE CEILING RECOMMENDATIONS")
    print("=" * 80)
    
    # Calculate adaptive ceilings
    jerk_adaptive = np.percentile(all_jerk, 95)  # 95th percentile
    accel_adaptive = np.percentile(all_accel_normalized, 95)
    resonance_adaptive = np.percentile(all_resonance, 95)
    
    print(f"\nRecommended Adaptive Ceilings (95th percentile) - NORMALIZED:")
    print(f"  JERK_MAX_MS3: {jerk_adaptive:.0f} (current: {limits['jerk_max_ms3']:.0f})")
    print(f"  ACCEL_RMS_MAX_MS2: {accel_adaptive:.1f} (current: {limits['accel_rms_max_ms2']:.1f})")
    print(f"  RESONANCE_MAX_HZ: {resonance_adaptive:.1f} (current: {limits['resonance_ranges']['forearm'][3]:.0f})")
    
    print(f"\nSubjects that would benefit from adaptive ceilings:")
    adaptive_beneficiaries = [s for s in ceiling_exceeders 
                             if s['max_jerk_ms3'] <= jerk_adaptive and 
                                s['accel_rms_normalized_ms2'] <= accel_adaptive and
                                s['resonance_freq_hz'] <= resonance_adaptive]
    
    if adaptive_beneficiaries:
        for subj in adaptive_beneficiaries:
            print(f"  {subj['subject']}: Would pass with adaptive ceilings")
    else:
        print("  No subjects would benefit from 95th percentile adaptive ceilings")

if __name__ == "__main__":
    main()
