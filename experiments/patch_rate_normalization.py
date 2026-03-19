#!/usr/bin/env python3
"""
Patch S2S Physics Engine with Rate Normalization
=============================================
Temporarily patches s2s_physics_v1_3.py to add rate normalization
for high-rate IMU data, then runs certification test.
"""

import os
import sys
import shutil
import subprocess

def backup_physics_engine():
    """Create backup of original physics engine"""
    src = "s2s_standard_v1_3/s2s_physics_v1_3.py"
    backup = "s2s_standard_v1_3/s2s_physics_v1_3.py.backup"
    
    if os.path.exists(backup):
        print("Backup already exists")
        return True
    
    try:
        shutil.copy2(src, backup)
        print(f"Created backup: {backup}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def apply_rate_normalization_patch():
    """Apply rate normalization patch to physics engine"""
    src_file = "s2s_standard_v1_3/s2s_physics_v1_3.py"
    
    try:
        # Read the original file
        with open(src_file, 'r') as f:
            content = f.read()
        
        # Find the check_jerk function and add rate normalization
        # Insert rate normalization after jerk calculation
        
        # Patch 1: Add rate normalization after jerk_raw calculation
        jerk_patch = '''            jerk_raw = np_diff(s2)  # Third derivative: position → vel → accel → jerk
            
            # RATE NORMALIZATION: Scale jerk for high-rate IMU data
            sample_rate = 1.0 / dt if dt > 0 else 50.0
            if sample_rate > 100:  # High-rate IMU (>100Hz)
                rate_factor = (sample_rate / 50.0) ** 3
                jerk_raw = jerk_raw / rate_factor
                d["rate_normalized"] = f"jerk scaled by 1/{rate_factor:.0f} for {sample_rate:.0f}Hz"
            
            jerk = jerk_raw.tolist()'''
        
        # Replace the original jerk calculation
        original_jerk = '''            jerk_raw = np_diff(s2)  # Third derivative: position → vel → accel → jerk
            jerk = jerk_raw.tolist()'''
        
        if original_jerk in content:
            content = content.replace(original_jerk, jerk_patch)
            print("Applied jerk rate normalization patch")
        else:
            print("Could not find jerk calculation to patch")
            return False
        
        # Patch 2: Add rate normalization to check_resonance function for acceleration
        # Find the check_resonance function
        resonance_start = content.find("def check_resonance(")
        if resonance_start == -1:
            print("Could not find check_resonance function")
            return False
        
        # Find where acceleration is processed in check_resonance
        accel_processing = content.find("    az = [s[2] if len(s) > 2 else 0.0 for s in accel]", resonance_start)
        if accel_processing == -1:
            print("Could not find acceleration processing in check_resonance")
            return False
        
        # Add rate normalization after acceleration processing
        resonance_patch = '''    az = [s[2] if len(s) > 2 else 0.0 for s in accel]
    
    # RATE NORMALIZATION: Scale acceleration for high-rate IMU data
    sample_rate = imu_raw.get("sample_rate_hz", 50.0)
    if sample_rate > 100:  # High-rate IMU (>100Hz)
        rate_factor_accel = sample_rate / 50.0
        az = [z / rate_factor_accel for z in az]
        d["rate_normalized_accel"] = f"accel scaled by 1/{rate_factor_accel:.0f} for {sample_rate:.0f}Hz"'''
        
        original_resonance = '''    az = [s[2] if len(s) > 2 else 0.0 for s in accel]'''
        
        if original_resonance in content[resonance_start:resonance_start + 1000]:
            content = content[:resonance_start] + content[resonance_start:].replace(original_resonance, resonance_patch, 1)
            print("Applied acceleration rate normalization patch")
        else:
            print("Could not find acceleration processing in check_resonance")
            return False
        
        # Write patched file
        with open(src_file, 'w') as f:
            f.write(content)
        
        print("Rate normalization patches applied successfully")
        return True
        
    except Exception as e:
        print(f"Error applying patch: {e}")
        return False

def restore_physics_engine():
    """Restore original physics engine from backup"""
    backup = "s2s_standard_v1_3/s2s_physics_v1_3.py.backup"
    src = "s2s_standard_v1_3/s2s_physics_v1_3.py"
    
    try:
        if os.path.exists(backup):
            shutil.copy2(backup, src)
            print(f"Restored original physics engine from {backup}")
            return True
        else:
            print("No backup found to restore")
            return False
    except Exception as e:
        print(f"Error restoring backup: {e}")
        return False

def run_certification_test(subject="s1"):
    """Run certification test on specified subject"""
    print(f"\n{'='*60}")
    print(f"Running Certification Test on Subject {subject}")
    print(f"{'='*60}")
    
    try:
        # Run certification on single subject
        cmd = ["python3", "certify_ninapro_db5.py", "--subject", subject]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/timbo/S2S")
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running certification: {e}")
        return False

def extract_rejection_rate(output):
    """Extract rejection rate from certification output"""
    lines = output.split('\n')
    for line in lines:
        if "rejection rate" in line.lower():
            # Look for percentage
            import re
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                return float(match.group(1))
    return None

def main():
    print("=" * 80)
    print("Rate Normalization Patch Test")
    print("=" * 80)
    
    # Step 1: Backup original physics engine
    print("\n1. Creating backup of physics engine...")
    if not backup_physics_engine():
        print("Failed to create backup, aborting")
        return
    
    # Step 2: Apply rate normalization patch
    print("\n2. Applying rate normalization patches...")
    if not apply_rate_normalization_patch():
        print("Failed to apply patches, aborting")
        restore_physics_engine()
        return
    
    # Step 3: Run certification test with patches
    print("\n3. Running certification test with rate normalization...")
    success = run_certification_test("s1")
    
    # Step 4: Restore original physics engine
    print("\n4. Restoring original physics engine...")
    restore_physics_engine()
    
    print("\n" + "=" * 80)
    print("PATCH TEST COMPLETE")
    print("=" * 80)
    
    if success:
        print("✅ Rate normalization test completed successfully")
        print("Check the output above for new rejection rate vs original 43%")
    else:
        print("❌ Rate normalization test failed")
    
    print("\nOriginal physics engine has been restored.")

if __name__ == "__main__":
    main()
