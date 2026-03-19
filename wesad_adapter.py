#!/usr/bin/env python3
"""
WESAD Dataset Adapter for S2S Certification
============================================
Prepared for Kristof Van Laerhoven (UbiComp Siegen)
Response to email: March 19, 2026

Dataset: WESAD (Wearable Stress and Affect Detection)
Link: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

Sensors:
- Chest: ACC (3-axis, 700Hz), ECG (700Hz), EMG (700Hz), EDA (700Hz), 
         RESP (700Hz), TEMP (700Hz)
- Wrist: ACC (3-axis, 32Hz), BVP/PPG (64Hz), EDA (4Hz), TEMP (4Hz)

S2S Certification Features:
1. Multi-sensor fusion (chest + wrist simultaneously)
2. Kinematic chain validation (chest-wrist coupling)
3. Biological signal verification (heart rate + stress markers)
4. Physics-based quality grading

Expected Results:
- Higher F1 improvement than PAMAP2 (+4.23%)
- WESAD has richer biological signals (PPG + EDA + ACC + RESP)
- More physics laws can be checked (7 biomechanical + 3 biological)

Output:
- Certified windows with GOLD/SILVER/BRONZE/REJECTED tiers
- Human-in-loop score (0-100) for stress detection quality
- Physics violation reports for debugging dataset issues
"""

import os
import sys
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add S2S to path
sys.path.insert(0, os.path.expanduser("~/S2S"))

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier
from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner

# WESAD dataset configuration
WESAD_ROOT = os.path.expanduser("~/wesad_data")  # Download from UbiComp
SUBJECTS = [f"S{i}" for i in range(2, 18)]  # S2-S17 (16 subjects)

# Sampling rates
CHEST_HZ = 700
WRIST_ACC_HZ = 32
WRIST_BVP_HZ = 64
WRIST_EDA_HZ = 4
WRIST_TEMP_HZ = 4

# Labels
LABELS = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignored",
    6: "ignored",
    7: "ignored",
}

# Window size for certification (5 seconds)
WINDOW_SEC = 5
CHEST_WINDOW = int(CHEST_HZ * WINDOW_SEC)  # 3500 samples
WRIST_WINDOW = int(WRIST_ACC_HZ * WINDOW_SEC)  # 160 samples


class WESADAdapter:
    """
    Adapter to certify WESAD dataset with S2S physics validation.
    
    Provides multi-sensor fusion of chest + wrist sensors with
    biological signal validation.
    """
    
    def __init__(self, output_dir: str = "wesad_certified"):
        self.physics_engine = PhysicsEngine()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_windows": 0,
            "certified": 0,
            "rejected": 0,
            "gold": 0,
            "silver": 0,
            "bronze": 0,
            "by_condition": {},
        }
    
    def load_subject(self, subject_id: str) -> Optional[Dict]:
        """Load WESAD subject data from pickle file."""
        pkl_path = Path(WESAD_ROOT) / subject_id / f"{subject_id}.pkl"
        
        if not pkl_path.exists():
            print(f"⚠️  Subject not found: {pkl_path}")
            return None
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        return data
    
    def certify_chest_acc(
        self,
        accel: np.ndarray,
        timestamps_ns: List[int],
    ) -> Dict:
        """Certify chest accelerometer data (700Hz)."""
        
        # Extract 3-axis acceleration
        if accel.shape[1] >= 3:
            acc_3d = accel[:, :3]
        else:
            return {"status": "FAIL", "tier": "REJECTED", "reason": "INSUFFICIENT_AXES"}
        
        # Create IMU structure (no gyro for WESAD chest)
        imu_raw = {
            "timestamps_ns": timestamps_ns,
            "accel": acc_3d.tolist(),
            "gyro": np.zeros_like(acc_3d).tolist(),  # No gyro
        }
        
        # Certify with physics engine
        result = self.physics_engine.certify(
            imu_raw=imu_raw,
            segment="chest",
        )
        
        return result
    
    def certify_wrist_acc(
        self,
        accel: np.ndarray,
        timestamps_ns: List[int],
    ) -> Dict:
        """Certify wrist accelerometer data (32Hz)."""
        
        # Extract 3-axis acceleration
        if accel.shape[1] >= 3:
            acc_3d = accel[:, :3]
        else:
            return {"status": "FAIL", "tier": "REJECTED", "reason": "INSUFFICIENT_AXES"}
        
        # Create IMU structure
        imu_raw = {
            "timestamps_ns": timestamps_ns,
            "accel": acc_3d.tolist(),
            "gyro": np.zeros_like(acc_3d).tolist(),  # No gyro
        }
        
        # Certify with physics engine
        result = self.physics_engine.certify(
            imu_raw=imu_raw,
            segment="wrist",
        )
        
        return result
    
    def extract_ppg_features(
        self,
        bvp: np.ndarray,
        hz: float = WRIST_BVP_HZ,
    ) -> Dict:
        """
        Extract PPG/BVP features for biological validation.
        
        Returns heart rate, HRV, and pulse quality metrics.
        """
        # Simple peak detection for heart rate
        # (In production, use scipy.signal.find_peaks with better parameters)
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(bvp, distance=int(hz * 0.5))  # Min 0.5s between peaks
        
        if len(peaks) < 2:
            return {
                "has_pulse": False,
                "heart_rate_bpm": None,
                "hrv_rmssd_ms": None,
            }
        
        # Calculate inter-beat intervals (IBI)
        ibi = np.diff(peaks) / hz  # seconds
        
        # Heart rate (BPM)
        hr_bpm = 60.0 / np.mean(ibi)
        
        # HRV - RMSSD (root mean square of successive differences)
        ibi_ms = ibi * 1000  # convert to milliseconds
        rmssd = np.sqrt(np.mean(np.diff(ibi_ms) ** 2))
        
        # Validate physiological range
        valid_hr = 30 <= hr_bpm <= 220
        valid_pulse = len(peaks) >= 5
        
        return {
            "has_pulse": valid_pulse and valid_hr,
            "heart_rate_bpm": hr_bpm if valid_hr else None,
            "hrv_rmssd_ms": rmssd if valid_pulse else None,
            "n_beats": len(peaks),
        }
    
    def fuse_chest_wrist(
        self,
        chest_cert: Dict,
        wrist_cert: Dict,
        ppg_features: Dict,
    ) -> Dict:
        """
        Fuse chest and wrist certifications using multi-sensor fusion.
        
        This demonstrates S2S's key strength: kinematic chain validation
        across distributed sensors.
        """
        fusion = FusionCertifier(
            device_id="wesad_study",
            session_id="chest_wrist_fusion",
            hierarchical=True,  # Use optimized O(n) fusion
        )
        
        # Add chest IMU
        fusion.add_imu_cert(chest_cert)
        
        # Add wrist IMU
        fusion.add_stream("wrist_imu", wrist_cert, sensor_type="IMU")
        
        # Add PPG/BVP as biological signal
        ppg_cert = {
            "tier": "GOLD" if ppg_features.get("has_pulse") else "REJECTED",
            "frame_start_ts_ns": chest_cert.get("frame_start_ts_ns", 0),
            "frame_end_ts_ns": chest_cert.get("frame_end_ts_ns", 0),
            "vitals": {
                "heart_rate_bpm": ppg_features.get("heart_rate_bpm"),
                "hrv_rmssd_ms": ppg_features.get("hrv_rmssd_ms"),
            },
            "flags": [] if ppg_features.get("has_pulse") else ["NO_PULSE"],
        }
        fusion.add_ppg_cert(ppg_cert)
        
        # Run fusion
        result = fusion.certify()
        
        return result
    
    def process_subject(
        self,
        subject_id: str,
        max_windows: int = 100,
    ) -> List[Dict]:
        """
        Process a single WESAD subject.
        
        Extracts windows, certifies chest + wrist simultaneously,
        and saves certified results.
        """
        print(f"\nProcessing {subject_id}...")
        
        # Load subject data
        data = self.load_subject(subject_id)
        if data is None:
            return []
        
        # Extract signals
        chest = data['signal']['chest']
        wrist = data['signal']['wrist']
        labels = data['label']
        
        chest_acc = chest['ACC']  # (N, 3) at 700Hz
        wrist_acc = wrist['ACC']  # (M, 3) at 32Hz
        wrist_bvp = wrist['BVP']  # (K,) at 64Hz
        
        # Align labels to chest sampling rate (700Hz)
        # Labels are at 700Hz for chest
        
        certificates = []
        
        # Process windows
        n_chest = len(chest_acc)
        n_windows = min((n_chest - CHEST_WINDOW) // CHEST_WINDOW, max_windows)
        
        for i in range(n_windows):
            # Extract chest window (700Hz, 5 seconds = 3500 samples)
            chest_start = i * CHEST_WINDOW
            chest_end = chest_start + CHEST_WINDOW
            
            chest_window = chest_acc[chest_start:chest_end]
            chest_ts = [int(j * 1e9 / CHEST_HZ) for j in range(len(chest_window))]
            
            # Extract corresponding wrist window (32Hz, 5 seconds = 160 samples)
            # Downsample ratio: 700/32 ≈ 21.875
            wrist_start = int(chest_start * WRIST_ACC_HZ / CHEST_HZ)
            wrist_end = wrist_start + WRIST_WINDOW
            
            if wrist_end > len(wrist_acc):
                break
            
            wrist_window = wrist_acc[wrist_start:wrist_end]
            wrist_ts = [int(j * 1e9 / WRIST_ACC_HZ) for j in range(len(wrist_window))]
            
            # Extract BVP window (64Hz, 5 seconds = 320 samples)
            bvp_start = int(chest_start * WRIST_BVP_HZ / CHEST_HZ)
            bvp_end = bvp_start + int(WRIST_BVP_HZ * WINDOW_SEC)
            
            if bvp_end > len(wrist_bvp):
                break
            
            bvp_window = wrist_bvp[bvp_start:bvp_end]
            
            # Get label for this window (majority vote)
            label_window = labels[chest_start:chest_end]
            majority_label = int(np.bincount(label_window.astype(int)).argmax())
            condition = LABELS.get(majority_label, "unknown")
            
            # Certify chest
            chest_cert = self.certify_chest_acc(chest_window, chest_ts)
            
            # Certify wrist
            wrist_cert = self.certify_wrist_acc(wrist_window, wrist_ts)
            
            # Extract PPG features
            ppg_features = self.extract_ppg_features(bvp_window)
            
            # Fuse chest + wrist + PPG
            fusion_cert = self.fuse_chest_wrist(chest_cert, wrist_cert, ppg_features)
            
            # Add metadata
            fusion_cert.update({
                "subject_id": subject_id,
                "window_index": i,
                "condition": condition,
                "chest_tier": chest_cert.get("tier"),
                "wrist_tier": wrist_cert.get("tier"),
                "ppg_hr_bpm": ppg_features.get("heart_rate_bpm"),
                "ppg_hrv_ms": ppg_features.get("hrv_rmssd_ms"),
            })
            
            certificates.append(fusion_cert)
            
            # Update statistics
            self.stats["total_windows"] += 1
            tier = fusion_cert.get("tier")
            if tier == "REJECTED":
                self.stats["rejected"] += 1
            else:
                self.stats["certified"] += 1
                if tier == "GOLD":
                    self.stats["gold"] += 1
                elif tier == "SILVER":
                    self.stats["silver"] += 1
                elif tier == "BRONZE":
                    self.stats["bronze"] += 1
            
            # Track by condition
            if condition not in self.stats["by_condition"]:
                self.stats["by_condition"][condition] = {"total": 0, "certified": 0}
            self.stats["by_condition"][condition]["total"] += 1
            if tier != "REJECTED":
                self.stats["by_condition"][condition]["certified"] += 1
        
        # Save subject certificates
        output_file = self.output_dir / f"{subject_id}_certified.json"
        with open(output_file, 'w') as f:
            json.dump(certificates, f, indent=2)
        
        print(f"  ✅ {len(certificates)} windows certified")
        print(f"  📁 Saved: {output_file}")
        
        return certificates
    
    def process_all_subjects(self, max_windows_per_subject: int = 100):
        """Process all WESAD subjects."""
        print("="*70)
        print("WESAD DATASET CERTIFICATION")
        print("="*70)
        
        all_certificates = []
        
        for subject_id in SUBJECTS:
            certs = self.process_subject(subject_id, max_windows=max_windows_per_subject)
            all_certificates.extend(certs)
        
        # Save summary
        self.save_summary()
        
        return all_certificates
    
    def save_summary(self):
        """Save certification summary and statistics."""
        summary = {
            "dataset": "WESAD",
            "total_subjects": len(SUBJECTS),
            "statistics": self.stats,
            "certification_rate": self.stats["certified"] / max(self.stats["total_windows"], 1),
            "tier_distribution": {
                "GOLD": self.stats["gold"] / max(self.stats["certified"], 1),
                "SILVER": self.stats["silver"] / max(self.stats["certified"], 1),
                "BRONZE": self.stats["bronze"] / max(self.stats["certified"], 1),
            },
        }
        
        summary_file = self.output_dir / "wesad_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*70)
        print("CERTIFICATION SUMMARY")
        print("="*70)
        print(f"Total windows:     {self.stats['total_windows']}")
        print(f"Certified:         {self.stats['certified']} ({summary['certification_rate']*100:.1f}%)")
        print(f"  GOLD:            {self.stats['gold']}")
        print(f"  SILVER:          {self.stats['silver']}")
        print(f"  BRONZE:          {self.stats['bronze']}")
        print(f"Rejected:          {self.stats['rejected']}")
        print("\nBy condition:")
        for cond, stats in self.stats["by_condition"].items():
            cert_rate = stats["certified"] / max(stats["total"], 1) * 100
            print(f"  {cond:15} {stats['certified']:4}/{stats['total']:4} ({cert_rate:.1f}%)")
        print(f"\n📁 Summary saved: {summary_file}")
        print("="*70)


def main():
    """Run WESAD certification for Kristof's benchmark."""
    
    # Check if WESAD data exists
    if not Path(WESAD_ROOT).exists():
        print("="*70)
        print("WESAD DATASET NOT FOUND")
        print("="*70)
        print(f"\nExpected location: {WESAD_ROOT}")
        print("\nDownload WESAD dataset from:")
        print("https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/")
        print("\nExtract to the expected location or update WESAD_ROOT in this script.")
        print("="*70)
        return
    
    # Run certification
    adapter = WESADAdapter(output_dir="wesad_certified")
    adapter.process_all_subjects(max_windows_per_subject=100)
    
    print("\n✅ WESAD certification complete!")
    print("\nNext steps:")
    print("1. Review results in wesad_certified/")
    print("2. Compare F1 scores vs uncertified baseline")
    print("3. Send results to Kristof Van Laerhoven")
    print("4. Publish paper with UbiComp collaboration")


if __name__ == "__main__":
    main()
