# S2S Experiments Analysis: Level 5 Models vs Corruption Analysis
**Date:** March 17, 2026  
**Focus:** Relationship between Physical AI models and corruption analysis scripts

---

## 🧪 **Experiment Categories Overview**

### **Level 5 Physical AI Models (8 files)**
**Purpose:** Train neural networks using physics-certified data for robot control
- `level5_full.py` - Full system with all data sources
- `level5_dualhead.py` - Dual-head architecture
- `level5_weighted.py` - Weighted training approach
- `level5_cnn.py` - CNN-based model
- `level5_ptt_ppg_cnn.py` - PTT-PPG CNN model
- `experiment_level5_transfer.py` - Transfer learning
- `experiment_level5_selfsupervised.py` - Self-supervised learning
- `experiment_level5_cnn_binary.py` - Binary classification

### **Corruption & Quality Analysis (4 files)**
**Purpose:** Systematically test physics engine boundaries and corruption detection
- `module1_corruption_fingerprinter.py` - Corruption fingerprinting
- `module2_frankenstein_mixer.py` - Boundary detection via mixing
- `module3_curriculum_generator.py` - Curriculum learning
- `module4_cloud_trainer.py` - Cloud-based training

---

## 🔗 **Relationship Analysis**

### **Data Flow Dependencies**
```
Corruption Analysis → Physics Engine → Boundary Detection → Constants Update
                                                            ↓
                                    Certified Data → Level 5 Training → Physical AI Models
```

### **Key Relationships**

#### **1. Corruption Fingerprinter → Physics Engine**
**Connection:** `module1_corruption_fingerprinter.py` directly imports and tests `PhysicsEngine`
**Purpose:** Discover physics law boundaries through systematic corruption
**Data Types Tested:**
- `corrupt_dropped_packets()` - Simulates sensor packet loss
- `corrupt_sensor_drift()` - Temperature drift simulation
- `corrupt_spike_injection()` - Electrical interference
- `corrupt_gravity_shift()` - Gravity misalignment
- `corrupt_clock_jitter()` - Unstable sampling clock
- `corrupt_synthetic_smoothing()` - Over-smoothing artifacts

#### **2. Frankenstein Mixer → Physics Engine**
**Connection:** `module2_frankenstein_mixer.py` uses binary search to find exact boundaries
**Purpose:** Find precise contamination ratios where physics laws fail
**Method:**
- Mixes certified GOLD/SILVER data with REJECTED data at varying ratios
- Binary search for boundary where tier changes from valid to invalid
- Records which laws fail at which contamination levels

#### **3. Level 5 Models → Certified Data**
**Connection:** All Level 5 models use physics-certified data as input
**Purpose:** Train robot control using only physics-validated data
**Data Sources:**
- NinaPro DB5 (certified EMG+accel)
- Amputee data (certified motion)
- RoboTurk (certified teleoperation)

---

## 🚨 **Synthetic Pattern Analysis**

### **Frankenstein Mixer: Synthetic Patterns Introduced**

#### **Yes - Introduces Synthetic Patterns Not in Real World**

**1. Artificial Contamination Gradients**
```python
# From module2_frankenstein_mixer.py lines 59-61
mixed_accel = (1 - ratio) * good_accel + ratio * bad_accel
mixed_gyro = (1 - ratio) * good_gyro + ratio * bad_gyro
```
**Issue:** Linear mixing creates unrealistic data that doesn't occur naturally
**Real World:** Sensor degradation is typically non-linear and abrupt
**Frankenstein:** Smooth, predictable contamination gradient

**2. Perfect Boundary Detection**
```python
# Binary search for exact ratio where physics breaks
for _ in range(int(np.log2(1 / precision)) + 1):
    mid = (lo + hi) / 2
    # Test at exact midpoint
```
**Issue:** Real-world corruption doesn't have precise mathematical boundaries
**Real World:** Corruption patterns are messy and multi-factor
**Frankenstein:** Clean, single-dimension boundaries

### **Corruption Fingerprinter: Synthetic Patterns**

#### **Controlled, Unnatural Corruption Types**
**1. Dropped Packets**
```python
# Replace dropped samples with last valid value
for idx in sorted(drop_idx):
    if idx > 0:
        corrupted[idx] = corrupted[idx - 1]
```
**Issue:** Too clean - real packet loss is more chaotic
**Real World:** Random gaps, interpolation artifacts, timing issues
**Synthetic:** Neat last-value holding

**2. Linear Drift**
```python
# Add linear drift - simulates sensor temperature drift
drift = np.linspace(0, intensity * 5.0, len(corrupted))
corrupted[:, 0] += drift  # drift on X axis
```
**Issue:** Perfect linear drift is unrealistic
**Real World:** Drift is exponential, temperature-dependent, multi-axis
**Synthetic:** Mathematical perfection

**3. Spike Injection**
```python
spike_magnitude = intensity * 50.0  # up to 50 m/s² spikes
corrupted[spike_idx] += np.random.randn(n_spikes, 3) * spike_magnitude
```
**Issue:** Controlled magnitude and distribution
**Real World:** Spikes are unpredictable, varying magnitudes
**Synthetic:** Gaussian distribution with fixed bounds

---

## 📊 **Boundary Detection Results**

### **From `results_frankenstein_boundaries.json`:**

#### **Physics Law Boundaries Discovered**
1. **Resonance Frequency**: Breaks at 29.2% contamination (±20.3%)
2. **IMU Internal Consistency**: Breaks at 30.6% contamination (±24.6%)
3. **Jerk Bounds**: Breaks at 53.7% contamination (±11.0%)

#### **Key Insights**
- **Jerk bounds most robust** (highest contamination tolerance)
- **Resonance frequency most sensitive** (lowest contamination tolerance)
- **Significant variation** in boundaries across different good/bad pairs
- **No single threshold** works for all corruption types

---

## 🔄 **Constants Integration Analysis**

### **Are Benchmark Results Used to Set Core Constants?**

#### **Evidence from Code Analysis:**

**1. Constants File Structure**
```python
# From constants.py lines 60-70
V1_2_CV_GOLD = 0.1      # Coefficient of variation
V1_2_CV_SILVER = 0.2     # Coefficient of variation  
V1_2_CV_BRONZE = 0.5     # Coefficient of variation
```

**2. Frankenstein Boundary Results**
```json
"jerk_bounds": {
  "mean_boundary": 0.5370065789473685,
  "std_boundary": 0.11007302244063916,
  "min_boundary": 0.4453125,
  "max_boundary": 0.984375
}
```

#### **Integration Assessment:**

**❌ No Direct Integration Found**
- Constants appear to be **manually set** based on domain expertise
- No evidence of automatic updating from experiment results
- Boundary detection results stored separately in JSON files
- No code that reads `results_frankenstein_boundaries.json` to update constants

**⚠️ Potential Integration Gap**
- Rich boundary data from experiments not used to calibrate core system
- Manual constants may not reflect empirical findings
- Opportunity for automated constant tuning based on experimental results

---

## 🎯 **Key Findings**

### **1. Synthetic Pattern Introduction**
**Answer: YES** - Frankenstein mixer introduces synthetic patterns:

**Unrealistic Aspects:**
- Linear contamination gradients (real corruption is abrupt)
- Perfect binary search boundaries (real boundaries are fuzzy)
- Controlled corruption types (real corruption is chaotic)
- Single-dimension analysis (real corruption is multi-factor)

**Risk to Physics Engine:**
- Core engine may be over-optimized for these synthetic patterns
- Real-world edge cases may be missed
- False confidence in boundary detection

### **2. Constants Integration Gap**
**Answer: NO** - Benchmark results not automatically integrated:

**Current State:**
- Constants manually set based on domain expertise
- Rich experimental data stored separately
- No feedback loop from experiments to core constants
- Missed opportunity for data-driven parameter tuning

**Recommendation:**
- Implement automated constant updating from experimental results
- Use boundary detection to set dynamic thresholds
- Create feedback loop between experiments and core system

### **3. Experiment-Model Relationship**
**Level 5 Models Depend On:**
- Physics-certified data quality
- Corruption analysis insights for data cleaning
- Boundary detection for robust training
- Quality floor guarantees for reliable training

**Current Integration:**
- Level 5 models use certified data as input
- No direct integration with corruption analysis findings
- Missed opportunity for corruption-aware training

---

## 🛠️ **Recommendations**

### **1. Improve Realism of Synthetic Corruption**
- Add non-linear corruption patterns
- Implement multi-factor corruption combinations
- Use real-world corruption signatures
- Add temporal dynamics to corruption

### **2. Integrate Experimental Results**
- Create automated constant updating from boundary detection
- Implement feedback loop from experiments to core physics
- Use empirical data to set dynamic thresholds
- Add continuous learning from experimental results

### **3. Enhance Level 5 Training**
- Incorporate corruption-aware training strategies
- Use boundary detection for data augmentation
- Implement robust training with synthetic corruption
- Add quality-aware loss functions

### **4. Bridge Experiment-Model Gap**
- Create pipeline from corruption analysis to model training
- Use boundary detection for robust model design
- Implement experimental result integration in training pipeline
- Add continuous evaluation against synthetic patterns

---

**Conclusion:** The experiments introduce valuable synthetic patterns for testing but lack integration with the core system. Level 5 models would benefit from corruption analysis insights, and constants should be updated based on experimental findings rather than manual expertise alone.
