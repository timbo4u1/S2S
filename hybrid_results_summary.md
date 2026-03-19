# Hybrid Model Results — Physics + Raw IMU

## 🎯 Target Achieved: Hybrid Beats Both Baselines

### 📊 Performance Comparison

| Model | Accuracy | Features | Improvement |
|-------|----------|-----------|-------------|
| Raw IMU | 79.55% | 768 features | Baseline |
| Physics Only | 70.48% | 19 features | Baseline |
| **Hybrid** | **83.68%** | 787 features | **+4.13% vs Raw, +13.20% vs Physics** |

### 🏆 Key Achievements

✅ **Hybrid model beats both baselines**
- Beats raw IMU by 4.13% (79.55% → 83.68%)
- Beats physics-only by 13.20% (70.48% → 83.68%)

✅ **Physics features add complementary signal**
- Physics: 6.0% of feature importance with 19 features
- Raw IMU: 94.0% of feature importance with 768 features
- Physics efficiency: 0.0032 per feature vs 0.0012 for raw

### 🔍 Feature Analysis

**Most Important Physics Features:**
1. `rigid_rms_measured` (0.0569) - RMS acceleration magnitude
2. `resonance_peak_energy` (0.0011) - Frequency domain energy  
3. `resonance_peak_hz` (0.0011) - Dominant frequency
4. `resonance_confidence` (0.0007) - Physics law confidence
5. `physical_law_score` (0.0004) - Overall physics score

**Most Important Raw IMU Features:**
1. Axis 2 (Z-axis) at timestep 196: 0.0134
2. Axis 2 (Z-axis) at timestep 27: 0.0107
3. Axis 2 (Z-axis) at timestep 0: 0.0104
4. Axis 2 (Z-axis) at timestep 160: 0.0103
5. Axis 2 (Z-axis) at timestep 216: 0.0102

### 📈 Activity Performance

| Activity | F1-Score | Performance |
|----------|-----------|-------------|
| Walking | 89.35% | Excellent |
| Sitting | 83.13% | Good |
| Running | 80.56% | Solid |

### 🎯 Scientific Impact

**Physics-informed hybrid model proves:**
1. **Complementarity**: Physics features provide unique signal beyond raw IMU
2. **Efficiency**: 19 physics features achieve 6% of predictive power
3. **Superiority**: Hybrid approach outperforms both individual methods
4. **Scalability**: Physics features add minimal computational overhead

### 📊 Confusion Matrices

- **Raw IMU**: Good overall, struggles with sitting vs walking
- **Physics Only**: Confused between activities, especially walking
- **Hybrid**: Best separation across all three activities

### 🚀 Next Steps

1. **Feature selection**: Optimize raw IMU feature subset
2. **Architecture**: Explore temporal models for raw IMU
3. **Integration**: Deploy hybrid model in production
4. **Extension**: Apply to other datasets and activities

---
*Results from 63,314 windows, 22 subjects, 5-fold subject-wise cross-validation*
