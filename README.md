# S2S — Physics Certification for Motion Data

[![PyPI](https://img.shields.io/pypi/v/s2s-certify)](https://pypi.org/project/s2s-certify/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18878307.svg)](https://doi.org/10.5281/zenodo.18878307)
[![Tests](https://img.shields.io/badge/tests-110%20passing-brightgreen.svg)](https://github.com/timbo4u1/S2S/actions)
[![S2S CI](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml/badge.svg)](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml)
[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](README.md)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](README.md)
```bash
pip install s2s-certify   # zero dependencies
```

Most motion datasets contain data that violates Newton's laws.
S2S finds it — before your robot trains on it.

It applies 11 physics equations to every sensor window: F=ma coupling,
rigid-body kinematics, jerk bounds, Hurst persistence. No ML. No training.
Equations that have been true since 1687.

**Proven on 5 independent datasets. 10,352 windows certified. 110 tests passing.**

---

## It improves model performance

| Dataset | Hz | Result |
|---------|-----|--------|
| UCI HAR | 50Hz | **+2.51% F1** vs corrupted baseline |
| PAMAP2 | 100Hz | **+4.23% F1** kinematic chain vs single sensor |
| WISDM 2019 | 20Hz | **+1.74% F1** vs corrupted baseline |

Less data. Higher quality. Better model. Every time.

---

## Try it now — no install

| | |
|---|---|
| 📊 [Interactive Data Explorer](https://timbo4u1.github.io/S2S/viz.html) | 104,160 certified records |
| 📱 [Phone IMU Demo](https://timbo4u1.github.io/S2S) | Real-time from your phone |
| 🎥 [Pose Camera Demo](https://timbo4u1.github.io/S2S/pose.html) | 17-joint live via webcam |

---

## What Is This

IMU and EMG signals from real humans obey physics. Synthetic data, corrupted recordings, and robotic motion do not. S2S checks eleven physics laws — jerk bounds, segment resonance, rigid-body kinematics, Hurst exponent persistence — and tells you whether your signal is biologically plausible before any model trains on it.

It is not a classifier. It does not learn. It applies equations that have been true since Newton.

**Proven on five independent datasets across five sampling rates (20Hz → 2000Hz). 110 tests passing.**

---

---

## Results

### Does it improve model performance?

Yes — across every dataset tested.

| Dataset | Hz | Improvement | Condition |
|---------|-----|-------------|-----------|
| UCI HAR | 50Hz | **+2.51% F1** | vs corrupted baseline |
| PAMAP2 | 100Hz | **+4.23% F1** | kinematic chain vs single sensor |
| WISDM 2019 | 20Hz | **+1.74% F1** | vs corrupted baseline |

Less data, higher quality, better model. Physics score is a reliable proxy for training value.

### Does it detect biological origin?

Yes — proven in both directions.

**Positive (real humans):** 10 NinaPro DB5 subjects, 2000Hz forearm EMG+ACC → all graded **HUMAN**. Hurst exponent H = 0.60–0.79 across all subjects.

**Negative (synthetic signals):** Pure 50Hz sine (robot motion), white noise (sensor fault), step function (bang-bang control) → all graded **NOT_BIOLOGICAL**. Hurst H < 0.55 in every case.

The separator: **Hurst exponent H < 0.7 = not biological.** Human motor control produces long-range correlated movement. Machines do not — unless specifically engineered to replicate it.

### Does it certify biological signals?

| Signal | Result |
|--------|--------|
| PPG pass rate | **96.3%** on real subjects |
| Heart rate | mean **106 BPM** (correct for activity) |
| HRV RMSSD | mean **21ms** (real human variability) |
| EMG→accel lag | **117.5ms** mean (neuromuscular literature: 50–200ms) |

Real pulse, real HRV, real lag. Synthetic data cannot fake all three simultaneously.

---

## Biological Origin Detection (v1.5.0)

`certify_session()` answers: did a human produce this session?
```python
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

pe = PhysicsEngine()

# Process windows
for window in session_windows:
    pe.certify(imu_raw={"timestamps_ns": ts, "accel": window}, segment="forearm")

# Session-level verdict
result = pe.certify_session()

print(result["biological_grade"])           # HUMAN / NOT_BIOLOGICAL / LOW_BIOLOGICAL_FIDELITY
print(result["recommendation"])             # ACCEPT / REVIEW / REJECT
print(result["biological_diversity_score"]) # 0-100 within human reference range
print(result["hurst"])                      # Hurst exponent — primary origin detector
```

**Grade thresholds:**

| Grade | Hurst | BFS | Meaning |
|-------|-------|-----|---------|
| `NOT_BIOLOGICAL` | H < 0.70 | any | Synthetic, robotic, or corrupted |
| `LOW_BIOLOGICAL_FIDELITY` | H ≥ 0.70 | BFS < 0.35 | Marginal — review before training |
| `HUMAN` | H ≥ 0.70 | BFS ≥ 0.35 | Confirmed biological origin |
| `SUPERHUMAN` | H ≥ 0.70 | BFS > 0.70 | Valid — prosthetics / enhanced systems |

**Important:** BFS is a floor detector, not a quality ranking. Higher BFS means higher signal diversity — not a better subject. The Hurst floor is the actual biological gate.

---

## Quick Start
```python
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

pe = PhysicsEngine()
result = pe.certify(
    imu_raw={"timestamps_ns": timestamps, "accel": accel_window},
    segment="forearm"
)

print(result["tier"])               # GOLD / SILVER / BRONZE / REJECTED
print(result["physical_law_score"]) # 0-100
print(result["laws_passed"])        # which physics laws passed
print(result["laws_failed"])        # which failed and why
```
```bash
# Live API — no install needed
curl -X POST https://s2s-65sy.onrender.com/certify \
  -H "Content-Type: application/json" \
  -d '{"accel": [[ax,ay,az],...], "sample_rate_hz": 50}'
```

---

## Tier System

| Tier | Score | Meaning |
|------|-------|---------|
| **GOLD** | ≥ 87 | All physics laws passed. Train on this. |
| **SILVER** | 75–86 | Trusted. Minor deviations within noise bounds. |
| **BRONZE** | 60–74 | Marginal. Usable; flag for review. |
| **REJECTED** | < floor | Removed. Physics violated. |

Floor = p25 of clean score distribution per dataset (adaptive per Hz).

---

## Eleven Physics Laws

### Per-Window (single sensor)

| # | Law | Catches |
|---|-----|---------|
| 1 | Newton F=ma — EMG→accel lag 117.5ms | Synthetic data without neuromuscular delay |
| 2 | Segment resonance ω=√(K/I) | Tremor at impossible frequency for body segment |
| 3 | Rigid-body kinematics a=α×r+ω²×r | Gyro and accel generated independently |
| 4 | Ballistocardiography F=ρQv | IMU missing cardiac recoil |
| 5 | Joule heating Q=0.75×P×t | Sustained EMG without thermal rise |
| 6 | Motor control jerk ∂³x/∂t³ ≤ 500 m/s³ | Robotic motion, keyframe artefacts |
| 7 | IMU consistency Var(accel) ~ Var(gyro) | Accel and gyro from separate generators |

### Session-Level (biological origin)

| # | Law | Catches |
|---|-----|---------|
| 8 | Hurst exponent H > 0.7 | Non-biological signal (synthetic/robotic) |
| 9 | Locomotion coherence — freq spread < 2.5Hz | Sensors recording different activities |
| 10 | Segment coupling — chest-ankle r > 0.3 | Independent synthetic channels |
| 11 | Cross-sensor jerk timing — ankle leads 0–200ms | Reversed or zero lag, not real heel-strike |

---

## Auto-Hz Detection

S2S reads sampling rate from timestamp intervals and signal amplitude range — no configuration needed.

| Hz | Signal range | Profile | Device |
|----|-------------|---------|--------|
| ≥ 400Hz | < 1.0 normalized | normalized_500hz | PTT-PPG wrist |
| ≤ 150Hz | > 10 raw ADC | raw_adc_100hz | PAMAP2 |
| other | other | default | fallback |

Before auto-Hz: PAMAP2 Level 4 HIL = 38.4. After: **65.3**. Same data, correct profile.

---

## Active Learning Pipeline

| Module | What it does | Result |
|--------|-------------|--------|
| **Corruption Fingerprinter** | Identifies which laws break first under corruption | Resonance breaks first (77%) |
| **Frankenstein Mixer** | Finds exact contamination boundaries per law | IMU consistency breaks at 30.6% |
| **Curriculum Generator** | Balanced training data at every quality level | 2,000 samples, auto-discovery |
| **Cloud Trainer** | Trains quality prediction on curriculum | 85.5% accuracy, +27.7% over baseline |

---

## Datasets

| Dataset | Hz | Sensors | Windows | Levels validated |
|---------|-----|---------|---------|-----------------|
| WISDM 2019 | 20Hz | Wrist accel | 46,946 | 1, 2 |
| PAMAP2 | 100Hz | Hand+Chest+Ankle IMU | 13,094 | 1, 2, 4 |
| UCI HAR | 50Hz | Body accel+gyro | 10,299 | 1, 2 |
| PhysioNet PTT-PPG | 500Hz | Wrist PPG+IMU+Thermal | 1,164 | 2, 3, 4, 5 |
| NinaPro DB5 | 2000Hz | Forearm EMG+Accelerometer | 1,470+ | BFS, origin detection |

---

## Hybrid AI (Experimental)

| Model | Accuracy | Features |
|-------|----------|---------|
| Raw IMU | 79.55% | 768 |
| Physics only | 70.48% | 19 |
| **Hybrid** | **83.68%** | 787 |

2 of 19 physics features contribute meaningfully. The other 17 need better feature engineering. Honest assessment: promising, not finished.

---

## Paper

**S2S: Physics-Certified Sensor Data — Four Proven Levels, Eleven Laws, Five Independent Datasets**

[→ Read paper (PDF)](docs/paper/S2S_Paper_v5.pdf) · [→ DOI: 10.5281/zenodo.18878307](https://doi.org/10.5281/zenodo.18878307)

---

## Project Structure
```
s2s_standard_v1_3/    # Physics engine — zero dependencies, pure Python
experiments/          # Proof scripts + results JSON
tests/                # 110 tests, all passing
docs/paper/           # S2S_Paper_v5.pdf
```

---

## License

BSL-1.1 — free for research and non-commercial use. Contact for commercial licensing.
