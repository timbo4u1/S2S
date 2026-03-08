# S2S — Physics-Certified Sensor Data

**Physics-certified motion data for prosthetics, robotics, and Physical AI.**

S2S is a physics validation layer for human motion sensor data. Before training a prosthetic hand, surgical robot, or humanoid — run your IMU data through S2S. It verifies the data obeys 11 biomechanical laws and issues a certificate. Bad data gets rejected before it reaches your model.

[![PyPI](https://img.shields.io/pypi/v/s2s-certify)](https://pypi.org/project/s2s-certify/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18878307.svg)](https://doi.org/10.5281/zenodo.18878307)
[![Tests](https://img.shields.io/badge/tests-110%20passing-brightgreen.svg)](https://github.com/timbo4u1/S2S/actions)
[![S2S CI](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml/badge.svg)](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml)
[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](README.md)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](README.md)

---

## Live Demos
- 📊 [Interactive Data Explorer](https://timbo4u1.github.io/S2S/viz.html) — 104,160 real certified records, hover to explore
- 📱 [Phone IMU Demo](https://timbo4u1.github.io/S2S) — real-time physics certification on your phone
- 🎥 [Pose Camera Demo](https://timbo4u1.github.io/S2S/pose.html) — 17-joint live certification

No install needed. All processing runs on your device. No data sent anywhere.

---

## The Problem

Physical AI (robots, prosthetics, exoskeletons) is trained on motion data. But most datasets contain synthetic data that violates physics, corrupted recordings, and mislabeled actions — with no way to verify the data came from a real human moving in physically valid ways.

A robot trained on bad data learns bad motion. A prosthetic hand trained on uncertified data fails its user.

---

## Four Proven Training Benefits

S2S improves model performance at every stage of the training pipeline. All results validated across **three independent datasets** at three different sampling rates.

### Level 1 — Quality Floor ✅ PROVEN on 3 datasets

| Dataset | Hz | Corruption | S2S Recovery | Net vs Clean |
|---------|-----|-----------|-------------|--------------|
| WISDM 2019 | 20Hz | 35% corrupted | 154% recovered | **+1.74% F1** |
| PAMAP2 | 100Hz | 35% corrupted | confirmed | **+0.95% F1** |
| UCI HAR | 50Hz | 35% corrupted | 135% recovered | **+2.51% F1** |

> Physics floor removes bad data and **beats the clean baseline across three independent datasets at three different sampling rates.**

### Level 2 — Physics Quality Floor Generalises ✅ PROVEN on 3 datasets

| Dataset | Hz | Data used | vs All data |
|---------|-----|----------|-------------|
| WISDM 2019 | 20Hz | 41% of windows | **+1.74% F1** |
| PAMAP2 | 100Hz | 88% of windows | **+0.95% F1** |
| UCI HAR | 50Hz | 49% of windows | **+2.51% F1** |

Also proven: kinematic chain consistency on PAMAP2 (hand + chest + ankle IMU):

| Condition | F1 | Δ |
|-----------|-----|---|
| Single chest IMU | 0.7969 | baseline |
| 3 IMUs naive concat | 0.8308 | +3.39% |
| 3 IMUs + chain filter | 0.8399 | +0.91% over naive |
| Net vs single sensor | **+4.23% F1** | ← headline |

> Less data, higher quality, better model. The physics score is a reliable proxy for training value — confirmed across devices, sampling rates, and activity types.

### Level 3 — Biological Signal Certification ✅ PROVEN

Tested on PhysioNet PTT-PPG — 4 real subjects, 1164 windows, 500Hz wrist device, walk/sit/run.

| Signal | Result |
|--------|--------|
| PPG pass rate | **96.3%** on real human subjects |
| Heart rate | mean **106 BPM** (physiologically correct for activity) |
| HRV RMSSD | mean **21ms** (real human variability) |
| Skin temperature | **33.6°C** (confirmed real human range) |

> Real pulse, real HRV, real temperature — verified simultaneously. Synthetic data cannot fake all three.

### Level 4 — Multi-Sensor Fusion Coherence ✅ PROVEN

| Dataset | HIL Score | Pass Rate | Tiers |
|---------|-----------|-----------|-------|
| PTT-PPG 500Hz wrist | **68.7/100** | 100% | 438 SILVER + 726 BRONZE |
| PAMAP2 100Hz (auto-Hz) | **65.3/100** | 100% | 87 SILVER + 13 BRONZE |

Real sensors: PPG infrared + PPG red + IMU accel+gyro + skin temperature — all from the same wrist hardware.

> If HR rose with activity, skin temperature stayed in human range, and IMU timing matched PPG — simultaneously — a human was there.

---

## Auto-Hz Device Detection

S2S automatically detects device profile from two numbers already in the data — sampling Hz (from median timestamp intervals) and signal amplitude range (from first window). No user configuration needed.

| Hz range | Signal range | Profile | Example |
|----------|-------------|---------|---------|
| ≥400Hz | <1.0 normalized | normalized_500hz | PTT-PPG |
| ≤150Hz | >10 raw ADC | raw_adc_100hz | PAMAP2 |
| other | other | default | fallback |

Before auto-Hz: PAMAP2 Level 4 HIL = 38.4. After: **65.3**. Same data, correct profile.

---

## Validated on Real Human Data

**WISDM 2019** (51 subjects, 20Hz, wrist accel, 18 activities):

| Level | Result |
|-------|--------|
| Level 1 | **+1.74% F1** vs corrupted, 154% recovery |
| Level 2 | **+1.74% F1** vs all data, 41% of windows used |

**PAMAP2** (9 subjects, 100Hz, hand+chest+ankle IMU, 12 activities):

| Level | Result |
|-------|--------|
| Level 1 | **+0.95% F1** vs corrupted |
| Level 2 | **+4.23% F1** kinematic chain vs single sensor |
| Level 4 | HIL **65.3/100**, 100% pass, 87 SILVER |

**UCI HAR** (30 subjects, 50Hz, body accel+gyro, 6 activities):

| Level | Result |
|-------|--------|
| Level 1 | **+2.51% F1** vs corrupted, 135% recovery |
| Level 2 | **+2.51% F1** vs all data, 49% of windows used |

**PhysioNet PTT-PPG** (4 subjects, 500Hz, wrist PPG+IMU+thermal, walk/sit/run):

| Level | Result |
|-------|--------|
| Level 2 IMU | 61.7% pass rate, avg score 37.2/100 |
| Level 3 PPG | **96.3% pass rate**, HR 106 BPM, HRV 21ms |
| Level 4 Fusion | HIL **68.7/100**, 100% pass, 438 SILVER |

---

## 11 Physics Laws

### Single-Sensor Laws (Levels 1–3)

| # | Law | What It Catches |
|---|-----|-----------------|
| 1 | Newton's Second Law (F=ma, 75ms EMG delay) | Synthetic data missing lagged EMG-accel correlation |
| 2 | Segment Resonance (ω=√(K/I)) | Tremor at impossible frequency for body segment |
| 3 | Rigid Body Kinematics (a=α×r+ω²×r) | Gyro and accel generated independently |
| 4 | Ballistocardiography (F=ρQv) | IMU missing cardiac recoil |
| 5 | Joule Heating (Q=0.75×P×t) | Sustained EMG without thermal elevation |
| 6 | Motor Control Jerk (∂³x/∂t³ ≤ 5000 m/s³) | Robotic or keyframe animation artefacts |
| 7 | IMU Consistency (Var(accel) ~ f(Var(gyro))) | Accel and gyro from independent generators |

### Multi-Sensor Chain Laws (Level 4)

| # | Law | What It Catches |
|---|-----|-----------------|
| 8 | Locomotion Coherence (freq spread <2.5Hz) | Sensors recording different activities |
| 9 | Segment Coupling (chest-ankle r >0.3) | Independent synthetic channels |
| 10 | Gyro-Accel Coupling (per IMU) | Rotation without corresponding acceleration |
| 11 | Cross-Sensor Jerk Timing (ankle leads chest 0–200ms) | Reversed or zero lag — not real heel-strike |

---

## Tier System

| Tier | Score | Meaning |
|------|-------|---------|
| GOLD | ≥87 | All physics laws passed. Pristine. |
| SILVER | 75–86 | Trusted. Minor deviations within noise. |
| BRONZE | 60–74 | Marginal. Candidate for reconstruction at ≤50Hz. |
| RECONSTRUCTED | — | Repaired, re-scored ≥75, spectral sim ≥0.8. Weight 0.5. |
| REJECTED | <floor | Removed from pipeline. |

Floor = p25 of clean score distribution per dataset (adaptive).

---

## Install
```bash
pip install s2s-certify
```

Zero dependencies. Pure Python 3.9+. Works on any platform.

---

## Quick Start
```python
from s2s_certify import certify

result = certify(accel_window, sample_rate_hz=20)

print(result['tier'])        # GOLD / SILVER / BRONZE / REJECTED
print(result['score'])       # 0–100
print(result['laws_passed']) # which physics laws passed
```
```bash
s2s-certify your_imu_data.csv
s2s-certify your_imu_data.csv --output report.json
```

---

## Datasets Validated

| Dataset | Hz | Sensors | Windows | Used for |
|---------|-----|---------|---------|----------|
| WISDM 2019 | 20Hz | Wrist accel | 46,946 | Levels 1, 2 |
| PAMAP2 | 100Hz | Hand+Chest+Ankle IMU | 13,094 | Levels 1, 2, 4 |
| UCI HAR | 50Hz | Body accel+gyro | 10,299 | Levels 1, 2 |
| PhysioNet PTT-PPG | 500Hz | Wrist PPG+IMU+Thermal | 1,164 | Levels 2, 3, 4 |

---

## Paper

**S2S: Physics-Certified Sensor Data — Four Proven Training Benefits Across Three Independent Datasets**

[→ Read paper (PDF)](docs/paper/S2S_Paper_v5.pdf) | [→ DOI: 10.5281/zenodo.18878307](https://doi.org/10.5281/zenodo.18878307)

---

## Project Structure
```
s2s_standard_v1_3/     # Physics engine (zero dependencies)
experiments/           # All experiments + results JSON
tests/                 # 110 tests, all passing
docs/paper/            # S2S_Paper_v5.pdf
```

---

## License

BSL-1.1 — free for research and non-commercial use. Contact for commercial licensing.
