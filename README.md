# S2S — Physics-Certified Sensor Data

**Physics-certified motion data for prosthetics, robotics, and Physical AI.**

S2S is a physics validation layer for human motion sensor data. Before training a prosthetic hand, surgical robot, or humanoid — run your IMU data through S2S. It verifies the data obeys 11 biomechanical laws and issues a certificate. Bad data gets rejected before it reaches your model.

[![PyPI](https://img.shields.io/badge/pypi-v1.5.0-orange)](https://pypi.org/project/s2s-certify/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18878307.svg)](https://doi.org/10.5281/zenodo.18878307) [![S2S CI](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml/badge.svg)](https://github.com/timbo4u1/S2S/actions) [![License](https://img.shields.io/badge/License-BSL--1.1-blue)](LICENSE) [![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml) [![dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml)

---

## Live Demos

[→ IMU Demo — open on your phone](https://timbo4u1.github.io/S2S/) · Real-time certification using phone accelerometer + gyroscope

[→ Pose Demo — camera + skeleton](https://timbo4u1.github.io/S2S/) · 17-joint body tracking with live physics certification

[→ Physical AI Demo — RoboTurk](docs/demo_physical_ai.html) · Robot arm learning from 686MB of certified teleoperation data

No install needed. All processing runs on your device. No data sent anywhere.

---

## The Problem

Physical AI (robots, prosthetics, exoskeletons) is trained on motion data. But most datasets contain synthetic data that violates physics, corrupted recordings, and mislabeled actions — with no way to verify the data came from a real human moving in physically valid ways.

A robot trained on bad data learns bad motion. A prosthetic hand trained on uncertified data fails its user.

---

## Four Proven Training Benefits

S2S is not just a filter. It improves model performance at every stage of the training pipeline. All results validated across two independent datasets (WISDM 20Hz, PAMAP2 100Hz).

### Level 1 — Quality Floor

Remove data that fails physics before training.

| Dataset | Corruption | Recovery | Result |
|---|---|---|---|
| WISDM 20Hz | 35% corrupted | 108% of damage recovered | +0.23% net vs clean |
| PAMAP2 100Hz | 35% corrupted | Confirmed cross-dataset | +0.51% F1 |

Physics floor removes bad data and beats the clean baseline with 46% less data.

### Level 2 — Curriculum Training

Train in physics quality order: GOLD → SILVER → BRONZE.

| Dataset | Result |
|---|---|
| WISDM 20Hz | +1.03% F1 vs clean baseline, 46% less data |

The model learns the ceiling first. Marginal data is introduced only after the model understands perfect motion.

### Level 3 — Adaptive Reconstruction

Repair marginal (BRONZE) records using frequency-appropriate methods. Every repaired record carries full provenance.

| Hz | Method | Result |
|---|---|---|
| ≤50Hz (20Hz WISDM) | Kalman RTS smoother | +1.44% F1 |
| ≥100Hz (PAMAP2) | Savitzky-Golay | Spectral sim=0.997 — signal needs no repair |

Dual acceptance: physics re-score ≥75 AND spectral similarity ≥0.8. Both must pass.

At low Hz, noise is separable from signal — Kalman removes it. At high Hz, every micro-movement is real — smoothing destroys features. S2S adapts automatically.

### Level 4 — Kinematic Chain Consistency (headline result)

Verify that multiple sensors tell a consistent biomechanical story.

| Condition | F1 | Δ |
|---|---|---|
| Single chest IMU | 0.7969 | baseline |
| 3 IMUs naive concat | 0.8308 | +3.39% |
| 3 IMUs + chain filter | 0.8399 | +0.91% over naive |
| **Net vs single sensor** | **+4.23% F1** | **← headline** |

Tested on PAMAP2 12-class activity recognition (hand + chest + ankle IMU, 100Hz).

Why this catches synthetic data: Synthetic motion generators produce each sensor channel independently. Real walking produces a 50–100ms ankle-to-chest jerk lag from heel-strike propagating up the skeleton. This timing cannot be faked without full rigid-body simulation. S2S Level 4 is the first physics-based cross-sensor consistency check for IMU data.

---

## 🤖 Physical AI — Robot Learning from Certified Data

**[→ Live Demo: demo_physical_ai.html](docs/demo_physical_ai.html)**

686MB of real RoboTurk teleoperation data certified and used to train a physical AI model. Three real commands learned from 260 episodes, 23,125 steps:

| Command | Episodes | Windows | GOLD | SILVER | BRONZE | REJ | Avg Score |
|---|---|---|---|---|---|---|---|
| object search | 79 | 454 | 102 | 237 | 112 | 3 | 64.8 |
| create tower | 54 | 401 | 117 | 206 | 75 | 3 | 65.9 |
| layout laundry | 83 | 288 | 65 | 166 | 55 | 2 | 65.4 |

All numbers verified from `roboturk_audit.json`. Top failing law: `imu_internal_consistency` — translation and rotation channels have different teleoperation latency. S2S detects the decoupling before the model trains.

**Level 5 Physical AI training (corrected gyro — rotation_delta double-diff):**

| Metric | Start | End |
|---|---|---|
| Quality accuracy (3-class) | 0.655 | 0.677 |
| Smoothness correlation | 0.916 | 0.941 |
| Training cycles | — | 20 |
| Certified pairs | — | 4,145 |

*Previous training used zero gyro — fixed 2026-03-13. Gyro method now matches `certify_roboturk.py` exactly.*

---

## 11 Physics Laws

### Single-Sensor Laws (Levels 1–3)

| # | Law | What It Catches |
|---|---|---|
| 1 | Newton's Second Law (F=ma, 75ms EMG delay) | Synthetic data missing lagged EMG-accel correlation |
| 2 | Segment Resonance (ω=√(K/I)) | Tremor at impossible frequency for body segment |
| 3 | Rigid Body Kinematics (a=α×r+ω²×r) | Gyro and accel generated independently |
| 4 | Ballistocardiography (F=ρQv) | IMU missing cardiac recoil |
| 5 | Joule Heating (Q=0.75×P×t) | Sustained EMG without thermal elevation |
| 6 | Motor Control Jerk (∂³x/∂t³ ≤ 5000 m/s³) | Robotic or keyframe animation artefacts |
| 7 | IMU Consistency (Var(accel) ~ f(Var(gyro))) | Accel and gyro from independent generators |

### Multi-Sensor Chain Laws (Level 4)

| # | Law | What It Catches |
|---|---|---|
| 8 | Locomotion Coherence (freq spread <2.5Hz) | Sensors recording different activities |
| 9 | Segment Coupling (chest-ankle r >0.3) | Independent synthetic channels |
| 10 | Gyro-Accel Coupling (per IMU) | Rotation without corresponding acceleration |
| 11 | Cross-Sensor Jerk Timing (ankle leads chest 0–200ms) | Reversed or zero lag — not real heel-strike |

---

## Tier System

| Tier | Score | Meaning |
|---|---|---|
| GOLD | ≥87 | All physics laws passed. Pristine. |
| SILVER | 75–86 | Trusted. Minor deviations within noise. |
| BRONZE | 60–74 | Marginal. Candidate for reconstruction at ≤50Hz. |
| RECONSTRUCTED | — | Repaired, re-scored ≥75, spectral sim ≥0.8. Weight 0.5. |
| REJECTED | <floor | Removed from pipeline. |

Floor = p25 of clean score distribution per dataset (adaptive). GOLD always means the same thing everywhere.

---

## Install

Zero dependencies. Pure Python 3.9+. Works on any platform.

```bash
pip install s2s-certify
```

## Quick Start

```python
from s2s_certify import certify

result = certify(accel_window, sample_rate_hz=20)
print(result['tier'])         # GOLD / SILVER / BRONZE / REJECTED
print(result['score'])        # 0–100
print(result['laws_passed'])  # which of 7 single-sensor laws passed
```

```python
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

pe = PhysicsEngine()
result = pe.certify(
    imu_raw={"timestamps_ns": timestamps, "accel": accel_window},
    segment="forearm"
)
print(result["tier"])               # GOLD / SILVER / BRONZE / REJECTED
print(result["physical_law_score"]) # 0-100
print(result["laws_passed"])
print(result["laws_failed"])
```

```bash
# Live API — no install needed
curl -X POST https://s2s-65sy.onrender.com/certify \
  -H "Content-Type: application/json" \
  -d '{"accel": [[ax,ay,az],...], "sample_rate_hz": 50}'
```

## Biological Origin Detection (v1.5.0)

```python
for window in session_windows:
    pe.certify(imu_raw={"timestamps_ns": ts, "accel": window}, segment="forearm")

result = pe.certify_session()
print(result["biological_grade"])           # HUMAN / NOT_BIOLOGICAL / LOW_BIOLOGICAL_FIDELITY
print(result["recommendation"])             # ACCEPT / REVIEW / REJECT
print(result["biological_diversity_score"]) # 0-100
print(result["hurst"])                      # Hurst exponent — primary origin detector
```

---

## Datasets Validated

| Dataset | Hz | Sensors | Windows | Used for |
|---|---|---|---|---|
| WISDM 2019 | 20Hz | Wrist accel | 46,946 | Levels 1, 2, 3 |
| PAMAP2 | 100Hz | Hand+Chest+Ankle IMU | 13,094 | Levels 1, 3, 4 |
| UCI HAR | 50Hz | Body accel+gyro | 10,299 | Levels 1, 2 |
| PhysioNet PTT-PPG | 500Hz | Wrist PPG+IMU+Thermal | 1,164 | Levels 2, 3, 4, 5 |
| NinaPro DB5 | 2000Hz | Forearm EMG+Accelerometer | 9,552 | BFS, origin detection |
| RoboTurk (Open-X) | 15Hz | 7-DOF end-effector | 901 | Physics audit, Level 5 Physical AI |

---

## Paper

**S2S: Physics-Certified Sensor Data — Four Proven Training Benefits Across Two Independent Datasets**

[→ Read paper (PDF)](docs/paper/S2S_Paper_v5.pdf) · [→ DOI: 10.5281/zenodo.18878307](https://doi.org/10.5281/zenodo.18878307)

---

## Project Structure

```
s2s_standard_v1_3/          # Physics engine — zero dependencies, pure Python
experiments/                # All level experiments + results JSON
  level5_corrected_best.pt          # Level 5 model (corrected gyro)
  results_level5_corrected.json     # Smoothness 0.941, quality 0.677
  real_trajectories.json            # Real RoboTurk XYZ positions per command
  roboturk_audit.json               # Full 260-episode audit
  command_motion_map.json           # Command → predicted motion vectors
  results_level4_pamap2.json        # +4.23% chain result
  results_level3_adaptive_wisdm.json # +1.44% Kalman result
docs/
  demo_physical_ai.html             # Physical AI demo — real RoboTurk data
  paper/S2S_Paper_v5.pdf
dashboard/app.py            # Streamlit human review UI
tests/                      # 110 tests, all passing
```

---

## License

BSL-1.1 — free for research and non-commercial use. Contact for commercial licensing.

Apache 2.0 from 2028-01-01.
