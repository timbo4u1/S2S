# S2S — Physical Motion Certification

**Physics-certified motion data for prosthetics, robotics, and Physical AI.**

S2S is a physics validation layer for human motion sensor data. Before training a prosthetic hand, surgical robot, or humanoid — run your IMU data through S2S. It verifies the data obeys 7 biomechanical laws and issues a cryptographic certificate. Bad data gets rejected before it reaches your model.

[![Tests](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml/badge.svg)](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml)
[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](README.md)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](README.md)

---

## Live Demos

**[→ IMU Demo — open on your phone](https://timbo4u1.github.io/S2S)** · Real-time certification using phone accelerometer + gyroscope

**[→ Pose Demo — camera + skeleton](https://timbo4u1.github.io/S2S/pose.html)** · 17-joint body tracking with live physics certification

No install needed. All processing runs on your device. No data sent anywhere.

---

## The Problem

Physical AI (robots, prosthetics, exoskeletons) is trained on motion data. But most datasets contain synthetic data that violates physics, corrupted recordings, and mislabeled actions — with no way to verify the data came from a real human moving in physically valid ways.

A robot trained on bad data learns bad motion. A prosthetic hand trained on uncertified data fails its user.

## The Solution

S2S applies **7 biomechanical physics laws** to every sensor record:

| Law | Source | What it checks |
|-----|--------|----------------|
| Newton F=ma coupling | Newton 1687 | EMG force precedes acceleration by 75ms |
| Rigid body kinematics | Euler | a = αxr + ωxωxr holds at every joint |
| Resonance frequency | Flash-Hogan 1985 | Forearm tremor is 8-12Hz |
| Jerk bounds | Flash-Hogan 1985 | Human motion <= 500 m/s3 |
| IMU consistency | Sensor physics | Accel and gyro from same chip must couple |
| BCG heartbeat | Starr 1939 | Heart stroke visible in wrist IMU |
| Joule heating | Ohm 1827 | EMG power matches thermal output |

Every passing record is **Ed25519 signed** — tamper-evident, machine-verifiable.

## Real Result (iPhone 11, real human hand)
```
Real human hand:  rigid_body r=0.35   imu_consistency r=0.35   SILVER 69/100
Synthetic data:   rigid_body r=-0.01  imu_consistency r=-0.01  BRONZE 53/100
```

S2S correctly distinguishes real human motion from synthetic using physics alone.

## Motion Domain Taxonomy

Based on Flash-Hogan 1985, Bernstein 1967, Fitts 1954, Wolpert 1998 (MOSAIC model):

| Domain | Jerk | Coupling r | Robot use |
|--------|------|------------|-----------|
| PRECISION | 80 m/s3 | 0.30 | Surgical robots, prosthetic hands, assembly |
| SOCIAL | 180 m/s3 | 0.15 | Service robots, HRI |
| LOCOMOTION | 300 m/s3 | 0.15 | Bipedal robots, prosthetic legs |
| DAILY_LIVING | 150 m/s3 | 0.20 | Home robots, elder care |
| SPORT | 500 m/s3 | 0.10 | Athletic training, motion analysis |

## Quick Start
```bash
git clone https://github.com/timbo4u1/S2S.git
cd S2S
python3 -c "from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine; print('OK')"
```

### Certify your IMU data
```python
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

result = PhysicsEngine().certify(
    imu_raw={
        "timestamps_ns": [...],
        "accel":         [...],
        "gyro":          [...],
    },
    segment="forearm"
)
print(result['physics_tier'])        # SILVER
print(result['physics_score'])       # 69
print(result['laws_passed'])
```

### Record with iPhone (free)
```bash
# 1. Install "Sensor Logger" app (free, iOS/Android)
# 2. Enable: AccelerometerUncalibrated + GyroscopeUncalibrated + Gravity
# 3. Record motion, export CSV, run:
python3 collect_action.py --accel AccelerometerUncalibrated.csv --gyro GyroscopeUncalibrated.csv --gravity Gravity.csv --label reach_forward --person your_name --out dataset/
```

### Build certified dataset from public data
```bash
# UCI HAR: 30 subjects, 6 activities, 50Hz (24MB)
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
unzip "UCI HAR Dataset.zip"
python3 s2s_dataset_adapter.py --dataset uci_har --input "UCI HAR Dataset/" --out s2s_dataset/

# PAMAP2: 9 subjects, 18 activities, 100Hz (500MB)
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
unzip PAMAP2_Dataset.zip
python3 s2s_dataset_adapter.py --dataset pamap2 --input PAMAP2_Dataset/ --out s2s_dataset/

# WISDM: 51 subjects, 18 activities, 20Hz (295MB)
curl -L -o wisdm-dataset.zip "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"
unzip wisdm-dataset.zip -d wisdm-dataset/
python3 wisdm_adapter.py --input wisdm-dataset/ --out s2s_dataset/ --device phone

# Train domain classifier
python3 train_classifier.py --dataset s2s_dataset/ --test
```

## Package Structure
```
s2s_standard_v1_3/
├── s2s_physics_v1_3.py          7 biomechanical physics laws
├── s2s_stream_certify_v1_3.py   Real-time IMU stream certification
├── s2s_emg_certify_v1_3.py      EMG certification
├── s2s_fusion_v1_3.py           5-sensor fusion + HIL score
├── s2s_signing_v1_3.py          Ed25519 signing + verification
├── s2s_api_v1_3.py              REST API server
└── ... (12 modules total)

collect_action.py                Record labeled actions from phone
s2s_dataset_adapter.py          Convert UCI HAR and PAMAP2
wisdm_adapter.py                Convert WISDM (51 subjects, 18 activities)
train_classifier.py             5-domain classifier (pure Python, zero deps)

docs/
├── index.html                   Live IMU demo (phone sensors)
└── pose.html                    Live pose demo (camera + skeleton)
```

## Certification Tiers

| Tier | Score | Meaning |
|------|-------|---------|
| GOLD | 90-100 | All laws pass — hardware-grade |
| SILVER | 60-89 | Core laws pass — suitable for training |
| BRONZE | 30-59 | Partial pass — use with caution |
| REJECTED | 0-29 | Physics violations detected |

## Supported Open Datasets

| Dataset | Subjects | Activities | Hz | Direct S2S |
|---------|----------|------------|-----|------------|
| UCI HAR | 30 | 6 | 50 | Yes |
| PAMAP2 | 9 | 18 | 100 | Yes |
| WISDM 2019 | 51 | 18 | 20 | Yes |
| Your iPhone | 1 | any | 100 | Yes |

## Use Cases

**Prosthetics** — Certify EMG training data for myoelectric hands. Uncertified data = hand that does not respond correctly.

**Surgical robots** — PRECISION domain ensures training data meets surgical accuracy standards.

**Humanoid robots** — Certify teleoperation recordings before training whole-body motion policies.

**Exoskeletons** — Verify force-generation training data before deployment.

**Research** — Use physics score as training loss: L = L_task + lambda * (1 - physics_score/100)

## License

**Business Source License 1.1 (BSL-1.1)**

- Free: research, education, personal projects
- Free: study, modify, contribute
- Commercial use requires a license from the author
- Converts to Apache 2.0 automatically on 2028-01-01

Commercial licensing: timur.davkarayev@gmail.com

## Status

v1.4 — 12 modules, zero dependencies, production ready.

**Domain Classifier:** 76.6% accuracy (5-fold CV, FINE_MOTOR mode), trained on 103,352 certified records from UCI HAR + PAMAP2 + WISDM across 5 domains (LOCOMOTION, DAILY_LIVING, PRECISION, SOCIAL, SPORT).

Validated on real iPhone 11 IMU data. SILVER certified, 4/4 physics laws passing.

**Preprint:** [hal-05531246v1](https://hal.science/hal-05531246v1) — HAL Open Science, February 28, 2026

Open to hardware partnerships and research collaborations.

---

*If you use S2S in research, please cite this repository.*
