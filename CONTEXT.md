# S2S — Project Context File
> **For AI assistants & new collaborators:** Paste this file at the start of any session.
> It contains everything needed to understand and continue work on S2S without re-explaining.

---

## What is S2S?

S2S ("Scan2Sell") is a **physics-certified motion data toolkit** for Physical AI.
It validates raw IMU/EMG sensor data against 7 biomechanical physics laws and issues
a cryptographic Ed25519 certificate for each passing record. The goal: make motion
data trustworthy for training robots, prosthetics, and exoskeletons.

**Repo:** https://github.com/timbo4u1/S2S  
**Author:** Timur Davkarayev (timur.davkarayev@gmail.com)  
**Version:** v1.3  
**Language:** Python 3.9+, zero external dependencies  
**License:** BSL-1.1 → Apache 2.0 on 2028-01-01  

---

## The 7 Biomechanical Laws

| Law | Source | What it checks |
|-----|--------|---------------|
| Newton F=ma coupling | Newton 1687 | EMG force precedes acceleration by 75ms |
| Rigid body kinematics | Euler | a = α×r + ω²×r holds at every joint |
| Resonance frequency | Flash-Hogan 1985 | Forearm tremor is 8-12Hz |
| Jerk bounds | Flash-Hogan 1985 | Human motion ≤ 500 m/s³ |
| IMU consistency | Sensor physics | Accel and gyro from same chip must couple |
| BCG heartbeat | Starr 1939 | Heart stroke visible in wrist IMU |
| Joule heating | Ohm 1827 | EMG power matches thermal output |

---

## Module Structure

```
s2s_standard_v1_3/
├── constants.py                — TLV registry, quality tiers, sensor profiles
├── s2s_physics_v1_3.py        — 7 biomechanical physics laws (core engine)
├── s2s_stream_certify_v1_3.py — Real-time IMU stream certification
├── s2s_emg_certify_v1_3.py   — EMG (electromyography) certification
├── s2s_lidar_certify_v1_3.py  — LiDAR certification
├── s2s_thermal_certify_v1_3.py— Thermal camera certification
├── s2s_ppg_certify_v1_3.py   — PPG / heart rate certification
├── s2s_fusion_v1_3.py        — 5-sensor fusion + HIL score
├── s2s_signing_v1_3.py       — Ed25519 signing + verification
├── s2s_registry_v1_3.py      — Device registry
├── s2s_api_v1_3.py           — REST API server
└── convert_to_s2s.py         — CSV → .s2s binary format

collect_action.py              — Record labeled actions from iPhone
s2s_dataset_adapter.py        — Convert UCI HAR, PAMAP2, Berkeley MHAD
wisdm_adapter.py              — WISDM dataset adapter
train_classifier.py           — Domain classifier training script
model/
  s2s_domain_classifier.json  — Trained classifier
  training_report.json        — Training metrics
docs/
  index.html                  — Live demo (IMU)
  pose.html                   — Live demo (pose)
```

---

## Core API

```python
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

result = PhysicsEngine().certify(
    imu_raw={
        "timestamps_ns": [...],   # nanosecond timestamps
        "accel":         [...],   # [[ax,ay,az], ...] m/s²
        "gyro":          [...],   # [[gx,gy,gz], ...] rad/s
    },
    segment="forearm"
)

result['tier']               # 'GOLD' | 'SILVER' | 'BRONZE' | 'REJECTED'
result['physical_law_score'] # 0-100
result['laws_passed']        # list of law names that passed
result['overall_score']      # float 0-100
```

---

## Certification Tiers

| Tier | Score | Meaning |
|------|-------|---------|
| GOLD | 90-100 | All laws pass — hardware-grade |
| SILVER | 60-89 | Core laws pass — suitable for training |
| BRONZE | 30-59 | Partial pass — use with caution |
| REJECTED | 0-29 | Physics violations detected |

---

## Motion Domains

| Domain | Jerk ≤ | Coupling r ≥ | Use case |
|--------|--------|--------------|----------|
| PRECISION | 80 m/s³ | 0.30 | Surgical robots, prosthetics |
| POWER | 200 m/s³ | 0.30 | Exoskeletons, warehouse arms |
| SOCIAL | 180 m/s³ | 0.15 | Service robots, HRI |
| LOCOMOTION | 300 m/s³ | 0.15 | Bipedal robots, prosthetic legs |
| DAILY_LIVING | 150 m/s³ | 0.20 | Home robots, elder care |
| SPORT | 500 m/s³ | 0.10 | Athletic training |

---

## Known Weaknesses (Active Development)

1. **Domain classifier accuracy: 65.9%** — below acceptable for production (target: 85%+)
2. **No test suite** — zero pytest coverage currently
3. **Pure Python math** — no numpy/GPU, slow on large batches
4. **No ML integration layer** — `s2s_ml_interface` module designed but not yet implemented
5. **Single commit on GitHub** — 19 local commits not yet pushed (git push pending)
6. **No CI/CD** — no GitHub Actions workflow yet
7. **No PyPI package** — must clone repo manually

---

## Active Roadmap

Full roadmap: see `S2S_GITHUB_ROADMAP.md` or GitHub Issues.

**Current priority order:**
1. Push full commit history (19 local commits)
2. Add missing files (wisdm_adapter, train_classifier, model/, docs/)
3. Write pytest tests for all 7 laws
4. Add GitHub Actions CI
5. Retrain domain classifier using S2S physics feature vectors
6. Build `s2s_ml_interface` module (PyTorch integration)
7. Add NumPy fast-path to PhysicsEngine
8. Streamlit certification dashboard
9. Reproducible benchmark: certified vs uncertified training on UCI HAR
10. PyPI package: `pip install s2s-certify`

---

## Physics-Informed ML Loss Term

```python
# Inside training loop:
loss_task = criterion(outputs, labels_batch)
loss_phys = torch.mean(1.0 - physics_scores / 100.0)
loss = loss_task + lambda_phys * loss_phys   # lambda_phys = 0.1
```

---

## Validated On

- iPhone 11 real IMU data: SILVER certified, 4/4 laws passing
- UCI HAR dataset: ~9,050 records certified, ~1,310 rejected
- PAMAP2 dataset: 100% certification rate

---

## Session Notes
*(Add notes here during each working session so continuity is maintained)*

- 2026-03-03: Full roadmap created. 13 GitHub issues defined. Priority: push commits first.

