# S2S — Physics Certification for Motion Data

**Bad robot and human motion training data costs you months. S2S finds it in seconds.**

[![PyPI](https://img.shields.io/badge/pypi-v1.5.0-orange)](https://pypi.org/project/s2s-certify/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18878307.svg)](https://doi.org/10.5281/zenodo.18878307) [![License](https://img.shields.io/badge/License-BSL--1.1-blue)](LICENSE) [![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml) [![dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml) [![tests](https://img.shields.io/badge/tests-110%2F110-brightgreen)](tests/)

```python
from s2s_standard_v1_3 import PhysicsEngine

result = PhysicsEngine().certify(
    imu_raw={
        "timestamps_ns": timestamps,
        "accel": accel_data,   # [[x,y,z], ...] m/s²
        "gyro":  gyro_data,    # [[x,y,z], ...] rad/s
    },
    segment="forearm"
)

print(result["tier"])                  # GOLD / SILVER / BRONZE / REJECTED
print(result["physical_law_score"])    # 0–100
print(result["laws_passed"])
print(result["laws_failed"])
```

---

## Why this exists

Most motion datasets contain bad data — corrupted recordings, synthetic signals that violate physics, mislabeled actions. You cannot see it by looking at the numbers. Your model trains on it anyway.

S2S asks a different question. Not "does this data look human?" but "does this data obey the physics of human movement?" A perfect statistical fake fails if it violates Newton's Second Law, segment resonance, or rigid body kinematics.

**Validated results across 6 datasets:**

| Dataset | Hz | Sensors | Result |
|---|---|---|---|
| UCI HAR | 50Hz | IMU | +2.51% F1 vs corrupted baseline |
| PAMAP2 | 100Hz | IMU ×3 | +4.23% F1 kinematic chain vs single sensor |
| WISDM 2019 | 20Hz | IMU | +1.74% F1 vs corrupted baseline |
| WESAD | 700Hz | IMU + BVP | +3.1% F1 stress classification |
| RoboTurk Open-X | 15Hz | Robot arm | 21.9% of teleoperation data rejected as physically invalid |

---

## Install

```bash
pip install s2s-certify

# With PyTorch ML integration:
pip install "s2s-certify[ml]"

# With Streamlit dashboard:
pip install "s2s-certify[dashboard]"
```

Zero runtime dependencies. Pure Python 3.9+.

---

## Quick CLI

```bash
s2s-certify yourfile.csv
s2s-certify yourfile.csv --output report.json --segment forearm
```

Auto-detects columns: `acc_x/acc_y/acc_z`, `ax/ay/az`, `gyro_x/gyro_y/gyro_z`, `gx/gy/gz`, etc.

---

## 5-Layer Architecture

S2S implements a complete Physical AI pipeline from raw sensor data to natural language intent:

| Layer | What | Status |
|---|---|---|
| 1 | Physics certification — does data obey Newton? | ✅ Complete |
| 2 | Quality prediction — dual-head AI on certified data | ✅ Complete |
| 3 | Motion retrieval — text query → certified motion | ✅ Complete |
| 4 | Action sequencing — chain, fill gaps, recognize intent | ✅ Complete |
| 5 | Scenario understanding — video + language input | ❌ Planned |

---

## Layer 1 — Physics Certification

7 biomechanical laws validated at runtime. Each law only runs when the required sensors are present — missing sensors do not penalise the score.

| Law | Equation | Requires | What it catches |
|---|---|---|---|
| Newton's Second Law | F = ma | IMU + EMG | EMG force must lead acceleration by ~75ms |
| Segment Resonance | ω = √(K/I) | IMU | Physiological tremor 8–12Hz for forearm |
| Rigid Body Kinematics | a = α×r + ω²×r | IMU + gyro | Gyro and accel must co-vary on a rigid body |
| Ballistocardiography | F = ρQv | IMU + PPG | Heartbeat recoil visible in wrist IMU at PPG rate |
| Joule Heating | Q = 0.75×P×t | EMG + thermal | EMG bursts must produce thermal elevation |
| Motor Control Jerk | d³x/dt³ ≤ 500 m/s³ | IMU | Human motion limit (Flash & Hogan 1985) |
| IMU Internal Consistency | Var(accel) ~ Var(gyro) | IMU + gyro | Independent generators produce zero coupling |

### Body segment parameters

| Segment | I (kg·m²) | K (N·m/rad) | Tremor band (Hz) | Jerk check |
|---|---|---|---|---|
| `forearm` | 0.020 | 1.5 | 8–12 | ✓ |
| `upper_arm` | 0.065 | 2.5 | 5–9 | ✓ |
| `hand` | 0.004 | 0.3 | 10–16 | ✓ |
| `finger` | 0.0003 | 0.05 | 15–25 | ✓ |
| `head` | 0.020 | 1.2 | 3–8 | ✓ |
| `walking` | 10.0 | 50.0 | 1–3 | skipped |

### Tier system

| Tier | Condition |
|---|---|
| GOLD | score ≥ 75 AND passed ≥ n_laws − 1 |
| SILVER | score ≥ 55 |
| BRONZE | score ≥ 35 |
| REJECTED | >30% laws failed OR score < 35 |

---

## Layer 2 — Quality Prediction

Dual-head model trained on certified pairs:

```python
from s2s_standard_v1_3.s2s_ml_interface import S2SFeatureExtractor, physics_loss

# 15-dim feature vector: 7 pass/fail + 7 scores + 1 overall
extractor = S2SFeatureExtractor(segment="forearm")
features  = extractor(imu_raw)

# Physics-informed training loss
loss = task_loss + physics_loss(scores, lambda_phys=0.1)
# Formula: λ × mean(1 − score/100)
```

Results: quality prediction 72.2% accuracy, smoothness correlation r=0.941.

---

## Layer 3 — Motion Retrieval

10,686 certified windows across 6 datasets, 64-dim embeddings.

```python
# From experiments/step3_retrieval_v2.py
# "pick up cup" → returns nearest GOLD certified motions
# Ranking: semantic 0.30 + embedding cosine 0.25 + physics 0.25 + smoothness 0.15 + tier 0.05
```

---

## Layer 4 — Action Sequencing

### 4a — Next action prediction

Trained on 21,896 consecutive certified window pairs from 5 datasets (NinaPro, Amputee, RoboTurk, PAMAP2, WESAD):

```python
from experiments.layer4_sequence_model import predict_next

next_features = predict_next(current_13dim_features)
# mean correlation r=0.958  |  smoothness r=0.927  |  jerk r=0.958
```

### 4b — Gap filling

Given start and end keyframes, fills intermediate windows. Trained on 87,529 quadruplet samples at t=1/3, 1/2, 2/3:

```python
from experiments.layer4b_gap_filling import fill_gap

intermediates = fill_gap(start_features, end_features, n_steps=3)
```

Results vs linear interpolation:

| Position | Neural r | Linear r | Improvement |
|---|---|---|---|
| t=0.33 | 0.944 | 0.889 | +0.055 |
| t=0.50 | 0.960 | 0.909 | +0.051 |
| t=0.67 | 0.945 | 0.889 | +0.056 |

The neural network captures the non-linear bell-shaped velocity profile of human arm motion (Flash & Hogan 1985) that linear interpolation misses.

### 4c — Intent recognition

Maps motion sequences to natural language labels:

```python
from experiments.layer4c_intent_recognition import query_by_text, classify_motion

# Text → motion
matches = query_by_text("pick up cup", top_k=5)
# Returns: "pick object" (0.484), "drinking" (0.411), "hand grasp" (0.287)

# Motion → intent
intent = classify_motion(motion_features, top_k=3)
```

Results: top-1 34.5%, top-3 60.2%, top-5 75.9% on 71 labels.

---

## All 6 sensor certifiers

### IMU — PhysicsEngine

```python
from s2s_standard_v1_3 import PhysicsEngine
result = PhysicsEngine().certify(imu_raw={...}, segment="forearm")
```

### EMG — EMGStreamCertifier

```python
from s2s_standard_v1_3.s2s_emg_certify_v1_3 import EMGStreamCertifier

ec   = EMGStreamCertifier(n_channels=8, sampling_hz=1000.0)  # min 500Hz
cert = ec.push_frame(ts_ns, [ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7])
```

GOLD: SNR ≥ 20dB + burst fraction ≥ 5% (threshold = 3× baseline RMS).

### PPG — PPGStreamCertifier

```python
from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier

pc   = PPGStreamCertifier(n_channels=2, sampling_hz=100.0)   # min 25Hz
cert = pc.push_frame(ts_ns, [red, ir])
print(cert["vitals"]["heart_rate_bpm"])
print(cert["vitals"]["hrv_rmssd_ms"])
print(cert["vitals"]["breathing_bpm"])
```

### LiDAR — LiDARStreamCertifier

```python
from s2s_standard_v1_3.s2s_lidar_certify_v1_3 import LiDARStreamCertifier

# 1D time-of-flight:
lc   = LiDARStreamCertifier(mode='scalar', device_id='tof_01')
cert = lc.push_frame(ts_ns, [distance_m])

# 3D point cloud:
lc3d = LiDARStreamCertifier(mode='pointcloud', n_points_per_frame=360)
cert = lc3d.push_frame(ts_ns, [x0,y0,z0, x1,y1,z1, ...])
```

GOLD: variance ≥ 0.01m² AND frame delta ≥ 0.5mm. Synthetic: frame delta < 1e-9m.

### Thermal — ThermalStreamCertifier

```python
from s2s_standard_v1_3.s2s_thermal_certify_v1_3 import ThermalStreamCertifier

tc   = ThermalStreamCertifier(frame_width=32, frame_height=24)
cert = tc.push_frame(ts_ns, flat_pixels)  # 768 floats in °C
print(cert["human_presence"]["human_present"])
print(cert["human_presence"]["body_heat_fraction"])
```

Human detected: ≥5% pixels in 28–38.5°C. GOLD: ≥15% body pixels + spatial range ≥5°C + frame delta ≥0.05°C.

### Multi-sensor Fusion — FusionCertifier

```python
from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier

fc = FusionCertifier(device_id="glove_v2")
fc.add_imu_cert(imu_cert)
fc.add_emg_cert(emg_cert)
fc.add_ppg_cert(ppg_cert)
fc.add_thermal_cert(thermal_cert)
fc.add_lidar_cert(lidar_cert)
result = fc.certify()
print(result["human_in_loop_score"])   # 0–100
fc.reset()                             # clear for reuse
```

Human-in-Loop score: 40pts stream quality + 40pts pairwise coherence + 20pts biological bonuses. Any SUSPECT_SYNTHETIC flag → score=0, REJECTED.

---

## Real-time streaming

```python
from s2s_standard_v1_3.s2s_stream_certify_v1_3 import StreamCertifier
import time

sc = StreamCertifier(
    sensor_names=["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]
)
cert = sc.push_frame(ts_ns=time.time_ns(), values=[ax, ay, az, gx, gy, gz])
```

Or pipe from device:

```bash
python3 -m s2s_standard_v1_3.s2s_stream_certify_v1_3 --mode tcp --port 9876
```

---

## REST API

```bash
python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080 \
    --sign-key keys/server.private.pem \
    --registry registry.json
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health + active sessions |
| GET | `/version` | Version + sensors |
| POST | `/certify/imu` | Batch IMU certification |
| POST | `/certify/emg` | Batch EMG certification |
| POST | `/certify/lidar` | Batch LiDAR (scalar or pointcloud) |
| POST | `/certify/thermal` | Batch thermal frames |
| POST | `/certify/ppg` | Batch PPG certification |
| POST | `/certify/fusion` | Fuse 2–5 stream certs |
| POST | `/stream/frame` | Push frame to persistent session |
| DELETE | `/stream/{id}` | End session |

---

## Cryptographic signing

```python
from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner, CertVerifier

signer, verifier = CertSigner.generate()
signer.save_keypair("keys/device_001")

signed = signer.sign_cert(cert_dict)
ok, reason = verifier.verify_cert(signed)
```

```bash
python3 -m s2s_standard_v1_3.s2s_signing_v1_3 keygen --out keys/device_001
python3 -m s2s_standard_v1_3.s2s_signing_v1_3 verify cert.json --pubkey keys/device_001.public.pem
```

---

## Device registry

```python
from s2s_standard_v1_3.s2s_registry_v1_3 import DeviceRegistry

reg = DeviceRegistry("registry.json")
reg.register(
    device_id="glove_v2_001",
    sensor_profile="imu_9dof",
    owner="you@example.com",
    expected_jitter_ns=4500.0,
    public_key_pem=signer.export_public_pem(),
    trust_tier="PROVISIONAL",
)
ok, reason, device = reg.validate_cert(cert_dict)
reg.promote("glove_v2_001")
```

---

## Data ingestion

```bash
# Convert CSV to .s2s binary
python3 -m s2s_standard_v1_3.convert_to_s2s input.csv -o output.s2s \
    --remove-nans \
    --inject-jitter-stdns 4500 \
    --sign-key keys/device_001.private.pem

# Batch certify .s2s files
python3 -m s2s_standard_v1_3.s2s_emg_certify_v1_3  data/*.s2s --out-json-dir certs/emg/
python3 -m s2s_standard_v1_3.s2s_ppg_certify_v1_3  data/*.s2s --out-json-dir certs/ppg/
```

---

## Biological origin detection

```python
pe = PhysicsEngine()
for window in session_windows:
    pe.certify(imu_raw=window, segment="forearm")

verdict = pe.certify_session()
print(verdict["biological_grade"])   # HUMAN / LOW_BIOLOGICAL_FIDELITY / NOT_BIOLOGICAL
print(verdict["hurst"])              # ≥0.7 = biological motor control
print(verdict["recommendation"])     # ACCEPT / REVIEW / REJECT
```

BFS = 0.3×(1/CV) + 0.4×(1−kurtosis_norm) + 0.3×(1−Hurst). Validated on all 10 NinaPro DB5 subjects.

---

## Validated datasets

| Dataset | Hz | Device | Segment | Windows |
|---|---|---|---|---|
| NinaPro DB5 | 2000Hz | Forearm EMG+IMU | forearm | 9,552 |
| PAMAP2 | 100Hz | Body IMU (3 sensors) | forearm | 13,094 |
| UCI HAR | 50Hz | Waist IMU | walking | 10,299 |
| WISDM 2019 | 20Hz | Wrist phone | forearm | 46,946 |
| PhysioNet PTT-PPG | 500Hz | Wrist PPG+IMU | forearm | 1,164 |
| RoboTurk Open-X | 15Hz | 7-DOF robot arm | forearm | 901 |
| WESAD | 700Hz | Chest+Wrist IMU+BVP | upper_arm | 8,995 |

---

## Project structure

```
s2s_standard_v1_3/
  s2s_physics_v1_3.py          ← PhysicsEngine — 7 laws, zero dependencies
  s2s_emg_certify_v1_3.py      ← EMG certifier (≥500Hz)
  s2s_ppg_certify_v1_3.py      ← PPG certifier — HR, HRV, breathing
  s2s_lidar_certify_v1_3.py    ← LiDAR (1D scalar + 3D pointcloud)
  s2s_thermal_certify_v1_3.py  ← Thermal — human body heat detection
  s2s_fusion_v1_3.py           ← 5-sensor fusion, Human-in-Loop score
  s2s_stream_certify_v1_3.py   ← Real-time streaming (stdin / TCP :9876)
  s2s_api_v1_3.py              ← REST API server (stdlib, zero deps)
  s2s_signing_v1_3.py          ← Ed25519 signing + HMAC-SHA256 fallback
  s2s_registry_v1_3.py         ← Device registry — trust tiers, revocation
  s2s_ml_interface.py          ← PyTorch Dataset, feature extractor, physics loss
  convert_to_s2s.py            ← CSV → .s2s binary
  cli.py                       ← s2s-certify CLI entry point
  constants.py                 ← TLV registry, tiers, sensor profiles

experiments/
  layer4_sequence_model.py     ← Layer 4a: next action prediction (r=0.958)
  layer4b_gap_filling.py       ← Layer 4b: gap filling (+5.5% vs linear)
  layer4c_intent_recognition.py← Layer 4c: motion→intent (top-5 75.9%)
  extract_sequences.py         ← Extract consecutive certified pairs
  extract_sequences_pamap2_wesad.py ← Add PAMAP2+WESAD to training data
  step3_retrieval_v2.py        ← Motion retrieval, semantic embeddings
  level5_dualhead.py           ← Dual-head quality+motion prediction
  corruption_experiment.py     ← Quality floor proof
  results_layer4.json          ← Layer 4a results
  results_layer4b.json         ← Layer 4b results
  results_layer4c.json         ← Layer 4c results
  results_wesad_f1.json        ← WESAD +3.1% F1 result

wesad_adapter.py               ← WESAD certification (chest+wrist+BVP)
wesad_f1_benchmark.py          ← F1 comparison certified vs all windows
s2s_pipeline.py                ← NinaPro pipeline CLI
wisdm_adapter.py               ← WISDM 2019 adapter
amputee_adapter.py             ← EMG amputee adapter
s2s_dataset_adapter.py         ← UCI HAR, PAMAP2, Berkeley MHAD adapter

tests/
  test_physics_laws.py         ← 110/110 tests passing
  test_emg_ppg.py
  test_fusion.py
```

---

## Live demos

[→ IMU Demo](https://timbo4u1.github.io/S2S/) · [→ Physical AI Demo](https://timbo4u1.github.io/S2S/demo_physical_ai.html) · [→ Live API](https://s2s-65sy.onrender.com)

---

## Paper

[→ PDF](docs/paper/S2S_Paper_v5.pdf) · [DOI: 10.5281/zenodo.18878307](https://doi.org/10.5281/zenodo.18878307)

---

## License

BSL-1.1 — free for research and non-commercial use. Apache 2.0 from 2028-01-01.
