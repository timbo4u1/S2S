# S2S — Physics Certification for Motion Data

**Bad robot and human motion training data costs you months. S2S finds it in seconds.**

[![PyPI](https://img.shields.io/badge/pypi-v1.6.0-orange)](https://pypi.org/project/s2s-certify/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18878307.svg)](https://doi.org/10.5281/zenodo.18878307) [![License](https://img.shields.io/badge/License-BSL--1.1-blue)](LICENSE) [![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml) [![dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml) [![tests](https://img.shields.io/badge/tests-110%2F110-brightgreen)](tests/)

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

S2S asks: does this data *obey the physics of human movement?* A perfect statistical fake fails if it violates Newton's Second Law, segment resonance, or rigid body kinematics.

**Validated results across 7 datasets:**

| Dataset | Hz | Sensors | Result |
|---|---|---|---|
| UCI HAR | 50Hz | IMU | +2.51% F1 vs corrupted baseline |
| PAMAP2 | 100Hz | IMU ×3 | +4.23% F1 kinematic chain vs single sensor |
| WISDM 2019 | 20Hz | IMU | +1.74% F1 vs corrupted baseline |
| WESAD | 700Hz | IMU + BVP | +3.1% F1 stress classification |
| RoboTurk Open-X | 15Hz | Robot arm | 21.9% of teleoperation data rejected as physically invalid |
| NinaPro DB5 | 2000Hz | Forearm EMG+IMU | 9,552 windows certified |
| PhysioNet PTT-PPG | 500Hz | Wrist PPG+IMU | 1,164 windows certified |

---

## Install

```bash
pip install s2s-certify
pip install "s2s-certify[ml]"        # with PyTorch
pip install "s2s-certify[dashboard]" # with Streamlit
```

Zero runtime dependencies. Pure Python 3.9+.

---

## Quick CLI

```bash
s2s-certify yourfile.csv
s2s-certify yourfile.csv --output report.json --segment forearm
```

---

## Architecture — how it all fits together

S2S has two parts: a **production system** (Layers 1–4) and a **research scaffolding** (Modules 1–4) that was used to build and validate it.

```
WHAT IT DOES (Layers)          HOW IT WAS BUILT (Modules)
─────────────────────────────────────────────────────────

Layer 1: Physics Certification  ← Module 1: Corruption Fingerprinter
  7 biomechanical laws             mapped which corruptions break which law
  GOLD/SILVER/BRONZE/REJECTED      proved thresholds are not arbitrary

Layer 2: Quality Prediction     ← Module 2: Frankenstein Mixer
  Dual-head AI on certified data   found exact physics boundary (binary search)
  smoothness r=0.941               proved S2S finds a real data quality cliff

                                ← Module 3: Curriculum Generator
                                   generated training data at all quality levels

                                ← Module 4: Cloud Trainer
                                   trained quality predictor on curriculum

Layer 3: Motion Retrieval          (no separate module — built directly)
  "pick up cup" → certified motion
  10,686 certified windows

Layer 4: Action Sequencing         (no separate module — built directly)
  4a predict next  r=0.958
  4b gap filling   +5.5% vs linear
  4c intent        top-5 75.9%

Layer 5: Scenario Understanding    ❌ planned
  video + language input
```

---

## Layer 1 — Physics Certification

7 biomechanical laws validated at runtime:

| Law | Equation | Requires | What it catches |
|---|---|---|---|
| Newton's Second Law | F = ma | IMU + EMG | EMG force must lead acceleration by ~75ms |
| Segment Resonance | ω = √(K/I) | IMU | Physiological tremor 8–12Hz for forearm |
| Rigid Body Kinematics | a = α×r + ω²×r | IMU + gyro | Gyro and accel must co-vary on a rigid body |
| Ballistocardiography | F = ρQv | IMU + PPG | Heartbeat recoil visible in wrist IMU at PPG rate |
| Joule Heating | Q = 0.75×P×t | EMG + thermal | EMG bursts must produce thermal elevation |
| Motor Control Jerk | d³x/dt³ ≤ 500 m/s³ | IMU | Human motion limit (Flash & Hogan 1985) |
| IMU Internal Consistency | Var(accel) ~ Var(gyro) | IMU + gyro | Independent generators produce zero coupling |

Missing sensors are skipped — they do not penalise the score.

### Body segment parameters

| Segment | I (kg·m²) | K (N·m/rad) | Tremor band (Hz) |
|---|---|---|---|
| `forearm` | 0.020 | 1.5 | 8–12 |
| `upper_arm` | 0.065 | 2.5 | 5–9 |
| `hand` | 0.004 | 0.3 | 10–16 |
| `finger` | 0.0003 | 0.05 | 15–25 |
| `head` | 0.020 | 1.2 | 3–8 |
| `walking` | 10.0 | 50.0 | 1–3 |

### Tier system

| Tier | Condition |
|---|---|
| GOLD | score ≥ 75 AND passed ≥ n_laws − 1 |
| SILVER | score ≥ 55 |
| BRONZE | score ≥ 35 |
| REJECTED | >30% laws failed OR score < 35 |

---

## Layer 2 — Quality Prediction

Dual-head model trained on 3,487 certified pairs (quality head 72.2%, smoothness r=0.941):

```python
from s2s_standard_v1_3.s2s_ml_interface import S2SFeatureExtractor, physics_loss

extractor = S2SFeatureExtractor(segment="forearm")
features  = extractor(imu_raw)   # 15-dim: 7 pass/fail + 7 scores + 1 overall

# Physics-informed training regularisation
loss = task_loss + physics_loss(scores, lambda_phys=0.1)
```

---

## Layer 3 — Motion Retrieval

10,686 certified windows across 6 datasets, semantic text-to-motion search:

```python
# "pick up cup" → nearest GOLD certified motions
# Ranking: semantic 0.30 + embedding cosine 0.25 + physics 0.25 + smoothness 0.15 + tier 0.05
```

---

## Layer 4 — Action Sequencing

### 4a — Next action prediction (mean r=0.958)

Trained on 21,896 consecutive pairs from 5 datasets:

```python
from experiments.layer4_sequence_model import predict_next
next_features = predict_next(current_features)
```

### 4b — Gap filling (+5.5% over linear interpolation)

Trained on 87,529 quadruplet samples at t=1/3, 1/2, 2/3. Captures the non-linear bell-shaped velocity profile of human arm motion (Flash & Hogan 1985):

```python
from experiments.layer4b_gap_filling import fill_gap
intermediates = fill_gap(start_features, end_features, n_steps=3)
```

| Position | Neural r | Linear r | Improvement |
|---|---|---|---|
| t=0.33 | 0.944 | 0.889 | +0.055 |
| t=0.50 | 0.960 | 0.909 | +0.051 |
| t=0.67 | 0.945 | 0.889 | +0.056 |

### 4c — Intent recognition (top-5 accuracy 75.9%)

```python
from experiments.layer4c_intent_recognition import query_by_text, classify_motion
matches = query_by_text("pick up cup", top_k=5)
# Returns: "pick object" (0.484), "drinking" (0.411), "hand grasp" (0.287)
```

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
ec   = EMGStreamCertifier(n_channels=8, sampling_hz=1000.0)
cert = ec.push_frame(ts_ns, [ch0..ch7])
```

### PPG — PPGStreamCertifier
```python
from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier
pc   = PPGStreamCertifier(n_channels=2, sampling_hz=100.0)
cert = pc.push_frame(ts_ns, [red, ir])
print(cert["vitals"]["heart_rate_bpm"])
print(cert["vitals"]["hrv_rmssd_ms"])
```

### LiDAR — LiDARStreamCertifier
```python
from s2s_standard_v1_3.s2s_lidar_certify_v1_3 import LiDARStreamCertifier
lc   = LiDARStreamCertifier(mode='scalar')
cert = lc.push_frame(ts_ns, [distance_m])
```

### Thermal — ThermalStreamCertifier
```python
from s2s_standard_v1_3.s2s_thermal_certify_v1_3 import ThermalStreamCertifier
tc   = ThermalStreamCertifier(frame_width=32, frame_height=24)
cert = tc.push_frame(ts_ns, flat_pixels)  # 768 floats in °C
print(cert["human_presence"]["human_present"])
```

### Multi-sensor Fusion — FusionCertifier
```python
from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier
fc = FusionCertifier(device_id="glove_v2")
fc.add_imu_cert(imu_cert)
fc.add_emg_cert(emg_cert)
fc.add_ppg_cert(ppg_cert)
result = fc.certify()
print(result["human_in_loop_score"])   # 0–100
fc.reset()                             # clear for reuse
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

---

## REST API

```bash
python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080 \
    --sign-key keys/server.private.pem \
    --registry registry.json
```

| Method | Path | Description |
|---|---|---|
| POST | `/certify/imu` | Batch IMU certification |
| POST | `/certify/emg` | Batch EMG certification |
| POST | `/certify/lidar` | Batch LiDAR (scalar or pointcloud) |
| POST | `/certify/thermal` | Batch thermal frames |
| POST | `/certify/ppg` | Batch PPG certification |
| POST | `/certify/fusion` | Fuse 2–5 stream certs |
| POST | `/stream/frame` | Push frame to persistent session |
| GET | `/health` | Health + active sessions |

---

## Cryptographic signing

```python
from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner, CertVerifier

signer, verifier = CertSigner.generate()
signer.save_keypair("keys/device_001")
signed = signer.sign_cert(cert_dict)
ok, reason = verifier.verify_cert(signed)
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

## Full project structure

```
s2s_standard_v1_3/               ← Production package (pip install s2s-certify)
  s2s_physics_v1_3.py            ← PhysicsEngine — 7 laws, zero dependencies
  s2s_emg_certify_v1_3.py        ← EMG certifier (≥500Hz)
  s2s_ppg_certify_v1_3.py        ← PPG — HR, HRV, breathing
  s2s_lidar_certify_v1_3.py      ← LiDAR (1D scalar + 3D pointcloud)
  s2s_thermal_certify_v1_3.py    ← Thermal — human body heat detection
  s2s_fusion_v1_3.py             ← 5-sensor fusion, Human-in-Loop score 0–100
  s2s_stream_certify_v1_3.py     ← Real-time streaming (stdin / TCP :9876)
  s2s_api_v1_3.py                ← REST API server (stdlib, zero deps)
  s2s_signing_v1_3.py            ← Ed25519 signing + HMAC-SHA256 fallback
  s2s_registry_v1_3.py           ← Device registry — trust tiers, revocation
  s2s_ml_interface.py            ← PyTorch Dataset, feature extractor, physics loss
  convert_to_s2s.py              ← CSV → .s2s binary
  cli.py                         ← s2s-certify CLI entry point
  constants.py                   ← TLV registry, tiers, sensor profiles

experiments/
  ── Research Modules (built and validated Layers 1–2) ──
  module1_corruption_fingerprinter.py  ← maps which corruptions break which law
  module2_frankenstein_mixer.py        ← binary search for exact physics boundary
  module3_curriculum_generator.py      ← generates training data at all quality levels
  module4_cloud_trainer.py             ← trains quality predictor on curriculum

  ── Layer validation experiments ──
  corruption_experiment.py             ← quality floor proof (Layer 1)
  level2_pamap2_curriculum.py          ← Layer 2 validation on PAMAP2
  level2_pamap2_adaptive_tiers.py      ← adaptive tier thresholds
  level3_pamap2_kalman.py              ← Layer 3 retrieval on PAMAP2
  level3_uci_kalman.py                 ← Layer 3 retrieval on UCI HAR
  level3_wisdm_kalman.py               ← Layer 3 retrieval on WISDM
  level4_multisensor_fusion.py         ← multi-sensor kinematic chain
  level5_dualhead.py                   ← dual-head quality+motion model
  level5_weighted.py                   ← semi-supervised weighted training
  level5_full.py                       ← full Level 5 training pipeline

  ── Layer 4 (action sequencing) ──
  extract_sequences.py                 ← extract consecutive certified pairs
  extract_sequences_pamap2_wesad.py    ← add PAMAP2+WESAD to training data
  layer4_sequence_model.py             ← 4a: next action prediction (r=0.958)
  layer4b_gap_filling.py               ← 4b: gap filling (+5.5% vs linear)
  layer4c_intent_recognition.py        ← 4c: motion→intent (top-5 75.9%)

  ── Retrieval ──
  step3_retrieval.py                   ← text-to-motion retrieval v1
  step3_retrieval_v2.py                ← retrieval with semantic embeddings
  auto_router.py                       ← automatic dataset routing

  ── Results ──
  results_layer4.json                  ← 4a: r=0.958
  results_layer4b.json                 ← 4b: +5.5% vs linear
  results_layer4c.json                 ← 4c: top-5 75.9%
  results_wesad_f1.json                ← WESAD +3.1% F1
  results_level4_pamap2.json           ← multi-sensor +5.0% F1
  results_level5_dualhead.json         ← quality 72.2%, smoothness 0.941
  roboturk_audit.json                  ← RoboTurk 78.1% pass rate

Dataset adapters (root level)
  wesad_adapter.py                     ← WESAD (chest+wrist+BVP)
  wesad_f1_benchmark.py                ← F1 comparison certified vs all
  wisdm_adapter.py                     ← WISDM 2019 (51 subjects, 20Hz)
  amputee_adapter.py                   ← EMG amputee (200Hz, 17 movements)
  s2s_dataset_adapter.py               ← UCI HAR, PAMAP2, Berkeley MHAD
  certify_roboturk.py                  ← RoboTurk Open-X certification
  s2s_pipeline.py                      ← NinaPro full pipeline CLI

tests/
  test_physics_laws.py                 ← 110/110 tests passing
  test_emg_ppg.py
  test_fusion.py

model/
  s2s_quality_model_pytorch.pkl        ← trained quality predictor (Module 4)
  s2s_domain_classifier.json           ← motion domain classifier

docs/
  demo_physical_ai.html
  paper/S2S_Paper_v5.pdf
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
