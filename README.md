# S2S — Physics Certification for Motion Data

**Bad robot and human motion training data costs you months. S2S finds it in seconds.**

[![PyPI](https://img.shields.io/badge/pypi-v1.5.0-orange)](https://pypi.org/project/s2s-certify/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18878307.svg)](https://doi.org/10.5281/zenodo.18878307) [![License](https://img.shields.io/badge/License-BSL--1.1-blue)](LICENSE) [![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml) [![dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml)

```python
from s2s_standard_v1_3 import PhysicsEngine

pe = PhysicsEngine()
result = pe.certify(
    imu_raw={
        "timestamps_ns": timestamps,   # nanosecond timestamps
        "accel": accel_data,           # [[x,y,z], ...] in m/s²
        "gyro":  gyro_data,            # [[x,y,z], ...] in rad/s (optional)
    },
    segment="forearm"
)

print(result["tier"])                  # GOLD / SILVER / BRONZE / REJECTED
print(result["physical_law_score"])    # 0–100
print(result["laws_passed"])           # which physics checks passed
print(result["laws_failed"])           # which checks failed and why
```

---

## Why this exists

Most motion datasets contain bad data — corrupted recordings, synthetic signals that violate physics, mislabeled actions. You can't see it by looking at the numbers. Your model trains on it anyway.

S2S asks a different question than statistical anomaly detection. It doesn't ask "does this data *look* human?" It asks "does this data *obey* the physics of human movement?" A perfect statistical fake fails here if it violates Newton's Second Law, segment resonance, or rigid body kinematics.

**Proven results across 6 datasets:**

| Dataset | Hz | Result |
|---|---|---|
| UCI HAR | 50Hz | +2.51% F1 vs corrupted baseline |
| PAMAP2 | 100Hz | +4.23% F1 kinematic chain vs single sensor |
| WISDM 2019 | 20Hz | +1.74% F1 vs corrupted baseline |
| RoboTurk (Open-X) | 15Hz | 21.9% of robot teleoperation data rejected as physically invalid |

---

## Install

```bash
pip install s2s-certify
```

Zero runtime dependencies. Pure Python 3.9+. NumPy optional (10–50× speedup when installed).

```bash
# With ML support (PyTorch + NumPy):
pip install "s2s-certify[ml]"

# With dashboard:
pip install "s2s-certify[dashboard]"
```

---

## Quick CLI

```bash
# Certify any IMU CSV — auto-detects accelerometer and gyro columns
s2s-certify yourfile.csv

# With options
s2s-certify yourfile.csv --output report.json --segment forearm
```

Auto-detects columns like `acc_x/acc_y/acc_z`, `ax/ay/az`, `gyro_x/gyro_y/gyro_z`, `gx/gy/gz`, etc. Outputs tier, score, coupling r, jerk P95, laws passed/failed, and optional Ed25519 signature field. Saves JSON report with `--output`.

---

## The 7 Physics Laws

S2S encodes 7 physical laws. Each law only runs when the required sensors are present — missing sensors don't penalise the score.

| Law | Equation | Requires | What it catches |
|---|---|---|---|
| Newton's Second Law | F = ma | IMU + EMG | EMG force must LEAD acceleration by ~75ms. Synthetic data has zero lagged correlation. |
| Segment Resonance | ω = √(K/I) | IMU | Physiological tremor at 8–12Hz for forearm (I=0.020 kg·m², K=1.5 N·m/rad). Cannot be faked with wrong parameters. |
| Rigid Body Kinematics | a = α×r + ω²×r | IMU + gyro | Gyro and accel measure the same motion from different physical principles. Independently generated fake data has zero coupling. |
| Ballistocardiography | F = ρQv | IMU + PPG cert | Each heartbeat ejects ~70ml at ~1m/s → measurable IMU recoil at exactly the PPG heart rate. |
| Joule Heating | Q = 0.75 × P × t | EMG cert + thermal cert | Sustained EMG bursts must produce thermal elevation — 75% of metabolic energy becomes heat. |
| Motor Control Jerk | d³x/dt³ ≤ 500 m/s³ | IMU | Flash & Hogan (1985) minimum-jerk model. Robotic keyframe artifacts and bang-bang control exceed this limit. |
| IMU Internal Consistency | Var(accel) ~ Var(gyro) | IMU + gyro | For a rigid body, accel and gyro variance must co-vary. Independent generators produce zero coupling. |

### Body segment parameters (exact from source)

| Segment | I (kg·m²) | K (N·m/rad) | Tremor band (Hz) | Jerk check |
|---|---|---|---|---|
| `forearm` | 0.020 | 1.5 | 8–12 | ✓ |
| `upper_arm` | 0.065 | 2.5 | 5–9 | ✓ |
| `hand` | 0.004 | 0.3 | 10–16 | ✓ |
| `finger` | 0.0003 | 0.05 | 15–25 | ✓ |
| `head` | 0.020 | 1.2 | 3–8 | ✓ |
| `walking` | 10.0 | 50.0 | 1–3 | **skipped** |

Note: `walking` skips the jerk check — the 500 m/s³ limit is not established for gait in the literature.

---

## Tier system

| Tier | Condition |
|---|---|
| GOLD | score ≥ 75 AND passed ≥ n_laws − 1 |
| SILVER | score ≥ 55 |
| BRONZE | score ≥ 35 |
| REJECTED | >30% of checked laws failed OR score < 35 |

---

## Motion Domains

S2S classifies motion into 6 domains for routing data to the correct AI model:

| Domain | Jerk limit | IMU coupling r | Use case |
|---|---|---|---|
| PRECISION | ≤ 80 m/s³ | ≥ 0.30 | Surgical robots, prosthetics |
| POWER | ≤ 200 m/s³ | ≥ 0.30 | Exoskeletons, warehouse arms |
| SOCIAL | ≤ 180 m/s³ | ≥ 0.15 | Service robots, HRI |
| LOCOMOTION | ≤ 300 m/s³ | ≥ 0.15 | Bipedal robots, prosthetic legs |
| DAILY_LIVING | ≤ 150 m/s³ | ≥ 0.20 | Home robots, elder care |
| SPORT | ≤ 500 m/s³ | ≥ 0.10 | Athletic training |

---

## Biological Origin Detection

After processing a full session, call `certify_session()` to detect whether data came from a real human:

```python
pe = PhysicsEngine()
for window in session_windows:
    pe.certify(imu_raw=window, segment="forearm")

verdict = pe.certify_session()
print(verdict["biological_grade"])   # HUMAN / LOW_BIOLOGICAL_FIDELITY / NOT_BIOLOGICAL
print(verdict["hurst"])              # H ≥ 0.7 = biological motor control
print(verdict["bfs"])                # Biological Fingerprint Score (0–1)
print(verdict["recommendation"])     # ACCEPT / REVIEW / REJECT
```

`BFS = 0.3×(1/CV) + 0.4×(1−kurtosis_norm) + 0.3×(1−Hurst)` — a floor detector, not a ranking. Hurst < 0.7 → NOT_BIOLOGICAL regardless of BFS. Outlier trimming: top+bottom 15% removed before computing CV, kurtosis, and Hurst.

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

ec = EMGStreamCertifier(
    n_channels=8,
    sampling_hz=1000.0,   # minimum 500Hz required
)
cert = ec.push_frame(ts_ns, [ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7])
```

GOLD: SNR ≥ 20dB + burst fraction ≥ 5% (threshold = 3× baseline RMS, lower 30% of envelope). Electrode contact check: variance ≥ 1e-4. Saturation check: >95% samples at rail → SATURATED. Multi-channel tier: all GOLD → GOLD; ≥70% GOLD/SILVER → SILVER; ≥50% active → BRONZE.

### PPG — PPGStreamCertifier

```python
from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier

pc = PPGStreamCertifier(
    n_channels=2,
    sampling_hz=100.0,    # minimum 25Hz required
)
cert = pc.push_frame(ts_ns, [red, ir])
print(cert["vitals"]["heart_rate_bpm"])
print(cert["vitals"]["hrv_rmssd_ms"])
print(cert["vitals"]["breathing_bpm"])
```

Auto-detects device profile (PTT-PPG 500Hz normalized vs PAMAP2 100Hz raw ADC). HR range: 0.5–3.5 Hz (30–210 BPM). HRV RMSSD ≥ 15ms → GOLD, ≥ 5ms → SILVER. Breathing modulation: 0.15–0.4 Hz. Synthetic: RMSSD < 0.001ms → SUSPECT_SYNTHETIC.

### LiDAR — LiDARStreamCertifier

```python
from s2s_standard_v1_3.s2s_lidar_certify_v1_3 import LiDARStreamCertifier

# 1D time-of-flight
lc = LiDARStreamCertifier(mode='scalar', device_id='tof_01')
cert = lc.push_frame(ts_ns, [distance_m])

# 3D point cloud
lc3d = LiDARStreamCertifier(mode='pointcloud', n_points_per_frame=360)
cert = lc3d.push_frame(ts_ns, [x0,y0,z0, x1,y1,z1, ...])
```

GOLD: variance ≥ 0.01m² AND frame delta ≥ 0.5mm RMS. Synthetic: frame-to-frame delta < 1e-9m (perfectly static). Saturation: distances ≥ 40m flagged. Planarity > 0.99 → PERFECTLY_PLANAR_SCENE flag.

### Thermal — ThermalStreamCertifier

```python
from s2s_standard_v1_3.s2s_thermal_certify_v1_3 import ThermalStreamCertifier

tc = ThermalStreamCertifier(frame_width=32, frame_height=24, device_id='lepton_01')
cert = tc.push_frame(ts_ns, flat_pixels)  # 32×24 = 768 floats in °C

print(cert["human_presence"]["human_present"])       # True/False
print(cert["human_presence"]["body_heat_fraction"])  # fraction of pixels in 28–38.5°C
```

Human detected: ≥5% pixels in 28–38.5°C range (`confident_human`: ≥15%). GOLD: confident human + spatial range ≥ 5°C + frame delta ≥ 0.05°C + no flags. Synthetic: frame-to-frame change < 1e-6°C → SUSPECT_SYNTHETIC. Sensor fault: pixels above 100°C or below -20°C.

### Multi-sensor Fusion — FusionCertifier

```python
from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier

fc = FusionCertifier(device_id="glove_v2", session_id="s001")
fc.add_imu_cert(imu_cert)
fc.add_emg_cert(emg_cert)
fc.add_ppg_cert(ppg_cert)
fc.add_thermal_cert(thermal_cert)
fc.add_lidar_cert(lidar_cert)
result = fc.certify()
print(result["human_in_loop_score"])   # 0–100
```

**Human-in-Loop score** = 40pts stream quality + 40pts pairwise coherence + 20pts biological bonuses. 8 coherence pairs checked: IMU↔EMG, IMU↔PPG, IMU↔Thermal, IMU↔LiDAR, EMG↔PPG, EMG↔Thermal, PPG↔Thermal, LiDAR↔Thermal. Any SUSPECT_SYNTHETIC flag from any stream → score=0, REJECTED.

### Real-time streaming — StreamCertifier

```python
from s2s_standard_v1_3.s2s_stream_certify_v1_3 import StreamCertifier

sc = StreamCertifier(
    sensor_names=["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]
)
cert = sc.push_frame(ts_ns=time.time_ns(), values=[0.1, -0.2, 9.8, 0.01, -0.01, 0.0])
```

Or pipe JSON-line frames from a device SDK:

```bash
# stdin mode
python3 -m s2s_standard_v1_3.s2s_stream_certify_v1_3 --mode stdin

# TCP server (default port 9876)
python3 -m s2s_standard_v1_3.s2s_stream_certify_v1_3 --mode tcp --port 9876
```

Frame format: `{"ts_ns": int, "channels": {"accel_x": float, ...}}`

Window defaults: 256 frames, step 32. Min window: 64.

---

## REST API

```bash
python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080

# With Ed25519 signing + device registry:
python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080 \
    --sign-key keys/server.private.pem \
    --registry registry.json
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check + active sessions + signing status |
| GET | `/version` | API version + available sensors + all endpoint list |
| GET | `/sessions` | List active streaming sessions |
| GET | `/stream/{id}/status` | Session stats + last cert |
| POST | `/certify/imu` | Batch IMU certification |
| POST | `/certify/emg` | Batch EMG certification |
| POST | `/certify/lidar` | Batch LiDAR (mode: scalar or pointcloud) |
| POST | `/certify/thermal` | Batch thermal frames (pass width + height) |
| POST | `/certify/ppg` | Batch PPG certification |
| POST | `/certify/fusion` | Fuse 2–5 stream certs → unified cert |
| POST | `/stream/frame` | Push one frame to a named persistent session |
| DELETE | `/stream/{id}` | End a streaming session |

All endpoints accept and return JSON. CORS enabled. Registry validation: unregistered device → 403. Signed certs add `_signature`, `_signing_key_id`, `_signing_mode`, `_signed_at_ns` fields.

---

## Cryptographic Signing

S2S certificates are signed with Ed25519 (HMAC-SHA256 stdlib fallback if `cryptography` not installed).

```python
from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner, CertVerifier, attach_signer_to_certifier

# Generate a fresh keypair:
signer, verifier = CertSigner.generate()

# Load from file:
signer   = CertSigner.from_pem_file("keys/device_001.private.pem")
verifier = CertVerifier.from_pem_file("keys/device_001.public.pem")

# Sign a cert:
signed_cert = signer.sign_cert(cert_dict)
# Adds: _signature (base64url), _signing_key_id (16-char hex), _signing_mode, _signed_at_ns

# Verify:
ok, reason = verifier.verify_cert(signed_cert)

# Auto-sign every cert from a stream certifier:
attach_signer_to_certifier(stream_certifier, signer)
```

```bash
# Key management CLI:
python3 -m s2s_standard_v1_3.s2s_signing_v1_3 keygen --out keys/device_001
# Creates: keys/device_001.private.pem + keys/device_001.public.pem

python3 -m s2s_standard_v1_3.s2s_signing_v1_3 verify cert.json --pubkey keys/device_001.public.pem
python3 -m s2s_standard_v1_3.s2s_signing_v1_3 sign   cert.json --privkey keys/device_001.private.pem
```

---

## Device Registry

Maps device IDs to sensor profiles, public keys, jitter specs, and trust tiers. Foundation for marketplace data provenance — buyers verify data came from a registered trusted device.

```python
from s2s_standard_v1_3.s2s_registry_v1_3 import DeviceRegistry

reg = DeviceRegistry("registry.json")

reg.register(
    device_id            = "glove_v2_001",
    sensor_profile       = "imu_9dof",     # imu_9dof / imu_6dof / emg_8ch / lidar_1d / thermal / ppg
    owner                = "timur@scan2sell.io",
    expected_jitter_ns   = 4500.0,          # vendor RMS jitter spec
    jitter_tolerance_pct = 20.0,            # ±20% tolerance
    public_key_pem       = signer.export_public_pem(),
    trust_tier           = "TRUSTED",       # TRUSTED / PROVISIONAL / UNTRUSTED
)

ok, reason, device = reg.validate_cert(cert_dict)
# Checks: registration, revocation, jitter fingerprint, Ed25519 signature

reg.promote("glove_v2_001")             # PROVISIONAL → TRUSTED
reg.revoke("bad_device", reason="tampered")
reg.update_public_key("device", new_pem)  # key rotation
```

```bash
# Registry CLI:
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 register \
    --device-id glove_001 --profile imu_9dof --owner you@example.com \
    --jitter-ns 4500 --pubkey keys/glove_001.public.pem --trust PROVISIONAL

python3 -m s2s_standard_v1_3.s2s_registry_v1_3 list
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 validate cert.json
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 promote glove_001
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 revoke  glove_001 --reason "tampered firmware"
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 summary
```

---

## Data ingestion

### CSV → .s2s binary

```bash
# Basic
python3 -m s2s_standard_v1_3.convert_to_s2s input.csv -o output.s2s

# With NaN removal, jitter injection, signing
python3 -m s2s_standard_v1_3.convert_to_s2s input.csv -o output.s2s \
    --remove-nans \
    --inject-jitter-stdns 4500 \
    --seed 42 \
    --sign-key keys/device_001.private.pem \
    --verbose
```

**Options:**
- `--remove-nans` — synchronized removal of any row containing NaN across any channel
- `--fill-nans {zero|mean|ffill}` — fill strategy instead of removal
- `--inject-jitter-stdns <float>` — Gaussian jitter on timestamps (realistic hardware noise)
- `--seed <int>` — reproducible jitter RNG
- `--delimiter <char>` — override CSV delimiter detection
- `--columns <col1 col2 ...>` — select specific columns by name or index
- `--annotate-suspect` — flag perfectly-regular timestamps in metadata
- `--overwrite` — allow overwriting existing output file

**Fail-fast behaviour:** NaN values in sensor columns abort conversion unless `--remove-nans` or `--fill-nans` is specified. Ambiguous delimiter aborts unless `--delimiter` is given.

Metadata embedded in every `.s2s` file: sensor map, entry count, jitter metrics before/after injection, NaN summary, tool provenance, CLI args (signing key path excluded).

### Batch certify .s2s files

```bash
python3 -m s2s_standard_v1_3.s2s_emg_certify_v1_3     data/*.s2s --out-json-dir certs/emg/
python3 -m s2s_standard_v1_3.s2s_ppg_certify_v1_3     data/*.s2s --out-json-dir certs/ppg/
python3 -m s2s_standard_v1_3.s2s_lidar_certify_v1_3   data/*.s2s --out-json-dir certs/lidar/ --mode scalar
python3 -m s2s_standard_v1_3.s2s_thermal_certify_v1_3 data/*.s2s --out-json-dir certs/thermal/ --width 32 --height 24
```

---

## ML integration

```python
from s2s_standard_v1_3.s2s_ml_interface import S2SFeatureExtractor, physics_loss

# 15-dim feature vector: 7 pass/fail + 7 per-law scores + 1 overall (all in [0,1])
extractor = S2SFeatureExtractor(segment="forearm")
features  = extractor(imu_raw)        # numpy array, shape (15,)

# Physics-informed regularisation term:
# total_loss = task_loss + physics_loss(scores, lambda_phys=0.1)
loss = physics_loss(physics_scores, lambda_phys=0.1)
# Formula: λ × mean(1 − score/100)
```

```python
from s2s_standard_v1_3.s2s_ml_interface import MotionDataset, S2SDataLoader

dataset = MotionDataset(imu_list, labels, segment="forearm", precompute=True)
loader  = S2SDataLoader(imu_list, labels, batch_size=32)

for features, labels in loader:
    outputs = model(features)    # features shape: [batch, 15]
```

```python
# Batch certification:
features, scores, tiers = certify_batch(imu_batch)
```

---

## Pipeline CLI

```bash
# Single subject (NinaPro-style .mat directory)
python3 s2s_pipeline.py --data ~/ninapro_db5/s5

# All subjects
python3 s2s_pipeline.py --data ~/ninapro_db5/ --all-subjects

# Demo (all NinaPro DB5 subjects)
python3 s2s_pipeline.py --demo
```

Window: 2000 samples (1s), step: 1000. Outputs tier distribution, Hurst exponent, BFS, ACCEPT/REVIEW/REJECT per subject. Report: `experiments/pipeline_report.json`.

---

## Validated datasets

| Dataset | Hz | Device | Segment | Windows |
|---|---|---|---|---|
| NinaPro DB5 | 2000Hz | Forearm EMG+IMU | forearm | 9,552 |
| PAMAP2 | 100Hz | Body IMU (3 sensors) | forearm | 13,094 |
| UCI HAR | 50Hz | Waist IMU | walking | 10,299 |
| WISDM 2019 | 20Hz | Wrist phone | forearm | 46,946 |
| PhysioNet PTT-PPG | 500Hz | Wrist PPG+IMU | forearm | 1,164 |
| RoboTurk (Open-X) | 15Hz | 7-DOF robot arm | forearm | 901 |

---

## Project structure

```
s2s_standard_v1_3/
  s2s_physics_v1_3.py          ← PhysicsEngine — 7 laws, zero dependencies
  s2s_emg_certify_v1_3.py      ← EMG certifier (≥500Hz)
  s2s_ppg_certify_v1_3.py      ← PPG certifier — heart rate, HRV, breathing
  s2s_lidar_certify_v1_3.py    ← LiDAR certifier (1D scalar + 3D pointcloud)
  s2s_thermal_certify_v1_3.py  ← Thermal certifier — human body heat detection
  s2s_fusion_v1_3.py           ← 5-sensor fusion, Human-in-Loop score (0–100)
  s2s_stream_certify_v1_3.py   ← Real-time streaming (stdin / TCP :9876)
  s2s_api_v1_3.py              ← REST API server (stdlib only, no dependencies)
  s2s_signing_v1_3.py          ← Ed25519 signing + HMAC-SHA256 fallback
  s2s_registry_v1_3.py         ← Device registry — jitter fingerprint + trust tiers
  s2s_ml_interface.py          ← PyTorch Dataset, DataLoader, feature extractor, physics loss
  convert_to_s2s.py            ← CSV → .s2s binary (fail-fast, NaN handling, jitter injection)
  cli.py                       ← `s2s-certify` CLI entry point
  constants.py                 ← TLV registry, tier constants, sensor profiles, jitter thresholds
  __init__.py                  ← exports PhysicsEngine only

s2s_pipeline.py                ← Full NinaPro pipeline CLI
certify_roboturk.py            ← RoboTurk Open-X certification
certify_openx.py               ← Open-X dataset certification
wisdm_adapter.py               ← WISDM 2019 adapter (51 subjects, 20Hz)
amputee_adapter.py             ← EMG amputee adapter (200Hz, 17 movements)
s2s_dataset_adapter.py         ← UCI HAR, PAMAP2, Berkeley MHAD, MoVi adapter

experiments/
  level5_corrected_best.pt     ← Current best model (smoothness r=0.941)
  retrieval_index_v3.json      ← 10,686 certified windows across 6 datasets
  retrieval_embeddings_v3.npy  ← 64-dim motion embeddings

tests/
  test_physics_laws.py         ← 21 tests covering all 7 laws
  test_emg_ppg.py
  test_fusion.py

docs/
  demo_physical_ai.html
  paper/S2S_Paper_v5.pdf
```

---

## Live demos

[→ IMU Demo](https://timbo4u1.github.io/S2S/) | [→ Physical AI Demo](https://timbo4u1.github.io/S2S/demo_physical_ai.html) | [→ Live API](https://s2s-65sy.onrender.com)

---

## Paper

[→ Read paper (PDF)](docs/paper/S2S_Paper_v5.pdf) · [DOI: 10.5281/zenodo.18878307](https://doi.org/10.5281/zenodo.18878307)

---

## License

BSL-1.1 — free for research and non-commercial use. Converts to Apache 2.0 on 2028-01-01.
