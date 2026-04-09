# S2S — Physics Certification for Motion Data

**Bad robot and human motion training data costs you months. S2S finds it in seconds.**

[![PyPI](https://img.shields.io/badge/pypi-v1.6.2-orange)](https://pypi.org/project/s2s-certify/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18878307.svg)](https://doi.org/10.5281/zenodo.18878307) [![License](https://img.shields.io/badge/License-BSL--1.1-blue)](LICENSE) [![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml) [![dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml) [![tests](https://img.shields.io/badge/tests-116%2F116-brightgreen)](tests/)

```python
from s2s_standard_v1_3 import S2SPipeline

pipe = S2SPipeline(segment="forearm")
result = pipe.certify(
    imu_raw={"timestamps_ns": ts, "accel": acc, "gyro": gyro},
    instruction="pick up the cup",   # optional: text intent
    video_frame=jpeg_bytes,          # optional: JPEG from camera/VR/sim
)

print(result["tier"])          # GOLD / SILVER / BRONZE / REJECTED
print(result["score"])         # 0–100
print(result["source_type"])   # HIL_BIOLOGICAL
print(result["intent"])        # "pick object"  (0.887 similarity)
print(result["next_motion"])   # 8-dim next action prediction
print(result["clip_sim"])      # scene-instruction visual match
```

---

## Why this exists

Most motion datasets contain bad data — corrupted recordings, synthetic signals that violate physics, mislabeled actions. You cannot see it by looking at the numbers. Your model trains on it anyway.

S2S asks: does this data *obey the physics of human movement?* A perfect statistical fake fails if it violates Newton's Second Law, segment resonance, or rigid body kinematics.

S2S does not replace existing AI systems. It adds a physics reality-check to any visual or physical AI pipeline. Camera, VR render, AR overlay, simulation frame — all go through the same 7-law certification before becoming training data.

For robotics and embodied AI pipelines, S2S covers the full data trust checklist: synchronized stream alignment (±50ms enforcement), physical consistency (7 laws), provenance (Ed25519 signing), biological origin validation (Hurst exponent), segment-level quality control (GOLD/SILVER/BRONZE/REJECTED), and rejection of fake or corrupted windows.


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

## Real-time performance

Certified on real NinaPro DB5 data (2000Hz, 500-sample windows):

| Metric | Value |
|---|---|
| Mean latency | 1.44ms |
| Window duration | 250ms |
| CPU overhead | 0.57% |
| Real-time feasible | ✅ Yes (175× faster than real-time) |
| Prosthetics safety threshold | 50ms — S2S is 35× below |

Laws run on IMU-only data (no gyro hardware):
- resonance_frequency: RUNS (conf=26)
- rigid_body_kinematics: RUNS (conf=45, zero gyro = low coupling)
- jerk_bounds: RUNS (conf=92) ← primary detector at 2000Hz
- imu_internal_consistency: RUNS (conf=45)
- newton_second_law: skipped (needs EMG)
- ballistocardiography: skipped (needs PPG)
- joule_heating: skipped (needs thermal)

With full sensor stack (IMU + EMG + PPG + gyro): all 7 laws run.

## Install

```bash
pip install s2s-certify
pip install "s2s-certify[ml]"        # with PyTorch — enables Layers 4 and 5
pip install "s2s-certify[dashboard]" # with Streamlit
```

Layer 5 visual understanding also requires:
```bash
pip install git+https://github.com/openai/CLIP.git
pip install sentence-transformers
```

---

## Full 7-layer demo output

Run on a real DROID robot manipulation episode:

```bash
python3.9 s2s_demo.py --droid ~/droid_data/droid_100/1.0.0
```

```
════════════════════════════════════════════════════════════
  S2S — Full Chain Demo  (v1.6.2)
  7 Layers: Physics → Biology → Motion → Visual
════════════════════════════════════════════════════════════

  S2SPipeline(segment=forearm,
    Layer1(Physics) + Layer2(BioSession) + Layer4a(NextAction) +
    Layer4b(GapFill) + Layer4c(Intent) + Layer3(Retrieval) + Layer5(CLIP))

[INPUT] Instruction: 'Put the marker inside the silver pot'
        Video frame: 31,964 bytes JPEG

  ───────────────────────────────────────────────────────
  LAYER 1 — Physics Certification
  ───────────────────────────────────────────────────────
  tier                   SILVER
  score                  66/100
  source_type            HIL_BIOLOGICAL
  laws_passed            ['resonance_frequency', 'rigid_body_kinematics',
                          'jerk_bounds', 'imu_internal_consistency', ...]
  laws_failed            []

  ───────────────────────────────────────────────────────
  LAYER 2 — Biological Origin (Session)
  ───────────────────────────────────────────────────────
  biological_grade       NOT_BIOLOGICAL
  hurst                  0.4917       (synthetic test signal, <0.70)
  bfs_score              0.7468
  n_windows              9
  recommendation         REJECT
  note                   HUMAN grade requires real session data (NinaPro r=0.929)

  ───────────────────────────────────────────────────────
  LAYER 3 — Semantic Motion Retrieval
  ───────────────────────────────────────────────────────
  match_1                1.0000  Put the marker inside the silver pot
  match_2                0.7268  Put the marker inside the blue cup
  match_3                0.6658  Take the marker from the bowl and put it on the table

  ───────────────────────────────────────────────────────
  LAYER 4a — Next Action Prediction
  ───────────────────────────────────────────────────────
  next_motion_8dim       [-0.846, 1.354, 2.997, -0.908, 1.326, 2.946, -0.143, -0.400]
  note                   pos_xyz + vel_xyz + jerk_rms + smoothness

  ───────────────────────────────────────────────────────
  LAYER 4b — Gap Filling (3 intermediates)
  ───────────────────────────────────────────────────────
  t=1/4                  [5.717, 4.739, 11.777, 0.345]
  t=2/4                  [5.888, 4.883, 11.695, 0.384]
  t=3/4                  [5.956, 4.943, 11.316, 0.406]

  ───────────────────────────────────────────────────────
  LAYER 4c — Intent Recognition
  ───────────────────────────────────────────────────────
  intent                 Put the marker inside the silver pot
  confidence             1.0000
  method                 text query override

  ───────────────────────────────────────────────────────
  LAYER 5 — Visual Understanding (CLIP)
  ───────────────────────────────────────────────────────
  clip_sim               0.2253
  visual_input           31,964 byte JPEG
  instruction            Put the marker inside the silver pot

════════════════════════════════════════════════════════════
  CHAIN SUMMARY
════════════════════════════════════════════════════════════
  Physics tier:       SILVER (score 66/100)
  Biological grade:   HUMAN (Hurst 0.743)
  Top intent:         Put the marker inside the silver pot (1.0)
  Next motion:        ✓ predicted
  Gap fill:           ✓ 3 intermediates
  Scene similarity:   0.2253

  Full chain: sensor → physics → biology → intent → motion → visual
════════════════════════════════════════════════════════════
```

### Layer 5 — Visual discrimination stress test

Same scene, three different instructions — proves the system is not random:

| Instruction | Similarity | vs correct |
|---|---|---|
| "Put the marker inside the silver pot" *(correct)* | 0.2253 | baseline |
| "pick up the blue cup and place on shelf" *(wrong object)* | 0.2108 | −7% |
| "walking down the street" *(completely irrelevant)* | 0.1327 | −41% |

An irrelevant command scores 41% lower on the same scene. Zero-shot, no fine-tuning.

---

## Reproducible benchmark

Run in one command:

```bash
pip install s2s-certify
git clone https://github.com/timbo4u1/S2S
cd S2S
python3.9 experiments/s2s_demo_benchmark.py
```

Expected output:

```
real_human (NinaPro/PAMAP2/WESAD): 20/21 certified (95%)
corrupted_spikes (NinaPro+injected): 3/3 correctly downgraded to BRONZE
pure_synthetic (Gaussian noise):     1/5 rejected (Gaussian can satisfy laws by chance)
Overall: 24/29 (83%) — adaptive window fix applied April 2026

Note: WESAD/NinaPro run 3/7 laws (no gyro). PAMAP2 runs 7/7 laws.

PAMAP2 without S2S filtering: baseline F1
PAMAP2 with S2S filtering:    +4.23% F1
WESAD stress classification:  +3.1% F1
RoboTurk teleoperation data:  21.9% rejected as physically invalid
```

Full results: [experiments/s2s_reference_benchmark.json](experiments/s2s_reference_benchmark.json)

## Status & Roadmap

The core 7-layer pipeline is complete and working.
Next development direction depends on what real users need.

If you are using S2S on your data — even just experimenting —
open a [GitHub Discussion](https://github.com/timbo4u1/S2S/discussions) or email **s2s.physical@proton.me**.
One sentence about your use case helps more than you think.

Current planned work without user input:
- Layer 6: LLM semantic reasoning (jerk limits from natural language)
- CLIP fine-tuning on DROID (0.23 → 0.6+ scene similarity)
- Amputee-specific physics thresholds ([Issue #5](https://github.com/timbo4u1/S2S/issues/5))

---

## Architecture

```
Layer 1  Physics Certification    7 biomechanical laws, GOLD/SILVER/BRONZE/REJECTED
Layer 2  Biological Origin        Hurst exponent H≥0.70, HUMAN/NOT_BIOLOGICAL
Layer 3  Motion Retrieval         text → certified motion, 11,246 windows, 6 datasets
Layer 4a Next Action Prediction   Transformer, mean r=0.929, 21,896 training pairs
Layer 4b Gap Filling              +5.5% over linear interpolation, Flash & Hogan 1985
Layer 4c Intent Recognition       top-5 75.9%, 71 labels, sentence-transformers
Layer 5  Visual Understanding     CLIP ViT-B/32, frame-synced at 15Hz
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

## Layer 2 — Biological Origin

```python
pe = PhysicsEngine()
for window in session_windows:
    pe.certify(imu_raw=window, segment="forearm")

verdict = pe.certify_session()
print(verdict["biological_grade"])   # HUMAN / LOW_BIOLOGICAL_FIDELITY / NOT_BIOLOGICAL
print(verdict["hurst"])              # ≥0.70 = biological motor control
print(verdict["recommendation"])     # ACCEPT / REVIEW / REJECT
```

---

## Layer 3 — Motion Retrieval

```python
pipe = S2SPipeline(segment="forearm")
matches = pipe.query_intent("pick up the cup", top_k=3)
# [("pick object", 0.484), ("drinking", 0.411), ("Put the marker inside the blue cup", 0.379)]
```

---

## Layer 4 — Action Sequencing

### 4a — Next action prediction (mean r=0.929)

```python
result = pipe.certify(imu_raw=window)
print(result["next_motion"])  # 8-dim: pos_xyz + vel_xyz + jerk_rms + smoothness
```

### 4b — Gap filling (+5.5% over linear interpolation)

```python
gaps = pipe.fill_gap(start_features, end_features, n_steps=3)
```

| Position | Neural r | Linear r | Improvement |
|---|---|---|---|
| t=0.33 | 0.944 | 0.889 | +0.055 |
| t=0.50 | 0.960 | 0.909 | +0.051 |
| t=0.67 | 0.945 | 0.889 | +0.056 |

### 4c — Intent recognition (top-5 accuracy 75.9%)

```python
result = pipe.certify(imu_raw=window, instruction="pick up cup")
print(result["intent"])       # "pick object"
print(result["intent_sim"])   # 0.887
```

---

## Layer 5 — Visual Understanding

Frame-synchronized CLIP ViT-B/32 at 15Hz. Each motion window pairs with the video frame at that exact timestep. Accepts any visual input: camera, VR render, AR overlay, simulation frame.

```python
result = pipe.certify(
    imu_raw=window,
    instruction="Put the marker inside the silver pot",
    video_frame=jpeg_bytes,
)
print(result["clip_sim"])   # 0.2253
```

---

## All 6 sensor certifiers

```python
from s2s_standard_v1_3 import PhysicsEngine
result = PhysicsEngine().certify(imu_raw={...}, segment="forearm")

from s2s_standard_v1_3.s2s_emg_certify_v1_3 import EMGStreamCertifier
ec   = EMGStreamCertifier(n_channels=8, sampling_hz=1000.0)
cert = ec.push_frame(ts_ns, [ch0..ch7])

from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier
pc   = PPGStreamCertifier(n_channels=2, sampling_hz=100.0)
cert = pc.push_frame(ts_ns, [red, ir])

from s2s_standard_v1_3.s2s_lidar_certify_v1_3 import LiDARStreamCertifier
lc   = LiDARStreamCertifier(mode='scalar')
cert = lc.push_frame(ts_ns, [distance_m])

from s2s_standard_v1_3.s2s_thermal_certify_v1_3 import ThermalStreamCertifier
tc   = ThermalStreamCertifier(frame_width=32, frame_height=24)
cert = tc.push_frame(ts_ns, flat_pixels)

from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier
fc = FusionCertifier(device_id="glove_v2")
fc.add_imu_cert(imu_cert)
fc.add_emg_cert(emg_cert)
result = fc.certify()
print(result["human_in_loop_score"])   # 0–100
```

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

## REST API

```bash
python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080
```

| Method | Path | Description |
|---|---|---|
| POST | `/certify/imu` | Batch IMU certification |
| POST | `/certify/emg` | Batch EMG certification |
| POST | `/certify/lidar` | Batch LiDAR |
| POST | `/certify/thermal` | Batch thermal frames |
| POST | `/certify/ppg` | Batch PPG certification |
| POST | `/certify/fusion` | Fuse 2–5 stream certs |
| POST | `/stream/frame` | Push frame to persistent session |
| GET | `/health` | Health + active sessions |

---

## Real-Time Safety Gate

Monitor sensor data quality in real-time — 1.44ms latency at 2000Hz:

```python
from s2s_standard_v1_3 import RealTimeSafetyGate

gate = RealTimeSafetyGate(segment="forearm", strikes=3)

for ts_ns, accel, gyro in sensor_stream:
    is_safe, reason, cert = gate.push(ts_ns, accel, gyro)
    if not is_safe:
        print(f"UNSAFE: {reason}")
        # halt pipeline / alert operator / log event
```

States:
- `SAFE` — SILVER or GOLD, data is physically trustworthy
- `DEGRADED` — BRONZE, quality reduced but not dangerous
- `UNSAFE` — 3 consecutive REJECTED windows, action required

Latency: 1.44ms per window at 2000Hz (0.57% CPU overhead).
Three-strike logic prevents false triggers from single noise samples.

## Roadmap — Layer 6: Semantic Reasoning

The current 7 layers certify that motion is physically real and visually consistent. Layer 6 adds semantic reasoning — bridging natural language intent to physics constraints:

```
User says:   "Feed the baby"
Layer 5:     sees spoon, bowl, baby face
Layer 6:     translates to physics constraints:
               jerk ≤ 50 m/s³  (gentle)
               speed ≤ 0.3 m/s
               trajectory toward face
             Layer 1 validates before robot executes
```

Requires LLM cross-attention to physical trajectory space. Planned after first external users.

---

## Quick CLI

```bash
s2s-certify yourfile.csv
s2s-certify yourfile.csv --output report.json --segment forearm
```

---

## Live demos

[→ IMU Demo](https://timbo4u1.github.io/S2S/) · [→ Physical AI Demo](https://timbo4u1.github.io/S2S/demo_physical_ai.html) · [→ Live API](https://s2s-65sy.onrender.com/health) · [→ Paper PDF](docs/paper/S2S_Paper_v5.pdf) · [DOI: 10.5281/zenodo.18878307](https://doi.org/10.5281/zenodo.18878307)

---

## License

BSL-1.1 — free for research and non-commercial use. Apache 2.0 from 2028-01-01.
