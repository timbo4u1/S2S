# S2S Quickstart

## Install

```bash
pip install s2s-certify
# With PyTorch ML integration:
pip install "s2s-certify[ml]"
```

Zero runtime dependencies. Python 3.9+.

---

## 1. Certify a CSV file (CLI)

```bash
s2s-certify yourfile.csv
s2s-certify yourfile.csv --output report.json --segment forearm
```

Auto-detects columns: `acc_x/acc_y/acc_z`, `ax/ay/az`, `accel_x/accel_y/accel_z`, `gyro_x/gyro_y/gyro_z`, `gx/gy/gz`, etc. Outputs tier, score, coupling r, jerk P95, laws passed/failed.

---

## 2. Certify IMU data in Python

```python
from s2s_standard_v1_3 import PhysicsEngine

pe = PhysicsEngine()
result = pe.certify(
    imu_raw={
        "timestamps_ns": timestamps,   # list of ints (nanoseconds)
        "accel": accel_xyz,            # [[ax,ay,az], ...] in m/s²
        "gyro":  gyro_xyz,             # [[gx,gy,gz], ...] in rad/s (optional but recommended)
    },
    segment="forearm"   # forearm / upper_arm / hand / finger / head / walking
)

print(result["tier"])                   # GOLD / SILVER / BRONZE / REJECTED
print(result["physical_law_score"])     # 0–100
print(result["laws_passed"])
print(result["laws_failed"])
```

### Check biological origin across a full session

```python
pe = PhysicsEngine()
for window in all_windows:
    pe.certify(imu_raw=window, segment="forearm")

verdict = pe.certify_session()
print(verdict["biological_grade"])   # HUMAN / LOW_BIOLOGICAL_FIDELITY / NOT_BIOLOGICAL
print(verdict["hurst"])              # ≥0.7 = human motor control
print(verdict["recommendation"])     # ACCEPT / REVIEW / REJECT
```

---

## 3. Certify EMG

```python
from s2s_standard_v1_3.s2s_emg_certify_v1_3 import EMGStreamCertifier

ec = EMGStreamCertifier(n_channels=8, sampling_hz=1000.0)  # minimum 500Hz

cert = ec.push_frame(ts_ns, [ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7])
if cert:
    print(cert["tier"])
```

---

## 4. Certify PPG

```python
from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier

pc = PPGStreamCertifier(n_channels=2, sampling_hz=100.0)   # minimum 25Hz

cert = pc.push_frame(ts_ns, [red_channel, ir_channel])
if cert:
    print(cert["vitals"]["heart_rate_bpm"])
    print(cert["vitals"]["hrv_rmssd_ms"])
    print(cert["vitals"]["breathing_bpm"])
```

---

## 5. Certify LiDAR

```python
from s2s_standard_v1_3.s2s_lidar_certify_v1_3 import LiDARStreamCertifier

# 1D time-of-flight:
lc = LiDARStreamCertifier(mode='scalar', device_id='tof_01')
cert = lc.push_frame(ts_ns, [distance_meters])

# 3D point cloud:
lc3d = LiDARStreamCertifier(mode='pointcloud', n_points_per_frame=360)
cert = lc3d.push_frame(ts_ns, [x0,y0,z0, x1,y1,z1, ...])
```

---

## 6. Certify Thermal

```python
from s2s_standard_v1_3.s2s_thermal_certify_v1_3 import ThermalStreamCertifier

tc = ThermalStreamCertifier(frame_width=32, frame_height=24, device_id='lepton_01')
cert = tc.push_frame(ts_ns, flat_pixels)   # 32×24 = 768 floats in °C

if cert:
    print(cert["human_presence"]["human_present"])
    print(cert["human_presence"]["body_heat_fraction"])
    print(cert["tier"])
```

---

## 7. Fuse multiple sensors

```python
from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier

fc = FusionCertifier(device_id="glove_v2", session_id="session_001")
fc.add_imu_cert(imu_cert)
fc.add_emg_cert(emg_cert)
fc.add_ppg_cert(ppg_cert)
fc.add_thermal_cert(thermal_cert)
fc.add_lidar_cert(lidar_cert)
result = fc.certify()

print(result["human_in_loop_score"])    # 0–100
print(result["tier"])
```

Human-in-Loop score: 40pts stream quality + 40pts pairwise coherence + 20pts biological bonuses. Any SUSPECT_SYNTHETIC flag → score=0, REJECTED.

---

## 8. Real-time streaming

```python
from s2s_standard_v1_3.s2s_stream_certify_v1_3 import StreamCertifier
import time

sc = StreamCertifier(
    sensor_names=["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]
)

cert = sc.push_frame(ts_ns=time.time_ns(), values=[ax, ay, az, gx, gy, gz])
if cert:
    print(cert["tier"], cert["score"])
```

Or pipe JSON from device SDK:

```bash
# stdin (one JSON frame per line):
python3 -m s2s_standard_v1_3.s2s_stream_certify_v1_3 --mode stdin

# TCP server (default :9876):
python3 -m s2s_standard_v1_3.s2s_stream_certify_v1_3 --mode tcp --port 9876
```

Frame format: `{"ts_ns": 1234567890, "channels": {"accel_x": 0.1, "accel_y": -0.2, "accel_z": 9.8}}`

---

## 9. REST API

```bash
python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080

# With signing + registry:
python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080 \
    --sign-key keys/server.private.pem \
    --registry registry.json
```

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/certify/imu \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "glove_001",
    "frames": [
      {"ts_ns": 1000000, "values": [0.1, -0.2, 9.8, 0.01, -0.01, 0.0]},
      {"ts_ns": 2000000, "values": [0.2, -0.1, 9.7, 0.02, -0.02, 0.1]}
    ]
  }'

# Thermal — pass width + height:
curl -X POST http://localhost:8080/certify/thermal \
  -H "Content-Type: application/json" \
  -d '{"device_id":"lepton_01","width":32,"height":24,"frames":[{"ts_ns":1000000,"pixels":[...768 floats...]}]}'

# LiDAR pointcloud:
curl -X POST http://localhost:8080/certify/lidar \
  -H "Content-Type: application/json" \
  -d '{"device_id":"lidar_01","mode":"pointcloud","frames":[{"ts_ns":1000000,"values":[x0,y0,z0,...]}]}'

# Push to persistent session:
curl -X POST http://localhost:8080/stream/frame \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s001","sensor_type":"imu","ts_ns":1000000,"values":[0.1,-0.2,9.8,0.01,-0.01,0.0]}'

# Session status:
curl http://localhost:8080/stream/s001/status

# End session:
curl -X DELETE http://localhost:8080/stream/s001
```

---

## 10. Convert CSV → .s2s binary

```bash
# Basic:
python3 -m s2s_standard_v1_3.convert_to_s2s input.csv -o output.s2s

# With NaN removal + jitter injection + signing:
python3 -m s2s_standard_v1_3.convert_to_s2s input.csv -o output.s2s \
    --remove-nans \
    --inject-jitter-stdns 4500 \
    --seed 42 \
    --sign-key keys/device_001.private.pem \
    --verbose

# Fill NaNs instead of removing:
python3 -m s2s_standard_v1_3.convert_to_s2s input.csv -o output.s2s --fill-nans ffill

# Specific columns:
python3 -m s2s_standard_v1_3.convert_to_s2s input.csv -o output.s2s \
    --columns acc_x acc_y acc_z gyro_x gyro_y gyro_z
```

NaN values abort conversion unless `--remove-nans` or `--fill-nans` is specified. Ambiguous delimiter aborts unless `--delimiter` is given.

---

## 11. Cryptographic signing

```python
from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner, CertVerifier

# Generate keypair (once per device):
signer, verifier = CertSigner.generate()
signer.save_keypair("keys/device_001")
# → keys/device_001.private.pem  (keep secret)
# → keys/device_001.public.pem   (share)

# Sign a cert:
signed = signer.sign_cert(cert_dict)
# Adds: _signature, _signing_key_id, _signing_mode, _signed_at_ns

# Verify:
ok, reason = verifier.verify_cert(signed)
```

```bash
python3 -m s2s_standard_v1_3.s2s_signing_v1_3 keygen --out keys/device_001
python3 -m s2s_standard_v1_3.s2s_signing_v1_3 verify cert.json --pubkey keys/device_001.public.pem
python3 -m s2s_standard_v1_3.s2s_signing_v1_3 sign   cert.json --privkey keys/device_001.private.pem
```

---

## 12. Device registry

```python
from s2s_standard_v1_3.s2s_registry_v1_3 import DeviceRegistry

reg = DeviceRegistry("registry.json")
reg.register(
    device_id          = "glove_v2_001",
    sensor_profile     = "imu_9dof",    # imu_9dof / imu_6dof / emg_8ch / lidar_1d / thermal / ppg
    owner              = "you@example.com",
    expected_jitter_ns = 4500.0,
    public_key_pem     = signer.export_public_pem(),
    trust_tier         = "PROVISIONAL",
)

ok, reason, device = reg.validate_cert(cert_dict)
reg.promote("glove_v2_001")     # PROVISIONAL → TRUSTED
reg.revoke("bad_device", reason="firmware tampered")
```

```bash
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 register \
    --device-id glove_001 --profile imu_9dof --owner you@example.com --jitter-ns 4500
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 list
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 validate cert.json
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 promote  glove_001
python3 -m s2s_standard_v1_3.s2s_registry_v1_3 summary
```

---

## 13. ML integration (PyTorch)

```python
from s2s_standard_v1_3.s2s_ml_interface import S2SFeatureExtractor, physics_loss

# 15-dim feature vector: 7 pass/fail + 7 scores + 1 overall (all in [0,1])
extractor = S2SFeatureExtractor(segment="forearm")
features  = extractor(imu_raw)    # numpy array, shape (15,)

# Physics-regularised training loss:
loss = task_loss + physics_loss(scores, lambda_phys=0.1)
# Formula: λ × mean(1 − score/100)
```

```python
from s2s_standard_v1_3.s2s_ml_interface import MotionDataset, S2SDataLoader

loader = S2SDataLoader(imu_list, labels, batch_size=32)
for features, labels in loader:
    outputs = model(features)    # features shape [batch, 15]
```

---

## 14. Batch certify .s2s files

```bash
python3 -m s2s_standard_v1_3.s2s_emg_certify_v1_3     data/*.s2s --out-json-dir certs/emg/
python3 -m s2s_standard_v1_3.s2s_ppg_certify_v1_3     data/*.s2s --out-json-dir certs/ppg/
python3 -m s2s_standard_v1_3.s2s_lidar_certify_v1_3   data/*.s2s --out-json-dir certs/lidar/ --mode scalar
python3 -m s2s_standard_v1_3.s2s_thermal_certify_v1_3 data/*.s2s --out-json-dir certs/thermal/ --width 32 --height 24
```

---

## Complete end-to-end example

```python
import time
from s2s_standard_v1_3 import PhysicsEngine
from s2s_standard_v1_3.s2s_emg_certify_v1_3  import EMGStreamCertifier
from s2s_standard_v1_3.s2s_ppg_certify_v1_3  import PPGStreamCertifier
from s2s_standard_v1_3.s2s_fusion_v1_3       import FusionCertifier
from s2s_standard_v1_3.s2s_signing_v1_3      import CertSigner

signer, verifier = CertSigner.generate()

imu_cert = PhysicsEngine().certify(imu_raw=your_imu_data, segment="forearm")

ec = EMGStreamCertifier(n_channels=8, sampling_hz=1000.0)
emg_cert = None
for ts_ns, samples in emg_stream:
    emg_cert = ec.push_frame(ts_ns, samples) or emg_cert

pc = PPGStreamCertifier(n_channels=2, sampling_hz=100.0)
ppg_cert = None
for ts_ns, samples in ppg_stream:
    ppg_cert = pc.push_frame(ts_ns, samples) or ppg_cert

fc = FusionCertifier(device_id="glove_v2", session_id="session_001")
fc.add_imu_cert(imu_cert)
if emg_cert: fc.add_emg_cert(emg_cert)
if ppg_cert: fc.add_ppg_cert(ppg_cert)
fusion_result = fc.certify()

signed_cert = signer.sign_cert(fusion_result)
ok, reason = verifier.verify_cert(signed_cert)

print(signed_cert["tier"])
print(signed_cert["human_in_loop_score"])
print(f"Signature valid: {ok} ({reason})")
```

---

## Tier reference

| Tier | Condition |
|---|---|
| GOLD | score ≥ 75 AND passed ≥ n_laws − 1 |
| SILVER | score ≥ 55 |
| BRONZE | score ≥ 35 |
| REJECTED | >30% laws failed OR score < 35 |

## Supported segments

`forearm` · `upper_arm` · `hand` · `finger` · `head` · `walking`

---

[→ Full README](README.md) · [→ GitHub](https://github.com/timbo4u1/S2S) · [→ Live API](https://s2s-65sy.onrender.com)
