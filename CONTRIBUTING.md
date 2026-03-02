# Contributing to S2S

Thank you for your interest in S2S. This document explains how to contribute.

## What We Need Most

### 1. Real IMU Data (highest value)
Record your own motion data using `collect_action.py` and share it. Every real human recording strengthens the validation story.

```bash
# Install "Sensor Logger" app on your phone (free, iOS/Android)
# Enable: AccelerometerUncalibrated + GyroscopeUncalibrated + Gravity
# Record 30 seconds of any action, export CSV, then:

python3 collect_action.py \
  --accel  AccelerometerUncalibrated.csv \
  --gyro   GyroscopeUncalibrated.csv \
  --gravity Gravity.csv \
  --label  walk_forward \
  --person your_name \
  --out    my_data/
```

Share the resulting `.s2s` files via GitHub issue or email.

### 2. Test the Pose Demo
Open https://timbo4u1.github.io/S2S/pose.html and report:
- Which phone/browser you used
- Whether the skeleton tracked correctly
- What GOLD/SILVER/BRONZE score you got for walking, sitting, waving

### 3. New Dataset Adapters
If you have access to a motion dataset not yet supported, write an adapter following the pattern in `s2s_dataset_adapter.py`. Any dataset with IMU (accel + gyro) at any Hz is useful.

### 4. Classifier Improvements
The domain classifier is in `train_classifier.py`. Current accuracy: 65.9%.

Known issue: PRECISION and DAILY_LIVING overlap at 20Hz.
Proposed fix: merge into FINE_MOTOR domain.

To test:
```bash
python3 train_classifier.py --dataset s2s_dataset/ --test
```

---

## Code Standards

- **Zero dependencies** — no pip packages. stdlib only. This is enforced by CI.
- **Pure Python** — no Cython, no C extensions
- **Single file modules** — each certifier is self-contained
- **No global state** — all classes take config in `__init__`

## Running Tests

```bash
# Check physics engine
python3 -c "from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine; print('OK')"

# Check classifier
python3 train_classifier.py --dataset s2s_dataset/ --test

# Full CI (same as GitHub Actions)
# See .github/workflows/ci.yml
```

## Submitting Changes

1. Fork the repo
2. Create a branch: `git checkout -b my-improvement`
3. Make your change
4. Test it
5. Open a pull request with a clear description

## Contact

Questions, data sharing, hardware partnerships: timur.davkarayev@gmail.com
