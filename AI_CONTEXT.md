# AI_CONTEXT.md — S2S Architectural Memory

**Read this before making any suggestions about the S2S codebase.**
This file records decisions made over months of development.
Do not suggest changing things marked INTENTIONAL.

---

## What S2S is

S2S (SCAN2SELL Physical Motion Certification) certifies human motion sensor
data against biomechanical physics laws before using it to train robots and
prosthetics. The core principle: bad training data that violates Newton's laws
produces bad robots. S2S finds it before training.

**PyPI:** `pip install s2s-certify` (v1.6.0)
**DOI:** 10.5281/zenodo.18878307
**License:** BSL-1.1 → Apache 2.0 on 2028-01-01

---

## The 5-Layer Architecture

```
Layer 1  PhysicsEngine          s2s_standard_v1_3/s2s_physics_v1_3.py
Layer 2  Quality Prediction     s2s_standard_v1_3/s2s_ml_interface.py
Layer 3  Motion Retrieval       experiments/step3_retrieval_v2.py
Layer 4a Next Action Predict    experiments/layer4_sequence_model.py
Layer 4b Gap Filling            experiments/layer4b_gap_filling.py
Layer 4c Intent Recognition     experiments/layer4c_intent_recognition.py
Layer 5  Scenario Understanding experiments/layer5_scenario.py
```

**All layers are complete and tested as of March 2026.**

---

## Validated results — do not dispute without running the code

| Dataset | Result | File |
|---|---|---|
| UCI HAR | +2.51% F1 | experiments/results_level3_uci.json |
| PAMAP2 | +4.23% F1 chain vs single | experiments/results_level4_pamap2.json |
| WISDM | +1.74% F1 | experiments/ |
| WESAD | +3.1% F1 stress classification | experiments/results_wesad_f1.json |
| RoboTurk | 21.9% data rejected as invalid | experiments/roboturk_audit.json |
| Layer 4a | mean r=0.929, smoothness r=0.919 | experiments/results_layer4.json |
| Layer 4b | +5.5% over linear interp | experiments/results_layer4b.json |
| Layer 4c | top-5 75.9% on 71 labels | experiments/results_layer4c.json |
| Layer 5 | 0.2302 CLIP scene sim, 15Hz sync | experiments/results_layer5_clip.json |

---

## INTENTIONAL design decisions — do not change

### 1. Timestamp jitter is required
All adapters (WESAD, DROID, extract_sequences) add Gaussian jitter
~200,000ns to timestamps. This is NOT a bug. Real sensors always have
timing noise. Zero-jitter timestamps trigger SUSPECT_SYNTHETIC correctly.
**Do not remove jitter.**

### 2. The 0.3 rejection threshold is correct
```python
elif len(failed) / max(n, 1) > 0.3:
    tier = "REJECTED"
```
Two laws failing (2/7 = 0.286) does NOT automatically pass because the
physics SCORE independently drops below 35, causing REJECTED via the score
path. Score and ratio are independent checks. Both must pass for SILVER+.
**This is not a macaron mismatch. Do not change the threshold.**

### 3. DROID robot data is NOT in human motion training
sequences_real.npz contains sources 0-4 (NinaPro, Amputee, RoboTurk,
PAMAP2, WESAD). Source 5 (DROID) was tested and removed because robot arm
motion degrades human motion prediction from r=0.958 to r=0.750.
S2S certifies human motion to teach robots — not the reverse.
**Do not add DROID back to sequences_real.npz.**

### 4. DROID language instructions ARE in the retrieval index
experiments/retrieval_index_v3.json contains 560 DROID instructions
("Put the marker inside the silver pot" etc.) with source_type=ROBOT_TELEOP.
This is correct — language retrieval benefits from robot task labels.
**The firewall is: human motion training ≠ robot language labels.**

### 5. WeAR dataset (10Hz, accel-only) is not certifiable
The WEAR dataset (18 subjects, 10Hz) was tested and rejected correctly.
S2S requires ≥40Hz for jerk bounds law. 10Hz data is not a bug — it's
below the physics minimum. Result reported honestly to Kristof Van Laerhoven.

### 6. WESAD has no gyroscope — BRONZE is correct
WESAD chest+wrist sensors have no gyro. Rigid body kinematics and IMU
internal consistency laws are skipped. BRONZE (not GOLD) is the correct
maximum tier for gyro-less data. 58% certification rate, +3.1% F1.
**BRONZE on WESAD is not a failure.**

### 7. Layer 4c top-1 accuracy 24% is acceptable
71 labels, minimum 2 samples per label. Top-5 accuracy is 75.9%.
The low top-1 is a data imbalance problem (new robot labels have 2 samples).
The semantic query retrieval works correctly: "pick up cup" → 0.887.
**Do not retrain with synthetic data to inflate top-1.**

### 8. sentence-transformers model is all-MiniLM-L6-v2
Downloaded once, cached at ~/.cache/huggingface/. Timeout warnings on slow
connections are normal — it falls back to cached version.
**Do not change the model.**

### 9. CLIP model is ViT-B/32
Downloaded once, cached at ~/.cache/clip/ViT-B-32.pt (338MB).
CLIP zero-shot similarity of 0.23 is normal for arbitrary robot scenes.
Ranking is correct even if absolute values are low.
**Do not interpret 0.23 as a failure.**

---

## File structure — what goes where

```
s2s_standard_v1_3/    ← Production package. pip install s2s-certify installs this.
                         NEVER put experiment code here.

experiments/          ← Research and training scripts. Not installed by pip.
                         Layer 4a/4b/4c models live here.

tests/                ← 110/110 tests passing. Run: python3.9 -m pytest tests/ -v

wesad_certified/      ← Generated data. In .gitignore. Recreate with wesad_adapter.py
droid_certified/      ← Generated data. In .gitignore. Recreate with layer5_scenario.py

model/                ← Trained sklearn models from Module 4.
docs/                 ← Paper PDF and demo HTML.
```

---

## Data sources and their source IDs

| ID | Dataset | Hz | Segment | Notes |
|---|---|---|---|---|
| 0 | NinaPro DB5 | 2000Hz | forearm | ~/ninapro_db5/ |
| 1 | EMG Amputee | 200Hz | forearm | ~/S2S_Project/EMG_Amputee/ |
| 2 | RoboTurk Open-X | 15Hz | forearm | ~/S2S/openx_data/ |
| 3 | PAMAP2 | 100Hz | forearm+chest | ~/S2S_data/pamap2/ |
| 4 | WESAD | 700Hz | upper_arm | ~/wesad_data/WESAD/ |
| 5 | DROID (robot only) | 15Hz | forearm | ~/droid_data/ — NOT in training |

---

## What the Modules (1-4) are

Modules 1-4 are research scaffolding that BUILT Layers 1-2. They are not
user-facing and are not installed by pip. They live in experiments/.

- Module 1: Corruption Fingerprinter — maps which corruptions break which law
- Module 2: Frankenstein Mixer — binary search for exact physics boundary
- Module 3: Curriculum Generator — generates training data at all quality levels
- Module 4: Cloud Trainer — trains quality predictor on curriculum

**Do not confuse Modules with Layers.**

---

## Python environment

- Python 3.9 for all S2S code (python3.9)
- pytest installed in 3.9
- numpy, scikit-learn, torch, sentence-transformers, clip all installed
- DO NOT use python3.14 (system default on this Mac)

---

## Researcher contacts

- **Kristof Van Laerhoven** (UbiComp Siegen) — WESAD/WEAR/Hangtime datasets
  - Sent: WEAR result (10Hz not certifiable, hardware limit)
  - Sent: WESAD result (+3.1% F1, 58% cert rate)
  - Email: kvl@eti.uni-siegen.de

---

## Key commands

```bash
# Run all tests
cd ~/S2S && python3.9 -m pytest tests/ -v

# Certify a CSV file
s2s-certify yourfile.csv --segment forearm

# Run WESAD certification
python3.9 wesad_adapter.py --root ~/wesad_data/WESAD --out wesad_certified/

# Run Layer 5 full chain
python3.9 experiments/layer5_scenario.py --clip --droid ~/droid_data/droid_100/1.0.0 --max 5

# Query intent
python3.9 experiments/layer4c_intent_recognition.py --query "pick up cup"
```

---

## What is NOT done yet

1. **Layer 5 fine-tuning** — CLIP is zero-shot. Fine-tuning on DROID would
   push scene similarity from 0.23 to 0.6+. Requires GPU (Google Colab).

2. **SensorCertifierBase** — EMG, PPG, LiDAR, Thermal share ~1,040 lines.
   Deduplication into a base class. Low priority — all tests pass.

3. **Amputee-specific thresholds** — Issue #5 on GitHub. Residual limb has
   different I and K constants. Research task.

4. **LLM integration** — Cross-attention from LLM token stream to physical
   trajectory. This is Layer 6 territory.

5. **source_type enum in PhysicsEngine output** — Add "HIL_BIOLOGICAL" vs
   "ROBOT_TELEOP" tag to cert JSON. One line change.

---

*Last updated: March 2026. Session: Layer 5 complete.*
