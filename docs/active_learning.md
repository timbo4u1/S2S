# S2S Active Learning — 4 Modules

## Where This Fits in S2S

```
S2S (existing)                    S2S Active Learning (new)
─────────────────                 ──────────────────────────
PhysicsEngine.certify()    ──▶    Module 1: Corruption Fingerprinter
  └─ 7 physics laws               └─ Learns what bad data looks like
  └─ GOLD/SILVER/BRONZE/REJECTED     from the inside

experiments/ (existing)    ──▶    Module 2: Frankenstein Mixer
  └─ Levels 1-5 results            └─ Finds exact physics boundaries
  └─ PTT-PPG, PAMAP2, NinaPro        by mixing good + bad data

data/ (existing)           ──▶    Module 3: Curriculum Generator
  └─ Certified windows             └─ Generates training data at
                                      every quality level automatically

model/ (new)               ◀──    Module 4: Cloud Trainer (Colab)
  └─ s2s_quality_model.pkl         └─ Trains neural net on curriculum
                                      returns small portable model
```

## The Idea

S2S currently **waits** to be fed certified data.

These modules make S2S an **active learner** — it generates its own
training curriculum by deliberately corrupting clean data, finding
physics boundaries, and training a model on the full quality spectrum.

**Analogy:** A child doesn't wait to be taught what's dangerous.
They touch things, taste things, fall down. Experience = learning.
Module 1-3 are the child exploring. Module 4 is the adult brain
that learns from all that exploration.

## Running Order

```bash
# Step 1: Fingerprint corruptions (Mac, ~10 min)
cd ~/S2S
python3 experiments/module1_corruption_fingerprinter.py
# Output: experiments/results_corruption_fingerprints.json

# Step 2: Find physics boundaries (Mac, ~15 min)
python3 experiments/module2_frankenstein_mixer.py
# Output: experiments/results_frankenstein_boundaries.json

# Step 3: Generate curriculum (Mac, ~20 min)
python3 experiments/module3_curriculum_generator.py
# Output: experiments/curriculum_dataset.npy
#         experiments/curriculum_labels.npy
#         experiments/curriculum_tiers.npy

# Step 4a: Train baseline on Mac (sklearn, fast)
python3 experiments/module4_cloud_trainer.py
# Output: model/s2s_quality_model_sklearn.pkl

# Step 4b: Train better model on Colab (PyTorch + GPU, free)
# 1. Upload curriculum files to Google Drive
# 2. Open module4_cloud_trainer.py in colab.research.google.com
# 3. Run all cells
# 4. Download model/s2s_quality_model_pytorch.pkl
```

## What Each Module Proves

| Module | Scientific question | Expected finding |
|--------|-------------------|-----------------|
| 1 | Which corruption breaks which physics law? | spike_injection → jerk_bounds first |
| 2 | How much contamination before physics fails? | ~25% bad data = rigid_body breaks |
| 3 | Can S2S generate its own training data? | Yes — full quality spectrum |
| 4 | Can ML learn to predict physics quality? | Yes — beats 50Hz heuristic thresholds |

## Why This Matters

Current S2S: binary filter (pass/fail)
After Module 4: continuous quality score (0-100)

The continuous score enables:
- Soft loss term in training: `L = L_task + λ × (1 - quality_score)`
- Weighted sampling: high quality data gets higher training weight
- Uncertainty quantification: "this window is 73/100 — borderline"

## Requirements

```
pip install numpy scikit-learn torch  # Module 4 PyTorch
# All others: zero dependencies (same as S2S core)
```

## Files

```
experiments/
  module1_corruption_fingerprinter.py  ← Run first
  module2_frankenstein_mixer.py        ← Run second
  module3_curriculum_generator.py      ← Run third
  module4_cloud_trainer.py             ← Run on Colab or Mac

  results_corruption_fingerprints.json  ← Module 1 output
  results_frankenstein_boundaries.json  ← Module 2 output
  curriculum_dataset.npy                ← Module 3 output
  curriculum_labels.npy
  curriculum_tiers.npy

model/
  s2s_quality_model_sklearn.pkl   ← Fast, runs anywhere
  s2s_quality_model_pytorch.pkl   ← Better, needs torch
```

---
S2S — github.com/timbo4u1/S2S | DOI: 10.5281/zenodo.18878307
