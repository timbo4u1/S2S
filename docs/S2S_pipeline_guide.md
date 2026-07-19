# S2S Pipeline — Easy Guide Map

**What is S2S?**
A quality control system for motion sensor data. Before you train an AI model on IMU, EMG, or PPG data, you run S2S. It tells you which windows of data are trustworthy and which are not — and exactly why.

**One sentence:** S2S reads raw sensor data and stamps each window GOLD, SILVER, BRONZE, or REJECTED based on whether it obeys the laws of physics and biology.

---

## The Big Picture

```
Your raw sensor data
(IMU accelerometer, gyroscope, EMG, PPG)
          │
          ▼
    ┌─────────────┐
    │   LAYER 1   │  ← Did the physics happen correctly?
    │  16 laws    │     e.g. does the arm actually move like an arm?
    └──────┬──────┘
           │  GOLD / SILVER / BRONZE pass through
           │  REJECTED stops here
           ▼
    ┌─────────────┐
    │   LAYER 2   │  ← Is this signal from a living human?
    │  Hurst / SE │     e.g. does the motion have biological complexity?
    └──────┬──────┘
           │  HUMAN grade passes through
           │  NOT_BIOLOGICAL stops here
           ▼
    ┌─────────────┐
    │   LAYER 3   │  ← Does the motion match what it claims to be?
    │  Retrieval  │     e.g. does "pinch gesture" actually look like a pinch?
    └──────┬──────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌──────────┐
│ LAYER 4 │  │ LAYER 5  │  ← Does the scene match the instruction?
│ Utility │  │ Vision   │     e.g. does the camera show the right context?
└─────────┘  └──────────┘
```

**Layer 4 is different from the others.** It does not check data quality.
It uses already-certified data to predict next actions, fill motion gaps,
and recognize intent. Think of it as the output side, not the filter side.

---

## Layer 1 — Does the Physics Work?

This is the main gate. 16 independent checks run on every window.

### How the score works

Each law returns a confidence score (0-100). The final score is the average.
Too many failures → REJECTED. High score, few failures → GOLD.

```
More than 30% of laws failed          → REJECTED
No temporal AND no spatial structure  → REJECTED  (synthetic noise)
Gaussian innovations detected         → REJECTED  (OU generator)
Score ≥ 75, almost all laws passed    → GOLD
Score ≥ 55                            → SILVER
Score ≥ 35                            → BRONZE
Below that                            → REJECTED
```

### The 16 laws — what each one checks

**Mechanics (Laws 1-6): Does the motion obey Newton?**

| Law | Plain English |
|---|---|
| 1 Newton | If you see muscle activation, you should see arm movement |
| 2 Resonance | The arm trembles at specific frequencies — not random ones |
| 3 Rigid Body | Accelerometer and gyroscope must agree (same rigid body) |
| 4 Heartbeat | A wrist sensor should feel your heartbeat as a tiny recoil |
| 5 Joule Heat | Active muscles produce heat — EMG and temperature must match |
| 6 Jerk | Human arms cannot accelerate faster than 500 m/s³ |

**Consistency (Laws 7-12): Is the signal internally coherent?**

| Law | Plain English |
|---|---|
| 7 IMU Consistency | Accelerometer and gyroscope variance must scale together |
| 8 Continuity | The end of one window and start of the next must connect |
| 9 Cross-Axis | The three axes (x, y, z) must move together — not independently |
| 10 Pointwise Jerk | No single sample can jump impossibly fast |
| 11 Spectral Flatness | Real motion has a peaked frequency spectrum, not a flat one |
| 12 Autocorrelation | Motion has memory — each sample relates to the previous one |

**Hardware Flags (Laws 13-15): Did the hardware behave?**

These are soft flags — they reduce the score but do not immediately reject.

| Law | Plain English |
|---|---|
| 13 Sensor Freeze | Consecutive identical readings = sensor may be stuck |
| 14 Powerline | A spike at exactly 50 or 60 Hz = electrical interference |
| 15 Splice | If the first half and second half of a window differ too much, two sessions were joined |

**Distributional (Law 16): Is the randomness biological?**

| Law | Plain English |
|---|---|
| 16 Innovation Kurtosis | Real biological motion has heavy-tailed residuals. Mathematical generators (like OU processes) always produce Gaussian residuals. This is unfakeable by parameter tuning. |

### The Triple Coherence Firewall

Three laws together block all known synthetic generators:

```
Law  9 — spatial:        axes must be correlated    (blocks: iid Gaussian)
Law 12 — temporal:       signal must have memory    (blocks: white noise)
Law 16 — distributional: residuals must be non-Gaussian (blocks: coupled OU)
```

A generator that defeats all three simultaneously does not exist yet.
It would need correlated axes, temporal memory, AND non-Gaussian residuals —
at which point it requires real biological data as a template.

### Synthetic detection results (v1.7.9)

| Signal | Caught | Blocked by |
|---|---|---|
| Gaussian iid noise | 10/10 | Law 2 (resonance), Law 12 |
| Pure sine wave | 10/10 | Law 9 (axes don't correlate naturally) |
| Clipped / saturated | 10/10 | Law 2 |
| Powerline 60Hz | 10/10 | Law 16 (Gaussian residuals) |
| Coupled OU standard | 10/10 | Law 16 |
| Coupled OU aggressive | 9/10 | Law 16 |
| Coupled OU slow drift | 10/10 | Law 16 |

**Real data false positive rate: 0/30**

---

## Layer 2 — Is This Signal From a Human?

Runs after all windows are processed, across the whole session.

### What it measures

**Hurst Exponent (H):** Measures how self-similar the signal is across time.
Real human motion: H between 0.70 and 0.95.
Too low (H < 0.70): random, no biological persistence.
Too high (H > 0.95): frozen or mechanical, no biological variability.

**Sample Entropy (SE):** Measures complexity.
Real human motion: SE > 0.3.
Too low: signal is too regular — synthetic or hardware fault.

**BFS Score:** Combined biological fidelity score.
Must be > 0.35 to grade as HUMAN.

### What it catches that Layer 1 misses

A frozen sensor with zero gyro input can slip past Layer 1
(Law 16 skips without gyro, Law 13 is only a soft flag).
Layer 2 catches it because frozen data has H ≈ 1.0 — too persistent
to be biological motion.

---

## Layer 3 — Does the Motion Match Its Label?

Encodes the motion using sentence-transformers and finds the nearest
match in the certified motion library by cosine similarity.

### SEMANTIC_MISMATCH flag

If the best semantic match has similarity < 0.6 to the claimed label,
the flag SEMANTIC_MISMATCH is added to the result.

**Example:** Someone records a real NinaPro "pinch" gesture but labels it
"power grasp" (intentional mislabeling or annotation error).
Layer 3 retrieves "pinch" at 0.95 similarity.
"Power grasp" scores only 0.31.
SEMANTIC_MISMATCH is flagged.

Layer 3 does not reject — it flags. The downstream system decides
whether to discard or review flagged windows.

---

## Layer 4 — Using Certified Data (Not a Filter)

Layer 4 sits beside the certification chain, not inside it.
It only runs on data that has passed Layers 1-3.

| Sub-layer | What it does |
|---|---|
| 4a — Next Action | Given a certified motion sequence, predicts what motion comes next |
| 4b — Gap Fill | Given two certified endpoints, fills the motion between them using minimum-jerk trajectory |
| 4c — Intent | Given a certified sequence and a text description, recognizes the human intent |

Layer 4 does not produce GOLD/SILVER/BRONZE/REJECTED.
It produces predictions. Think of it as the intelligence built on top
of the quality foundation.

---

## Layer 5 — Does the Scene Match?

Optional. Requires a camera frame alongside the sensor data.

Uses CLIP to compute cosine similarity between the scene image
and the natural language instruction.

If clip_sim < 0.25: VISUAL_MISMATCH flag added (soft, does not change tier).

**Example:** A robot receives a "pick up the cup" instruction
while the camera shows an empty table. VISUAL_MISMATCH is flagged.

---

## What Cannot Be Caught (Honest Limits)

| Scenario | Why it passes | What to do |
|---|---|---|
| Real data, correct label, correct scene | This is valid data — it should pass | Nothing. This is the goal. |
| Replay attack with correct label | Signal is genuinely biological | Access control on data collection |
| Label noise at scale | S2S catches semantic mismatch but not all errors | Human review of flagged windows |

The only undefeatable input is real biological data used correctly.
That is the intended outcome.

---

## Output of a Single Window

```python
{
  "tier":               "SILVER",
  "physical_law_score": 67,
  "laws_passed":        ["jerk_bounds", "temporal_autocorrelation", ...],
  "laws_failed":        ["resonance_frequency"],
  "flags":              ["PHYSICS_VIOLATION:resonance_frequency"],
  "law_details":        { "resonance_frequency": { "reason": "HZ_TOO_LOW" } }
}
```

---

## Quick Reference — Version History

| Version | What was added |
|---|---|
| v1.0 | Laws 1-7, basic physics |
| v1.5 | Law 8, session continuity |
| v1.7.0 | Laws 9-12, dual coherence firewall |
| v1.7.5 | Laws 13-15, hardware soft flags |
| v1.7.9 | Law 16, innovation kurtosis — closes the OU generator gap |

**Current state:** 16 laws · 187 tests passing · 36/36 benchmark · 0% false positives on real data

---

## One-Command Start

```bash
pip install s2s-certify

# Certify a single dataset folder
s2s-refinery --input /your/data/folder --output report.csv

# Or in Python
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
pe = PhysicsEngine()
result = pe.certify(imu_raw=your_window, segment="forearm")
print(result["tier"], result["physical_law_score"])
```

---

*github.com/timbo4u1/S2S · pypi.org/project/s2s-certify · v1.7.9*
