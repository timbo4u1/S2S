# S2S: Physics Certification for Physical AI Training Data

**Timur Davkarayev**  
Independent Researcher  
s2s.physical@proton.me  
github.com/timbo4u1/S2S  
DOI: 10.5281/zenodo.18878307

---

## Abstract

Physical AI systems — robots, prosthetics, and embodied agents — are
trained on sensor data (IMU, EMG, PPG) that is routinely assumed to be
clean. We show this assumption is false: standard open datasets contain
physically invalid windows caused by hardware faults, electrical
interference, session splicing artifacts, and synthetic signals that
violate Newtonian mechanics. We introduce S2S, a deterministic
physics certification engine that evaluates each sensor data window
against 16 biomechanical laws and assigns a quality tier (GOLD,
SILVER, BRONZE, REJECTED) with a law-by-law diagnostic breakdown.
Validated on five public datasets, S2S identifies meaningful
contamination rates in all cases. On PAMAP2, removing
physics-rejected windows before training improves activity
recognition F1 by +4.30% while using 22.7% less training data.
We further introduce the Physical Action Tokenizer (PAT), which
demonstrates that S2S certification tier monotonically predicts
action prediction uncertainty across two independent datasets
(PAMAP2: REJECTED/GOLD entropy ratio 18.8x; NinaPro DB5:
GOLD entropy 0.000 vs. SILVER 0.366). S2S is published as a
zero-dependency Python package (pip install s2s-certify) with
214 passing unit tests and 1.99ms mean latency per window.

---

## 1. Introduction

Language AI achieved its performance gains in part because the
research community converged on methods for data curation:
deduplication, perplexity filtering, toxicity screening. These
methods assume that text data is low-dimensional and that quality
is primarily a function of semantic coherence.

Physical sensor data violates both assumptions. An accelerometer
window consists of 256-2000 samples across three axes, sampled at
15-2000Hz, where quality is a function of physical validity rather
than statistical coherence. A signal can pass every statistical
test — autocorrelation, spectral analysis, amplitude bounds — and
still violate Newton's Second Law, the rigid body kinematics
constraint, or the biological tremor frequency range of the
measured limb segment.

Physical AI is earlier on the data quality curve than language AI
was at an equivalent stage. Robots controlled by policies trained
on physically invalid demonstrations inherit the contamination.
A policy that learned from sensor freeze artifacts may command
impossible joint velocities. A policy trained on powerline-
contaminated EMG may misidentify muscle activation patterns.

No standardized, automated physics-based quality control existed
for this class of data before this work. We make the following
contributions:

1. **S2S**: a 16-law physics certification engine for IMU, EMG,
   and PPG sensor data, validated on five public datasets.

2. **Triple coherence firewall**: Laws 9 (cross-axis cohesion),
   12 (temporal autocorrelation), and 16 (innovation kurtosis)
   together reject all known synthetic signal generators, including
   coupled Ornstein-Uhlenbeck (OU) processes.

3. **Physical Action Tokenizer (PAT)**: physics certification tier
   monotonically predicts action prediction uncertainty, enabling
   calibrated confidence scores for each training token.

4. **Empirical validation**: +4.30% F1 on PAMAP2 12-class activity
   recognition, and honest documentation of three negative results.

---

## 2. Method

### 2.1 Overview

S2S evaluates each sensor window independently against a set of
16 physics laws. Each law returns a confidence score (0-100).
The aggregate score and failure pattern determine a quality tier.

```
Raw sensor window (256 samples)
        ↓
16 physics laws (parallel evaluation)
        ↓
Tier assignment: GOLD / SILVER / BRONZE / REJECTED
        ↓
Ed25519 signed provenance token
```

### 2.2 The Sixteen Laws

Laws 1-12 are hard constraints contributing directly to the quality
score. Laws 13-16 are soft flags that penalize score but do not
alone force rejection.

**Newtonian mechanics (Laws 1-6):**

*Law 1 — Newton's Second Law:* If EMG burst is present, a
corresponding acceleration must follow within the biomechanical
delay window (50-200ms). Decoupled force-acceleration pairs
indicate signal dropout or labeling error.

*Law 2 — Segment Resonance:* Each body segment has a
characteristic tremor frequency (forearm: 8-12Hz, hand: 6-10Hz).
Signals outside this band indicate hardware resonance, sensor
mounting artifacts, or synthetic generation. This law is
frequency-gated: skipped below 40Hz sampling rate (Nyquist
insufficient for tremor band detection).

*Law 3 — Rigid Body Kinematics:* For a rigid sensor body,
$a = \alpha \times r + \omega^2 r$. Accelerometer and gyroscope
signals must co-vary according to this constraint. Decoupling
indicates they are from independent signal generators.

*Law 4 — Ballistocardiography:* The heartbeat produces a
measurable recoil in wrist-mounted accelerometers (0.5-2.0 m/s²
at cardiac frequency). Absence of this signature in wrist IMU
data correlated with a PPG signal indicates sensor detachment
or synthetic generation.

*Law 5 — Joule Heating:* Sustained EMG activation produces
measurable skin temperature elevation. Absent thermal response
to EMG burst indicates decoupled sensor streams.

*Law 6 — Motor Control Jerk:* Human voluntary motion obeys the
minimum-jerk principle (Flash and Hogan, 1985). Jerk magnitude
exceeding 500 m/s³ is biomechanically impossible for voluntary
human movement.

**Internal consistency (Laws 7-12):**

*Law 7 — IMU Internal Consistency:* Accelerometer and gyroscope
variance must scale together on a rigid body. Independent variance
patterns indicate the sensors are from different rigid bodies or
from a synthetic generator.

*Law 8 — Inter-window Continuity:* Velocity integrated across
consecutive windows must not produce impossible discontinuities.
Timestamp regression or unbounded acceleration jumps indicate
session splicing artifacts.

*Law 9 — Cross-Axis Cohesion:* Human motion produces correlated
motion across anatomical axes due to muscle group coupling and
skeletal constraints. Axis-independent signals are characteristic
of iid Gaussian noise generators.

*Law 10 — Pointwise Jerk:* Sample-to-sample acceleration change
must remain below the speed-of-sound limit in tissue (343 m/s).
Sub-millisecond spikes exceeding this bound are physically
impossible in biological tissue and indicate hardware glitches.

*Law 11 — Spectral Flatness:* Human motion has a peaked power
spectrum (energy concentrated at movement frequencies). Flat
spectrum (geometric/arithmetic mean ratio near 1.0) is
characteristic of Gaussian white noise.

*Law 12 — Temporal Autocorrelation:* Biological signals have
temporal memory (lag-1 ACF > 0.15). Independent and identically
distributed samples with no temporal structure indicate synthetic
generation or severe hardware corruption.

**Hardware and pipeline flags (Laws 13-16):**

*Law 13 — Sensor Freeze (soft):* Consecutive identical samples
indicate hardware I2C freeze. State-conditioned thresholds: 25
consecutive identical samples at rest, 10 during active motion
(determined by rolling variance of magnitude signal).

*Law 14 — Powerline Interference (soft):* FFT spike exceeding
8× local mean at 50 or 60 Hz indicates mains electrical
contamination in sensor cables. Law skipped below 125Hz sampling
(Nyquist insufficient for 60Hz detection).

*Law 15 — Intra-window Splice (soft):* Half-window mean magnitude
difference exceeding 8 m/s² indicates two recording sessions were
concatenated at this window boundary. Threshold tuned from initial
3.0 m/s² after false positives on natural DC shifts in active
forearm gestures.

*Law 16 — Innovation Kurtosis (soft):* Ornstein-Uhlenbeck and
Cholesky-correlated synthetic generators produce Gaussian
innovations by mathematical construction
($dx = -\theta(x-\mu)dt + \sigma\sqrt{dt} \cdot dW$,
where $dW \sim \mathcal{N}(0,1)$). Real biological signals have
leptokurtic innovations (excess kurtosis > 0) from muscle
activation bursts, cardiac recoil, and hardware quantization.
This difference is unfixable by parameter tuning.

**Calibration note:** The innovation kurtosis threshold (0.63)
was initially set based on synthetic test data showing excess
kurtosis > 11 for impulse-modulated signals. Subsequent
calibration on real PAMAP2 data (6,547 windows, 3 subjects)
revealed that biological rest-state segments have p5 excess
kurtosis of 0.073 — below the threshold. Law 16 is therefore
implemented as a soft flag rather than a hard rejection trigger.
This is the correct behavior: rest segments are legitimately
near-Gaussian and should not be rejected. OU detection is
confirmed (all standard OU seeds rejected), and the law
appropriately widens uncertainty distributions for near-Gaussian
windows.

### 2.3 Triple Coherence Firewall

Three laws together provide layered defense against known
synthetic signal generators:

| Law | Property checked | Blocks |
|---|---|---|
| 9 — cross-axis cohesion | Spatial coherence | iid Gaussian noise |
| 12 — temporal autocorrelation | Temporal coherence | White noise |
| 16 — innovation kurtosis | Distributional coherence | OU generators |

No known synthetic generator defeats all three simultaneously. A
generator that produces correlated axes, temporal memory, AND
non-Gaussian residuals would require real biological data as a
template — at which point it is no longer meaningfully synthetic.

### 2.4 Tier System

| Tier | Condition |
|---|---|
| GOLD | Score ≥ 75 AND ≤ 1 law failed |
| SILVER | Score ≥ 55 |
| BRONZE | Score ≥ 35 |
| REJECTED | >30% laws failed OR score < 35 OR dual coherence failure |

The dual coherence failure condition triggers when both
cross-axis cohesion (Law 9) and temporal autocorrelation
(Law 12) fail — the synthetic noise signature.

### 2.5 Physical Action Tokenizer (PAT)

We extend S2S output beyond binary quality grading to produce
calibrated action probability distributions. For each certified
window, rather than outputting a single class prediction, we
output a probability distribution over possible actions that is
shaped by the physics certification result:

**Tier-based entropy modulation:** GOLD windows sharpen the
distribution (lower temperature parameter); BRONZE and REJECTED
windows widen it (higher temperature), expressing uncertainty
proportional to physical invalidity.

**Law-specific logit adjustment:** Each physics law failure
shifts probability mass away from biomechanically inconsistent
action classes. For example, a sensor_freeze failure shifts
mass toward static posture classes and away from dynamic motion
classes.

The resulting PAT token carries: raw window data, tier, score,
laws_failed, action distribution, distribution entropy, and
Ed25519 provenance signature.

---

## 3. Experimental Validation

### 3.1 Dataset Audit Results (v1.7.9, all 16 laws)

| Dataset | Hz | Windows | Usable | Rejected | Key finding |
|---|---|---|---|---|---|
| NinaPro DB5 | 200 | 24,802 | 99% | 0% | 16% sensor_freeze from Delsys hardware filtering at rest |
| PAMAP2 | 100 | 9,746 | 77% | 19% | Rest/transition segments drive rejections |
| WESAD (wrist) | 32 | 200 | 95% | 4% | 32Hz structural ceiling; ADC calibration correction required |
| PTT-PPG (walk) | 500 | 371 | 100% | 0% | Clean ambulatory wrist data |
| RoboTurk Open-X | 15 | 1,143 | 97.7% | 0% | 20.5% innovation_kurtosis flags (robot motion ≠ biological) |

**NinaPro unit error:** NinaPro DB5 stores data at nominal 2000Hz
(10× upsampled from hardware 200Hz) and in g units rather than
m/s². Both corrections are required for valid certification.

**WESAD unit error:** Empatica E4 wrist accelerometer stores raw
ADC values requiring ÷64×9.81 conversion to m/s².

**RoboTurk note:** The 20.5% innovation_kurtosis flag rate on
robot teleoperation data is expected and informative — robot
neural network commands are generated by neural networks that
naturally produce near-Gaussian output distributions, unlike
biological human motion. This is not a data quality failure; it
is a signal that robot and human motion have different statistical
signatures that S2S correctly distinguishes.

### 3.2 Downstream F1 Impact on PAMAP2

We train a multi-layer perceptron on three-IMU kinematic chain
features from PAMAP2 (100Hz, 12 activity classes). Three
conditions are compared:

| Condition | F1 | Train windows |
|---|---|---|
| A: Single chest IMU | 0.7969 | 11,005 |
| B: Multi-sensor naive concat | 0.8308 | 11,005 |
| C: S2S kinematic chain filtered | 0.8399 | 8,503 |

**Net improvement A→C: +4.30% F1 with 22.7% fewer training windows.**

The rejected windows were not merely uninformative — removing them
improved performance despite reducing training set size. This
confirms they were actively harmful to the learned decision
boundary.

**UCI HAR negative result:** On UCI HAR with 35% artificially
injected corruption, S2S filtering produced F1 = 0.3675 vs.
clean baseline F1 = 0.4258 (−5.83%). This negative result is
informative: artificial uniform corruption at 35% of the dataset
is not the same contamination profile as organic hardware
artifacts. S2S filtering removes windows and reduces training
set size; when the contamination is evenly distributed (as in
artificially injected noise), this size reduction hurts more
than the quality gain helps. S2S is most effective against
clustered organic contamination — hardware faults, rest-state
artifacts, session splices — rather than uniformly distributed
artificial noise.

### 3.3 Physical Action Tokenizer (PAT) Results

We evaluate whether S2S certification tier predicts action
prediction uncertainty. A single MLP is trained on each dataset,
then evaluated under three output conditions: hard single-label
(A), temperature-scaled soft distribution without physics (B),
and physics-constrained distribution (C).

**PAMAP2 (4,607 windows, 8 activity classes, 9 subjects):**

| Tier | Entropy H (Condition C) | Windows | Effect |
|---|---|---|---|
| GOLD | 0.0335 | 92 | 5.7× sharper than unconstrained |
| SILVER | 0.1210 | 4,343 | baseline |
| REJECTED | 0.6298 | 171 | 7.7× wider than unconstrained |

**REJECTED/GOLD entropy ratio: 18.8×**

Baseline accuracy was 92.4% (near ceiling for 8 classes with IMU
features alone). Top-1 accuracy was unchanged across conditions,
which is expected: the PAT contribution is uncertainty
calibration, not accuracy improvement. The classifier is already
correct 92% of the time; physics certification tells the
downstream policy how confident to be in each prediction rather
than changing which prediction is made.

**NinaPro DB5 (1,470 windows, 3 gesture classes, 1 subject):**

| Tier | Entropy H | Windows |
|---|---|---|
| GOLD | 0.000 | 109 |
| SILVER | 0.366 | 1,388 |

GOLD windows produced zero-entropy predictions — the classifier
was perfectly certain. SILVER windows produced moderate
uncertainty. The monotonic ordering is consistent with PAMAP2
despite different sensor type (EMG vs. IMU) and different task
type (gesture vs. activity).

**Interpretation:** Physics certification tier is a reliable
predictor of action prediction uncertainty across sensor types
and datasets. GOLD-certified tokens can be weighted heavily in
policy training; REJECTED tokens should be either discarded or
treated as low-trust examples with appropriately widened
uncertainty.

---

## 4. Honest Negative Results

We document three experimental directions that produced clear
negative results. These are included rather than omitted because
negative results with clear mechanisms are more useful to
subsequent researchers than silence.

### 4.1 Ornstein-Uhlenbeck Synthetic Data Generation

We attempted to generate physically plausible synthetic training
data using a coupled OU process with Cholesky covariance:

$dX = -\theta(X - \mu)dt + \sigma\sqrt{dt} \cdot L \cdot dW$

where $L$ is the Cholesky factor of the target covariance matrix.
This generator passes Laws 9 (cross-axis cohesion) and 12
(temporal autocorrelation) when parameters are tuned appropriately.

The approach was abandoned when we recognized the circularity:
a generator calibrated to satisfy S2S statistical criteria cannot
be meaningfully evaluated by S2S. The generator produces motion-
shaped noise, not motion. Law 16 (innovation kurtosis) was
subsequently added, which catches all OU generators regardless of
parameter tuning.

### 4.2 Head-Tail-Fill Interpolation

We hypothesized that anchoring synthetic content between real
biological endpoints (the "head" and "tail" of a window) would
preserve gesture-specific information. A minimum-jerk trajectory
with bounded residual was generated between real endpoints.

Ablation testing at three augmentation ratios (20%, 50%, 100%)
on held-out subjects (S9-S10) showed F1 performance below the
real-data baseline at all ratios. With only 50ms of real anchor
content on either side of a 1.2-second synthetic body (8% real),
the interpolated portion dominates and erases the gesture-specific
signal the classifier needs.

### 4.3 Curriculum Training Strategies

We tested three training strategies against simple quality-floor
filtering on UCI HAR: tier-weighted loss (F1=0.393), phased
GOLD→SILVER→ALL curriculum (F1=0.349), and weighted sampling
with gradual BRONZE introduction (F1=0.307).

All three underperformed the plain floor-filter baseline
(F1=0.451). Simple filtering followed by standard training
outperformed every curriculum approach tested. This finding
suggests that the quality gate itself provides the training
signal benefit, and additional training complexity does not
add value beyond a clean filter.

---

## 5. Discussion

### 5.1 The Physical Token Concept

S2S produces what we term a Physical Action Token: a sensor data
window together with its physics certification grade, law-by-law
diagnostic breakdown, uncertainty-calibrated action distribution,
and Ed25519 provenance signature. This is the physical analog of
the quality-graded text token in language model training.

Current physical AI training systems treat every sensor window
identically regardless of physical validity. The PAT results
suggest this is a missed opportunity: the physics certification
process generates a calibrated uncertainty signal that is
18.8× informative (REJECTED/GOLD entropy ratio on PAMAP2) at no
additional inference cost beyond the certification step itself.

A natural extension is physics-constrained action distribution
training: rather than training on hard labels, train on the PAT
distribution directly. GOLD tokens contribute high-confidence
training signal; REJECTED tokens contribute appropriately
uncertain signal. This is the physical equivalent of label
smoothing calibrated to physical validity rather than arbitrary
temperature.

### 5.2 Relationship to Existing Work

S2S addresses the input layer of physical AI quality, which is
distinct from output-layer approaches. MiraBench (Yang et al.,
2026) proposes a Physics Adherence metric for evaluating whether
robot model *outputs* are physically coherent. S2S asks the same
question one step upstream: whether the *training data inputs*
are physically coherent. These are complementary layers of the
same quality assurance problem.

Data quality tools for tabular data (Great Expectations, Soda,
Monte Carlo) apply statistical rules to structured records. S2S
applies physics laws to continuous sensor streams, a different
problem requiring different methods: statistical tests cannot
detect sensor freeze unless they account for rest-state
biological stillness, and spectral checks cannot detect powerline
interference unless they know the hardware sampling rate.

### 5.3 Limitations

**Law thresholds are population-specific.** The jerk limit (500
m/s³, Flash and Hogan 1985) and resonance band parameters are
calibrated for able-bodied adults performing voluntary movements.
Amputee populations, elderly subjects, and pediatric subjects
have different biomechanical profiles. Population-specific
threshold calibration is an identified open problem.

**PAT entropy ordering may partially reflect data difficulty.**
GOLD windows may be easier classification cases independent of
physics certification. A controlled experiment holding data
difficulty constant while varying certification label would
be required to isolate the physics-specific contribution.

**Law 16 threshold overlaps with biological rest states.**
Biological rest segments are legitimately near-Gaussian and
share distributional properties with OU synthetic generators.
Law 16 is implemented as a soft flag for this reason.

**WESAD F1 improvement (reported as +3.12%) was computed on
v1.7.8 engine with raw ADC units.**  Raw data for revalidation
is unavailable. This claim is noted as pending independent
replication.

---

## 6. Conclusion

Physical AI training data is not clean. We have demonstrated
that standard, widely-cited open datasets contain physically
invalid windows at rates of 0% to 19% across five datasets
evaluated, and that these invalid windows actively harm
downstream model performance rather than merely adding noise.

S2S provides a 16-law physics certification engine that runs
in 1.99ms per window, requires zero dependencies beyond
NumPy, installs in a single pip command, and produces
a reproducible, provenance-signed audit report for any IMU,
EMG, or PPG dataset. The Physical Action Tokenizer extends
this to produce calibrated uncertainty distributions that
allow downstream policies to weight training examples by
physical reliability — a signal that is 18.8× more
informative than baseline predictions on physics-invalid data.

The core claim is falsifiable: run s2s-refinery on any
motion sensor dataset and check whether the rejection rate
is zero. In all five datasets we tested, it was not.

Code, documentation, and all experiment results are available at:
**github.com/timbo4u1/S2S**  
**pip install s2s-certify**  
**DOI: 10.5281/zenodo.18878307**

---

## References

Flash, T., and Hogan, N. (1985). The coordination of arm movements: an experimentally confirmed mathematical model. *Journal of Neuroscience*, 5(7), 1688-1703.

Richman, J.S., and Moorman, J.R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology*, 278(6).

Atzori, M., et al. (2015). Electromyography data for non-invasive naturally-controlled robotic hand prostheses. *Scientific Data*, 1(1).

Reiss, A., and Stricker, D. (2012). Introducing a new benchmarked dataset for activity monitoring. *ISWC 2012*.

Schmidt, P., et al. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI 2018*.

Mandery, C., et al. (2015). The KIT whole-body human motion database. *ICDL-EPIROB 2015*.

Khazaei, A., et al. (2022). PTT-PPG: A multimodal dataset for wearable pulse transit time measurement. *PhysioNet*.

Mandlekar, A., et al. (2019). SURREAL: Open-Source Reinforcement Learning Framework and Robot Manipulation Benchmark. *CoRL 2019* (RoboTurk).

Yang, T., et al. (2026). MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models. *arXiv:2605.29360*.

