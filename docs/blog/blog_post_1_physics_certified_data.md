# Why Physical AI Needs Physics-Certified Training Data

*Posted by Timur Davkarayev · March 2026*

---

A prosthetic hand that fails its user is not a software problem. It is a data problem.

Every myoelectric prosthetic hand is trained on EMG recordings of human muscle signals. The hand learns: when these electrical patterns appear, move these fingers this way. The training data is the foundation. If the foundation is corrupt, the hand fails — not occasionally, but systematically, in the same wrong ways, every time.

The problem is that nobody certifies motion training data before it goes into models.

## What "Bad" Motion Data Looks Like

There are three categories of bad data that regularly appear in motion datasets:

**Synthetic data that violates physics.** Many labs augment real recordings with computer-generated motion. Synthetic data is cheap, easy to generate at scale, and completely unconstrained by the laws of physics. A synthetic wrist movement can have jerk values of 2,000 m/s³ — four times higher than the biomechanical maximum for human motion (Flash & Hogan, 1985). A robot trained on this data learns movement patterns that no human ever makes. When it tries to mirror a real user, it fails.

**Corrupted sensor recordings.** IMU chips from the same rigid body must satisfy a coupling constraint — the accelerometer and gyroscope readings from the same device are related by rigid body kinematics. When a sensor is loose, miscalibrated, or recording at the wrong rate, this coupling breaks. The data looks valid numerically but describes physically impossible motion.

**Mislabeled actions.** Sitting still and eating soup look almost identical in a wrist IMU at 20Hz. Both show low jerk, low coupling, slow oscillation. Without domain-level physics checks, a classifier confidently labels eating as rest — and the prosthetic hand trained on this data fails to respond correctly when the user reaches for a spoon.

## The Certification Approach

S2S applies seven biomechanical physics laws to every sensor record before it enters a training dataset:

**Jerk bounds** (Flash & Hogan, 1985) — Human voluntary motion has a maximum jerk of approximately 500 m/s³. Values above this indicate synthetic augmentation, sensor error, or impact artifacts. Every S2S record has jerk verified.

**Rigid body kinematics** (Euler) — For any rigid segment, the relationship a = α×r + ω²×r must hold. When it does not, the sensor is moving relative to the body — loose attachment, slip, or calibration drift. S2S rejects records that violate this.

**IMU coupling consistency** — Accelerometer and gyroscope from the same chip must show correlated motion. Decorrelation indicates sensor fault or data corruption. S2S measures this as a Pearson correlation and requires r ≥ 0.15.

**Resonance frequency** (Flash & Hogan, 1985) — The forearm has a characteristic tremor frequency of 8-12Hz. This appears in every real human wrist recording as a small oscillation. Synthetic data typically lacks this. S2S checks for it.

**Newton F=ma coupling** — When EMG data is present, muscle activation must precede the resulting acceleration by 50-100ms. This is the electromechanical delay of human muscle. S2S verifies this temporal relationship when EMG is available.

**BCG heartbeat signature** (Starr, 1939) — The heartbeat produces a small mechanical pulse visible in a wrist-mounted accelerometer. This ballistic cardiogram is present in every living human's wrist IMU at rest. Synthetic data lacks it. S2S uses this as a liveness check.

**Joule heating consistency** (Ohm, 1827) — EMG signal power should match the thermal output measured by a skin-contact thermometer. When it does not, one of the sensors is faulty.

## A Real Comparison

Here is what S2S produces on real versus synthetic data from the same action (reach forward):

```
Real human hand (iPhone 11 IMU):
  rigid_body_kinematics: r=0.35  PASS
  imu_coupling:          r=0.35  PASS
  jerk_bounds:           187 m/s³ PASS
  resonance_frequency:   9.2 Hz  PASS
  Score: SILVER 69/100

Synthetic augmented data:
  rigid_body_kinematics: r=-0.01 FAIL
  imu_coupling:          r=-0.01 FAIL
  jerk_bounds:           2,340 m/s³ FAIL
  resonance_frequency:   not detected FAIL
  Score: BRONZE 53/100
```

S2S correctly separates real from synthetic using physics alone — no labels, no ground truth, no manual review.

## Why This Matters for Prosthetics Specifically

A myoelectric prosthetic hand typically trains on 50-200 hours of EMG recordings. At a 100Hz sample rate, that is 18-72 million records. No human can manually review 72 million records for physical validity.

S2S can process 103,000 records per training run in minutes, on a laptop, with zero dependencies. Every record that exits the pipeline carries an Ed25519 cryptographic signature — machine-verifiable proof that the record passed physics certification at a specific timestamp.

When the prosthetic hand fails, you can ask: which records trained this behavior? Were they certified? If not certified, why not? This is auditable training data for the first time.

## The Open Question

S2S currently achieves 65.9% domain classification accuracy on a dataset of 103,331 records from three public sources (UCI HAR, PAMAP2, WISDM). The classifier separates five motion domains: PRECISION, SOCIAL, LOCOMOTION, DAILY_LIVING, and SPORT.

The main limitation is sample rate. At 20Hz (the rate of consumer phone IMUs), the PRECISION and DAILY_LIVING domains are physically indistinguishable — eating soup and precision assembly look identical. At 100Hz, they separate. The physics is correct; the sensor resolution is the bottleneck.

This is not a software problem that can be solved with more data or a better model. It is a measurement problem. The answer is better sensors — or accepting that 20Hz IMUs cannot distinguish fine manipulation from rest, and designing training pipelines accordingly.

## Code

S2S is open source under BSL-1.1 (free for research, converts to Apache 2.0 in 2028):

```
https://github.com/timbo4u1/S2S
```

Live demo — open on your phone:
```
https://timbo4u1.github.io/S2S
```

Pose certification demo (camera + skeleton):
```
https://timbo4u1.github.io/S2S/pose.html
```

Preprint: hal-05531246v1, HAL Open Science, February 2026.

---

*If you are working on prosthetics, surgical robotics, or humanoid motion training data and want to discuss physics certification, contact timur.davkarayev@gmail.com.*
