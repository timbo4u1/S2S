# C-MHAD diagnostic validation protocol

Written and committed BEFORE any S2S certification run on C-MHAD.
Engine: 25453b3c47780ea502a12d23abaeaff9400e9c64 (origin/main when this file was committed).
Any engine change after this commit invalidates the run. Fix it in a separate PR, rerun everything, report both runs.

## Questions
1. What does S2S report on unmodified C-MHAD waist-IMU recordings?
2. Does S2S detect known faults injected into copies of those recordings?
Not tested here: downstream accuracy, action tokenization, annotation alignment.

## Data handling
- 120 trials, 12 subjects, 50 Hz, waist sensor.
- Accel kept in m/s^2 exactly as recorded. Resting |a| is about 11.4 m/s^2, not 9.81. Per-trial values are in trial_audit.csv. No rescaling.
- Gyro converted from deg/s to rad/s.
- Real timestamps converted to ns relative to the first sample. The 51 internal one-sample dropouts stay as timestamp gaps. No interpolation, no zero padding.
- segment="waist": resonance is expected NOT_ASSESSED (no calibrated parameters). Powerline is expected NOT_ASSESSED (50 Hz is below the 125 Hz minimum). Jerk uses the arm limit, so jerk results are provisional.
- Windows: 256 samples, stride 128. A "gap window" contains a real dropout.

## Part A: baseline audit (unmodified data)
One engine per trial, windows in order (continuity state carries across windows).
Report per law PASS/FAIL/NOT_ASSESSED counts, assessed-coverage distribution, tier distribution, and flag rate on gap vs non-gap windows.
These are observed flag rates, not false-positive rates: nothing labels windows clean or corrupt.

## Part B: fault injection
Windows: all non-gap windows. A fresh engine per window (no cross-window state). Each fault is injected into a copy, and the original is certified too (paired). RNG seed 20261101.

| fault | definition | severities |
|---|---|---|
| freeze | repeat one sample for k samples, all 6 channels, random position | k = 8, 16, 32 |
| spike | add offset to one accel axis for one sample | 5, 20, 50 m/s^2 |
| gap | delete m consecutive samples, keep real timestamps | m = 1, 3, 10 |
| scale | multiply accel | x1.2, x2, x9.81 |
| clip | clip each accel axis at a percentile of its absolute values in the window | p95, p90, p80 |
| jitter | Gaussian noise on timestamps, kept monotonic | sigma = 1, 3, 6 ms |
| splice | second half replaced by a window from another subject | one level |
| axis_swap | swap accel x and z (gyro too) | one level |

Detection, defined in advance:
- Primary: the faulted copy has a law in laws_failed that its paired original does not.
- Secondary: the faulted copy has a lower tier than its original.
- Targeted: the expected law fails. Expected: freeze -> sensor_freeze; spike -> pointwise_jerk or jerk_bounds; splice -> intra_window_splice. No specific law is expected for gap, jitter, scale, clip or axis_swap within a single window (inter_window_continuity needs a previous window).

Predictions recorded in advance:
- freeze k=8 is below the documented thresholds (10 samples in motion, 25 at rest), so a miss is expected.
- axis_swap and x1.2 scale preserve magnitude and are likely undetected.
- A one-sample gap is likely undetected.
- A splice from another subject at similar magnitude is likely below the 8 m/s^2 mean-magnitude threshold.

Report recall per fault x severity with 95% intervals from a subject-level bootstrap (12 subjects), plus per-subject spread.

## Rules
- No threshold or engine tuning on C-MHAD.
- Originals are never modified. Faulted copies exist only in memory.
- Null and negative results are reported as they come out.

## Amendment 1 (committed before any S2S run on C-MHAD)
Supersedes the matching items above; everything else stands.
1. Part A: non-overlapping windows (stride 256), one engine per trial, so continuity compares adjacent windows. Part B keeps stride 128 with a fresh engine per window.
2. Part A gap vs non-gap: only about 50 gap windows. Report counts and intervals, draw no conclusion. The dropout question is answered by the Part B gap fault.
3. Added Part B faults, the unit errors this project has actually hit: accel divided by 9.81 (g read as m/s^2); gyro left in deg/s (x57.2958, conversion skipped).
4. Windows stay 256 samples. gap: take 256+m samples, delete m consecutive ones inside. freeze: k = total identical consecutive samples.
5. Detection denominators exclude pairs where detection is impossible by construction (original already fails the target law, or is already REJECTED). Report the excluded count per cell.
6. Added predictions from the documented freeze thresholds (10 identical samples in motion, 25 at rest): k=16 detected only in windows judged in motion; k=32 detected in both; k=8 missed.
