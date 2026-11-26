# Milestone 3F.2A — Independent Verification Report

Read-only verification of the MM-Fit w19 acquisition against local files at `~/S2S_data/mmfit_3f2/`. Reproducer: `experiments/cross_embodiment/milestone_3f2a/verification/verify_mmfit_w19.py` (9 derived artifacts in same directory). No source files modified; nothing downloaded; no engine changes.

## 1. File inventory and integrity — VERIFIED

10 wearable `.npy` files (9 inertial/mag + 1 derived-HR) + labels CSV + video present and sha256-recorded; pose_2d/pose_3d separately **PARTIAL** (exist in source zip central directory; range download incomplete — correctly not claimed absent); `sw_r_hr`, `sp_l_*`, depth, EMG, raw PPG correctly classified RAW_UNAVAILABLE/UNKNOWN (absent from source session, not download failures). `w19_rgb_frame_timestamps.csv` found **0 bytes** — PTS derived independently via ffprobe. A stray partial `mm-fit.zip` (1.63 GB, killed download) remains unused in the data dir.

## 2. Wearable stream inspection — VERIFIED

All inertial/mag streams are `(N,5)` float64 `(Frame, Timestamp, X,Y,Z)`; HR is `(N,3)` `(Frame,Timestamp,BPM)`. Zero non-finite cells in every stream. Measured rates reproduce prior audit: sw ~103–104 Hz, sp ~212 Hz, sp mag ~100 Hz, eb ~79.9 Hz, HR ~0.96 Hz. Timestamps are **Unix epoch ms** (verified from raw values ≈1.596e12). Frame column spans 898–52561 across all inertial streams; HR 1118–52540.

## 3. Device/sensor identity — PARTIALLY VERIFIED

Filename prefixes map to device classes per dataset docs (`sw_l/sw_r` watches, `sp_r` pocket phone, `eb_l` earbud). **`sp_r` physical model remains UNKNOWN** — measured mag ~100 Hz is consistent with Samsung S7 (100 Hz nominal), not Huawei P20 (65 Hz); do not relabel. Per-device streams must not be merged into a synthetic IMU.

## 4. Timestamp/sample integrity — VERIFIED with nuance

Per-stream stats in `timestamp_integrity.csv`. High-rate streams have **duplicate epoch-ms timestamps** (e.g. 53,531 in `sp_r_gyr`) and dt=0 intervals — consistent with ms quantization at >100 Hz, **not** duplicated samples; HR is strictly monotonic. No negative intervals; reversals=0. Frame indices monotonic non-decreasing with gaps (frame_unique < span). Distinction recorded: timestamp dups ≠ frame dups ≠ value dups.

## 5. Frame-index → video mapping — VERIFIED WITH BOUNDARY UNCERTAINTY

Video: h264, 1280×720, 30 fps, **52,561 frames**, PTS strictly increasing, median interval 33.33 ms, zero duplicate PTS — independently reproduced. Wearable `Frame` max = 52,561 vs video max index 52,560 → **off-by-one boundary**: either 1-based camera counter or a terminal out-of-range marker. Direct index correspondence is plausible but carries **±1 frame (~33 ms)** uncertainty; zero/one-based indexing not definitively established → correspondence status `VERIFIED_WITH_BOUNDARY_UNCERTAINTY`, not exact.

## 6. Cross-modal synchronization — REPRODUCED, ONE CORRECTION

Wearable↔wearable epoch-ms offsets reproduce exactly: mean −7.40…+15.13 ms, std ≤13.28 ms, max |offset| 73 ms (`clock_sync_metrics.csv`).

**Correction to prior audit**: the video's PTS is *relative* (starts at 0) — it carries **no absolute clock**. Therefore wearable↔video alignment is **frame-index only**, not timestamp-based. Drift wearable↔video is **unmeasurable** beyond the 33 ms frame period; wearable↔wearable drift is bounded by ms quantization but unquantified. Alignment uncertainty: per-frame jitter (std ≤13.3 ms) + up to ±1 frame index (~33 ms) + unquantified drift. No acceptable threshold invented.

## 7. Video/depth — RGB VERIFIED, DEPTH UNRESOLVED

RGB decode path verified via ffprobe frame dump (52,561 rows, no errors). Depth: no depth artifact in acquired files or the w19 zip member list → **UNKNOWN/NOT_ASSESSED**, not assumed from "RGB-D" marketing.

## 8. PhysicsEngine v1.8 compatibility — CONDITIONAL

`certify(imu_raw={timestamps_ns, accel, gyro})` schema confirmed. Blockers documented in `physics_engine_compatibility.json`: axis units are **UNKNOWN** in MM-Fit files — running unit-assuming laws would invent semantics; engine must be fed per-device streams (no synthetic IMU merge); HR is derived BPM with no raw-PPG path; mag has no documented engine role. Law-level results on raw MM-Fit values → **NOT_ASSESSED** until units confirmed.

## 9. Labels — VERIFIED, count corrected

`w19_labels.csv` has **26 rows** (prior report said 25 — corrected), zero overlapping pairs, sorted, frame domain = camera/video frame index, 9 distinct activities. Set-level annotations, not per-rep. Participant id for w19 is undocumented → participant-level split leakage risk until the 10-participant mapping is established.

## 10. Training readiness

See `training_readiness_matrix.json`: A (IMU) / B (video, single session) / C (IMU+video) / D (+mag, sparse) / E (+HR) = CONDITIONAL GO pending units/split-mapping; F (full multimodal incl. depth+pose) = NEEDS_EVIDENCE.

## Per-modality decisions

| Modality | Decision |
|---|---|
| IMU accel/gyro → PhysicsEngine | **CONDITIONAL GO** — schema fits, units UNKNOWN block law-level claims |
| Magnetometer | **CONDITIONAL GO** — data verified, no documented engine role |
| Derived BPM | **CONDITIONAL GO** — usable as derived scalar; never raw-PPG checks |
| RGB video integrity | **CONDITIONAL GO** — decode verified; integrity checks unimplemented |
| Depth | **NOT_ASSESSED** — no artifact acquired |
| Wearable↔video sync | **CONDITIONAL GO** — frame-index mechanism verified w/ ±1 frame + unquantified drift bounds |
| Multimodal training | **CONDITIONAL GO** for A–E on w19-scale data; **NEEDS_EVIDENCE** for full modality set |

No universal approval claimed; actual acquired data supports bounded multimodal integration, not the dataset card's unqualified "synchronized" claim.
