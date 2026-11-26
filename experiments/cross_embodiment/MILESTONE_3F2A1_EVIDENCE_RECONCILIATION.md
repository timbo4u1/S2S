# Milestone 3F.2A.1 — Final MM-Fit Timestamp, Provenance and Evidence Reconciliation

Resolves independent-review findings on the w19 acquisition. Read-only apart from regenerated verification artifacts under `experiments/cross_embodiment/milestone_3f2a/verification/` (now 13 artifacts + `verify_mmfit_w19.py`). No engine/schema/adapter changes; no downloads; nothing staged.

## 1. Timestamp duplicate investigation — RESOLVED (mixed mechanisms)

`timestamp_duplication_mechanism.csv` (per-stream, sampled ≤5,000 duplicate groups each):

| Stream | Mechanism | zero-Δt | Evidence |
|---|---|---|---|
| `eb_l_acc/gyr` | **QUANTIZATION_ONLY** | 45.4% | all sampled dup groups have *different* XYZ → real ~90 Hz samples sharing ms timestamps |
| `sp_r_acc` | MIXED | 6.2% | 4,798 identical vs 202 different (of 5,000) → held/batch dominates |
| `sp_r_gyr` | MIXED | 14.6% | 3,633 identical vs 1,367 different |
| `sp_r_mag` | MIXED | 0.6% | 510 identical vs 249 different |
| `sw_l/sw_r acc/gyr` | QUANTIZATION_ONLY | 0.2–0.8% | all different values; rare ms collisions at ~103 Hz |
| `sw_l_hr` | NONE | 0% | strictly monotonic |

Record density reproduces nominal rates (eb ~91/s, sp ~211/s, sw ~103–104/s, hr ~1/s). **Conclusion:** duplication is *not* explained by ms quantization alone — smartphone streams also contain held/batch-delivered samples (identical values sharing a timestamp). Per-sample timestamp quality = **DEGRADED** for all high-rate streams; **OK** for HR. No resampling/interpolation invented; downstream consumers must tolerate duplicate timestamps or treat per-sample timing as approximate.

## 2. Synchronization metric terminology — CORRECTED

`clock_sync_metrics.csv` now carries `metric_definition = "first_sample_ts_diff_per_shared_frame_ms"`. All offset columns are renamed in the manifest (`first_sample_ts_diff_*`). These compare the **first sensor-sample epoch-ms timestamp associated with each shared Frame value** — they are *not* direct clock offsets. Verified retained: shared epoch-ms wearable representation, frame-index association, video relative PTS, ±1-frame indexing uncertainty, drift UNKNOWN. No exact wearable↔video clock claim is made (video has no absolute clock).

## 3. Frame mapping — VERIFIED with explicit ambiguity

Evidence chain: MM-Fit loader (`utils/dataset.py`) + EDA document the Frame column as the left RGB-D camera frame reference; labels and (documented) pose arrays use the same frame domain; video has 52,561 frames (indices 0–52,560) vs wearable max 52,561 → **zero-based vs one-based ambiguity preserved** (±1 frame ≈ 33 ms). No frame offset applied anywhere. Pose files remain PARTIAL so pose-index cross-check is pending.

## 4. PTS reproducibility — CORRECTED

`verify_mmfit_w19.py` now consumes `verification/w19_rgb_frame_timestamps.csv` (regenerated deterministically via ffprobe; sha256 `1b76b892…934c0d`, 52,561 rows) instead of `/tmp/w19_pts.csv`. `parse_pts_csv()` takes the first and last numeric CSV token per line — tolerant of SEI side-data strings and trailing empty fields (current export is clean `pts,dts,time` rows). Recorded: frame count, first/last PTS, interval stats in `video_metadata.json`. Manifest lists the PTS export as a derived artifact separate from source files.

## 5. Label and inventory corrections — APPLIED

Superseding values (old values remain traceable in the 3F.2A report): **label sets = 26** (was 25), **total repetitions = 258**, **activities = 9**, **NPY streams = 9 inertial/mag + 1 derived-HR = 10** (pose files PARTIAL). Confirmed directly from `w19_labels.csv` (26 rows, 0 overlaps, sorted).

## 6. Unit evidence and device mapping

- MM-Fit paper (IMWUT 2020, Fig. 1/example trace) shows a left-worn smartwatch accelerometer in **m/s²** and gyroscope in **rad/s**. This is *documentary example* evidence consistent with — but not file-verifying — the `sw_l`/`sw_r` streams. Recorded as `DOCUMENTED_EXAMPLE_ONLY`; law-level PhysicsEngine runs still require explicit unit confirmation.
- `sp_r`: Samsung S7 recorded as **plausible** mapping (measured 210 Hz acc/gyro + ~100 Hz mag matches S7 profile, not Huawei P20 @65 Hz) — **not verified hardware identity**; remains UNKNOWN pending source metadata.
- `eb_l`, `sp_r_mag` units: UNKNOWN. No calibration data exists in acquired files — none invented.

## 7. Depth source inventory — RECORDED, NOT ACQUIRED

Zenodo record **7607736** ("MM-Fit Dataset — video and depth") contains `w19_depth.mp4` = **19,014,084 bytes** (~19 MB). Status ladder: source available ✓ / locally acquired ✗ / decoded ✗ / timestamps assessed ✗ / RGB-depth sync assessed ✗. Not downloaded per constraints. (Record also re-lists `w19_rgb.mp4`.)

## 8. PhysicsEngine compatibility — SEPARATED

| Layer | Status |
|---|---|
| Input schema (`imu_raw={timestamps_ns,accel,gyro}`) | COMPATIBLE — mechanical epoch-ms→ns conversion only |
| Unit compatibility | PARTIAL — `sw_*` have documentary m/s²+rad/s example evidence; all others UNKNOWN → law-level checks on those = NOT_ASSESSED |
| Timestamp suitability | DEGRADED — duplicate/quantized ms timestamps; engine windowing must tolerate or document |
| Actual law-level assessment | NOT_RUN — controlled v1.8 test on `sw_l` permissible only as schema smoke test with units flagged DOCUMENTED_EXAMPLE; no threshold/engine changes; BPM never enters PPG paths |

## 9. Modality readiness (supersedes prior matrix)

- **Wearable-only integration**: CONDITIONAL GO — per-device streams verified; units partially documented; timestamp degradation must propagate as NOT_ASSESSED/uncertainty, not silent repair.
- **Human video integrity adapter**: CONDITIONAL GO — RGB decode/PTS verified; integrity checks still unimplemented.
- **Frame-level wearable/video association**: CONDITIONAL GO — frame-index mechanism verified with ±1-frame ambiguity explicit.
- **Sample-level multimodal synchronization**: NOT_ASSESSED — video lacks absolute clock; per-sample ts degraded; only frame-level association supported.
- **Multimodal model training**: CONDITIONAL GO for IMU(+video) experiments on documented windows; NEEDS_EVIDENCE for depth-inclusive or sample-exact-sync designs.

## Constraints honored

No production/source modifications; no downloads (Zenodo record metadata queried only); no training; no staging/commits; original audit reports preserved with corrections marked as superseding.
