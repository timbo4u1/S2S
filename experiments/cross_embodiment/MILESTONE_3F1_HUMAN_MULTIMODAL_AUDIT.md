# Milestone 3F.1 — Human Multimodal Video + Wearable Dataset Evidence Audit

Read-only audit. No engine/adapter/schema changes, no downloads of video populations, no training, no staging.

## 1. Existing human S2S architecture (verified in code)

- `s2s_pipeline.py` hosts the PhysicsEngine-style certification path (`certify()` producing per-law results consumed by `experiments/cross_embodiment/human_adapter.py` → `human_certify_result_to_assessment`, which maps law outcomes into `quality_schema` CheckResults).
- `human_adapter.py` already separates NOT_ASSESSED reasons and preserves engine evidence refs; it is the boundary between sensor-law diagnostics and the shared report schema.
- v1.8 quality laws operate on **numeric wearable/IMU-type signals** (finite values, units, sampling, noise, freeze/splice/resonance-type laws evidenced across `tests/test_physics_laws.py`, `test_sensor_freeze.py`, `test_splice.py`, `test_sample_entropy.py`, `test_spectral_flatness.py`, `test_wavelet_2d.py`, `test_powerline.py`, `test_column_detect.py`). PAMAP2 experiments exist under `experiments/` (LOSO splits, corruption detection, training-utility audits) using IMU-like tabular wearables.
- Applicability: IMU/accel/gyro/magnetometer → existing laws apply. EMG/PPG → partially (numeric signals; EMG laws unverified, PPG derived-BPM vs raw distinction needed). **RGB/depth video cannot pass through PhysicsEngine** — no pixel-level law exists; video needs modality-integrity checks instead.

## 2. Dataset candidates

### MM-Fit (PRIMARY — recommended)

Source: https://mmfit.github.io/ ; paper IMWUT 4(4) 2020 (doi:10.1145/3432701); video on Zenodo record 7672767 (v2, CC-BY-4.0).

Verified published properties:
- **21 workout sessions, 10 participants**, ~809 minutes, 616 exercise sets, 6,160 reps; 11 labels (squats, lunges, bicep curls, sit-ups, push-ups, triceps extensions, dumbbell rows, jumping jacks, shoulder press, lateral raises, non-activity).
- **Video**: RGB-D (Orbbec Astra Pro), multi-view, **30 Hz**; Zenodo video pack = **39.1 GB** (`w00_rgb.mp4` 2.2 GB … `w20_rgb.mp4` 0.96 GB; per-session MP4s, md5 listed).
- **Wearables** (all time-synchronised per dataset documentation):
  - 2× TicWatch Pro: accel 100 Hz, gyro 100 Hz, **BPM 1 Hz (derived heart rate, not raw PPG)**.
  - Samsung S7 (pocket): accel/gyro 210 Hz, magnetometer 100 Hz.
  - Huawei P20: accel/gyro 500 Hz, magnetometer 65 Hz.
  - Nokia eSense earbud: accel/gyro 90 Hz.
  - Canonical benchmark resamples inertial to 50 Hz; pose at 30 Hz.
- **Pose**: 2D and 3D pose estimates provided (derived, monocular in canonical benchmark).
- **NOT present**: raw PPG, EMG — do not assume availability; only derived BPM at 1 Hz.
- **Sync**: dataset claims time-synchronised multi-device capture; per-stream clock mapping mechanism (shared host clock vs post-hoc alignment) is **not fully documented in public metadata — treat as NEEDS_EVIDENCE until files inspected**.
- License: CC-BY-4.0 (Zenodo video); research use OK with citation. 10 subjects → participant-level splits feasible but small-N.

### TNT15 (SECONDARY — limited fit)

Source: TNT Hannover project page + TNT15_documentation.pdf (v1.2, Jan 2016); research-only license.

Verified: 8 calibrated RGB cameras + 10 IMUs (orientation+acceleration), 4 actors, 5 activities (walking, running-in-place, arm rotations, jumping/skiing, punching); ~4.5 min, ~13k frames @50 Hz; silhouettes, projection matrices, 3D meshes; citation required; **research purposes only**.

Limitations for S2S: tiny population (4 subjects, minutes-scale), no wearables beyond IMU, no EMG/PPG/heart-rate, restrictive license, motion-capture lab setting. Useful as a video+IMU cross-check, not as the primary multimodal corpus.

## 3. MM-Fit file/metadata audit notes

- Immutable-ish identifiers: Zenodo record 7672767 v2 (md5 per video file); project page is the canonical metadata source; dataset itself distributed via project page/Zenodo — record exact revision at download time.
- Per-session wearable files (accelerometer/gyro/BPM) are small; video is the 39.1 GB bulk.
- Time relationship video↔wearable: claimed synchronised; **explicit per-stream timestamp recoverability must be verified from actual files** (NEEDS_EVIDENCE).
- Calibration: RGB-D intrinsics/extrinsics expected from Orbbec; verify presence in metadata.

## 4. Proposed human multimodal evidence mapping

| Source | Generic mapping | Role | Clock domain | Units | Status |
|---|---|---|---|---|---|
| RGB frames | video modality payload | camera | `video/rgb_session` | px | NEEDS_EVIDENCE (decode) |
| Depth frames | depth modality payload | camera | `video/depth_session` | mm? UNKNOWN | NEEDS_EVIDENCE |
| Accel (×5 devices) | wearable arrays | observation | per-device clock | m/s² or g — UNKNOWN until metadata | READY_FOR_TEST |
| Gyro (×5) | wearable arrays | observation | per-device clock | rad/s or deg/s — UNKNOWN | READY_FOR_TEST |
| Magnetometer (×2 phones) | wearable arrays | observation | per-device clock | µT? UNKNOWN | NEEDS_EVIDENCE |
| BPM | scalar series | observation (derived) | watch clock | bpm | READY_FOR_TEST (derived, not raw PPG) |
| Raw PPG | **absent** | — | — | — | NOT_APPLICABLE (not in dataset) |
| EMG | **absent** | — | — | — | NOT_APPLICABLE |
| 2D/3D pose | derived skeleton modality | observation (derived) | video clock | UNKNOWN | NEEDS_EVIDENCE |

Sensor identity, device, fs, clock domain, calibration, participant/session ID, modality availability, and provenance must all be preserved; UNKNOWN wherever undocumented.

## 5. Human assessment separation

- **A. Existing v1.8 sensor laws**: apply to accel/gyro/mag/BPM/pose-scalar streams only.
- **B. Generic modality-integrity (new, later milestone)**: video decode-ability, frame count/dimensions, timestamp continuity, missing frames — NOT_ASSESSED until implemented; no thresholds invented.
- **C. Cross-modal checks**: sync verification, temporal alignment, pose↔sensor agreement — all NEEDS_EVIDENCE pending sync-metadata proof.

Classification: IMU streams READY_FOR_TEST; video/depth/pose NEEDS_EVIDENCE; raw PPG/EMG NOT_APPLICABLE for MM-Fit; TNT15 IMU+video NEEDS_EVIDENCE (tiny N, license).

## 6. Downstream experiment design (later milestone)

Matched participant-level splits (10 subjects → e.g. 7 train / 3 test, multiple fold assignments), identical windows across modalities, seeds ≥3, confidence intervals, baselines: IMU-only, video-only, IMU+video, IMU+video+BPM, +pose. Target: exercise label + repetition count. Missing-modality handling: zero-out/mask consistent with MM-Fit paper's strategy. Explicitly test whether video adds measurable value — more modalities ≠ assumed better.

## 7. Minimal acquisition plan (next milestone)

- Metadata/session docs + wearable files for all 21 sessions (small, MB-scale).
- **One representative session's RGB-D video** (~1–2 GB) for decode/sync verification — NOT the 39.1 GB population.
- Total recommended first pull: < 3 GB. Requires explicit approval before any bulk download; storage is fine (83 GB free).

## 8. Unresolved questions

1. Exact per-stream clock/timestamp representation and sync mechanism in MM-Fit files.
2. Whether depth stream is stored per-session and its units.
3. Pose provenance (which estimator/version) and its clock.
4. Zenodo vs project-page revision pinning for non-video files.
5. TNT15 download mechanics/license acceptance flow.

## 9. Decision

**CONDITIONAL GO** — proceed to human multimodal adapter development on **MM-Fit** with limitations: wearable IMU/BPM streams are READY_FOR_TEST; video/depth/pose are NEEDS_EVIDENCE pending bounded acquisition and sync verification; EMG/raw PPG are absent (NOT_APPLICABLE); TNT15 is a secondary video+IMU cross-check only (research-only license, 4 subjects). No synchronization is claimed without file-level evidence; no thresholds invented; this is not model certification.
