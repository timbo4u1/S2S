# Milestone 3F.2A — MM-Fit Bounded Acquisition and Synchronization Verification

Acquisition + evidence audit. No adapter implementation, no engine changes, no bulk video download. All files stored outside the git tree at `~/S2S_data/mmfit_3f2/`.

## 1. Sources and revisions

| Item | Identifier | Notes |
|---|---|---|
| Wearable/pose/label archive | `https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip` | 1,742,309,258 B, Last-Modified 2020-12-27, ETag `c72b4cb6…-102`; **range-request member extraction used** (not full download) |
| RGB video (session w19) | Zenodo record **7672767 v2**, DOI 10.5281/zenodo.7672767, file `w19_rgb.mp4`, md5 `5d5f8dd6434deffba8842cfefe0101db` | CC-BY-4.0; smallest of 21 session videos (415,360,385 B declared) |
| Format reference | `KDMStromback/mm-fit` repo `utils/dataset.py`, `EDA.ipynb` | documents `.npy` modality layout and frame-synchronization claim |

Acquisition method: ZIP central directory fetched via HTTP range; only session `w19` members extracted (~72 MB uncompressed). Video fetched in full (single representative session, within 3 GB budget). **Total acquired ≈ 0.5 GB — far under the 3 GB cap.**

## 2. Acquired file inventory (session w19)

Directory `mm-fit/w19/` inside the archive. SHA-256 for every file recorded in `milestone_3f2a/reproducibility_manifest.json` and `~/S2S_data/mmfit_3f2/w19_sha256.txt`.

| File | Uncompressed B | Contents |
|---|---|---|
| `w19_sw_l_acc.npy` / `_gyr` | 7,103,208 ×2 | left TicWatch accel+gyro, (177,577 × 5) f64 |
| `w19_sw_l_hr.npy` | 39,536 | left watch heart rate, (1,642 × 3) f64 — **derived BPM, not raw PPG** |
| `w19_sw_r_acc.npy` / `_gyr` | 7,170,648 ×2 | right TicWatch accel+gyro, (179,263 × 5) f64 |
| `w19_sp_r_acc/gyr/mag.npy` | 14,622,968 / 14,622,968 / 6,910,768 | one smartphone accel+gyro (365,571 × 5) + magnetometer (172,766 × 5) |
| `w19_eb_l_acc/gyr.npy` | 5,498,048 ×2 | Nokia eSense earbud accel+gyro, (137,448 × 5) f64 |
| `w19_labels.csv` | 705 | 25 exercise-set annotations `(start_frame, end_frame, reps, activity)` |
| `w19_pose_2d.npy`, `w19_pose_3d.npy` | 15.7 MB / 22.3 MB uncompressed | **derived pose estimates** — acquisition in progress at audit time; not required for sync verdict |
| `video/w19_rgb.mp4` | 415,360,385 declared | RGB video, ffprobe verification below |

**Absent in w19 (verified, not assumed):** `sw_r_hr` (no right-watch BPM), all `sp_l_*` (second smartphone streams missing for this session), no depth `.npy` in the wearable archive (depth is expected inside/alongside the RGB-D video or a separate stream — unresolved), no EMG, no raw PPG anywhere.

## 3. Wearable field semantics (verified from actual arrays)

Every inertial/mag file is `(N, 5)` float64 with columns **(Frame, Timestamp, X, Y, Z)**; HR is `(N, 3)` = **(Frame, Timestamp, Heart-rate)** — matching the dataset's own EDA documentation.

| Property | Verified value |
|---|---|
| Timestamp units | **Unix epoch milliseconds** (values ≈ 1,596,123,423,547–1,596,125,144,626 → 2020-07-31) |
| Frame column | **Left RGB-D camera frame index** — identical span 898→52561 across all inertial streams; HR spans 1118→52540 |
| Clock domain | Single shared epoch-ms clock across all wearable devices (max inter-stream start spread ≈ 34 ms) |
| Value units | **UNKNOWN** — accel/gyro/mag axis units not documented in inspected files; plausible m/s², rad/s, µT but unverified |
| HR semantics | derived beats-per-minute scalar (~0.96 Hz effective), values ≈ 105–110 — **NOT raw PPG; must not enter PPG-specific checks** |
| Missing samples | frame coverage 51,544–51,660 unique frames per stream vs 51,664 total video frames; gaps exist (frame_unique < span) |
| Non-finite values | **zero** in all inspected streams |
| Timestamp monotonicity | **not strictly monotonic** — dt_min = 0 ms (duplicate timestamps, e.g. 53,531 dup rows in `sp_r_gyr`); HR is strictly monotonic |
| Measured rates | sw ~103–104 Hz, sp acc/gyr ~212 Hz, sp mag ~100 Hz, eb ~79.9 Hz, hr ~0.96 Hz — vs nominal 100/210/100?/90/1 Hz (mag measured ≈100 Hz, not the 65 Hz Huawei figure — device-to-file mapping unresolved) |

## 4. Video inspection

`w19_rgb.mp4`: download completed and verified — md5 matches Zenodo exactly; ffprobe: **h264, 1280×720, yuv420p, 30 fps, 52,561 frames, 1752.0 s**. `nb_frames=52561` matches the sensor `Frame` column maximum (52561), and labels use the same frame domain — strong evidence the frame index is the video frame number. RGB only in this file; **depth stream not present in this artifact** — depth availability remains unresolved pending dataset docs or a depth-specific file.

## 5. Clock synchronization matrix (measured, session w19)

Method: per-stream first-timestamp-per-frame offsets vs `sw_l_acc` reference, over 38k–51k common frames.

| Source clock | Target clock | Mapping | Offset | Drift | Decision |
|---|---|---|---|---|---|
| any wearable `Timestamp` (epoch ms) | any other wearable `Timestamp` | **VERIFIED** — shared epoch-ms domain | mean −7.4 to +15.1 ms; per-frame std ≤13.3 ms; extremes −48…+73 ms | not separately measurable at ms quantization; bounded by timestamp resolution | **CAN ALIGN** via epoch-ms timestamps |
| wearable `Frame` column | left RGB-D video frame index | **VERIFIED by documentation + data consistency** (frame span 898–52561 ≈ video duration × 30 fps) | n/a (index mapping) | n/a | **CAN ALIGN** — frame index is the documented sync key |
| wearable streams | depth stream | **UNVERIFIED** — no depth artifact acquired | UNKNOWN | UNKNOWN | NOT_ASSESSED |
| RGB video | depth video | UNVERIFIED | UNKNOWN | UNKNOWN | NOT_ASSESSED |
| HR (1 Hz) | video/IMU | VERIFIED via shared epoch-ms + frame cols (offset mean +15 ms) | measured | bounded | CAN ALIGN (low rate) |

Honest caveats: timestamp column shows duplicate/ms-quantized values (not strictly monotonic for high-rate streams); per-frame offsets up to ~73 ms exist for the earbud; drift over the 28.7-min session is **not independently measured** — bounded by ms timestamp resolution but unquantified. Alignment is therefore "verified mechanism, bounded uncertainty," not sample-exact.

## 6. S2S v1.8 compatibility

- **READY_FOR_TEST (PhysicsEngine path):** `sw_l/sw_r acc+gyr`, `sp_r acc+gyr+mag`, `eb_l acc+gyr`, `sw_l_hr` — all numeric, finite, epoch-ms timestamped, frame-aligned. Preserve existing laws/thresholds/NOT_ASSESSED behavior unchanged.
- **HR handling:** derived BPM only — never route to raw-PPG checks.
- **NOT for PhysicsEngine:** `w19_rgb.mp4` (RGB), any depth, `pose_2d/3d` (derived representations) — need separate visual/modality-integrity checks (future milestone).
- **Labels:** frame-indexed CSV — usable as supervision targets; frame domain = video frame index.

## 7. Experiment readiness

| Modality | Status |
|---|---|
| IMU accel/gyro (5 device streams present in w19) | READY_FOR_TEST |
| Magnetometer (sp_r only in w19) | READY_FOR_TEST |
| Derived HR | READY_FOR_TEST (as derived scalar; not PPG) |
| RGB video | NEEDS_EVIDENCE (decode verified; integrity checks unimplemented) |
| Depth | NEEDS_EVIDENCE→NOT_ASSESSED (artifact not acquired) |
| Pose 2D/3D | NEEDS_EVIDENCE (derived; acquisition finishing) |
| Cross-modal sync | VERIFIED mechanism / bounded uncertainty — READY for alignment-aware experiments with stated tolerance |
| EMG, raw PPG | NOT_APPLICABLE (absent from dataset) |

## 8. Unresolved questions

1. Depth stream location/format (not in wearable archive; possibly muxed with RGB-D video or separate files).
2. Device→file mapping certainty (`sp_r` = which phone? mag rate suggests Samsung S7 @100 Hz, not Huawei @65 Hz).
3. Exact timestamp quantization/drift over full session.
4. Pose file completion at audit cutoff (pose_3d/pose_2d range-fetch in progress; formats documented as (3×N×33)/(2×N×19)).
5. Per-session device completeness varies (w19 lacks sp_l_* and sw_r_hr) — adapters must handle per-session modality availability.

## 9. Recommended next adapter design (3F.2B)

Human multimodal adapter emitting: per-device observation streams (accel/gyro/mag/HR) with epoch-ms clock domain + video-frame index as secondary locator; video/depth/pose as camera/derived modalities with PARTIAL availability semantics; labels as metadata; per-session modality inventory (not global assumptions); UNKNOWN units until documented. PhysicsEngine receives only numeric wearable streams.

## 10. Decision

**CONDITIONAL GO** — suitable for 3F.2B adapter implementation with specific limitations: wearable IMU/mag/derived-HR streams are verified READY_FOR_TEST with a demonstrated shared epoch-ms clock and frame-index sync mechanism; RGB video integrity is NEEDS_EVIDENCE (decode path exists, checks unimplemented); depth and pose remain partially unresolved; EMG/raw PPG are NOT_APPLICABLE; per-session device availability is asymmetric and must be modeled per session. Synchronization is verified as a *mechanism with bounded uncertainty*, not sample-exact — cross-modal alignment claims must carry that tolerance. No thresholds invented; no engine modified; ~0.5 GB acquired of the 3 GB budget.
