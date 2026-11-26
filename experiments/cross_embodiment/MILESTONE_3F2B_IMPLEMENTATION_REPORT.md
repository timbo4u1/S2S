# Milestone 3F.2B — MM-Fit Human Multimodal Evidence Adapter

Implements a modality-faithful evidence adapter for the bounded MM-Fit w19 acquisition. Modality ingestion + assessment only — **no model training, no PhysicsEngine/schema/adapter modifications**.

## Architecture

`experiments/cross_embodiment/milestone_3f2b/`:

| File | Role |
|---|---|
| `mmfit_adapter.py` | Stream loader (per-device, never merged), `DatasetRecord` builder, `CheckResult` helper, schema-faithful report assembly |
| `wearable_assessment.py` | Per-stream integrity + timestamp-quality + unit-semantics + PhysicsEngine-compat checks (4 per stream) |
| `video_assessment.py` | ffprobe metadata + PTS stats; independent video integrity and clock-domain checks |
| `frame_association.py` | Frame-index↔video association with ±1-frame uncertainty; depth acquisition state |
| `run_mmfit_assessment.py` | Orchestrator → `outputs/w19/` (7 deterministic JSON artifacts) |
| `tests/test_mmfit_adapter.py` | 16 tests covering requirements A–P |

## Modality mappings

- **9 inertial/mag streams** (`sw_l/sw_r` acc+gyr, `sp_r` acc+gyr+mag, `eb_l` acc+gyr): `(N,5)` f64 `Frame,Timestamp,X,Y,Z`; epoch-ms clock; camera frame index; units UNKNOWN (sw_* have documentary m/s²/rad/s example evidence only).
- **1 HR stream** (`sw_l_hr`): `(N,3)` derived BPM — explicitly NOT raw PPG.
- **RGB video**: h264 1280×720@30fps, 52,561 frames, relative PTS (no absolute clock).
- **Labels**: 26 sets / 258 reps / 9 activities, frame-index domain.
- **Absent honestly**: depth (RAW_UNAVAILABLE; source zenodo:7607736 `w19_depth.mp4` 19,014,084 B not acquired), pose (PARTIAL), `sw_r_hr`, `sp_l_*`, EMG, raw PPG.

## PhysicsEngine v1.8 compatibility

Engine **not imported or executed** by the adapter (verified by test). Per stream: `MMFIT_PHYSICS_SCHEMA_COMPAT` = PASS for acc/gyro (`imu_raw` shape compatible, epoch-ms→ns mechanical), NOT_ASSESSED for mag/HR (no documented path). `MMFIT_PHYSICS_LAWS` = **NOT_ASSESSED** for every stream — axis units unverified; v1.8 laws/thresholds/tiers unchanged. BPM never enters PPG paths.

## Timestamp findings

Per-stream stats reproduce 3F.2A.1: epoch-ms shared clock; duplication mechanisms preserved verbatim (`QUANTIZATION_ONLY` for sw/eb, `MIXED_QUANTIZATION_AND_HELD` for sp_r); zero-Δt fractions recorded; per-sample ts quality DEGRADED (WARNING) except HR (clean); zero reversals; no resampling.

## Video & frame association

Video integrity PASS (codec/resolution/frames/monotonic PTS verified; pixel decode not exhaustive). Clock-domain check NOT_ASSESSED — relative PTS is not an absolute clock. Frame association WARNING: mechanism verified, ±1 frame (~33 ms) boundary uncertainty, zero/one-based unresolved, drift NOT_MEASURED, no sample-exact joins.

## Assessment results

`outputs/w19/assessment_report.json`: **56 checks — PASS=22, WARNING=10, NOT_ASSESSED=24**, embodiment HUMAN, no score/tier. Report round-trips through `AssessmentReport.from_dict`; deterministic serialization verified by test (identical bytes across runs).

## Tests

- `test_mmfit_adapter.py`: **16 passed** (all streams load unmodified; no merging; UNKNOWN preserved; no physics PASS on unknown units; BPM≠PPG; missing modalities honest; dup/quantization findings preserved; no invented reversals; relative PTS; ±1-frame uncertainty; no sample-exact sync; depth RAW_UNAVAILABLE; 26/258/9 labels; round-trip; determinism; engine unchanged).
- Full S2S suite: **541 passed, 14 warnings, 137.7s** (includes human adapter 7, schema 22, robot generic/assessment/aloha/DROID suites — all unchanged and green).

## Unresolved issues

Units for sp_r/eb_l/mag; sp_r hardware identity; per-sample timestamp degradation; frame boundary mapping; video absolute clock absence; depth/pose acquisition; w19 participant id.

## Modality-specific readiness

| Modality | Decision |
|---|---|
| Each acc/gyro stream | CONDITIONAL — schema OK, units unverified → laws NOT_ASSESSED |
| Magnetometer | CONDITIONAL — no engine path documented |
| Derived HR | READY_FOR_TEST as derived scalar (not PPG) |
| RGB video integrity | READY_FOR_TEST |
| Depth | RAW_UNAVAILABLE |
| Frame-level association | CONDITIONAL (±1 frame) |
| Sample-level sync | NOT_ASSESSED |
| Multimodal training | NOT_ASSESSED until 3F.3 |

## Decision

- **GO** for wearable integration (evidence-faithful, honest degradation).
- **GO** for RGB video integrity assessment.
- **CONDITIONAL GO** for frame-level wearable/video association.
- **NOT_ASSESSED** for sample-exact synchronization and full multimodal training (3F.3).

No universal multimodal readiness claimed.

## 3F.2B.1 corrections (adapter_version 3f.2b.2)

- Video check renamed `MMFIT_RGB_VIDEO_METADATA_PTS_INTEGRITY`; expected 30 FPS enforced via `fractions.Fraction` (no `eval`); PTS-derived FPS cross-checked against reported FPS (0.5 tolerance); empty/invalid PTS handled gracefully; pixel-decode limitation explicit.
- Frame-association finding now references **all 10 wearable streams + labels CSV + MP4 + PTS export** (relative paths, SHA-256) — no longer only the first stream.
- `stream_stats()` validates Frame/Timestamp finiteness and integer-valuedness **before** int64 casting; NaN/Inf/non-integer inputs raise `ValueError` (cannot masquerade as timestamp reversals).
- Rate semantics split: `observed_record_rate_hz` / `observed_distinct_ts_rate_hz` / `verified_nominal_hz=None` / `candidate_reference_hz` (unverified, LOW confidence for `sp_r`).
- Duplicate mechanism split into `dup_observed_pattern` (measured) vs `dup_causal_interpretation` (inferred, LOW confidence).
- PTS export referenced by repo-relative path with SHA-256 in evidence refs; output stays deterministic.

**Findings preserved**: 10 streams unmerged, 26 labels/258 reps/9 activities, BPM≠PPG, relative PTS, ±1-frame association, depth RAW_UNAVAILABLE, physics laws NOT_ASSESSED on UNKNOWN units. Report: 56 checks, PASS=22/WARNING=10/NOT_ASSESSED=24 (unchanged).

**Tests**: adapter **30 passed** (14 new regressions); full suite **555 passed, 14 warnings, 144.6s**.

### Revised decisions
- Wearable assessment: **GO** (with documented timestamp/unit caveats)
- RGB video metadata/PTS: **GO**
- Frame association: **CONDITIONAL GO** (±1 frame)
- Sample-level sync: **NOT_ASSESSED**
- Multimodal training: **NOT_ASSESSED** until 3F.3

## 3F.2B.2.1 final corrections (adapter_version 3f.2b.2)

- Version `3f.2b.2` aligned across adapter, manifest, report, and all check versions; all 7 outputs regenerated.
- Timestamp validator now enforces signed-int64 representability with float-exact bounds (`v >= -2^63`, `v < 2^63`); out-of-range → `ValueError`; span computed in float to avoid overflow warnings.
- `video_stats()` safe for 0/1/2 PTS rows → explicit `insufficient`; `video_checks()` returns WARNING + `pts_insufficient_rows`.
- Bare `nominal_hz` removed from persisted inventory; `verified_nominal_hz=None`, `candidate_reference_hz` (source/confidence/`verified_for_this_device=False`) retained; candidates never gate checks.
- `sample_count_convention` added to record provenance (sum of stream records, not unique synchronized timestamps).
- Frame-association finding verified: **13 evidence refs** (10 streams + labels + MP4 + PTS), all sha256, all relative paths.
- Findings preserved: 56 checks PASS=22/WARNING=10/NOT_ASSESSED=24; deterministic output (identical md5 across runs); physics laws NOT_ASSESSED; BPM≠PPG.

**Tests**: adapter **40 passed**; focused (mmfit+schema+human) **69 passed**; full suite **565 passed, 14 warnings, 132.5s**.
