#!/usr/bin/env python3
"""Part A: frozen-engine baseline diagnostic audit on original C-MHAD IMU trials.

Reads original tab-delimited .xls files through cmhad_adapter.py. Does not modify
source data and does not perform annotation alignment or fault injection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

from cmhad_adapter import (  # noqa: E402
    PART_A_STRIDE,
    WIN,
    iter_windows,
    load_trial,
    trial_files,
    trial_ids,
)
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine  # noqa: E402

ENGINE_COMMIT = "25453b3c47780ea502a12d23abaeaff9400e9c64"
BOOTSTRAP_SEED = 20261101
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_TRIALS = 120
SEGMENT = "waist"

WINDOWS_CSV = EXPERIMENTS_DIR / "results_cmhad_part_a_windows.csv"
SUBJECTS_CSV = EXPERIMENTS_DIR / "results_cmhad_part_a_subjects.csv"
SUMMARY_JSON = EXPERIMENTS_DIR / "results_cmhad_part_a_summary.json"


def git_text(*args: str) -> str:
    p = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return p.stdout.strip()


def verify_engine_pin() -> str:
    """Require the complete engine package to match the protocol's frozen commit."""
    for diff_args in (
        ("diff", "--quiet", ENGINE_COMMIT, "HEAD", "--", "s2s_standard_v1_3"),
        ("diff", "--quiet", "HEAD", "--", "s2s_standard_v1_3"),
        ("diff", "--cached", "--quiet", "HEAD", "--", "s2s_standard_v1_3"),
    ):
        p = subprocess.run(["git", *diff_args], cwd=REPO_ROOT)
        if p.returncode != 0:
            raise RuntimeError(
                "S2S engine package differs from the frozen protocol engine or "
                "has local changes. Do not run Part A until reconciled."
            )
    return git_text("rev-parse", "HEAD")


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def subject_bootstrap_rate(
    per_subject: Dict[int, Dict[str, int]],
    numerator: str,
    denominator: str,
    seed_offset: int,
) -> Dict[str, object]:
    """Equal-weight subject mean and percentile CI; subjects are resampling units."""
    eligible = []
    for subject in sorted(per_subject):
        den = int(per_subject[subject][denominator])
        if den > 0:
            num = int(per_subject[subject][numerator])
            eligible.append((subject, num, den, num / den))

    pooled_num = sum(x[1] for x in eligible)
    pooled_den = sum(x[2] for x in eligible)
    if not eligible:
        return {
            "eligible_subjects": 0,
            "pooled_numerator": pooled_num,
            "pooled_denominator": pooled_den,
            "pooled_window_rate": None,
            "equal_weight_subject_mean_rate": None,
            "subject_bootstrap_95pct_ci": None,
            "per_subject_rates": [],
        }

    rates = [x[3] for x in eligible]
    point = statistics.mean(rates)
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    boot = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [rng.choice(rates) for _ in rates]
        boot.append(statistics.mean(draw))

    return {
        "eligible_subjects": len(eligible),
        "pooled_numerator": pooled_num,
        "pooled_denominator": pooled_den,
        "pooled_window_rate": pooled_num / pooled_den if pooled_den else None,
        "equal_weight_subject_mean_rate": point,
        "subject_bootstrap_95pct_ci": [percentile(boot, 2.5), percentile(boot, 97.5)],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED + seed_offset,
        "per_subject_rates": [
            {"subject": f"S{s:02d}", "flagged": n, "windows": d, "rate": r}
            for s, n, d, r in eligible
        ],
    }


def coverage_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "p05": percentile(values, 5),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "mean": statistics.mean(values),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing Part A result files; default is to stop if any exist",
    )
    args = parser.parse_args()

    outputs = [WINDOWS_CSV, SUBJECTS_CSV, SUMMARY_JSON]
    existing = [str(p.relative_to(REPO_ROOT)) for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing result files: " + ", ".join(existing)
            + ". Inspect them first; use --overwrite only for an intentional rerun."
        )

    run_head = verify_engine_pin()
    protocol_commit = git_text("log", "-1", "--format=%H", "--", "docs/cmhad_protocol.md")
    files = trial_files()
    if len(files) != EXPECTED_TRIALS:
        raise RuntimeError(f"Expected {EXPECTED_TRIALS} trial files; found {len(files)}")

    trial_keys = [trial_ids(path) for path in files]
    if len(set(trial_keys)) != len(trial_keys):
        raise RuntimeError("Duplicate subject/trial identifiers found")

    window_rows: List[Dict[str, object]] = []
    per_subject = defaultdict(lambda: {
        "gap_windows": 0,
        "gap_flagged": 0,
        "non_gap_windows": 0,
        "non_gap_flagged": 0,
        "total_windows": 0,
        "any_fail_windows": 0,
    })
    law_counts = defaultdict(Counter)
    tier_counts = Counter()
    assessment_counts = Counter()
    coverage_values: List[float] = []

    trial_count = 0
    total_gap_events = 0
    total_missing_slots = 0
    assigned_gap_events = 0
    unwindowed_gap_events = 0
    trailing_samples_omitted = 0
    engine_error_law_count = Counter()

    for path in files:
        subject, trial = trial_ids(path)
        tr = load_trial(path)
        ts_ns = np.asarray(tr["ts_ns"], dtype=np.int64)
        accel = np.asarray(tr["accel"], dtype=np.float64)
        gyro = np.asarray(tr["gyro"], dtype=np.float64)

        if len(ts_ns) != len(accel) or len(ts_ns) != len(gyro):
            raise RuntimeError(f"Timestamp/sensor length mismatch: {path}")
        if len(ts_ns) < WIN or not np.all(np.diff(ts_ns) > 0):
            raise RuntimeError(f"Invalid timestamps or insufficient samples: {path}")
        if not (np.isfinite(accel).all() and np.isfinite(gyro).all()):
            raise RuntimeError(f"Non-finite sensor values found: {path}")

        gap_at = np.asarray(tr["gap_at"], dtype=np.int64)
        gap_lost = np.asarray(tr["gap_lost"], dtype=np.int64)
        total_gap_events += int(len(gap_at))
        total_missing_slots += int(gap_lost.sum())

        windows = list(iter_windows(tr, stride=PART_A_STRIDE))
        if not windows:
            raise RuntimeError(f"No complete windows: {path}")
        last_end = windows[-1][1]
        trailing_samples_omitted += int(len(ts_ns) - last_end)
        assigned_gap_events += int(np.sum(gap_at < last_end))
        unwindowed_gap_events += int(np.sum(gap_at >= last_end))

        # Exactly one stateful engine per trial; windows remain chronological.
        engine = PhysicsEngine()
        trial_count += 1
        for window_index, (start, end, contains_gap) in enumerate(windows):
            imu_raw = {
                "timestamps_ns": ts_ns[start:end].tolist(),
                "accel": accel[start:end].tolist(),
                "gyro": gyro[start:end].tolist(),
            }
            result = engine.certify(
                imu_raw=imu_raw,
                segment=SEGMENT,
                device_id="C-MHAD-Shimmer3-waist",
                session_id=f"S{subject:02d}_T{trial:02d}",
            )

            statuses = result.get("law_statuses", {})
            if not statuses:
                raise RuntimeError(
                    f"Engine returned no law_statuses for S{subject:02d} T{trial:02d} window {window_index}"
                )
            checked = int(result.get("n_laws_checked", len(statuses)))
            assessed = int(result.get("n_laws_assessed", 0))
            coverage = assessed / checked if checked else 0.0
            failed = sorted(result.get("laws_failed", []))
            not_assessed = sorted(result.get("laws_not_assessed", []))
            any_fail = bool(failed)
            details = result.get("law_details", {})
            skip_reasons = {
                law: info.get("skip")
                for law, info in details.items()
                if isinstance(info, dict) and info.get("skip") is not None
            }
            for law, reason in skip_reasons.items():
                if str(reason).startswith("ERROR:"):
                    engine_error_law_count[law] += 1

            for law, status in statuses.items():
                if status not in ("PASS", "FAIL", "NOT_ASSESSED"):
                    raise RuntimeError(f"Unexpected status {status!r} for law {law}")
                law_counts[law][status] += 1

            tier = str(result.get("tier", "UNKNOWN"))
            tier_counts[tier] += 1
            assessment = str(result.get("assessment_status", "UNKNOWN"))
            assessment_counts[assessment] += 1
            coverage_values.append(coverage)

            ps = per_subject[subject]
            ps["total_windows"] += 1
            ps["any_fail_windows"] += int(any_fail)
            if contains_gap:
                ps["gap_windows"] += 1
                ps["gap_flagged"] += int(any_fail)
            else:
                ps["non_gap_windows"] += 1
                ps["non_gap_flagged"] += int(any_fail)

            window_rows.append({
                "subject": f"S{subject:02d}",
                "trial": f"T{trial:02d}",
                "window_index": window_index,
                "sample_start": start,
                "sample_end_exclusive": end,
                "n_samples": end - start,
                "first_timestamp_ns": int(ts_ns[start]),
                "last_timestamp_ns": int(ts_ns[end - 1]),
                "contains_internal_gap": int(contains_gap),
                "assessment_status": assessment,
                "tier": tier,
                "tier_scope": result.get("tier_scope", "UNKNOWN"),
                "physical_law_score": result.get("physical_law_score"),
                "n_laws_checked": checked,
                "n_laws_assessed": assessed,
                "assessed_coverage": coverage,
                "n_laws_passed": len(result.get("laws_passed", [])),
                "n_laws_failed": len(failed),
                "n_laws_not_assessed": len(not_assessed),
                "any_law_failed": int(any_fail),
                "laws_failed_json": json.dumps(failed, sort_keys=True),
                "laws_not_assessed_json": json.dumps(not_assessed, sort_keys=True),
                "law_statuses_json": json.dumps(statuses, sort_keys=True),
                "skip_reasons_json": json.dumps(skip_reasons, sort_keys=True),
            })

    # Build subject-level gap/non-gap rates and their subject-cluster bootstrap CIs.
    subject_rows = []
    bootstrap_input = {}
    for subject in sorted(per_subject):
        x = per_subject[subject]
        subject_rows.append({
            "subject": f"S{subject:02d}",
            **x,
            "gap_flag_rate": x["gap_flagged"] / x["gap_windows"] if x["gap_windows"] else "",
            "non_gap_flag_rate": x["non_gap_flagged"] / x["non_gap_windows"] if x["non_gap_windows"] else "",
        })
        bootstrap_input[subject] = x

    flag_rates = {
        "definition": "window flagged when one or more laws have status FAIL",
        "gap_windows": subject_bootstrap_rate(
            bootstrap_input, "gap_flagged", "gap_windows", seed_offset=0
        ),
        "non_gap_windows": subject_bootstrap_rate(
            bootstrap_input, "non_gap_flagged", "non_gap_windows", seed_offset=1
        ),
        "interpretation": (
            "Descriptive only. Gap windows are sparse and there are no independent "
            "ground-truth labels for clean/corrupt windows. These are not false-positive rates."
        ),
    }

    n_windows = len(window_rows)
    if n_windows != sum(tier_counts.values()):
        raise RuntimeError("Internal accounting error: tier counts do not sum to windows")
    for law, counts in law_counts.items():
        if sum(counts.values()) != n_windows:
            raise RuntimeError(f"Internal accounting error for law {law}: status counts != windows")

    summary = {
        "experiment": "C-MHAD Part A baseline diagnostic audit",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": run_head,
        "protocol_commit": protocol_commit,
        "frozen_engine_commit": ENGINE_COMMIT,
        "segment": SEGMENT,
        "window_samples": WIN,
        "part_a_stride_samples": PART_A_STRIDE,
        "windowing": "non-overlapping chronological windows; one PhysicsEngine instance per trial",
        "subjects": len(per_subject),
        "trials": trial_count,
        "windows": n_windows,
        "timestamp_gap_events_in_recordings": total_gap_events,
        "missing_sample_slots_inside_recordings": total_missing_slots,
        "gap_events_assigned_to_complete_windows": assigned_gap_events,
        "gap_events_outside_complete_windows": unwindowed_gap_events,
        "gap_windows": sum(int(r["contains_internal_gap"]) for r in window_rows),
        "non_gap_windows": sum(1 - int(r["contains_internal_gap"]) for r in window_rows),
        "trailing_observed_samples_not_windowed": trailing_samples_omitted,
        "per_law_status_counts": {
            law: {
                "PASS": int(counts.get("PASS", 0)),
                "FAIL": int(counts.get("FAIL", 0)),
                "NOT_ASSESSED": int(counts.get("NOT_ASSESSED", 0)),
            }
            for law, counts in sorted(law_counts.items())
        },
        "tier_distribution": {k: int(v) for k, v in sorted(tier_counts.items())},
        "assessment_status_distribution": {k: int(v) for k, v in sorted(assessment_counts.items())},
        "assessed_coverage_fraction_distribution": coverage_summary(coverage_values),
        "gap_vs_non_gap_any_failure_rates": flag_rates,
        "engine_error_skip_counts": {k: int(v) for k, v in sorted(engine_error_law_count.items())},
        "notes": [
            "A window is flagged when at least one assessable law returns FAIL.",
            "A window without a timestamp gap is not thereby certified clean.",
            "Gap/non-gap rates are descriptive observed flag rates, not false-positive rates.",
            "The quietest-gyro acceleration-magnitude statistic is a proxy, not verified rest or calibration ground truth.",
            "No action annotations were aligned or used in this experiment.",
            "No thresholds were tuned and no original data files were modified.",
            "Tails shorter than 256 samples were not windowed; their counts are reported above.",
            "No downstream classifier or action-recognition performance was evaluated.",
        ],
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with WINDOWS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(window_rows)
    with SUBJECTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(subject_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subject_rows)
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")

    print("C-MHAD Part A completed.")
    print(f"engine_commit={ENGINE_COMMIT}")
    print(f"repository_head={run_head}")
    print(f"protocol_commit={protocol_commit}")
    print(f"subjects={len(per_subject)} trials={trial_count} windows={n_windows}")
    print(
        f"gap_events={total_gap_events} missing_slots={total_missing_slots} "
        f"gap_events_windowed={assigned_gap_events} gap_windows={summary['gap_windows']} "
        f"non_gap_windows={summary['non_gap_windows']}"
    )
    print(f"trailing_samples_not_windowed={trailing_samples_omitted}")
    print("tier_distribution=" + json.dumps(summary["tier_distribution"], sort_keys=True))
    print("assessment_status=" + json.dumps(summary["assessment_status_distribution"], sort_keys=True))
    print("assessed_coverage=" + json.dumps(summary["assessed_coverage_fraction_distribution"], sort_keys=True))
    print("per_law_status_counts=" + json.dumps(summary["per_law_status_counts"], sort_keys=True))
    print("gap_vs_non_gap_flag_rates=" + json.dumps(flag_rates, sort_keys=True))
    print("engine_error_skip_counts=" + json.dumps(summary["engine_error_skip_counts"], sort_keys=True))
    print("outputs:")
    print(f"  {WINDOWS_CSV.relative_to(REPO_ROOT)}")
    print(f"  {SUBJECTS_CSV.relative_to(REPO_ROOT)}")
    print(f"  {SUMMARY_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
