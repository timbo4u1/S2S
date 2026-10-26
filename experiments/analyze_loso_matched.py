#!/usr/bin/env python3
"""Analyze matched-retention PAMAP2 LOSO result JSON files.

This utility averages repeated seeds within each held-out subject, then
bootstraps subjects (not seed repetitions) to form an exploratory interval.
It does not produce a formal significance test.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


COMPARISONS = {
    "B-A": ("B_multi_naive", "A_single_sensor"),
    "C-B": ("C_chain_filtered", "B_multi_naive"),
    "C-E": ("C_chain_filtered", "E_random_class_matched"),
    "D-C": ("D_chain_weighted", "C_chain_filtered"),
    "E-B": ("E_random_class_matched", "B_multi_naive"),
    "D-A": ("D_chain_weighted", "A_single_sensor"),
}


def load_result_files(paths: Sequence[str]) -> List[Dict[str, Any]]:
    results = []
    seen_seeds = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        meta = result.get("meta", {})
        seed = meta.get("seed")
        if not isinstance(seed, int):
            raise ValueError(f"{path}: meta.seed must be an integer")
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed {seed}; use exactly one result file per seed")
        if meta.get("evaluation_mode") != "leave_one_subject_out":
            raise ValueError(f"{path}: expected leave_one_subject_out evaluation")
        if meta.get("experiment") != "s2s_level4_pamap2":
            raise ValueError(f"{path}: unexpected experiment {meta.get('experiment')!r}")
        results.append(result)
        seen_seeds.add(seed)
    if not results:
        raise ValueError("Provide at least one result file")
    return sorted(results, key=lambda item: item["meta"]["seed"])


def fold_metric_map(result: Mapping[str, Any]) -> Dict[int, Dict[str, float]]:
    """Map each held-out subject to condition -> fixed 12-class macro-F1."""
    mapped = {}
    for fold in result.get("fold_results", []):
        subject = fold.get("test_subject")
        if not isinstance(subject, int):
            raise ValueError("Every fold must include an integer test_subject")
        if subject in mapped:
            raise ValueError(f"Subject {subject} appears in more than one fold")
        metrics = {}
        for condition in set(sum((list(pair) for pair in COMPARISONS.values()), [])):
            entry = fold.get(condition)
            if not isinstance(entry, dict):
                raise ValueError(f"Subject {subject}: missing condition {condition}")
            value = entry.get("macro_f1_12class")
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Subject {subject}: invalid macro_f1_12class for {condition}")
            metrics[condition] = float(value)
        mapped[subject] = metrics
    if not mapped:
        raise ValueError("No fold results found")
    return mapped


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linear-interpolated percentile for an already sorted non-empty sequence."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def analyze_results(
    results: Sequence[Mapping[str, Any]],
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 2026,
) -> Dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap_samples")
    if not results:
        raise ValueError("No results provided")

    seeds = [int(result["meta"]["seed"]) for result in results]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Result files must have unique seeds")

    maps = {seed: fold_metric_map(result) for seed, result in zip(seeds, results)}
    subject_sets = [set(value.keys()) for value in maps.values()]
    expected_subjects = subject_sets[0]
    if any(s != expected_subjects for s in subject_sets[1:]):
        raise ValueError("Result files have different held-out subject sets")
    subjects = sorted(expected_subjects)

    output = {
        "experiment": "s2s_level4_pamap2_matched_retention_analysis",
        "metric": "fixed 12-class macro-F1",
        "primary_comparison": "C-E",
        "seeds": sorted(seeds),
        "n_seeds": len(seeds),
        "subjects": subjects,
        "n_subjects": len(subjects),
        "bootstrap": {
            "unit": "held-out subject",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "interval": "percentile 95%; exploratory because only nine subjects are available",
        },
        "comparisons": {},
        "interpretation_limit": (
            "Seeds are repeated runs on the same held-out subjects, not independent subjects. "
            "Subject-bootstrap intervals are exploratory and are not a formal significance test."
        ),
    }

    rng = random.Random(bootstrap_seed)
    for label, (left, right) in COMPARISONS.items():
        per_seed_deltas = {}
        subject_seed_deltas = {subject: [] for subject in subjects}

        for seed in sorted(seeds):
            deltas = []
            for subject in subjects:
                delta = maps[seed][subject][left] - maps[seed][subject][right]
                subject_seed_deltas[subject].append(delta)
                deltas.append(delta)
            per_seed_deltas[str(seed)] = round(statistics.mean(deltas) * 100.0, 4)

        # Seeds are averaged within each subject before subject-level uncertainty.
        subject_means = {
            str(subject): statistics.mean(subject_seed_deltas[subject])
            for subject in subjects
        }
        subject_values = [subject_means[str(subject)] for subject in subjects]
        mean_delta = statistics.mean(subject_values)

        bootstrap_means = []
        for _ in range(bootstrap_samples):
            draw = [rng.choice(subject_values) for _ in subject_values]
            bootstrap_means.append(statistics.mean(draw) * 100.0)
        bootstrap_means.sort()

        output["comparisons"][label] = {
            "left_condition": left,
            "right_condition": right,
            "mean_delta_percentage_points": round(mean_delta * 100.0, 4),
            "subject_bootstrap_95ci_percentage_points": [
                round(percentile(bootstrap_means, 0.025), 4),
                round(percentile(bootstrap_means, 0.975), 4),
            ],
            "per_seed_mean_delta_percentage_points": per_seed_deltas,
            "per_subject_mean_delta_percentage_points": {
                subject: round(value * 100.0, 4)
                for subject, value in subject_means.items()
            },
            "seed_mean_spread_pp": round(
                max(per_seed_deltas.values()) - min(per_seed_deltas.values()), 4
            ),
        }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_files", nargs="+", help="One LOSO JSON result per unique seed")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--out", help="Optional path to write the analysis JSON")
    args = parser.parse_args(argv)

    try:
        results = load_result_files(args.result_files)
        report = analyze_results(results, args.bootstrap_samples, args.bootstrap_seed)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved analysis → {args.out}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
