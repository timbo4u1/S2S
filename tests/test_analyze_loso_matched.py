"""Tests for subject-level matched-retention analysis."""

import json

import pytest

from experiments.analyze_loso_matched import analyze_results, load_result_files


def make_result(seed, values_by_subject):
    folds = []
    for subject, values in values_by_subject.items():
        folds.append({
            "test_subject": subject,
            "A_single_sensor": {"macro_f1_12class": values["A"]},
            "B_multi_naive": {"macro_f1_12class": values["B"]},
            "C_chain_filtered": {"macro_f1_12class": values["C"]},
            "D_chain_weighted": {"macro_f1_12class": values["D"]},
            "E_random_class_matched": {"macro_f1_12class": values["E"]},
        })
    return {
        "meta": {
            "seed": seed,
            "experiment": "s2s_level4_pamap2",
            "evaluation_mode": "leave_one_subject_out",
        },
        "fold_results": folds,
    }


def test_c_minus_e_averages_seeds_within_subject_before_bootstrapping():
    results = [
        make_result(42, {
            101: {"A": .4, "B": .5, "C": .6, "D": .61, "E": .5},
            102: {"A": .3, "B": .4, "C": .5, "D": .51, "E": .4},
            103: {"A": .3, "B": .4, "C": .4, "D": .41, "E": .5},
        }),
        make_result(43, {
            101: {"A": .4, "B": .5, "C": .7, "D": .71, "E": .6},
            102: {"A": .3, "B": .4, "C": .6, "D": .61, "E": .5},
            103: {"A": .3, "B": .4, "C": .4, "D": .41, "E": .4},
        }),
    ]

    report = analyze_results(results, bootstrap_samples=1000, bootstrap_seed=2026)
    result = report["comparisons"]["C-E"]

    # Subject-level means: 0.10, 0.10, -0.05 -> mean +0.05 = +5 pp.
    assert result["mean_delta_percentage_points"] == pytest.approx(5.0)
    assert result["per_subject_mean_delta_percentage_points"] == {
        "101": 10.0,
        "102": 10.0,
        "103": -5.0,
    }
    assert set(result["per_seed_mean_delta_percentage_points"]) == {"42", "43"}
    assert result["subject_bootstrap_95ci_percentage_points"][0] <= 5.0
    assert result["subject_bootstrap_95ci_percentage_points"][1] >= 5.0
    assert report["n_subjects"] == 3
    assert report["n_seeds"] == 2
    assert "not independent subjects" in report["interpretation_limit"]


def test_analysis_is_deterministic():
    result = make_result(42, {
        101: {"A": .4, "B": .5, "C": .6, "D": .61, "E": .5},
        102: {"A": .3, "B": .4, "C": .5, "D": .51, "E": .4},
        103: {"A": .3, "B": .4, "C": .4, "D": .41, "E": .5},
    })
    first = analyze_results([result], bootstrap_samples=1000, bootstrap_seed=123)
    second = analyze_results([result], bootstrap_samples=1000, bootstrap_seed=123)
    assert first == second


def test_result_loader_rejects_duplicate_seeds(tmp_path):
    result = make_result(42, {
        101: {"A": .4, "B": .5, "C": .6, "D": .61, "E": .5},
    })
    paths = []
    for idx in range(2):
        path = tmp_path / ("seed_%s.json" % idx)
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(str(path))

    with pytest.raises(ValueError, match="Duplicate seed 42"):
        load_result_files(paths)


def test_analysis_rejects_mismatched_subject_sets():
    first = make_result(42, {
        101: {"A": .4, "B": .5, "C": .6, "D": .61, "E": .5},
        102: {"A": .3, "B": .4, "C": .5, "D": .51, "E": .4},
    })
    second = make_result(43, {
        101: {"A": .4, "B": .5, "C": .6, "D": .61, "E": .5},
        103: {"A": .3, "B": .4, "C": .5, "D": .51, "E": .4},
    })

    with pytest.raises(ValueError, match="different held-out subject sets"):
        analyze_results([first, second], bootstrap_samples=1000)
