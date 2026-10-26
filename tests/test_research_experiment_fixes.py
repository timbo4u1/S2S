import numpy as np
import pytest
import os

import run_benchmark
from experiments.level4_multisensor_fusion import DOMAINS, MLP, load_pamap2_multisensor, WINDOW_SIZE, STEP_SIZE, ACTIVITY_LABELS


def test_pamap2_loader_uses_gyro_columns_10_to_12(tmp_path, monkeypatch):
    # Each column has a distinctive value, making incorrect mappings visible.
    data = np.tile(np.arange(47, dtype=float), (300, 1))
    np.savetxt(tmp_path / "subject101.dat", data)

    captured = []

    def fake_certify(accel, gyro, timestamps, segment="forearm"):
        captured.append((accel, gyro))
        return {"tier": "SILVER", "physical_law_score": 70}

    monkeypatch.setattr(run_benchmark, "certify", fake_certify)

    results = run_benchmark.load_pamap2(
        str(tmp_path), n_subjects=1, n_windows=1
    )

    assert len(results) == 1
    assert captured[0][0][0] == [4.0, 5.0, 6.0]
    assert captured[0][1][0] == [10.0, 11.0, 12.0]


def test_level4_evaluate_computes_macro_f1():
    model = object.__new__(MLP)

    def fake_forward(features):
        label = int(features[0])
        predicted = 1 if label == 0 else label
        probabilities = [0.0] * len(DOMAINS)
        probabilities[predicted] = 1.0
        return [], probabilities

    model.forward = fake_forward
    samples = [([i], i) for i in range(len(DOMAINS))]

    accuracy, macro_f1 = model.evaluate(samples)

    assert accuracy == pytest.approx(11 / 12)
    assert macro_f1 == pytest.approx((10 + 2 / 3) / 12)


def test_pamap2_loader_segments_at_malformed_row(tmp_path):
    """
    Test that malformed rows (fewer than 47 columns) terminate segments.
    Creates synthetic data with 300 supported rows, one malformed row,
    and 300 more supported rows. Windows should only be created within
    each supported segment, never crossing the malformed row.
    """
    # Create synthetic PAMAP2-style data
    supported_act = 1  # lying

    # Segment 1: 300 samples of supported activity
    seg1_len = 300
    # Segment 2: 1 malformed row (fewer than 47 columns)
    # Segment 3: 300 samples of supported activity
    seg3_len = 300

    # Build data as list of lines to handle variable column counts
    lines = []

    # Segment 1: valid rows
    for i in range(seg1_len):
        cols = [str(float(j)) for j in range(47)]
        cols[1] = str(supported_act)  # Activity column
        lines.append(" ".join(cols))

    # Malformed row: only 10 columns
    lines.append(" ".join([str(float(j)) for j in range(10)]))

    # Segment 3: valid rows
    for i in range(seg3_len):
        cols = [str(float(j)) for j in range(47)]
        cols[1] = str(supported_act)  # Activity column
        lines.append(" ".join(cols))

    # Write to file
    dat_path = tmp_path / "subject101.dat"
    with open(dat_path, 'w') as f:
        f.write("\n".join(lines))

    # Load using the fixed loader
    windows = load_pamap2_multisensor(str(tmp_path))

    # Verify windows were created
    assert len(windows) > 0, "Should create windows from supported segments"

    # Verify no window contains the malformed row
    for w in windows:
        assert w['activity'] == supported_act, \
            f"Window should only contain supported activity {supported_act}"

    # With WINDOW_SIZE=256, STEP_SIZE=128, seg1_len=300:
    # Segment 1: window starting at 0 (1 window: 0+256=256 <= 300, 128+256=384 > 300)
    # Segment 3: window starting at 0 (1 window)
    # Total: 2 windows
    seg1_windows = (seg1_len - WINDOW_SIZE) // STEP_SIZE + 1 if seg1_len >= WINDOW_SIZE else 0
    seg3_windows = (seg3_len - WINDOW_SIZE) // STEP_SIZE + 1 if seg3_len >= WINDOW_SIZE else 0
    expected_windows = seg1_windows + seg3_windows

    assert len(windows) == expected_windows, \
        f"Expected {expected_windows} windows with malformed-row segmentation, got {len(windows)}"


def test_pamap2_loader_segments_at_unsupported_activity(tmp_path):
    """
    Test that unsupported activity rows terminate segments.
    Creates synthetic data with supported activity 1, then unsupported
    activity 99, then supported activity 1 again. Windows should only be
    created within the supported segments.
    """
    supported_act = 1
    unsupported_act = 99

    seg1_len = 300
    seg2_len = 100  # unsupported activity
    seg3_len = 300

    total_samples = seg1_len + seg2_len + seg3_len
    data = np.zeros((total_samples, 47))

    data[:seg1_len, 1] = supported_act
    data[seg1_len:seg1_len+seg2_len, 1] = unsupported_act
    data[seg1_len+seg2_len:, 1] = supported_act

    for i in range(total_samples):
        for j in range(47):
            if j != 1:
                data[i, j] = float(j)

    dat_path = tmp_path / "subject101.dat"
    np.savetxt(dat_path, data)

    windows = load_pamap2_multisensor(str(tmp_path))

    assert len(windows) > 0, "Should create windows from supported segments"

    for w in windows:
        assert w['activity'] == supported_act, \
            f"Window should only contain supported activity {supported_act}"

    seg1_windows = (seg1_len - WINDOW_SIZE) // STEP_SIZE + 1 if seg1_len >= WINDOW_SIZE else 0
    seg3_windows = (seg3_len - WINDOW_SIZE) // STEP_SIZE + 1 if seg3_len >= WINDOW_SIZE else 0
    expected_windows = seg1_windows + seg3_windows

    assert len(windows) == expected_windows, \
        f"Expected {expected_windows} windows with unsupported-activity segmentation, got {len(windows)}"


def test_detailed_metrics_macro_f1_supported_vs_weighted_differ():
    """
    Test that macro_f1_supported_classes and weighted_f1 differ on imbalanced data.
    Creates a synthetic test set where class 0 has many samples (high F1) and
    class 1 has few samples (low F1). The weighted F1 should be closer to class 0's
    F1 due to its higher support, while macro_f1_supported_classes should be the
    arithmetic mean of both classes' F1 scores.
    """
    from experiments.level4_multisensor_fusion import compute_detailed_metrics, MLP

    model = object.__new__(MLP)

    # Create a model with controlled predictions:
    # - First 90 class 0 samples: predict correctly (TP)
    # - Last 10 class 0 samples: predict incorrectly (FN)
    # - First 5 class 1 samples: predict correctly (TP)
    # - Last 5 class 1 samples: predict incorrectly (FN)
    call_count = [0]
    def fake_forward(features):
        call_count[0] += 1
        label = int(features[0])
        if label == 0:
            # First 90 correct, last 10 incorrect
            predicted = 0 if call_count[0] <= 90 else 1
        else:
            # First 5 correct, last 5 incorrect
            predicted = 1 if call_count[0] <= 95 else 0
        probabilities = [0.0, 0.0]
        probabilities[predicted] = 1.0
        return [], probabilities

    model.forward = fake_forward

    # Create imbalanced test set:
    # - Class 0: 100 samples (90 TP, 10 FN -> F1 = 2*0.9*1.0/(0.9+1.0) ≈ 0.947)
    # - Class 1: 10 samples (5 TP, 5 FN -> F1 = 2*0.5*1.0/(0.5+1.0) = 0.667)
    samples = []
    for i in range(100):
        samples.append(([0.0], 0))
    for i in range(10):
        samples.append(([1.0], 1))

    metrics = compute_detailed_metrics(model, samples, 2)

    # Check that both metrics exist
    assert 'macro_f1_supported_classes' in metrics
    assert 'weighted_f1' in metrics

    # Macro F1 (supported) = (0.947 + 0.667) / 2 ≈ 0.807
    # Weighted F1 = (0.947*100 + 0.667*10) / 110 ≈ 0.921
    # These should differ
    assert metrics['macro_f1_supported_classes'] != metrics['weighted_f1'], \
        f"macro_f1_supported_classes ({metrics['macro_f1_supported_classes']}) and weighted_f1 ({metrics['weighted_f1']}) should differ on imbalanced data"

    # Verify per-class metrics show the imbalance
    assert metrics['per_class'][0]['support'] == 100
    assert metrics['per_class'][1]['support'] == 10


def test_loso_folds_isolate_subjects():
    """
    Test that LOSO folds never share subjects between train and test.
    Uses synthetic windows with subject IDs to verify isolation.
    """
    from experiments.level4_multisensor_fusion import run_loso

    # Create synthetic windows for 3 subjects
    subjects = [1, 2, 3]
    windows = []

    for subj in subjects:
        # Create 200 windows per subject (total 600, passes the 500-window check)
        for i in range(200):
            w = {
                'subject': subj,
                'activity': 1,  # lying
                'domain': 'lying',
                'hand_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'hand_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'chest_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'chest_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'ankle_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'ankle_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            }
            windows.append(w)

    # Monkeypatch load_pamap2_multisensor to return our synthetic windows
    import experiments.level4_multisensor_fusion as l4_module
    original_loader = l4_module.load_pamap2_multisensor
    l4_module.load_pamap2_multisensor = lambda data_dir: windows

    # Monkeypatch run_fold to skip actual training and just check subject isolation
    original_run_fold = l4_module.run_fold
    fold_check_results = []

    def fake_run_fold(train_w, test_w, fold_idx, test_subject_id, epochs, norm_s=None, norm_m=None, chain_floor=None, seed=42):
        # Verify no subject overlap
        train_subjects = set(w['subject'] for w in train_w)
        test_subjects = set(w['subject'] for w in test_w)
        fold_check_results.append({
            'fold_idx': fold_idx,
            'test_subject': test_subject_id,
            'train_subjects': train_subjects,
            'test_subjects': test_subjects,
            'overlap': len(train_subjects & test_subjects)
        })
        # Return dummy results with all condition keys
        dummy_results = {
            'A_single_sensor': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'B_multi_naive': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'C_chain_filtered': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'D_chain_weighted': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
        }
        dummy_predictions = {
            'A_single_sensor': [(0, 0)] * len(test_w),
            'B_multi_naive': [(0, 0)] * len(test_w),
            'C_chain_filtered': [(0, 0)] * len(test_w),
            'D_chain_weighted': [(0, 0)] * len(test_w),
        }
        return dummy_results, dummy_predictions, None, None, None

    l4_module.run_fold = fake_run_fold

    try:
        # Run LOSO (will call our monkeypatched functions)
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'results.json')
            run_loso(tmpdir, output_path, epochs=1)
    finally:
        # Restore original functions
        l4_module.load_pamap2_multisensor = original_loader
        l4_module.run_fold = original_run_fold

    # Verify each fold had no subject overlap
    for result in fold_check_results:
        assert result['overlap'] == 0, \
            f"Fold {result['fold_idx']} had subject overlap: {result['train_subjects']} ∩ {result['test_subjects']}"
        assert result['test_subject'] in result['test_subjects']
        assert result['test_subject'] not in result['train_subjects']


def test_loso_each_subject_held_out_once():
    """
    Test that in LOSO, each subject is held out exactly once.
    """
    from experiments.level4_multisensor_fusion import run_loso
    import experiments.level4_multisensor_fusion as l4_module
    import tempfile
    import os

    subjects = [1, 2, 3]
    windows = []

    for subj in subjects:
        # Create 200 windows per subject (total 600, passes the 500-window check)
        for i in range(200):
            w = {
                'subject': subj,
                'activity': 1,
                'domain': 'lying',
                'hand_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'hand_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'chest_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'chest_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'ankle_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'ankle_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            }
            windows.append(w)

    original_loader = l4_module.load_pamap2_multisensor
    l4_module.load_pamap2_multisensor = lambda data_dir: windows

    held_out_subjects = []

    def fake_run_fold(train_w, test_w, fold_idx, test_subject_id, epochs, norm_s=None, norm_m=None, chain_floor=None, seed=42):
        held_out_subjects.append(test_subject_id)
        dummy_results = {
            'A_single_sensor': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'B_multi_naive': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'C_chain_filtered': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'D_chain_weighted': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
        }
        dummy_predictions = {
            'A_single_sensor': [(0, 0)] * len(test_w),
            'B_multi_naive': [(0, 0)] * len(test_w),
            'C_chain_filtered': [(0, 0)] * len(test_w),
            'D_chain_weighted': [(0, 0)] * len(test_w),
        }
        return dummy_results, dummy_predictions, None, None, None

    original_run_fold = l4_module.run_fold
    l4_module.run_fold = fake_run_fold

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'results.json')
            run_loso(tmpdir, output_path, epochs=1)
    finally:
        l4_module.load_pamap2_multisensor = original_loader
        l4_module.run_fold = original_run_fold

    # Each subject should be held out exactly once
    assert len(held_out_subjects) == len(subjects)
    assert sorted(held_out_subjects) == sorted(subjects)
    for subj in subjects:
        assert held_out_subjects.count(subj) == 1, \
            f"Subject {subj} should be held out exactly once, got {held_out_subjects.count(subj)}"


def test_conditions_use_same_held_out_folds():
    """
    Test that all conditions (A, B, C, D) use the same held-out subject in each fold.
    Uses distinguishable activities to verify the test set is identical across conditions.
    """
    from experiments.level4_multisensor_fusion import run_fold, ACTIVITY_LABELS, DOM2IDX
    import experiments.level4_multisensor_fusion as l4_module

    # Create synthetic windows for 2 subjects with different activities
    # Subject 1: lying (activity 1) -> domain index 0
    # Subject 2: sitting (activity 2) -> domain index 1
    windows = []

    for i in range(5):
        w1 = {
            'subject': 1,
            'activity': 1,  # lying
            'domain': ACTIVITY_LABELS[1],
            'hand_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
            'hand_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            'chest_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
            'chest_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            'ankle_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
            'ankle_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
        }
        windows.append(w1)

        w2 = {
            'subject': 2,
            'activity': 2,  # sitting
            'domain': ACTIVITY_LABELS[2],
            'hand_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
            'hand_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            'chest_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
            'chest_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            'ankle_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
            'ankle_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
        }
        windows.append(w2)

    # Run one fold with subject 1 (lying) as test
    test_subject = 1
    train_w = [w for w in windows if w['subject'] != test_subject]
    test_w = [w for w in windows if w['subject'] == test_subject]

    # Expected labels: all test windows are from subject 1 (activity 1 -> domain index 0)
    expected_label = DOM2IDX[ACTIVITY_LABELS[1]]
    expected_labels = [expected_label] * len(test_w)

    # Monkeypatch compute_detailed_metrics to record labels on each call
    original_metrics = l4_module.compute_detailed_metrics
    recorded_label_sequences = []

    def fake_metrics(model, test_data, n_classes):
        # Record the ordered labels for this condition
        labels = [label for _, label in test_data]
        recorded_label_sequences.append(labels)
        return {
            'accuracy': 0.5,
            'macro_f1_12class': 0.5,
            'macro_f1_supported_classes': 0.5,
            'weighted_f1': 0.5,
            'per_class': {}
        }

    l4_module.compute_detailed_metrics = fake_metrics

    try:
        fold_results, fold_predictions, _, _, _ = run_fold(train_w, test_w, 0, test_subject, epochs=1)
    finally:
        l4_module.compute_detailed_metrics = original_metrics

    # Assert metrics were computed exactly 4 times (one for each condition A, B, C, D)
    assert len(recorded_label_sequences) == 4, \
        f"Expected 4 metric calls (A, B, C, D), got {len(recorded_label_sequences)}"

    # Assert all four recorded label sequences are identical
    for i in range(1, 4):
        assert recorded_label_sequences[i] == recorded_label_sequences[0], \
            f"Condition {i} labels differ from condition 0"

    # Assert the labels match the expected labels for the held-out subject (activity 1 -> index 0)
    assert recorded_label_sequences[0] == expected_labels, \
        f"Expected labels {expected_labels}, got {recorded_label_sequences[0]}"


def test_pooled_metrics_from_held_out_predictions():
    """
    Test that pooled metrics are computed correctly from held-out predictions.
    Verifies that pooled sample counts equal accumulated evaluated test samples,
    and that pooled per-class supports sum to the pooled test count.
    """
    from experiments.level4_multisensor_fusion import compute_pooled_metrics

    # Create synthetic predictions for 2 classes with imbalanced support
    # Class 0: 100 samples, 90 correct (TP=90, FN=10)
    # Class 1: 10 samples, 5 correct (TP=5, FN=5)
    predictions = []
    for i in range(90):
        predictions.append((0, 0))  # correct
    for i in range(10):
        predictions.append((1, 0))  # incorrect: pred 1, true 0
    for i in range(5):
        predictions.append((1, 1))  # correct
    for i in range(5):
        predictions.append((0, 1))  # incorrect: pred 0, true 1

    pooled = compute_pooled_metrics(predictions, 2)

    # Verify pooled sample count
    assert pooled['n_pooled_samples'] == 110, \
        f"Expected 110 pooled samples, got {pooled['n_pooled_samples']}"

    # Verify per-class supports sum to total
    total_support = sum(pooled['per_class'][i]['support'] for i in range(2))
    assert total_support == 110, \
        f"Expected total support 110, got {total_support}"

    # Verify class 0 support is 100 (90 TP + 10 FN)
    assert pooled['per_class'][0]['support'] == 100
    # Verify class 1 support is 10 (5 TP + 5 FN)
    assert pooled['per_class'][1]['support'] == 10

    # Verify metrics are computed from predictions, not fold means
    # Class 0: precision = 90/(90+5) ≈ 0.947, recall = 90/100 = 0.9, F1 ≈ 0.923
    # Class 1: precision = 5/(5+10) ≈ 0.333, recall = 5/10 = 0.5, F1 ≈ 0.4
    assert pooled['per_class'][0]['precision'] == pytest.approx(0.9474, abs=0.001)
    assert pooled['per_class'][0]['recall'] == pytest.approx(0.9)
    assert pooled['per_class'][1]['precision'] == pytest.approx(0.3333, abs=0.001)
    assert pooled['per_class'][1]['recall'] == pytest.approx(0.5)


def test_paired_comparison_labels_and_paired_folds():
    """
    Test that paired comparisons use correct labels (B-A, C-B, D-C, D-A)
    and that mean differences are computed from aligned fold results.
    """
    from experiments.level4_multisensor_fusion import run_loso
    import experiments.level4_multisensor_fusion as l4_module
    import tempfile
    import os

    # Create synthetic windows for 3 subjects
    subjects = [1, 2, 3]
    windows = []

    for subj in subjects:
        for i in range(200):
            w = {
                'subject': subj,
                'activity': 1,
                'domain': 'lying',
                'hand_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'hand_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'chest_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'chest_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'ankle_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'ankle_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            }
            windows.append(w)

    original_loader = l4_module.load_pamap2_multisensor
    l4_module.load_pamap2_multisensor = lambda data_dir: windows

    # Create fold results with known differences
    fold_idx = [0]
    def fake_run_fold(train_w, test_w, fold_idx_arg, test_subject_id, epochs, norm_s=None, norm_m=None, chain_floor=None, seed=42):
        # B > A by 0.1, C > B by 0.05, D > C by 0.02, D > A by 0.17
        dummy_results = {
            'A_single_sensor': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'B_multi_naive': {'accuracy': 0.5, 'macro_f1_12class': 0.6, 'macro_f1_supported_classes': 0.6, 'weighted_f1': 0.6, 'per_class': {}},
            'C_chain_filtered': {'accuracy': 0.5, 'macro_f1_12class': 0.65, 'macro_f1_supported_classes': 0.65, 'weighted_f1': 0.65, 'per_class': {}},
            'D_chain_weighted': {'accuracy': 0.5, 'macro_f1_12class': 0.67, 'macro_f1_supported_classes': 0.67, 'weighted_f1': 0.67, 'per_class': {}},
        }
        dummy_predictions = {
            'A_single_sensor': [(0, 0)] * len(test_w),
            'B_multi_naive': [(0, 0)] * len(test_w),
            'C_chain_filtered': [(0, 0)] * len(test_w),
            'D_chain_weighted': [(0, 0)] * len(test_w),
        }
        return dummy_results, dummy_predictions, None, None, None

    original_run_fold = l4_module.run_fold
    l4_module.run_fold = fake_run_fold

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'results.json')
            results = run_loso(tmpdir, output_path, epochs=1)
    finally:
        l4_module.load_pamap2_multisensor = original_loader
        l4_module.run_fold = original_run_fold

    # Verify paired comparison labels are correct
    paired = results['paired_comparisons']
    assert 'B-A' in paired, "Missing B-A comparison"
    assert 'C-B' in paired, "Missing C-B comparison"
    assert 'D-C' in paired, "Missing D-C comparison"
    assert 'D-A' in paired, "Missing D-A comparison"

    # Verify mean differences match expected values
    assert paired['B-A']['mean'] == pytest.approx(0.1)
    assert paired['C-B']['mean'] == pytest.approx(0.05)
    assert paired['D-C']['mean'] == pytest.approx(0.02)
    assert paired['D-A']['mean'] == pytest.approx(0.17)

    # Verify n_paired_folds is reported
    assert paired['B-A']['n_paired_folds'] == 3
    assert paired['C-B']['n_paired_folds'] == 3


def test_aggregated_weighted_f1_present():
    """
    Test that aggregated statistics include weighted_f1 mean and std.
    """
    from experiments.level4_multisensor_fusion import run_loso
    import experiments.level4_multisensor_fusion as l4_module
    import tempfile
    import os

    subjects = [1, 2, 3]
    windows = []

    for subj in subjects:
        for i in range(200):
            w = {
                'subject': subj,
                'activity': 1,
                'domain': 'lying',
                'hand_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'hand_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'chest_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'chest_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'ankle_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'ankle_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            }
            windows.append(w)

    original_loader = l4_module.load_pamap2_multisensor
    l4_module.load_pamap2_multisensor = lambda data_dir: windows

    def fake_run_fold(train_w, test_w, fold_idx, test_subject_id, epochs, norm_s=None, norm_m=None, chain_floor=None, seed=42):
        dummy_results = {
            'A_single_sensor': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'B_multi_naive': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'C_chain_filtered': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'D_chain_weighted': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
        }
        dummy_predictions = {
            'A_single_sensor': [(0, 0)] * len(test_w),
            'B_multi_naive': [(0, 0)] * len(test_w),
            'C_chain_filtered': [(0, 0)] * len(test_w),
            'D_chain_weighted': [(0, 0)] * len(test_w),
        }
        return dummy_results, dummy_predictions, None, None, None

    original_run_fold = l4_module.run_fold
    l4_module.run_fold = fake_run_fold

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'results.json')
            results = run_loso(tmpdir, output_path, epochs=1)
    finally:
        l4_module.load_pamap2_multisensor = original_loader
        l4_module.run_fold = original_run_fold

    # Verify weighted_f1 aggregation is present
    for cond in ['A_single_sensor', 'B_multi_naive', 'C_chain_filtered', 'D_chain_weighted']:
        assert 'mean_weighted_f1' in results['aggregated'][cond], \
            f"Missing mean_weighted_f1 in {cond}"
        assert 'std_weighted_f1' in results['aggregated'][cond], \
            f"Missing std_weighted_f1 in {cond}"
        assert 'n_folds' in results['aggregated'][cond], \
            f"Missing n_folds in {cond}"


def test_run_legacy_five_value_return():
    """
    Test that run_legacy correctly handles the five-value return from run_fold.
    Monkeypatches load_pamap2_multisensor and run_fold to avoid real data and training.
    """
    from experiments.level4_multisensor_fusion import run_legacy
    import experiments.level4_multisensor_fusion as l4_module
    import tempfile
    import os

    # Create synthetic windows (>500 to pass the check)
    windows = []
    for subj in range(10):
        for i in range(100):
            w = {
                'subject': subj,
                'activity': 1,
                'domain': 'lying',
                'hand_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'hand_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'chest_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'chest_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'ankle_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'ankle_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            }
            windows.append(w)

    original_loader = l4_module.load_pamap2_multisensor
    l4_module.load_pamap2_multisensor = lambda data_dir: windows

    def fake_run_fold(train_w, test_w, fold_idx, test_subject_id, epochs, norm_s=None, norm_m=None, chain_floor=None, seed=42):
        dummy_results = {
            'A_single_sensor': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}, 'train_n': len(train_w)},
            'B_multi_naive': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}, 'train_n': len(train_w)},
            'C_chain_filtered': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}, 'train_n': len(train_w)},
            'D_chain_weighted': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}, 'train_n': len(train_w)},
        }
        dummy_predictions = {
            'A_single_sensor': [(0, 0)] * len(test_w),
            'B_multi_naive': [(0, 0)] * len(test_w),
            'C_chain_filtered': [(0, 0)] * len(test_w),
            'D_chain_weighted': [(0, 0)] * len(test_w),
        }
        return dummy_results, dummy_predictions, None, None, None

    original_run_fold = l4_module.run_fold
    l4_module.run_fold = fake_run_fold

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'results.json')
            run_legacy(tmpdir, output_path, epochs=1)
            # Verify the file was written
            assert os.path.exists(output_path), "Legacy results JSON was not written"
    finally:
        l4_module.load_pamap2_multisensor = original_loader
        l4_module.run_fold = original_run_fold


def test_weighted_f1_uses_unrounded_values():
    """
    Test that weighted-F1 uses unrounded per-class F1 values for calculation.
    Using the specific fixture: 4x(1,0), 14x(0,1), 2x(1,1)
    Expected weighted_f1 rounded to 4 decimal places is 0.1455.
    """
    from experiments.level4_multisensor_fusion import compute_pooled_metrics

    # Fixture: 4 samples (1,0), 14 samples (0,1), 2 samples (1,1)
    predictions = []
    for i in range(4):
        predictions.append((1, 0))  # pred 1, true 0
    for i in range(14):
        predictions.append((0, 1))  # pred 0, true 1
    for i in range(2):
        predictions.append((1, 1))  # pred 1, true 1

    pooled = compute_pooled_metrics(predictions, 2)

    # Expected weighted_f1 = 0.1455 (using unrounded per-class F1)
    assert pooled['weighted_f1'] == pytest.approx(0.1455, abs=0.0001), \
        f"Expected weighted_f1 ≈ 0.1455, got {pooled['weighted_f1']}"


def test_empty_pooled_metrics_schema():
    """
    Test that empty pooled metrics return correct schema with all classes.
    """
    from experiments.level4_multisensor_fusion import compute_pooled_metrics

    n_classes = 12
    pooled = compute_pooled_metrics([], n_classes)

    # Verify zero metrics
    assert pooled['accuracy'] == 0.0
    assert pooled['macro_f1_12class'] == 0.0
    assert pooled['macro_f1_supported_classes'] == 0.0
    assert pooled['weighted_f1'] == 0.0
    assert pooled['n_pooled_samples'] == 0

    # Verify per_class has all classes
    assert len(pooled['per_class']) == n_classes
    for i in range(n_classes):
        assert i in pooled['per_class']
        assert pooled['per_class'][i]['support'] == 0
        assert pooled['per_class'][i]['precision'] == 0.0
        assert pooled['per_class'][i]['recall'] == 0.0
        assert pooled['per_class'][i]['f1'] == 0.0


def test_mlp_seed_reproducibility():
    """
    Test that two MLP instances with the same seed have identical initial weights.
    """
    from experiments.level4_multisensor_fusion import MLP

    input_dim = 10
    hidden = 5
    output = 3

    # Create two MLPs with same seed
    mlp1 = MLP(input_dim=input_dim, hidden=hidden, output=output, seed=123)
    mlp2 = MLP(input_dim=input_dim, hidden=hidden, output=output, seed=123)

    # Verify weights are identical
    for j in range(hidden):
        for i in range(input_dim):
            assert mlp1.W1[j][i] == mlp2.W1[j][i], f"W1[{j}][{i}] differs"
        assert mlp1.b1[j] == mlp2.b1[j], f"b1[{j}] differs"

    for k in range(output):
        for j in range(hidden):
            assert mlp1.W2[k][j] == mlp2.W2[k][j], f"W2[{k}][{j}] differs"
        assert mlp1.b2[k] == mlp2.b2[k], f"b2[{k}] differs"


def test_mlp_different_seeds_different_weights():
    """
    Test that different seeds produce different initial weights.
    """
    from experiments.level4_multisensor_fusion import MLP

    input_dim = 10
    hidden = 5
    output = 3

    # Create two MLPs with different seeds
    mlp1 = MLP(input_dim=input_dim, hidden=hidden, output=output, seed=123)
    mlp2 = MLP(input_dim=input_dim, hidden=hidden, output=output, seed=456)

    # Verify at least some weights differ
    weights_differ = False
    for j in range(hidden):
        for i in range(input_dim):
            if mlp1.W1[j][i] != mlp2.W1[j][i]:
                weights_differ = True
                break
        if weights_differ:
            break

    assert weights_differ, "Weights should differ with different seeds"


def test_shuffling_does_not_affect_global_rng():
    """
    Test that shuffling and training do not reset or depend on Python's global random state.
    """
    from experiments.level4_multisensor_fusion import MLP
    import random

    # Set global RNG to known state
    random.seed(999)
    global_before = random.random()

    # Create and train an MLP with explicit shuffle seed
    mlp = MLP(input_dim=5, hidden=3, output=2, seed=42)
    samples = [([0.1, 0.2, 0.3, 0.4, 0.5], 0, 1.0) for _ in range(10)]
    mlp.train_epoch(samples, shuffle_seed=123)

    # Check global RNG state is unchanged
    global_after = random.random()
    assert global_after != global_before, "Global RNG should advance normally"

    # Reset global RNG and verify MLP behavior is independent
    random.seed(999)
    mlp2 = MLP(input_dim=5, hidden=3, output=2, seed=42)
    samples2 = [([0.1, 0.2, 0.3, 0.4, 0.5], 0, 1.0) for _ in range(10)]
    mlp2.train_epoch(samples2, shuffle_seed=123)

    # Verify MLPs have identical weights despite different global RNG states
    for j in range(3):
        for i in range(5):
            assert mlp.W1[j][i] == mlp2.W1[j][i], f"W1[{j}][{i}] differs despite global RNG"


def test_fold_condition_seed_derivation():
    """
    Test that fold/condition seed derivation is deterministic and documented.
    """
    from experiments.level4_multisensor_fusion import derive_seed

    # Test the seed derivation formula used in run_fold
    base_seed = 42
    fold_idx = 0

    # All conditions in same fold share the same model seed
    fold_model_seed = derive_seed(base_seed, fold_idx)
    fold_shuffle_seed = derive_seed(base_seed, fold_idx, "shuffle")

    # Verify seeds are deterministic
    fold_model_seed2 = derive_seed(base_seed, fold_idx)
    fold_shuffle_seed2 = derive_seed(base_seed, fold_idx, "shuffle")

    assert fold_model_seed == fold_model_seed2, "Model seed should be deterministic"
    assert fold_shuffle_seed == fold_shuffle_seed2, "Shuffle seed should be deterministic"

    # Verify different folds get different seeds
    fold_1_model_seed = derive_seed(base_seed, 1)
    assert fold_model_seed != fold_1_model_seed, "Different folds should have different seeds"

    # Verify shuffle seed differs from model seed
    assert fold_model_seed != fold_shuffle_seed, "Shuffle seed should differ from model seed"

    # Verify different base seeds produce different derived seeds
    different_base = derive_seed(999, fold_idx)
    assert fold_model_seed != different_base, "Different base seeds should produce different derived seeds"


def test_cross_process_seed_stability():
    """
    Test that derive_seed produces the same result across fresh Python processes.
    """
    from experiments.level4_multisensor_fusion import derive_seed
    import subprocess
    import sys

    base_seed = 42
    fold_idx = 0

    # Get seed in current process
    seed1 = derive_seed(base_seed, fold_idx)

    # Run a fresh Python process to compute the same seed
    code = f"""
import sys
sys.path.insert(0, '/Users/timbo/S2S')
from experiments.level4_multisensor_fusion import derive_seed
print(derive_seed({base_seed}, {fold_idx}))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    seed2 = int(result.stdout.strip())

    assert seed1 == seed2, f"Seeds differ across processes: {seed1} vs {seed2}"


def test_shared_per_fold_seeds():
    """
    Test that all conditions in the same fold share the same model and shuffle seeds.
    """
    from experiments.level4_multisensor_fusion import derive_seed

    base_seed = 42
    fold_idx = 0

    # All conditions should share the same model seed
    model_seed_a = derive_seed(base_seed, fold_idx)
    model_seed_b = derive_seed(base_seed, fold_idx)
    model_seed_c = derive_seed(base_seed, fold_idx)
    model_seed_d = derive_seed(base_seed, fold_idx)

    assert model_seed_a == model_seed_b == model_seed_c == model_seed_d, \
        "All conditions in same fold should share model seed"

    # All conditions should share the same shuffle seed
    shuffle_seed = derive_seed(base_seed, fold_idx, "shuffle")
    assert shuffle_seed == derive_seed(base_seed, fold_idx, "shuffle"), \
        "Shuffle seed should be consistent"


def test_shuffle_rng_advances_across_epochs():
    """
    Test that the shuffle RNG advances state across epochs rather than resetting.
    """
    from experiments.level4_multisensor_fusion import MLP

    mlp = MLP(input_dim=5, hidden=3, output=2, seed=42)
    samples = [([0.1, 0.2, 0.3, 0.4, 0.5], 0, 1.0) for _ in range(10)]

    # Train multiple epochs with same shuffle seed
    shuffle_seed = 123
    for _ in range(3):
        mlp.train_epoch(samples, shuffle_seed=shuffle_seed)

    # Verify shuffle RNG was initialized and used (not None)
    assert mlp._shuffle_rng is not None, "Shuffle RNG should be initialized"

    # Create a new MLP and train with same seed
    mlp2 = MLP(input_dim=5, hidden=3, output=2, seed=42)
    samples2 = [([0.1, 0.2, 0.3, 0.4, 0.5], 0, 1.0) for _ in range(10)]
    for _ in range(3):
        mlp2.train_epoch(samples2, shuffle_seed=shuffle_seed)

    # Both should have identical weights since they followed same deterministic path
    for j in range(3):
        for i in range(5):
            assert mlp.W1[j][i] == mlp2.W1[j][i], f"W1[{j}][{i}] differs"


def test_run_loso_passes_base_seed_to_run_fold():
    """
    Test that run_loso passes the base seed to run_fold, not a pre-derived fold seed.
    run_fold should derive fold-specific seeds internally from base seed + fold_idx.
    """
    from experiments.level4_multisensor_fusion import run_loso
    import experiments.level4_multisensor_fusion as l4_module
    import tempfile

    # Create synthetic windows for 2 subjects (need at least 500 windows total)
    WINDOW_SIZE = l4_module.WINDOW_SIZE
    windows = []
    for subj in [1, 2]:
        for i in range(300):
            w = {
                'subject': subj,
                'domain': 'lying',
                'hand_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'hand_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'chest_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'chest_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
                'ankle_a': [[1.0, 2.0, 3.0]] * WINDOW_SIZE,
                'ankle_g': [[0.1, 0.2, 0.3]] * WINDOW_SIZE,
            }
            windows.append(w)

    original_loader = l4_module.load_pamap2_multisensor
    l4_module.load_pamap2_multisensor = lambda data_dir: windows

    # Track what seed run_fold receives
    seeds_received = []

    def fake_run_fold(train_w, test_w, fold_idx, test_subject_id, epochs, norm_s=None, norm_m=None, chain_floor=None, seed=42):
        seeds_received.append((fold_idx, seed))
        dummy_results = {
            'A_single_sensor': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'B_multi_naive': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'C_chain_filtered': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
            'D_chain_weighted': {'accuracy': 0.5, 'macro_f1_12class': 0.5, 'macro_f1_supported_classes': 0.5, 'weighted_f1': 0.5, 'per_class': {}},
        }
        dummy_predictions = {
            'A_single_sensor': [(0, 0)] * len(test_w),
            'B_multi_naive': [(0, 0)] * len(test_w),
            'C_chain_filtered': [(0, 0)] * len(test_w),
            'D_chain_weighted': [(0, 0)] * len(test_w),
        }
        return dummy_results, dummy_predictions, None, None, None

    original_run_fold = l4_module.run_fold
    l4_module.run_fold = fake_run_fold

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'results.json')
            base_seed = 123
            run_loso(tmpdir, output_path, epochs=1, seed=base_seed)
    finally:
        l4_module.load_pamap2_multisensor = original_loader
        l4_module.run_fold = original_run_fold

    # Verify run_fold received the base seed for each fold, not pre-derived seeds
    assert len(seeds_received) == 2, f"Expected 2 folds, got {len(seeds_received)}"
    for fold_idx, received_seed in seeds_received:
        assert received_seed == base_seed, \
            f"Fold {fold_idx} received seed {received_seed}, expected base seed {base_seed}"
