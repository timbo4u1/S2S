import numpy as np
import pytest

import run_benchmark
from experiments.level4_multisensor_fusion import DOMAINS, MLP


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
