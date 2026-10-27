"""Regression tests for benchmark runner truthfulness."""

import run_benchmark


def test_tier_expectation_does_not_count_execution_error_as_detection():
    failed_execution = {
        "tier": "REJECTED",
        "execution_ok": False,
        "error": "RuntimeError: sentinel",
    }
    assert not run_benchmark.tier_is_expected(failed_execution, ("REJECTED",))


def test_legacy_result_without_execution_field_keeps_tier_compatibility():
    # Saved reference cases predate the execution_ok field.
    assert run_benchmark.tier_is_expected(
        {"tier": "BRONZE"}, ("GOLD", "SILVER", "BRONZE")
    )


def test_certify_exception_is_not_reported_as_rejected(monkeypatch):
    def raise_sentinel(*args, **kwargs):
        raise RuntimeError("sentinel certification failure")

    monkeypatch.setattr(run_benchmark.engine, "certify", raise_sentinel)
    result = run_benchmark.certify(
        acc=[[0.0, 0.0, 9.81]] * 8,
        gyro=[[0.0, 0.0, 0.0]] * 8,
        ts=list(range(8)),
    )

    assert result["tier"] == "ERROR"
    assert result["execution_ok"] is False
    assert "sentinel certification failure" in result["error"]


def test_certify_marks_successful_execution(monkeypatch):
    monkeypatch.setattr(
        run_benchmark.engine,
        "certify",
        lambda *args, **kwargs: {
            "tier": "SILVER",
            "physical_law_score": 65,
        },
    )
    result = run_benchmark.certify(
        acc=[[0.0, 0.0, 9.81]] * 8,
        gyro=[[0.0, 0.0, 0.0]] * 8,
        ts=list(range(8)),
    )

    assert result["execution_ok"] is True
    assert run_benchmark.tier_is_expected(result, ("SILVER",))
