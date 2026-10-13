import pytest

from cssd_ibscabc.algorithms.common import EvaluationBudget


def test_paper_time_budget_uses_elapsed_clock_and_records_process_cpu(
    monkeypatch,
    small_instance,
):
    elapsed_values = iter([10.0, 10.6, 11.1])
    process_values = iter([3.0, 3.2])
    monkeypatch.setattr(
        "cssd_ibscabc.algorithms.common.time.perf_counter",
        lambda: next(elapsed_values),
    )
    monkeypatch.setattr(
        "cssd_ibscabc.algorithms.common.time.process_time",
        lambda: next(process_values),
    )
    budget = EvaluationBudget(
        small_instance,
        max_evaluations=None,
        time_limit_seconds=1.0,
        algorithm="unit",
    )
    assert budget.elapsed == pytest.approx(0.6)
    assert budget.exhausted
    assert budget.process_elapsed == pytest.approx(0.2)
