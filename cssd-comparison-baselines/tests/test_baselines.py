from __future__ import annotations

from copy import deepcopy
from random import Random

import pytest

from cssd_baselines import solve_alns, solve_standard_abc, solve_tabu_search
from cssd_baselines import bridge
from cssd_baselines.common import BaselineConfig, SearchRuntime
from cssd_baselines.neighborhoods import large_neighbor, small_neighbor
from cssd_baselines.convergence import write_convergence_reports
from cssd_baselines.reporting import (
    ExperimentIntegrityError,
    aggregate_rows,
    load_ibscabc_results,
    print_summary,
    validate_complete_four_algorithm_results,
    write_four_algorithm_reports,
)

from cssd_ibscabc.core.feasibility import validate_schedule
from cssd_ibscabc.io.loader import load_instance
from cssd_ibscabc.representation.batch_structure import validate_structure


def _instance():
    path = sorted((bridge.ARTICLE_ROOT / "dataset" / "Instances 25").glob("*.txt"))[0]
    return load_instance(path)


def _config(seed: int = 77) -> BaselineConfig:
    return BaselineConfig(
        population_size=4,
        stagnation_limit=5,
        candidate_attempts=1,
        onlooker_count=4,
        neighborhood_size=4,
        tabu_tenure=5,
        max_evaluations=80,
        time_limit_seconds=None,
        cache_capacity=64,
        seed=seed,
    )


def test_all_baselines_are_deterministic_and_feasible():
    instance = _instance()
    for solver, label in (
        (solve_standard_abc, "Standard ABC"),
        (solve_alns, "ALNS"),
        (solve_tabu_search, "TS"),
    ):
        first = solver(instance, _config())
        second = solver(instance, _config())
        assert first.evaluations == second.evaluations == 80
        assert first.schedule.objective == second.schedule.objective
        assert first.history == second.history
        assert first.schedule.algorithm == label
        validate_schedule(instance, first.schedule)


def test_small_and_large_neighborhoods_preserve_structure():
    instance = _instance()
    runtime = SearchRuntime(instance, _config(), "test")
    source = runtime.initial_population()[0]
    rng = Random(11)
    small = small_neighbor(instance, source.solution, source.schedule, rng)
    assert small is not None
    validate_structure(instance, small.solution)
    large = large_neighbor(
        "window_rebuild", instance, source.solution, source.schedule, rng
    )
    assert large is not None
    validate_structure(instance, large.solution)


def test_required_metrics_are_aggregated():
    rows = [
        {
            "algorithm_label": "ALNS",
            "jobs": 100,
            "total_completion_time": 1000 + index,
            "average_time_in_system": 100 + index,
            "c_max": 200 + index,
            "utilization_stage1_percent": 60 + index,
            "utilization_stage2_percent": 70 + index,
            "runtime_seconds": 1 + index,
            "feasible": True,
        }
        for index in range(2)
    ]
    summary = aggregate_rows(rows)[0]
    assert summary["total_completion_time_paper"]
    assert summary["average_time_in_system_paper"]
    assert summary["c_max_paper"]
    assert summary["utilization_stage1_percent_paper"]
    assert summary["utilization_stage2_percent_paper"]


def _formal_four_algorithm_rows():
    ibscabc_paths = [
        bridge.REFERENCE_ROOT
        / "outputs"
        / run
        / "raw"
        / "results.jsonl"
        for run in (
            "e4_fixed_j25_20260807T083643121632Z",
            "e4_fixed_scale_20260730T055650782856Z",
        )
    ]
    missing = [path for path in ibscabc_paths if not path.is_file()]
    if missing:
        pytest.skip(
            "legacy J25/single-seed integration results are intentionally excluded "
            "from the current multi-seed reproducibility package"
        )
    ibscabc = [
        row for path in ibscabc_paths for row in load_ibscabc_results(path)
    ]
    baselines = []
    for algorithm, label in (
        ("standard_abc", "Standard ABC"),
        ("alns", "ALNS"),
        ("ts", "TS"),
    ):
        for source in ibscabc:
            row = deepcopy(source)
            row["task_id"] = f"{algorithm}:{source['task_id']}"
            row["algorithm"] = algorithm
            row["algorithm_label"] = label
            baselines.append(row)
    return ibscabc_paths, ibscabc, baselines


def test_four_algorithm_integrity_contract_and_reports(tmp_path):
    ibscabc_paths, ibscabc, baselines = _formal_four_algorithm_rows()
    integrity = validate_complete_four_algorithm_results(ibscabc + baselines)
    assert integrity["status"] == "PASS"
    assert integrity["result_count"] == 400
    assert integrity["group_count"] == 20

    summaries = write_four_algorithm_reports(tmp_path, baselines, ibscabc_paths)
    assert len(summaries) == 20
    assert {row["instance_count"] for row in summaries} == {20}
    assert (tmp_path / "tables" / "four_algorithm_summary.csv").is_file()
    assert (tmp_path / "tables" / "four_algorithm_integrity.json").is_file()

    convergence = write_convergence_reports(tmp_path, baselines, ibscabc_paths)
    assert convergence["status"] == "IBS_CABC_BEST_OR_TIED_ON_ALL_SCALES"
    for jobs in (25, 100, 250, 500, 1000):
        assert (tmp_path / "figures" / f"convergence_J{jobs}.png").is_file()
    assert (tmp_path / "tables" / "convergence_diagnostics.json").is_file()


def test_four_algorithm_integrity_rejects_incomplete_results():
    _, ibscabc, baselines = _formal_four_algorithm_rows()
    with pytest.raises(ExperimentIntegrityError, match="expected 400"):
        validate_complete_four_algorithm_results((ibscabc + baselines)[:-1])


def test_summary_prints_all_required_metric_headers(capsys):
    _, ibscabc, baselines = _formal_four_algorithm_rows()
    summaries = aggregate_rows(ibscabc + baselines)
    print_summary(summaries, title="四算法检查")
    output = capsys.readouterr().out
    for text in (
        "IBS_CABC",
        "Standard ABC",
        "ALNS",
        "TS",
        "J25",
        "J100",
        "J250",
        "J500",
        "J1000",
        "ΣCj2均值（标准差）",
        "平均TIS",
        "Cmax",
        "阶段1利用率/%",
        "阶段2利用率/%",
    ):
        assert text in output
