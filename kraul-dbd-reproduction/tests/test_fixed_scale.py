from __future__ import annotations

import numpy as np

from kraul_dbd_reproduction.config import PROJECT_ROOT
from kraul_dbd_reproduction.fixed_config import (
    FORMAL_FIXED_DATASET_SHA256,
    load_fixed_scale_spec,
)
from kraul_dbd_reproduction.fixed_runner import audit_fixed_instances, dry_run_fixed_scale
from kraul_dbd_reproduction.fixed_reporting import (
    PAPER_TABLE5_KRAUL_GA,
    _paper_table5_validation,
    _step_series,
    _with_derived_metrics,
)
from kraul_dbd_reproduction.ga import config_from_profile, run_genetic_algorithm
from kraul_dbd_reproduction.model import load_instance


def test_formal_fixed_scale_contract_covers_all_five_sizes() -> None:
    config = PROJECT_ROOT / "configs/fixed_scale_all.yaml"
    spec = load_fixed_scale_spec(config)
    audit = audit_fixed_instances(spec)
    assert spec.is_formal
    assert len(spec.instance_paths) == 100
    assert audit["dataset_sha256"] == FORMAL_FIXED_DATASET_SHA256
    assert audit["total_jobs"] == 37_500
    assert audit["counts_by_jobs"] == {25: 20, 100: 20, 250: 20, 500: 20, 1000: 20}

    preview = dry_run_fixed_scale(config)
    assert preview["tasks"] == 100
    assert preview["seeds"] == [101]
    assert preview["nominal_seconds_by_jobs"] == {
        25: 131.25,
        100: 1800.0,
        250: 10_725.0,
        500: 36_000.0,
        1000: 36_000.0,
    }


def test_ga_records_auditable_convergence_coordinates() -> None:
    instance = load_instance(
        (PROJECT_ROOT / "../dataset/Instances 25/J_25_F_5_seed_0.txt").resolve()
    )
    config = config_from_profile(
        "paper_final",
        {
            "population_size": 10,
            "max_elitism_percentage": 0.20,
            "local_search_moves": 8,
            "restart_interval": 5,
            "max_restart_attempts": 1,
        },
    )
    result = run_genetic_algorithm(
        instance,
        seed=101,
        time_limit_seconds=60.0,
        config=config,
        max_evaluations=20,
    )
    lengths = {
        len(result.history),
        len(result.incumbent_history),
        len(result.history_elapsed_seconds),
        len(result.history_evaluations),
    }
    assert lengths == {1}
    assert result.history_elapsed_seconds[0] > 0
    assert result.history_evaluations[0] == 10
    assert result.incumbent_history[0] <= result.history[0]


def test_convergence_endpoint_includes_the_final_population_evaluation() -> None:
    record = {
        "time_limit_seconds": 10.0,
        "algorithm_diagnostics": {
            "history": [100.0, 95.0],
            "incumbent_history": [100.0, 95.0],
            "history_elapsed_seconds": [2.0, 8.0],
            "history_evaluations": [100, 200],
            "best_seen_during_search": 90.0,
        },
    }
    values = _step_series(record, np.asarray([0.0, 50.0, 100.0]))
    assert np.allclose(values, [0.0, 0.0, 10.0])


def test_fixed_report_derives_makespan_from_legacy_raw_records() -> None:
    record = {
        "task_id": "legacy",
        "schedule": {"completion_stage2": [10, 17, 12]},
    }
    output = _with_derived_metrics(record)
    assert output["makespan"] == 17
    assert "makespan" not in record


def test_paper_table5_validation_uses_the_tis_identity_for_j250() -> None:
    rows = []
    for jobs, paper in PAPER_TABLE5_KRAUL_GA.items():
        implied_tis = 388.0 if jobs == 250 else paper["time_in_system_printed_mean"]
        rows.append(
            {
                "scale": f"J{jobs}",
                "jobs": jobs,
                "instances": 20,
                "objective_mean": paper["objective_mean"],
                "objective_sample_std": paper["objective_sample_std"],
                "release_sum_mean": paper["objective_mean"] - jobs * implied_tis,
                "average_time_in_system_mean": implied_tis,
                "average_time_in_system_sample_std": paper[
                    "time_in_system_printed_sample_std"
                ],
                "makespan_mean": paper["makespan_mean"],
                "makespan_sample_std": paper["makespan_sample_std"],
                "utilization_stage1_mean": paper["utilization_stage1_percent"] / 100.0,
                "utilization_stage2_mean": paper["utilization_stage2_percent"] / 100.0,
            }
        )
    validation, comparisons = _paper_table5_validation(rows)
    assert validation["status"] == "PASS"
    j250 = next(row for row in comparisons if row["scale"] == "J250")
    assert j250["time_in_system_paper_printed"] == 481
    assert j250["time_in_system_implied_by_paper_objective"] == 388
    assert j250["all_checks_pass"] is True
