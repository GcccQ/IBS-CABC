from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kraul_dbd_reproduction.config import (
    FORMAL_NOTEBOOK_SHA256,
    PROJECT_ROOT,
    _protocol,
    load_experiment_spec,
)
from kraul_dbd_reproduction.decoder import decode_permutation, validate_schedule
from kraul_dbd_reproduction.ga import config_from_profile, run_genetic_algorithm
from kraul_dbd_reproduction.metrics import schedule_metrics
from kraul_dbd_reproduction.model import load_instance, official_time_limit


DATA_ROOT = (PROJECT_ROOT / "../dataset/RealWorldInstances365").resolve()


def test_loader_applies_sules_rule_and_author_orders() -> None:
    instance = load_instance(DATA_ROOT / "J_419_F_5_seed_233.txt")
    assert instance.job_count == 419
    assert (instance.machines_stage1, instance.machines_stage2) == (12, 5)
    assert (instance.capacity_stage1, instance.capacity_stage2) == (15, 11000)
    assert instance.raw_jobs[0, 1] == 50
    assert instance.raw_jobs[0, 5] == 51
    assert instance.jobs[0, 1] == 101
    assert instance.jobs[0, 2] == 131
    assert np.array_equal(np.sort(instance.erd_order), np.arange(instance.job_count))


def test_ffcbspt_released_decoder_regression() -> None:
    instance = load_instance(DATA_ROOT / "J_419_F_5_seed_233.txt")
    schedule = decode_permutation(instance, instance.erd_order)
    validate_schedule(instance, schedule)
    assert schedule.objective == 563_699
    metrics = schedule_metrics(instance, schedule)
    assert metrics["time_in_system"]["sum"] == metrics["objective"] - metrics["release_sum"]
    assert sum(metrics["arrivals_hourly"]) == instance.job_count
    assert sum(metrics["completed_stage1_hourly"]) == instance.job_count
    assert sum(metrics["completed_stage2_hourly"]) == instance.job_count


def test_final_paper_and_notebook_profiles_are_not_conflated() -> None:
    paper = config_from_profile("paper_final")
    notebook = config_from_profile("notebook_example")
    assert (paper.population_size, paper.crossover_rate, paper.mutation_rate) == (100, 0.95, 0.40)
    assert paper.mutation_type == "insert"
    assert (notebook.crossover_rate, notebook.mutation_type) == (0.80, "flip")
    assert paper.selection_semantics == "released_objective_proportional"


def test_ga_is_seed_reproducible_under_an_evaluation_boundary() -> None:
    instance = load_instance(DATA_ROOT / "J_80_F_5_seed_281.txt")
    config = config_from_profile(
        "paper_final",
        {
            "population_size": 10,
            "max_elitism_percentage": 0.20,
            "local_search_moves": 3,
            "max_restart_attempts": 0,
        },
    )
    first = run_genetic_algorithm(
        instance,
        seed=101,
        time_limit_seconds=60,
        config=config,
        max_evaluations=10,
    )
    second = run_genetic_algorithm(
        instance,
        seed=101,
        time_limit_seconds=60,
        config=config,
        max_evaluations=10,
    )
    assert first.schedule.objective == second.schedule.objective
    assert np.array_equal(first.schedule.permutation, second.schedule.permutation)
    assert first.history == second.history
    assert first.evaluations == second.evaluations == 10
    assert first.termination_reason == second.termination_reason == "evaluation_limit"

    partial = run_genetic_algorithm(
        instance,
        seed=101,
        time_limit_seconds=60,
        config=config,
        max_evaluations=3,
    )
    assert partial.evaluations == 3
    assert partial.termination_reason == "evaluation_limit"
    validate_schedule(instance, partial.schedule)


def test_loader_rejects_invalid_numeric_and_family_data(tmp_path: Path) -> None:
    malformed = {
        "nan": "1 1 1 1 15 11000\n1 nan 20 0 1 5 5 1 10\n",
        "negative": "1 1 1 1 15 11000\n1 10 20 -1 1 5 5 1 10\n",
        "fractional": "1 1 1 1 15 11000\n1 10.5 20 0 1 5 5 1 10\n",
        "stage2_load": "1 1 1 1 15 11000\n1 10 20 0 1 5 5 1 11\n",
        "setup": (
            "2 1 1 1 15 11000\n"
            "1 10 20 0 1 5 5 1 10\n"
            "2 10 20 0 1 6 5 1 10\n"
        ),
    }
    for name, content in malformed.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_instance(path)


def test_schedule_validation_cross_checks_batches_and_job_arrays() -> None:
    instance = load_instance(DATA_ROOT / "J_80_F_5_seed_281.txt")
    schedule = decode_permutation(instance, instance.erd_order)
    machines = schedule.machine_stage1.copy()
    machines[0] = instance.machines_stage1 + 1
    with pytest.raises(ValueError, match="machine IDs disagree"):
        validate_schedule(instance, replace(schedule, machine_stage1=machines))

    first_batch = replace(
        schedule.stage1_batches[0], machine_id=instance.machines_stage1 + 1
    )
    with pytest.raises(ValueError, match="invalid machine ID"):
        validate_schedule(
            instance,
            replace(
                schedule,
                stage1_batches=(first_batch, *schedule.stage1_batches[1:]),
            ),
        )


def test_ga_rejects_a_restart_population_that_would_grow() -> None:
    with pytest.raises(ValueError, match="restart regeneration"):
        config_from_profile("paper_final", {"population_size": 8})


def test_protocol_rejects_nan_and_misspelled_fields() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _protocol({"mode": "official_time", "name": "bad", "tau": float("nan")})
    with pytest.raises(ValueError, match="unknown protocol fields"):
        _protocol(
            {
                "mode": "wall_clock",
                "name": "bad",
                "time_limit_seconds": 1,
                "max_evalutions": 10,
            }
        )


def test_formal_contract_and_source_notebook_are_locked() -> None:
    spec = load_experiment_spec(PROJECT_ROOT / "configs/dbd_proxy91.yaml")
    assert spec.is_formal
    assert len(spec.instance_paths) == 91
    assert spec.expected_jobs == 22_927
    assert official_time_limit(load_instance(spec.instance_paths[0])) == 1497.2685
    assert hashlib.sha256(spec.source_notebook.read_bytes()).hexdigest() == FORMAL_NOTEBOOK_SHA256
    notebook_text = spec.source_notebook.read_text(encoding="utf-8")
    assert "lower_bound = processin" in notebook_text
