from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

import cssd_dbd_case.runner as runner_module
from cssd_dbd_case.config import (
    FORMAL_ALGORITHMS,
    FORMAL_DATASET_SHA256,
    load_experiment_spec,
)
from cssd_dbd_case.runner import (
    _exclusive_run_lock,
    _read_successes,
    audit_instances,
    build_tasks,
    run_experiment,
    run_one,
)
from cssd_dbd_case.reporting import build_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_formal_contract_is_three_algorithms_and_dbd_only() -> None:
    spec = load_experiment_spec(PROJECT_ROOT / "configs" / "dbd_proxy91.yaml")
    audit = audit_instances(spec)
    tasks = build_tasks(spec)

    assert tuple(item.name for item in spec.algorithms) == FORMAL_ALGORITHMS
    assert spec.seeds == (101,)
    assert len(spec.instance_paths) == 91
    assert spec.day_length_minutes == 1440
    assert spec.workers == 8
    assert audit["total_jobs"] == 22_927
    assert audit["dataset_sha256"] == FORMAL_DATASET_SHA256
    assert len(tasks) == 273
    assert {task["planning_mode"] for task in tasks} == {"DBD"}
    assert {task["algorithm"] for task in tasks} == set(FORMAL_ALGORITHMS)


def test_formal_contract_rejects_parameter_drift(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "dbd_proxy91.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["experiment"]["algorithms"][1]["parameters"]["population_size"] = 21
    altered = tmp_path / "altered.yaml"
    altered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="formal parameters for alns were altered"):
        load_experiment_spec(altered)


def test_formal_contract_rejects_day_length_drift(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "dbd_proxy91.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["experiment"]["day_length_minutes"] = 720
    altered = tmp_path / "altered_day_length.yaml"
    altered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="formal proxy91 case-study contract was altered"):
        load_experiment_spec(altered)


def test_one_task_returns_feasible_paper_metrics() -> None:
    spec = load_experiment_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
    task = next(task for task in build_tasks(spec) if task["algorithm"] == "alns")
    result = run_one(task)

    assert result["feasible"] is True
    assert result["planning_mode"] == "DBD"
    assert result["evaluations"] <= 12
    assert result["metric_moments"]["time_in_system"]["count"] == result["jobs"]
    assert sum(result["arrivals_hourly"]) == result["jobs"]
    assert sum(result["completed_stage1_hourly"]) == result["jobs"]
    assert sum(result["completed_stage2_hourly"]) == result["jobs"]
    assert (
        result["metric_moments"]["time_in_system"]["sum"]
        == result["objective"] - result["release_sum"]
    )
    for stage in (1, 2):
        expected = result[f"stage{stage}_processing_minutes"] / (
            result[f"machines_stage{stage}"] * result[f"stage{stage}_active_horizon_minutes"]
        )
        assert result[f"utilization_stage{stage}"] == pytest.approx(expected)


def test_smoke_experiment_builds_complete_report(tmp_path: Path) -> None:
    run = run_experiment(
        PROJECT_ROOT / "configs" / "smoke.yaml",
        tmp_path,
        workers=1,
    )
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    integrity = json.loads((run / "report" / "integrity.json").read_text(encoding="utf-8"))

    assert manifest["task_count_total"] == 6
    assert manifest["task_count_successful"] == 6
    assert manifest["task_count_pending"] == 0
    assert manifest["budget_summary"]["all_tasks_evaluation_limit"] == 72
    assert (run / "task_manifest.json").is_file()
    assert integrity["status"] == "PASS"
    assert integrity["errors"] == []
    assert (run / "figures" / "figure_4_7_dbd_stage1.png").is_file()
    assert (run / "figures" / "figure_4_8_dbd_stage2.png").is_file()

    with (run / "tables" / "table_4_9_dbd_three_algorithms.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        table = list(csv.DictReader(handle))
    assert [row["method"] for row in table] == ["IBS-CABC", "ALNS", "TS"]
    assert {int(row["day_count"]) for row in table} == {2}
    assert {int(row["device_count"]) for row in table} == {664}
    assert all("stage1_utilization_daily_mean_percent" in row for row in table)
    assert all("stage1_utilization_pooled_active_horizon_percent" in row for row in table)

    with (run / "tables" / "hourly_stage2_dbd.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        hourly = list(csv.DictReader(handle))
    assert len(hourly) == 24
    assert sum(float(row["IBS-CABC_total"]) for row in hourly) == 664
    assert sum(float(row["ALNS_total"]) for row in hourly) == 664
    assert sum(float(row["TS_total"]) for row in hourly) == 664

    assert (
        run_experiment(
            PROJECT_ROOT / "configs" / "smoke.yaml",
            workers=1,
            resume=run,
        )
        == run
    )

    results_path = run / "raw" / "results.jsonl"
    original_results = results_path.read_text(encoding="utf-8")
    result_rows = [json.loads(line) for line in original_results.splitlines() if line.strip()]
    result_rows[0]["algorithm_label"] = "tampered"
    results_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    build_report(run)
    tampered_integrity = json.loads((run / "report" / "integrity.json").read_text(encoding="utf-8"))
    assert tampered_integrity["status"] == "INCOMPLETE"
    assert any("persisted task contract" in error for error in tampered_integrity["errors"])
    results_path.write_text(original_results, encoding="utf-8")
    build_report(run)

    tampered = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    tampered["audit"]["dataset_sha256"] = "tampered"
    (run / "manifest.json").write_text(
        json.dumps(tampered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset"):
        run_experiment(
            PROJECT_ROOT / "configs" / "smoke.yaml",
            workers=1,
            resume=run,
        )


def test_serial_interruption_is_checkpointed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(_task: object) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_module, "run_one", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(
            PROJECT_ROOT / "configs" / "smoke.yaml",
            tmp_path,
            workers=1,
        )

    run = next(path for path in tmp_path.iterdir() if path.is_dir())
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert progress["state"] == "interrupted"
    assert progress["task_count_successful"] == 0
    assert progress["task_count_pending"] == 6
    assert manifest["task_count_pending"] == 6
    assert manifest["last_interrupted_utc"]


def test_jsonl_tail_repair_and_duplicate_rejection(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        '{"task_id":"one","value":1}\n{"task_id":"two"',
        encoding="utf-8",
    )
    assert set(_read_successes(path)) == {"one"}
    assert path.read_text(encoding="utf-8") == '{"task_id":"one","value":1}\n'

    path.write_text('{"task_id":"one"}\n{"task_id":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task_id"):
        _read_successes(path)


def test_experiment_directory_lock_rejects_second_writer(tmp_path: Path) -> None:
    with _exclusive_run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another runner"):
            with _exclusive_run_lock(tmp_path):
                pass
