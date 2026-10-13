import json
from pathlib import Path

import pytest

from cssd_ibscabc.experiments import ExperimentRunner, load_experiment_config
from cssd_ibscabc.experiments.runner import _atomic_write_json
from cssd_ibscabc.reporting import build_report
from scripts.run_cp_sat_ablation import (
    _is_complete,
    _latest_complete,
    _latest_incomplete,
)


def _one_instance_config(split: Path) -> dict[str, object]:
    return {
        "experiment": {
            "name": "unit_pipeline",
            "selection": {"split": str(split)},
            "protocols": [{"name": "E20", "mode": "evaluations", "max_evaluations": 20}],
            "seeds": [1],
            "algorithms": [
                {
                    "name": "ibscabc",
                    "parameters": {
                        "colony_size": 4,
                        "archive_size": 8,
                        "assignment_repair_enabled": False,
                    },
                }
            ],
        }
    }


def test_resume_and_report_pipeline(data_root, tmp_path):
    split = tmp_path / "one.txt"
    instance = sorted((data_root / "Instances 25").glob("*.txt"))[0]
    split.write_text(str(instance.relative_to(data_root)).replace("\\", "/") + "\n")
    config = _one_instance_config(split)
    runner = ExperimentRunner(config, data_root, tmp_path / "outputs")
    output = runner.run()
    first_lines = (output / "raw" / "results.jsonl").read_text().splitlines()
    with (output / "raw" / "results.jsonl").open("ab") as handle:
        handle.write(b'{"task_id":')
    runner.run(output)
    second_lines = (output / "raw" / "results.jsonl").read_text().splitlines()
    assert len(first_lines) == len(second_lines) == 1
    assert all(json.loads(line)["feasible"] for line in second_lines)
    progress = json.loads((output / "progress.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert progress["status"] == "finished"
    assert progress["task_count_pending"] == 0
    assert manifest["truncated_torn_result_bytes"] > 0
    assert set(manifest["internal_thread_environment"].values()) == {"1"}
    report = build_report(output)
    assert report.exists()
    summary = json.loads((output / "report" / "performance_summary.json").read_text())
    assert summary["status"] == "INCOMPLETE"

    changed = json.loads(json.dumps(config))
    changed["experiment"]["seeds"] = [2]
    with pytest.raises(ValueError, match="resume refused"):
        ExperimentRunner(changed, data_root, tmp_path / "outputs").run(output)


def test_base_algorithm_defaults_are_applied_with_local_override():
    project = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project / "configs" / "smoke.yaml")
    algorithm = config["experiment"]["algorithms"][0]
    assert algorithm["name"] == "ibscabc"
    assert algorithm["parameters"]["colony_size"] == 4
    assert algorithm["parameters"]["ejection_depth"] == 3


def test_experiment_rejects_disabling_main_mechanism(data_root, tmp_path):
    split = tmp_path / "one.txt"
    instance = sorted((data_root / "Instances 25").glob("*.txt"))[0]
    split.write_text(str(instance.relative_to(data_root)).replace("\\", "/") + "\n")
    config = _one_instance_config(split)
    config["experiment"]["algorithms"][0]["parameters"]["bottleneck_aware"] = False
    with pytest.raises(ValueError, match="must keep bottleneck-aware"):
        ExperimentRunner(config, data_root, tmp_path / "outputs").tasks()


def test_experiment_rejects_removed_algorithm(data_root, tmp_path):
    split = tmp_path / "one.txt"
    instance = sorted((data_root / "Instances 25").glob("*.txt"))[0]
    split.write_text(str(instance.relative_to(data_root)).replace("\\", "/") + "\n")
    config = _one_instance_config(split)
    config["experiment"]["algorithms"] = [{"name": "kraul_ga_corrected"}]
    with pytest.raises(ValueError, match="unknown algorithms"):
        ExperimentRunner(config, data_root, tmp_path / "outputs").tasks()


def test_atomic_progress_write_retries_windows_reader_race(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "progress.json"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise PermissionError("simulated Windows sharing violation")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    _atomic_write_json(destination, {"status": "running"})
    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "running"
    }


def test_cp_sat_launcher_discovers_completed_and_interrupted_checkpoints(tmp_path):
    completed = tmp_path / "e7_cp_sat_ablation_001"
    interrupted = tmp_path / "e7_cp_sat_ablation_002"
    completed.mkdir()
    interrupted.mkdir()
    _atomic_write_json(
        completed / "manifest.json",
        {"task_count_pending": 0, "task_count_failed": 0},
    )
    _atomic_write_json(completed / "progress.json", {"status": "finished"})
    _atomic_write_json(
        interrupted / "manifest.json",
        {"task_count_pending": 17, "task_count_failed": 0},
    )
    _atomic_write_json(
        interrupted / "progress.json",
        {"status": "interrupted"},
    )
    assert _is_complete(completed) is True
    assert _is_complete(interrupted) is False
    assert _latest_complete(tmp_path) == completed
    assert _latest_incomplete(tmp_path) == interrupted
