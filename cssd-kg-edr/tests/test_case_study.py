import json
from pathlib import Path

from cssd_ibscabc.experiments import ExperimentRunner, load_experiment_config
from cssd_ibscabc.experiments.case_study import combine_daily_instances
from cssd_ibscabc.reporting import build_report


def _paths(data_root: Path, count: int = 2) -> list[Path]:
    return sorted((data_root / "Instances 25").glob("*.txt"))[:count]


def test_aio_combination_uses_one_calendar_and_contiguous_ids(data_root):
    first, second = _paths(data_root)
    combined = combine_daily_instances([first, second], day_length_minutes=1440)
    assert combined.job_count == 50
    assert [job.job_id for job in combined.jobs] == list(range(1, 51))
    assert [job.release for job in combined.jobs[:25]] == [
        job.release for job in combine_daily_instances([first]).jobs
    ]
    second_original = combine_daily_instances([second])
    assert [job.release for job in combined.jobs[25:]] == [
        job.release + 1440 for job in second_original.jobs
    ]


def test_paper_case_configs_contain_only_expected_own_algorithm_tasks(data_root):
    project = Path(__file__).resolve().parents[1]
    base = ExperimentRunner(
        load_experiment_config(project / "configs" / "case_aio_dbd.yaml"),
        data_root,
        project / "outputs",
    ).tasks()
    assert len(base) == 92
    assert sum(task["planning_mode"] == "DBD" for task in base) == 91
    assert sum(task["planning_mode"] == "AIO" for task in base) == 1
    assert {task["solver"] for task in base} == {"ibscabc"}

    sensitivity = ExperimentRunner(
        load_experiment_config(
            project / "configs" / "case_machine_sensitivity.yaml"
        ),
        data_root,
        project / "outputs",
    ).tasks()
    assert len(sensitivity) == 9
    assert {task["planning_mode"] for task in sensitivity} == {"AIO"}
    assert {task["solver"] for task in sensitivity} == {"ibscabc"}


def test_small_aio_dbd_pipeline_and_case_report(data_root, tmp_path):
    split = tmp_path / "two_days.txt"
    split.write_text(
        "\n".join(
            str(path.relative_to(data_root)).replace("\\", "/")
            for path in _paths(data_root)
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "experiment": {
            "name": "unit_aio_dbd",
            "selection": {"split": str(split)},
            "case_study": {"days": 2},
            "day_length_minutes": 1440,
            "planning_modes": ["DBD", "AIO"],
            "scenarios": [{"name": "base"}],
            "protocols": [
                {"name": "E8", "mode": "evaluations", "max_evaluations": 8}
            ],
            "seeds": [7],
            "algorithms": [
                {
                    "name": "ibscabc",
                    "parameters": {
                        "colony_size": 4,
                        "archive_size": 4,
                        "assignment_repair_enabled": False,
                    },
                }
            ],
        }
    }
    output = ExperimentRunner(config, data_root, tmp_path / "outputs").run()
    records = [
        json.loads(line)
        for line in (output / "raw" / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 3
    assert all(record["feasible"] for record in records)
    aio = next(record for record in records if record["planning_mode"] == "AIO")
    second_day = next(
        record
        for record in records
        if record["planning_mode"] == "DBD" and record["case_day_index"] == 1
    )
    assert aio["jobs"] == 50
    assert sum(aio["arrivals_hourly"]) == 50
    assert aio["metric_moments"]["time_in_system"]["count"] == 50
    assert second_day["calendar_objective"] == (
        second_day["objective"] + 1440 * second_day["jobs"]
    )

    report = build_report(output)
    assert report.name == "case_study_summary_zh.md"
    summary = json.loads(
        (output / "report" / "performance_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "PASS"
    assert {
        row["planning_mode"] for row in summary["rows"] if row["scenario"] == "base"
    } == {"AIO", "DBD"}
    assert (output / "tables" / "paper_table8_ibscabc_aio_dbd.csv").is_file()
