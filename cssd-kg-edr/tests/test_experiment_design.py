import json
from pathlib import Path

import yaml

from cssd_ibscabc.experiments import (
    ExperimentRunner,
    apply_selected_parameters,
    load_experiment_config,
)
from cssd_ibscabc.reporting import build_report


def _split(data_root: Path, destination: Path, count: int = 1) -> Path:
    instances = sorted((data_root / "Instances 25").glob("*.txt"))[:count]
    destination.write_text(
        "\n".join(
            str(path.relative_to(data_root)).replace("\\", "/")
            for path in instances
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def test_e1_to_e7_configs_have_exact_task_matrix(data_root, tmp_path):
    project = Path(__file__).resolve().parents[1]
    expected = {
        "e1_parameter_calibration.yaml": (540, 1, 9),
        "e2_small25.yaml": (20, 1, 1),
        "e3_real365.yaml": (365, 1, 1),
        "e4_fixed_scale.yaml": (80, 1, 1),
        "e5_machine_grid_j250.yaml": (1820, 91, 1),
        "e6_case_aio_dbd.yaml": (92, 1, 1),
        "e6_case_machine_sensitivity.yaml": (9, 9, 1),
        "e7_cp_sat_ablation.yaml": (120, 1, 2),
    }
    for filename, (task_count, scenarios, labels) in expected.items():
        config = load_experiment_config(project / "configs" / filename)
        tasks = ExperimentRunner(
            config,
            data_root,
            tmp_path / "outputs",
        ).tasks()
        assert len(tasks) == task_count
        assert len({task["scenario"] for task in tasks}) == scenarios
        assert len({task["algorithm"] for task in tasks}) == labels
        assert {task["solver"] for task in tasks} == {"ibscabc"}
        assert {
            task["parameters"]["assignment_repair_workers"] for task in tasks
        } == {1}


def test_cp_sat_ablation_tasks_are_strict_pairs(data_root, tmp_path):
    project = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project / "configs" / "e7_cp_sat_ablation.yaml")
    tasks = ExperimentRunner(config, data_root, tmp_path / "outputs").tasks()
    for index in range(0, len(tasks), 2):
        pair = tasks[index : index + 2]
        assert len(pair) == 2
        assert {task["algorithm"] for task in pair} == {
            "IBS-CABC",
            "IBS-CABC-w/o-CP-SAT",
        }
        assert len({task["instance_path"] for task in pair}) == 1
        assert len({task["seed"] for task in pair}) == 1
        full = next(task for task in pair if task["algorithm"] == "IBS-CABC")
        no_cp = next(
            task for task in pair if task["algorithm"] == "IBS-CABC-w/o-CP-SAT"
        )
        full_parameters = dict(full["parameters"])
        no_cp_parameters = dict(no_cp["parameters"])
        assert full_parameters.pop("assignment_repair_enabled") is True
        assert no_cp_parameters.pop("assignment_repair_enabled") is False
        assert full_parameters == no_cp_parameters


def test_local_precheck_is_fixed_small_and_exploratory(data_root, tmp_path):
    project = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project / "configs" / "local_precheck.yaml")
    tasks = ExperimentRunner(config, data_root, tmp_path / "outputs").tasks()
    assert len(tasks) == 8
    assert {int(task["seed"]) for task in tasks} == {101}
    assert {float(dict(task["protocol"])["tau"]) for task in tasks} == {0.05}
    assert {str(dict(task["protocol"])["mode"]) for task in tasks} == {
        "official_time"
    }
    assert {str(task["solver"]) for task in tasks} == {"ibscabc"}


def test_selected_parameter_file_overlays_later_experiment(tmp_path):
    project = Path(__file__).resolve().parents[1]
    selected = tmp_path / "selected_parameters.yaml"
    selected.write_text(
        yaml.safe_dump(
            {
                "algorithm": "ibscabc",
                "parameters": {
                    "colony_size": 12,
                    "archive_size": 12,
                    "abandonment_limit": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    config = apply_selected_parameters(
        load_experiment_config(project / "configs" / "e2_small25.yaml"),
        selected,
    )
    parameters = config["experiment"]["algorithms"][0]["parameters"]
    assert parameters["colony_size"] == 12
    assert parameters["archive_size"] == 12
    assert parameters["abandonment_limit"] == 6
    assert parameters["assignment_repair_workers"] == 1
    assert config["selected_parameter_source"]["path"] == str(selected.resolve())


def test_e1_report_selects_configuration_and_writes_parameter_file(
    data_root,
    tmp_path,
):
    split = _split(data_root, tmp_path / "two.txt", count=2)
    config = {
        "experiment": {
            "name": "unit_e1",
            "report_template": "e1",
            "selection": {"split": str(split)},
            "protocols": [
                {"name": "E4", "mode": "evaluations", "max_evaluations": 4}
            ],
            "seeds": [1, 2],
            "calibration_parameters": ["abandonment_limit"],
            "evidence_contract": {
                "instances": 2,
                "scenarios": 1,
                "algorithm_labels": 2,
            },
            "algorithms": [
                {
                    "name": "ibscabc",
                    "label": "P1",
                    "parameters": {
                        "colony_size": 4,
                        "archive_size": 4,
                        "abandonment_limit": 4,
                        "assignment_repair_enabled": False,
                    },
                },
                {
                    "name": "ibscabc",
                    "label": "P2",
                    "parameters": {
                        "colony_size": 4,
                        "archive_size": 4,
                        "abandonment_limit": 8,
                        "assignment_repair_enabled": False,
                    },
                },
            ],
        }
    }
    output = ExperimentRunner(config, data_root, tmp_path / "outputs").run()
    report = build_report(output)
    assert report.name == "e1_summary_zh.md"
    selected = yaml.safe_load(
        (output / "tables" / "selected_parameters.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert selected["algorithm"] == "ibscabc"
    assert selected["parameters"]["abandonment_limit"] in {4, 8}
    summary = json.loads(
        (output / "report" / "performance_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "PASS"


def test_e5_grid_report_writes_long_matrix_and_heatmap(data_root, tmp_path):
    split = _split(data_root, tmp_path / "one.txt")
    config = {
        "experiment": {
            "name": "unit_e5",
            "report_template": "e5",
            "selection": {"split": str(split)},
            "scenario_grid": {
                "machines_stage1": [2, 3],
                "machines_stage2": [1, 2],
            },
            "protocols": [
                {"name": "E4", "mode": "evaluations", "max_evaluations": 4}
            ],
            "seeds": [1],
            "evidence_contract": {
                "instances": 1,
                "scenarios": 4,
                "algorithm_labels": 1,
            },
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
    build_report(output)
    assert (output / "tables" / "table6_machine_grid_long.csv").is_file()
    assert (output / "tables" / "table6_gap_lb_matrix.csv").is_file()
    assert (output / "tables" / "table7_parameter_by_machine_status.csv").is_file()
    assert (output / "figures" / "table6_gap_lb_heatmap.png").is_file()


def test_precheck_report_writes_module_diagnostics(data_root, tmp_path):
    split = _split(data_root, tmp_path / "one.txt")
    config = {
        "experiment": {
            "name": "unit_precheck",
            "report_template": "precheck",
            "selection": {"split": str(split)},
            "protocols": [
                {"name": "E20", "mode": "evaluations", "max_evaluations": 20}
            ],
            "seeds": [1],
            "evidence_contract": {
                "instances": 1,
                "scenarios": 1,
                "algorithm_labels": 1,
            },
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
    build_report(output)
    assert (output / "tables" / "local_precheck_summary.csv").is_file()
    assert (
        output / "tables" / "local_precheck_module_diagnostics.csv"
    ).is_file()
    summary = json.loads(
        (output / "report" / "performance_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["evidence_status"] == "exploratory_only_not_for_paper_comparison"


def test_development_protocol_is_not_labeled_formal(data_root, tmp_path):
    split = _split(data_root, tmp_path / "one_e4.txt")
    config = {
        "experiment": {
            "name": "unit_development_e4",
            "report_template": "e4",
            "selection": {"split": str(split)},
            "protocols": [
                {"name": "development_E4", "mode": "evaluations", "max_evaluations": 4}
            ],
            "seeds": [1],
            "evidence_contract": {
                "instances": 1,
                "scenarios": 1,
                "algorithm_labels": 1,
            },
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
    build_report(output)
    summary = json.loads(
        (output / "report" / "performance_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["evidence_status"] == "development_screening_only"


def test_cp_sat_ablation_report_writes_paired_table_and_diagnostics(
    data_root,
    tmp_path,
):
    split = _split(data_root, tmp_path / "one_cp.txt")
    common = {
        "colony_size": 4,
        "archive_size": 4,
        "assignment_repair_seconds_per_family": 0.01,
        "assignment_repair_workers": 1,
        "assignment_repair_iterations": 1,
        "assignment_repair_joint_max_jobs": 600,
        "assignment_repair_max_jobs": 600,
        "assignment_repair_time_fraction": 0.30,
        "mathematical_repair_seconds": 0.02,
    }
    config = {
        "experiment": {
            "name": "unit_cp_sat_ablation",
            "report_template": "cp_sat_ablation",
            "paired_algorithm_order": True,
            "selection": {"split": str(split)},
            "protocols": [
                {"name": "E4", "mode": "evaluations", "max_evaluations": 4}
            ],
            "seeds": [1],
            "evidence_contract": {
                "instances": 1,
                "scenarios": 1,
                "algorithm_labels": 2,
            },
            "ablation": {
                "full_label": "IBS-CABC",
                "no_cp_label": "IBS-CABC-w/o-CP-SAT",
                "bootstrap_resamples": 100,
                "bootstrap_seed": 7,
            },
            "algorithms": [
                {
                    "name": "ibscabc",
                    "label": "IBS-CABC",
                    "parameters": common
                    | {"assignment_repair_enabled": True},
                },
                {
                    "name": "ibscabc",
                    "label": "IBS-CABC-w/o-CP-SAT",
                    "parameters": common
                    | {"assignment_repair_enabled": False},
                },
            ],
        }
    }
    output = ExperimentRunner(config, data_root, tmp_path / "outputs").run()
    report = build_report(output)
    assert report.name == "cp_sat_ablation_summary_zh.md"
    assert (output / "tables" / "table8_1_cp_sat_ablation.csv").is_file()
    assert (output / "tables" / "table8_1_cp_sat_ablation_zh.csv").is_file()
    assert (
        output / "aggregate" / "cp_sat_ablation_paired_instances.csv"
    ).is_file()
    assert (output / "figures" / "cp_sat_ablation_paired_gain.png").is_file()
    summary = json.loads(
        (output / "report" / "performance_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "PASS"
    assert summary["cp_sat_ablation"]["complete_instance_pairs"] == 1
