from pathlib import Path

import pytest

from cssd_ibscabc.experiments import ExperimentRunner, load_experiment_config
from cssd_ibscabc.experiments.protocols import published_time_limit
from cssd_ibscabc.experiments.splits import build_published_splits
from cssd_ibscabc.bounds import published_jobwise_lower_bound
from cssd_ibscabc.io import DataValidationError, audit_dataset, load_instance


def test_full_dataset_audit(data_root):
    audit = audit_dataset(data_root)
    assert audit.valid
    assert (audit.instance_count, audit.job_count) == (465, 126_931)
    assert (audit.job_min, audit.job_max) == (25, 1000)


def test_real365_size_range_matches_the_files(data_root):
    instances = [
        load_instance(path) for path in sorted((data_root / "RealWorldInstances365").glob("*.txt"))
    ]
    sizes = [instance.job_count for instance in instances]
    assert len(instances) == 365
    assert (min(sizes), max(sizes)) == (70, 431)


def test_published_splits_have_exact_table_groups(data_root, tmp_path):
    splits = build_published_splits(data_root, tmp_path)
    assert len(splits["paper_small25"]) == 20
    assert len(splits["paper_fixed"]) == 80
    assert len(splits["paper_fixed_j100"]) == 20
    assert len(splits["paper_fixed_j250"]) == 20
    assert len(splits["paper_fixed_j500"]) == 20
    assert len(splits["paper_fixed_j1000"]) == 20
    assert len(splits["paper_real365"]) == 365
    assert len(splits["paper_all465"]) == 465
    assert len(splits["smoke"]) == 1
    assert len(set(splits["paper_all465"])) == 465
    assert splits == build_published_splits(data_root, tmp_path)


def test_paper_main_config_creates_exactly_465_tasks(data_root, tmp_path):
    project = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project / "configs" / "paper_main.yaml")
    runner = ExperimentRunner(config, data_root, tmp_path / "outputs")
    tasks = runner.tasks()
    assert len(tasks) == 465
    assert {str(task["solver"]) for task in tasks} == {"ibscabc"}
    assert {int(task["seed"]) for task in tasks} == {101}
    assert {str(dict(task["protocol"])["mode"]) for task in tasks} == {"official_time"}


def test_published_time_formula_uses_tau_one_and_1800_second_cap(data_root):
    small = load_instance(sorted((data_root / "Instances 25").glob("*.txt"))[0])
    j100 = load_instance(
        sorted((data_root / "Instances100-1000" / "J_100").glob("*.txt"))[0]
    )
    j250 = load_instance(
        sorted((data_root / "Instances100-1000" / "J_250").glob("*.txt"))[0]
    )
    j500 = load_instance(
        sorted((data_root / "Instances100-1000" / "J_500").glob("*.txt"))[0]
    )
    assert published_time_limit(small) == pytest.approx(6.5625)
    assert published_time_limit(j100) == pytest.approx(90.0)
    assert published_time_limit(j250) == pytest.approx(536.25)
    assert published_time_limit(j500) == pytest.approx(1800.0)


def test_published_lower_bound_folds_setup_into_job_processing(small_instance):
    expected = sum(
        job.release + job.duration_stage1 + job.duration_stage2
        for job in small_instance.jobs
    )
    without_setup = sum(
        job.release + job.processing_stage1 + job.processing_stage2
        for job in small_instance.jobs
    )
    assert published_jobwise_lower_bound(small_instance) == expected
    assert without_setup < expected


def test_loader_rejects_nonfinite_values_and_empty_dataset(tmp_path):
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("1 1 1 1 15 11000\n1 50 nan 0 1 0 0 1 10\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="finite"):
        load_instance(invalid)
    empty = tmp_path / "empty"
    empty.mkdir()
    audit = audit_dataset(empty)
    assert not audit.valid
    assert audit.invalid_files
