import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import json
import numpy as np
import pytest

from contracts import IBSCABC_DIAGNOSTIC_FIELDS, METHODS, METRICS, atomic_json, digest, load_checkpoints, load_config, prepare, save_checkpoint
from report import aggregate_cp_sat, aggregate_instances, aggregate_scales, paired_statistics, holm


def fixture_data(n=2, repeats=3, scale=100):
    tasks, rows = [], []
    for i in range(n):
        for algorithm_index, algorithm in enumerate(METHODS):
            for seed in range(1, repeats + 1):
                task = {"task_id": f"J{scale}_{i}_{algorithm}_{seed}", "task_hash": "hash", "jobs": scale, "instance": f"i{i}", "instance_sha256": str(i), "algorithm": algorithm, "seed": seed, "time_limit_seconds": 90.0, "release_sum": 0, "mode": "smoke"}
                row = {**task, "status": "success", "feasible": True, **{m: 1.0 for m in METRICS}}
                row["objective"] = 1000 + 100 * i + 10 * algorithm_index + seed
                row["average_time_in_system"] = row["objective"] / scale
                row.update({"solver_reported_runtime_seconds": 1.0, "process_cpu_seconds": 1.0, "runtime_overrun_seconds": 0.0, "runtime_underrun_seconds": 89.0})
                if algorithm == "ibscabc":
                    row.update({field: (False if field == "assignment_repair_applicable" else 0.0) for field in IBSCABC_DIAGNOSTIC_FIELDS})
                    row["assignment_repair_time_share_percent"] = 0.0
                tasks.append(task)
                rows.append(row)
    return tasks, rows


def test_two_level_sd_and_rpr():
    tasks, records = fixture_data()
    instance, coverage = aggregate_instances(tasks, records)
    assert len(instance) == 8 and all(c["complete"] for c in coverage)
    assert instance[0]["objective_within_sd"] == pytest.approx(1)
    scale, complete = aggregate_scales(instance, {100: {"0", "1"}})
    assert complete == [100]
    alns = next(r for r in scale if r["algorithm"] == "alns")
    assert alns["objective_mean"] == pytest.approx(1052)
    assert alns["objective_between_instance_sd"] == pytest.approx(np.std([1002, 1102], ddof=1))
    assert alns["objective_mean_within_instance_sd"] == pytest.approx(1)
    assert alns["rpr"] == 1
    ga = next(r for r in scale if r["algorithm"] == "kraul_ga")
    assert ga["rpr"] == pytest.approx(1072 / 1052)
    assert ga["rpr"] != pytest.approx(np.mean([1022/1002, 1122/1102]), abs=1e-6)


def test_cp_sat_aggregation_uses_complete_solver_time_denominator():
    _, records = fixture_data(n=2, repeats=3)
    ibs = [row for row in records if row["algorithm"] == "ibscabc"]
    for index, row in enumerate(ibs):
        row["assignment_repair_applicable"] = True
        row["assignment_repair_model_calls"] = 2
        row["assignment_repair_rounds"] = 1
        row["assignment_repair_runtime_seconds"] = 0.25
        row["assignment_repair_max_solver_call_runtime_seconds"] = 0.2
        row["runtime_seconds"] = 2.0 if index == 0 else 1.0
    rows = aggregate_cp_sat(
        records,
        [100],
        {100: {"0", "1"}},
        {100: {1, 2, 3}},
        {"assignment_repair_seconds_per_family": 10.0},
    )
    assert rows[0]["runs"] == 6
    assert rows[0]["mean_model_calls_per_run"] == 2
    assert rows[0]["aggregate_cp_wall_time_share_percent"] == pytest.approx(
        100 * 1.5 / sum(row["runtime_seconds"] for row in ibs)
    )


def test_missing_seed_never_becomes_complete_instance():
    tasks, records = fixture_data()
    instance, coverage = aggregate_instances(tasks, records[:-1])
    assert len(instance) == 7
    assert sum(not c["complete"] for c in coverage) == 1
    scale, complete = aggregate_scales(instance, {100: {"0", "1"}})
    assert scale == [] and complete == []


def test_duplicate_seed_rejected():
    tasks, records = fixture_data()
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_instances(tasks, records + [records[0]])


def test_wilcoxon_uses_twenty_instances_not_six_hundred_runs():
    tasks, records = fixture_data(n=20, repeats=30)
    instance, _ = aggregate_instances(tasks, records)
    stats, pairs = paired_statistics(instance, [100], resamples=100)
    assert len(pairs) == 20
    assert stats[0]["instances"] == 20
    from scipy.stats import wilcoxon
    expected = wilcoxon(np.repeat(-10.0, 20), method="auto").pvalue
    assert stats[0]["wilcoxon_p_two_sided"] == pytest.approx(expected)
    assert stats[0]["holm_p_five_tests"] is None
    assert stats[0]["losses"] == 20


def test_zero_differences_are_ties():
    tasks, records = fixture_data(n=20, repeats=2)
    for row in records:
        row["objective"] = 1000 + row["seed"]
    instance, _ = aggregate_instances(tasks, records)
    stats, _ = paired_statistics(instance, [100], resamples=100)
    assert stats[0]["wilcoxon_p_two_sided"] == 1
    assert stats[0]["ties"] == 20 and stats[0]["rank_biserial"] == 0


def test_holm_monotonic_and_bounded():
    assert holm([0.001, 0.01, 0.03, 0.04, 0.9]) == pytest.approx([0.005, 0.04, 0.09, 0.09, 0.9])


def test_atomic_resume_keeps_only_completed_checkpoints(tmp_path):
    tasks, records = fixture_data()
    save_checkpoint(tmp_path, records[0], tasks[0])
    orphan = tmp_path / "checkpoints" / "unfinished.json.tmp"
    orphan.write_text('{"torn":', encoding="utf-8")
    loaded = load_checkpoints(tmp_path, tasks)
    assert list(loaded) == [tasks[0]["task_id"]]
    with pytest.raises(ValueError, match="overwrite"):
        save_checkpoint(tmp_path, records[0], tasks[0])
    changed = copy.deepcopy(tasks)
    changed[0]["task_hash"] = "changed"
    with pytest.raises(ValueError, match="disagrees"):
        load_checkpoints(tmp_path, changed)


def test_corrupt_checkpoint_rejected(tmp_path):
    tasks, records = fixture_data()
    save_checkpoint(tmp_path, records[0], tasks[0])
    path = next((tmp_path / "checkpoints").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["result"]["objective"] += 1
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoints(tmp_path, tasks)


def test_full_protocol_task_counts_and_budgets():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/fixed_30_20.yaml")
    manifest, tasks = prepare(config)
    assert len(tasks) == 8000
    assert len({t["task_id"] for t in tasks}) == 8000
    assert sum(t["time_limit_seconds"] for t in tasks) == 7263000
    assert [sum(t["jobs"] == j for t in tasks) for j in (100, 250, 500, 1000)] == [2400, 2400, 1600, 1600]
    for scale in (100, 250, 500, 1000):
        selected = [task for task in tasks if task["jobs"] == scale]
        expected = set(range(1, 31 if scale in (100, 250) else 21))
        by_algorithm = {
            algorithm: {
                (task["instance_sha256"], task["seed"])
                for task in selected
                if task["algorithm"] == algorithm
            }
            for algorithm in METHODS
        }
        assert all({seed for _, seed in pairs} == expected for pairs in by_algorithm.values())
        assert len({frozenset(pairs) for pairs in by_algorithm.values()}) == 1


def test_server32_protocol_keeps_same_8000_tasks_with_72_workers():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/fixed_30_20_server32.yaml"
    if not (root / "data/formal/instances").is_dir():
        pytest.skip("server bundle data are created by build_server_bundle.py")
    config = load_config(config_path)
    manifest, tasks = prepare(config)
    assert len(tasks) == 8000
    assert manifest["core"]["config"]["workers"] == 72
    assert manifest["core"]["config"]["worker_recycle_tasks"] == 8
    assert manifest["observed_environment"]["logical_cpus"] >= 1


def test_formal_protocol_disallows_incomplete_or_repeated_seed_plan():
    root = Path(__file__).resolve().parents[1]
    import yaml
    config = load_config(root / "configs/fixed_30_20.yaml")
    config["scales"][100]["seeds"] = [1, 1]
    # Use a temporary file in a test-owned directory, not the formal config.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "bad.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="unique"):
            load_config(path)


def test_all_four_scales_have_n20_and_overall_n80():
    instance_rows = []
    for scale in (100, 250, 500, 1000):
        tasks, records = fixture_data(n=20, repeats=30 if scale < 500 else 20, scale=scale)
        rows, _ = aggregate_instances(tasks, records)
        instance_rows.extend(rows)
    statistics, paired = paired_statistics(instance_rows, [100, 250, 500, 1000], resamples=100)
    assert [r["instances"] for r in statistics] == [20, 20, 20, 20, 80]
    assert len(paired) == 80
    assert all(r["holm_p_five_tests"] is not None for r in statistics)
    assert all(0 <= r["holm_p_five_tests"] <= 1 for r in statistics)


def test_infeasible_and_nan_checkpoints_rejected(tmp_path):
    tasks, records = fixture_data()
    bad = copy.deepcopy(records[0])
    bad["feasible"] = False
    with pytest.raises(ValueError, match="feasible"):
        save_checkpoint(tmp_path, bad, tasks[0])
    bad = copy.deepcopy(records[0])
    bad["objective"] = float("nan")
    with pytest.raises(ValueError, match="invalid numeric"):
        save_checkpoint(tmp_path, bad, tasks[0])


def test_output_directory_lock_prevents_concurrent_writer(tmp_path):
    from kraul_dbd_reproduction.runner import _run_lock
    with _run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another process"):
            with _run_lock(tmp_path):
                pass


def test_selected_algorithms_share_study_contract():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/fixed_30_20.yaml")
    left, _ = prepare(config, [100], ["alns"])
    right, _ = prepare(config, [250], ["ts"])
    assert left["study_hash"] == right["study_hash"]
    assert left["run_contract"] != right["run_contract"]


def test_resume_changed_manifest_is_rejected(tmp_path):
    from run import validate_resume
    tasks, _ = fixture_data()
    atomic_json(tmp_path / "manifest.json", {"run_contract": "original"})
    with pytest.raises(ValueError, match="Cannot resume"):
        validate_resume(tmp_path, {"run_contract": "changed"}, tasks)


def test_digest_is_stable_across_integer_key_json_round_trip():
    original = {"scales": {100: {"seeds": [1, 2]}, 250: {}, 500: {}, 1000: {}}}
    assert digest(original) == digest(json.loads(json.dumps(original)))


def test_manifest_checksums_survive_disk_round_trip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/smoke.yaml")
    manifest, tasks = prepare(config)
    atomic_json(tmp_path / "manifest.json", {**manifest, "created_utc": "test"})
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert digest(saved["core"]) == saved["study_hash"]
    assert digest({k: v for k, v in saved.items() if k not in {"created_utc", "run_contract", "observed_environment"}}) == saved["run_contract"]
    assert digest(json.loads(json.dumps(tasks))) == saved["task_manifest_hash"]


def test_checkpoint_loading_does_not_retain_all_histories(tmp_path):
    tasks, records = fixture_data()
    records[0]["evidence"] = {"history": [1, 2, 3]}
    save_checkpoint(tmp_path, records[0], tasks[0])
    assert "evidence" not in load_checkpoints(tmp_path, tasks)[tasks[0]["task_id"]]
    assert load_checkpoints(tmp_path, tasks, include_evidence=True)[tasks[0]["task_id"]]["evidence"] == records[0]["evidence"]
