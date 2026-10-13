from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from kraul_dbd_reproduction.merge import (
    METHOD_ORDER,
    _interval_minutes_by_hour,
    _validate_pair,
    merge_reports,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _manifest(contract: str) -> dict[str, object]:
    return {
        "run_contract": contract,
        "audit": {
            "dataset_sha256": "same-data",
            "total_jobs": 24,
            "day_count": 1,
            "sequence": [
                {"relative_path": "day.txt", "instance_sha256": "same-instance", "jobs": 24}
            ],
        },
        "experiment": {
            "contract": "formal_proxy91",
            "expected_days": 1,
            "expected_jobs": 24,
            "first_day_seed": 1,
            "last_day_seed": 1,
            "machines_stage1": 12,
            "machines_stage2": 5,
            "day_length_minutes": 1440,
            "seeds": [101],
            "workers": 8,
            "protocol": {
                "mode": "official_time",
                "name": "paper_time_tau1",
                "tau": 1.0,
                "time_limit_seconds": None,
                "max_evaluations": None,
            },
        },
    }


def _table_row(method: str) -> dict[str, object]:
    return {
        "method": method,
        "information_mode": "DBD",
        "random_seed": "" if method == "FFCBSPT" else 101,
        "day_count": 1,
        "device_count": 24,
        "time_in_system_paper": "1.00 (0.10)",
        "waiting_stage1_paper": "0.20 (0.10)",
        "waiting_stage2_paper": "0.30 (0.10)",
        "waiting_stage1_mean_min": 2.0,
        "waiting_stage2_mean_min": 3.0,
        "stage1_utilization_paper": "40.00%",
        "stage2_utilization_paper": "60.00%",
    }


def _hourly(methods: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hour in range(24):
        row: dict[str, object] = {
            "hour": hour,
            "Arrival_total": 1,
            "Arrival_average_per_day": 1.0,
        }
        for method in methods:
            row[f"{method}_total"] = 1
            row[f"{method}_average_per_day"] = 1.0
        rows.append(row)
    return rows


def _result_settings(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "success",
                "machines_stage1": 12,
                "machines_stage2": 5,
                "capacity_stage1": 15,
                "capacity_stage2": 11000,
                "day_length_minutes": 1440,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _job_timings(methods: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in methods:
        for job_id in range(1, 25):
            release = (job_id - 1) * 60
            start1 = release + 2
            completion1 = start1 + 10
            start2 = completion1 + 3
            rows.append(
                {
                    "method": method,
                    "seed": "" if method == "FFCBSPT" else 101,
                    "case_day_number": 1,
                    "job_id": job_id,
                    "release": release,
                    "start_stage1": start1,
                    "completion_stage1": completion1,
                    "start_stage2": start2,
                }
            )
    return rows


def test_merge_requires_and_combines_identical_experiment_contracts(tmp_path: Path) -> None:
    kraul = tmp_path / "kraul"
    three = tmp_path / "three"
    _json(kraul / "manifest.json", _manifest("kraul-contract"))
    legacy_manifest = _manifest("three-contract")
    legacy_manifest.pop("run_contract")
    legacy_manifest.update(
        {
            "config_sha256": "legacy-config",
            "split_sha256": "legacy-split",
            "task_manifest_sha256": "legacy-tasks",
            "code_sha256": {"runner": "legacy-code"},
        }
    )
    legacy_audit = legacy_manifest["audit"]
    legacy_audit["days"] = legacy_audit.pop("day_count")
    legacy_audit["instance_sequence"] = legacy_audit.pop("sequence")
    _json(three / "manifest.json", legacy_manifest)
    for root in (kraul, three):
        _json(root / "report" / "integrity.json", {"status": "PASS"})
        _result_settings(root / "raw" / "results.jsonl")
    _csv(
        kraul / "tables" / "table_4_9_ffcbspt_kraul_ga.csv",
        [_table_row("FFCBSPT"), _table_row("Kraul-GA")],
    )
    _csv(
        three / "tables" / "table_4_9_dbd_three_algorithms.csv",
        [_table_row("IBS-CABC"), _table_row("ALNS"), _table_row("TS")],
    )
    for stage in (1, 2):
        _csv(kraul / "tables" / f"hourly_stage{stage}_dbd.csv", _hourly(("FFCBSPT", "Kraul-GA")))
        _csv(three / "tables" / f"hourly_stage{stage}_dbd.csv", _hourly(("IBS-CABC", "ALNS", "TS")))
    _csv(
        kraul / "tables" / "job_level_schedules.csv",
        _job_timings(("FFCBSPT", "Kraul-GA")),
    )
    _csv(
        three / "tables" / "job_level_schedules.csv",
        _job_timings(("IBS-CABC", "ALNS", "TS")),
    )
    output = tmp_path / "merged"
    summary = merge_reports(kraul, three, output)
    assert summary.is_file()
    integrity = json.loads((output / "merge_integrity.json").read_text(encoding="utf-8"))
    assert tuple(integrity["methods"]) == METHOD_ORDER
    assert integrity["source_run_contracts"]["ibscabc_alns_ts"]["schema"] == "legacy_manifest"
    assert (output / "figure_4_7_dbd_five_algorithms_stage1.png").is_file()
    assert (output / "figure_4_8_dbd_five_algorithms_stage2.png").is_file()
    assert (output / "figure_dbd_five_algorithms_average_queue_stage1.png").is_file()
    assert (output / "figure_dbd_five_algorithms_average_queue_stage2.png").is_file()
    assert (output / "hourly_average_queue_stage1_dbd_five_algorithms.csv").is_file()
    assert (output / "hourly_average_queue_stage2_dbd_five_algorithms.csv").is_file()
    queue = integrity["average_queue_curves"]
    assert queue["maximum_waiting_time_conservation_error_minutes"] == pytest.approx(0.0)
    assert queue["maximum_table_waiting_sum_error_minutes"] == pytest.approx(0.0)


def test_queue_interval_is_folded_across_midnight_without_losing_area() -> None:
    values = _interval_minutes_by_hour(
        23 * 60 + 50,
        24 * 60 + 10,
        day_length_minutes=24 * 60,
    )
    assert values[23] == pytest.approx(10.0)
    assert values[0] == pytest.approx(10.0)
    assert sum(values) == pytest.approx(20.0)


def test_multiseed_merge_reports_seed_mean_sd_and_queue_ci(tmp_path: Path) -> None:
    kraul = tmp_path / "kraul"
    three = tmp_path / "three"
    seeds = (1, 2)
    manifest = _manifest("multiseed-contract")
    manifest["experiment"]["contract"] = "formal_proxy91_multiseed"
    manifest["experiment"]["seeds"] = list(seeds)
    for root in (kraul, three):
        _json(root / "manifest.json", manifest)
        _json(root / "report" / "integrity.json", {"status": "PASS"})
        _result_settings(root / "raw" / "results.jsonl")

    def seed_row(method: str, seed: int | None) -> dict[str, object]:
        waiting1 = 2.0 if seed is None else 1.0 + seed
        waiting2 = 3.0 if seed is None else 2.0 + seed
        row = _table_row(method)
        row["random_seed"] = "" if seed is None else seed
        row["waiting_stage1_mean_min"] = waiting1
        row["waiting_stage2_mean_min"] = waiting2
        return row

    def summary_row(method: str) -> dict[str, object]:
        row = _table_row(method)
        stochastic = method != "FFCBSPT"
        row.update(
            {
                "random_seed": "",
                "random_seeds": "1;2" if stochastic else "",
                "seed_count": 2 if stochastic else 0,
            }
        )
        return row

    _csv(
        kraul / "tables" / "table_4_9_ffcbspt_kraul_ga.csv",
        [seed_row("FFCBSPT", None), *(seed_row("Kraul-GA", seed) for seed in seeds)],
    )
    _csv(
        three / "tables" / "table_4_9_dbd_three_algorithms.csv",
        [
            seed_row(method, seed)
            for method in ("IBS-CABC", "ALNS", "TS")
            for seed in seeds
        ],
    )
    _csv(
        kraul / "tables" / "table_9_ffcbspt_kraul_ga_seed_summary.csv",
        [summary_row("FFCBSPT"), summary_row("Kraul-GA")],
    )
    _csv(
        three / "tables" / "table_9_dbd_three_algorithms_seed_summary.csv",
        [summary_row("IBS-CABC"), summary_row("ALNS"), summary_row("TS")],
    )
    for stage in (1, 2):
        _csv(kraul / "tables" / f"hourly_stage{stage}_dbd.csv", _hourly(("FFCBSPT", "Kraul-GA")))
        _csv(three / "tables" / f"hourly_stage{stage}_dbd.csv", _hourly(("IBS-CABC", "ALNS", "TS")))

    def timings(methods: tuple[str, ...], *, deterministic: bool) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        method_seeds: tuple[int | None, ...] = (None,) if deterministic else seeds
        for method in methods:
            for seed in method_seeds:
                waiting1 = 2.0 if seed is None else 1.0 + seed
                waiting2 = 3.0 if seed is None else 2.0 + seed
                for job_id in range(1, 25):
                    release = (job_id - 1) * 60
                    start1 = release + waiting1
                    completion1 = start1 + 10
                    rows.append(
                        {
                            "method": method,
                            "seed": "" if seed is None else seed,
                            "case_day_number": 1,
                            "job_id": job_id,
                            "release": release,
                            "start_stage1": start1,
                            "completion_stage1": completion1,
                            "start_stage2": completion1 + waiting2,
                        }
                    )
        return rows

    _csv(
        kraul / "tables" / "job_level_schedules.csv",
        timings(("FFCBSPT",), deterministic=True)
        + timings(("Kraul-GA",), deterministic=False),
    )
    _csv(
        three / "tables" / "job_level_schedules.csv",
        timings(("IBS-CABC", "ALNS", "TS"), deterministic=False),
    )

    output = tmp_path / "merged"
    merge_reports(kraul, three, output)
    with (output / "hourly_average_queue_stage1_dbd_five_algorithms.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        first = next(csv.DictReader(handle))
    mean = float(first["IBS-CABC_average_queue_jobs_mean"])
    low = float(first["IBS-CABC_average_queue_jobs_ci95_low"])
    high = float(first["IBS-CABC_average_queue_jobs_ci95_high"])
    assert mean == pytest.approx(2.5 / 60.0)
    assert low < mean < high
    assert (output / "table_9_dbd_five_algorithms_10seeds.csv").is_file()


def test_merge_rejects_different_protocols(tmp_path: Path) -> None:
    kraul = tmp_path / "kraul"
    three = tmp_path / "three"
    left = _manifest("left")
    right = _manifest("right")
    right["experiment"]["protocol"]["tau"] = 2.0
    _json(kraul / "manifest.json", left)
    _json(three / "manifest.json", right)
    for root in (kraul, three):
        _json(root / "report" / "integrity.json", {"status": "PASS"})
        _result_settings(root / "raw" / "results.jsonl")
    with pytest.raises(ValueError, match="protocol"):
        _validate_pair(kraul, three)
