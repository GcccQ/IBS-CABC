from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_DEVICE_METRICS = ("time_in_system", "waiting_stage1", "waiting_stage2")
_PAIR_FIELDS = (
    "instance",
    "instance_path",
    "instance_sha256",
    "jobs",
    "families",
    "machines_stage1",
    "machines_stage2",
    "capacity_stage1",
    "capacity_stage2",
    "protocol",
    "protocol_mode",
    "max_evaluations",
    "time_limit_seconds",
    "day_length_minutes",
    "release_sum",
    "lower_bound",
)


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_task_contract(
    root: Path,
    manifest: Mapping[str, object],
) -> tuple[dict[str, Mapping[str, object]], list[str]]:
    """Load and authenticate the exact task list persisted before computation."""

    errors: list[str] = []
    path = root / "task_manifest.json"
    if not path.is_file():
        return {}, ["task_manifest.json is missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"task_manifest.json cannot be read: {error}"]
    if not isinstance(payload, Mapping):
        return {}, ["task_manifest.json must contain an object"]
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        return {}, ["task_manifest.json has no task list"]

    calculated_hash = _json_hash(raw_tasks)
    if payload.get("sha256") != calculated_hash:
        errors.append("task_manifest.json self-hash does not match its task list")
    if manifest.get("task_manifest_sha256") != calculated_hash:
        errors.append("task_manifest.json does not match the run manifest")

    tasks: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(raw_tasks):
        if not isinstance(item, Mapping):
            errors.append(f"task manifest entry {index + 1} is not an object")
            continue
        task_id = str(item.get("task_id", ""))
        if not task_id:
            errors.append(f"task manifest entry {index + 1} has no task_id")
            continue
        stable = {key: value for key, value in item.items() if key != "task_id"}
        if task_id != _json_hash(stable)[:24]:
            errors.append(f"task manifest entry {index + 1} has an invalid task_id")
        if task_id in tasks:
            errors.append(f"task manifest contains duplicate task_id {task_id}")
            continue
        tasks[task_id] = item
    return tasks, errors


def _load_records(path: Path) -> tuple[list[dict[str, object]], int]:
    latest: dict[str, dict[str, object]] = {}
    line_count = 0
    if not path.is_file():
        return [], 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            line_count += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("result record must be an object")
                latest[str(record["task_id"])] = record
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid results JSONL line {line_number}") from error
    return list(latest.values()), line_count


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        if not fields:
            handle.write("status\nno_records\n")
        else:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _pooled_moment(
    records: Sequence[Mapping[str, object]],
    metric: str,
) -> tuple[float, float, int, float, float]:
    count = 0
    total = 0.0
    total_squares = 0.0
    for record in records:
        moments = record.get("metric_moments")
        if not isinstance(moments, Mapping):
            continue
        values = moments.get(metric)
        if not isinstance(values, Mapping):
            continue
        count += int(values.get("count", 0))
        total += float(values.get("sum", 0.0))
        total_squares += float(values.get("sum_squares", 0.0))
    mean = total / count if count else 0.0
    corrected = total_squares - total * total / count if count else 0.0
    sample_std = math.sqrt(max(0.0, corrected / (count - 1))) if count > 1 else 0.0
    return mean, sample_std, count, total, total_squares


def _sum_hourly(
    records: Sequence[Mapping[str, object]],
    field: str,
    bins: int,
) -> list[int]:
    totals = [0] * bins
    for record in records:
        values = record.get(field)
        if not isinstance(values, list) or len(values) != bins:
            raise ValueError(f"invalid {field} array in task {record.get('task_id')}")
        for index, value in enumerate(values):
            totals[index] += int(value)
    return totals


def _algorithm_specs(manifest: Mapping[str, object]) -> list[dict[str, str]]:
    experiment = manifest.get("experiment")
    if not isinstance(experiment, Mapping):
        return []
    raw = experiment.get("algorithms")
    if not isinstance(raw, list):
        return []
    specs: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, Mapping):
            specs.append(
                {
                    "name": str(item.get("name", "")),
                    "label": str(item.get("label", item.get("name", ""))),
                }
            )
    return specs


def _algorithm_parameters(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    experiment = manifest.get("experiment")
    raw = experiment.get("algorithms") if isinstance(experiment, Mapping) else None
    parameters: dict[str, Mapping[str, object]] = {}
    if not isinstance(raw, list):
        return parameters
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        values = item.get("parameters")
        if isinstance(values, Mapping):
            parameters[str(item.get("name", ""))] = values
    return parameters


def _expected_values(manifest: Mapping[str, object]) -> tuple[int, int, list[int], int]:
    experiment = manifest.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("manifest has no experiment contract")
    expected_days = int(experiment.get("expected_days", 0))
    expected_jobs = int(experiment.get("expected_jobs", 0))
    raw_seeds = experiment.get("seeds")
    if not isinstance(raw_seeds, list):
        raise ValueError("manifest experiment seeds must be a list")
    seeds = [int(seed) for seed in raw_seeds]
    bins = int(experiment.get("day_length_minutes", 1440)) // 60
    return expected_days, expected_jobs, seeds, bins


def _integrity_checks(
    manifest: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    line_count: int,
    task_contract: Mapping[str, Mapping[str, object]],
    task_contract_errors: Sequence[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = list(task_contract_errors)
    warnings: list[str] = []
    specs = _algorithm_specs(manifest)
    expected_names = [item["name"] for item in specs]
    expected_labels = {item["name"]: item["label"] for item in specs}
    expected_parameters = _algorithm_parameters(manifest)
    expected_days, expected_jobs, seeds, bins = _expected_values(manifest)
    expected_tasks = int(manifest.get("task_count_total", 0))
    contract_tasks = expected_days * len(expected_names) * len(seeds)

    if expected_tasks != contract_tasks:
        errors.append(
            f"manifest task count is {expected_tasks}, but the DBD contract implies {contract_tasks}"
        )
    if line_count != len(records):
        errors.append("results.jsonl contains duplicate task identifiers")
    if len(task_contract) != expected_tasks:
        errors.append(
            f"task manifest contains {len(task_contract)} unique tasks, expected {expected_tasks}"
        )
    record_ids = {str(record.get("task_id", "")) for record in records}
    if record_ids != set(task_contract):
        errors.append("result task identifiers do not exactly match the task manifest")
    if len(records) != expected_tasks:
        errors.append(f"unique result count is {len(records)}, expected {expected_tasks}")
    if int(manifest.get("task_count_pending", expected_tasks)) != 0:
        errors.append("manifest still contains pending tasks")
    if int(manifest.get("task_count_successful", 0)) != expected_tasks:
        errors.append("manifest successful-task count does not match the task contract")
    if int(manifest.get("task_count_failed_attempts", 0)):
        warnings.append("one or more failed attempts were retried; inspect raw/failures.jsonl")

    observed_names = {str(record.get("algorithm", "")) for record in records}
    if observed_names != set(expected_names):
        errors.append("result algorithms do not match the configured three-algorithm scope")
    if any(str(record.get("planning_mode", "")).upper() != "DBD" for record in records):
        errors.append("non-DBD result detected")
    if any(not bool(record.get("feasible", False)) for record in records):
        errors.append("one or more schedules are marked infeasible")

    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    paired: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
    experiment = manifest.get("experiment")
    expected_protocol = experiment.get("protocol") if isinstance(experiment, Mapping) else None
    audit = manifest.get("audit")
    instance_sequence = audit.get("instance_sequence") if isinstance(audit, Mapping) else None
    if not isinstance(instance_sequence, list) or len(instance_sequence) != expected_days:
        errors.append("manifest does not contain the complete ordered instance audit")
        instance_sequence = []
    for record in records:
        task_id = str(record.get("task_id", ""))
        expected_task = task_contract.get(task_id)
        if expected_task is not None:
            mismatched = [
                field
                for field in (
                    "experiment",
                    "contract",
                    "planning_mode",
                    "case_day_index",
                    "day_length_minutes",
                    "algorithm",
                    "algorithm_label",
                    "parameters",
                    "seed",
                )
                if record.get(field) != expected_task.get(field)
            ]
            if record.get("protocol_spec") != expected_task.get("protocol"):
                mismatched.append("protocol_spec")
            if (
                Path(str(record.get("instance_path", ""))).resolve()
                != Path(str(expected_task.get("instance_path", ""))).resolve()
            ):
                mismatched.append("instance_path")
            if mismatched:
                errors.append(
                    f"task {task_id} differs from its persisted task contract: "
                    + ", ".join(sorted(set(mismatched)))
                )
        algorithm = str(record.get("algorithm", ""))
        seed = int(record.get("seed", -1))
        day = int(record.get("case_day_index", -1))
        grouped[(algorithm, seed)].append(record)
        paired[(day, seed)].append(record)

        jobs = int(record.get("jobs", 0))
        if record.get("algorithm_label") != expected_labels.get(algorithm):
            errors.append(f"task {record.get('task_id')} has an incorrect algorithm label")
        if 0 <= day < len(instance_sequence) and isinstance(instance_sequence[day], Mapping):
            expected_instance = instance_sequence[day]
            if record.get("instance_sha256") != expected_instance.get("instance_sha256"):
                errors.append(
                    f"task {record.get('task_id')} does not match the locked instance hash"
                )
            if jobs != int(expected_instance.get("jobs", -1)):
                errors.append(f"task {record.get('task_id')} does not match the locked job count")
            if record.get("instance") != expected_instance.get("instance"):
                errors.append(
                    f"task {record.get('task_id')} does not match the locked instance name"
                )
        if record.get("parameters") != expected_parameters.get(algorithm):
            errors.append(f"task {record.get('task_id')} does not preserve solver parameters")
        if expected_protocol is not None and record.get("protocol_spec") != expected_protocol:
            errors.append(f"task {record.get('task_id')} does not preserve the budget protocol")
        protocol_mode = str(record.get("protocol_mode", ""))
        if protocol_mode == "official_time":
            machines = int(record.get("machines_stage1", 0)) + int(record.get("machines_stage2", 0))
            tau = (
                float(expected_protocol.get("tau", 1.0))
                if isinstance(expected_protocol, Mapping)
                else 1.0
            )
            expected_seconds = min((5000.0 + tau * jobs**2 * machines / 2.0) / 1000.0, 1800.0)
            actual_seconds = float(record.get("time_limit_seconds", math.nan))
            if not math.isclose(actual_seconds, expected_seconds, rel_tol=0.0, abs_tol=1e-12):
                errors.append(f"task {record.get('task_id')} has an incorrect paper time budget")
            if record.get("max_evaluations") is not None:
                errors.append(f"task {record.get('task_id')} mixes time and evaluation budgets")
        elif protocol_mode == "evaluations":
            maximum = int(record.get("max_evaluations", 0))
            configured_maximum = (
                int(expected_protocol.get("max_evaluations", 0))
                if isinstance(expected_protocol, Mapping)
                else 0
            )
            if (
                maximum <= 0
                or maximum != configured_maximum
                or int(record.get("evaluations", 0)) > maximum
                or record.get("time_limit_seconds") is not None
            ):
                errors.append(f"task {record.get('task_id')} violates its evaluation budget")
        elif protocol_mode == "wall_clock":
            configured_seconds = (
                float(expected_protocol.get("time_limit_seconds", 0.0))
                if isinstance(expected_protocol, Mapping)
                else 0.0
            )
            actual_seconds = float(record.get("time_limit_seconds", 0.0))
            if (
                actual_seconds <= 0
                or not math.isclose(actual_seconds, configured_seconds, abs_tol=1e-12)
                or record.get("max_evaluations") is not None
            ):
                errors.append(f"task {record.get('task_id')} has an invalid wall-clock budget")
        else:
            errors.append(f"task {record.get('task_id')} has an unknown budget mode")

        moments = record.get("metric_moments")
        for metric in _DEVICE_METRICS:
            values = moments.get(metric) if isinstance(moments, Mapping) else None
            if not isinstance(values, Mapping) or int(values.get("count", -1)) != jobs:
                errors.append(f"task {record.get('task_id')} has invalid {metric} moments")
        time_values = moments.get("time_in_system") if isinstance(moments, Mapping) else None
        flow_sum = (
            float(time_values.get("sum", math.nan))
            if isinstance(time_values, Mapping)
            else math.nan
        )
        identity = float(record.get("objective", 0)) - float(record.get("release_sum", 0))
        if not math.isclose(flow_sum, identity, rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"task {record.get('task_id')} violates the TIS objective identity")
        if int(record.get("objective", -1)) != int(record.get("total_completion_time", -2)):
            errors.append(f"task {record.get('task_id')} has inconsistent objective fields")
        if not math.isclose(
            flow_sum,
            float(record.get("total_flow_time", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append(f"task {record.get('task_id')} has inconsistent total flow time")
        if jobs > 0 and not math.isclose(
            flow_sum / jobs,
            float(record.get("average_time_in_system", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"task {record.get('task_id')} has an inconsistent mean TIS")
        expected_calendar_objective = int(record.get("objective", 0)) + (
            day * int(record.get("day_length_minutes", 0)) * jobs
        )
        if int(record.get("calendar_objective", -1)) != expected_calendar_objective:
            errors.append(f"task {record.get('task_id')} has an invalid calendar objective")
        if float(record.get("lower_bound", 0.0)) > float(record.get("objective", 0.0)):
            errors.append(f"task {record.get('task_id')} has a lower bound above its objective")

        for metric, record_field in (
            ("waiting_stage1", "waiting_stage1"),
            ("waiting_stage2", "waiting_stage2"),
        ):
            values = moments.get(metric) if isinstance(moments, Mapping) else None
            total = float(values.get("sum", math.nan)) if isinstance(values, Mapping) else math.nan
            if jobs > 0 and not math.isclose(
                total / jobs,
                float(record.get(record_field, math.nan)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append(f"task {record.get('task_id')} has an inconsistent {metric} mean")

        for stage in (1, 2):
            processing = int(record.get(f"stage{stage}_processing_minutes", -1))
            horizon = int(record.get(f"stage{stage}_active_horizon_minutes", -1))
            machines = int(record.get(f"machines_stage{stage}", 0))
            utilization = float(record.get(f"utilization_stage{stage}", math.nan))
            recomputed = (
                processing / (machines * horizon)
                if processing >= 0 and machines > 0 and horizon > 0
                else 0.0
            )
            if (
                processing < 0
                or horizon <= 0
                or not 0.0 <= utilization <= 1.0 + 1e-12
                or not math.isclose(utilization, recomputed, rel_tol=1e-12, abs_tol=1e-12)
            ):
                errors.append(f"task {record.get('task_id')} has invalid stage-{stage} utilization")
        for field in ("arrivals_hourly", "completed_stage1_hourly", "completed_stage2_hourly"):
            values = record.get(field)
            if not isinstance(values, list) or len(values) != bins:
                errors.append(f"task {record.get('task_id')} has invalid {field}")
            elif any(int(value) != value or int(value) < 0 for value in values):
                errors.append(f"task {record.get('task_id')} has non-integral {field} counts")
            elif sum(int(value) for value in values) != jobs:
                errors.append(f"task {record.get('task_id')} has an invalid {field} total")

    required_days = list(range(expected_days))
    for name in expected_names:
        for seed in seeds:
            members = grouped.get((name, seed), [])
            days = sorted(int(record.get("case_day_index", -1)) for record in members)
            if days != required_days:
                errors.append(f"{name}/seed={seed} does not contain exactly all DBD days")
            devices = sum(int(record.get("jobs", 0)) for record in members)
            if expected_jobs and devices != expected_jobs:
                errors.append(f"{name}/seed={seed} covers {devices} jobs, expected {expected_jobs}")

    for day in required_days:
        for seed in seeds:
            members = paired.get((day, seed), [])
            if {str(record.get("algorithm", "")) for record in members} != set(expected_names):
                errors.append(f"day={day + 1}/seed={seed} is not paired across all algorithms")
                continue
            reference = members[0]
            for record in members[1:]:
                unequal = [
                    field for field in _PAIR_FIELDS if record.get(field) != reference.get(field)
                ]
                if unequal:
                    errors.append(
                        f"day={day + 1}/seed={seed} differs across algorithms: "
                        + ", ".join(unequal)
                    )
                if record.get("arrivals_hourly") != reference.get("arrivals_hourly"):
                    errors.append(f"day={day + 1}/seed={seed} has inconsistent arrivals")
        all_day_records = [
            record for record in records if int(record.get("case_day_index", -1)) == day
        ]
        if all_day_records:
            reference_arrivals = all_day_records[0].get("arrivals_hourly")
            if any(
                record.get("arrivals_hourly") != reference_arrivals
                for record in all_day_records[1:]
            ):
                errors.append(f"day={day + 1} arrivals differ across random seeds")

    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def _aggregate(
    name: str,
    label: str,
    seed: int,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    members = sorted(records, key=lambda record: int(record.get("case_day_index", -1)))
    machines_stage1 = int(members[0].get("machines_stage1", 0))
    machines_stage2 = int(members[0].get("machines_stage2", 0))
    processing_stage1 = sum(int(record.get("stage1_processing_minutes", 0)) for record in members)
    processing_stage2 = sum(int(record.get("stage2_processing_minutes", 0)) for record in members)
    horizon_stage1 = sum(int(record.get("stage1_active_horizon_minutes", 0)) for record in members)
    horizon_stage2 = sum(int(record.get("stage2_active_horizon_minutes", 0)) for record in members)
    daily_mean_stage1 = 100.0 * fmean(
        float(record.get("utilization_stage1", 0.0)) for record in members
    )
    daily_mean_stage2 = 100.0 * fmean(
        float(record.get("utilization_stage2", 0.0)) for record in members
    )
    pooled_stage1 = (
        100.0 * processing_stage1 / (machines_stage1 * horizon_stage1)
        if machines_stage1 > 0 and horizon_stage1 > 0
        else 0.0
    )
    pooled_stage2 = (
        100.0 * processing_stage2 / (machines_stage2 * horizon_stage2)
        if machines_stage2 > 0 and horizon_stage2 > 0
        else 0.0
    )
    row: dict[str, object] = {
        "method": label,
        "algorithm": name,
        "information_mode": "DBD",
        "random_seed": seed,
        "day_count": len(members),
        "device_count": sum(int(record.get("jobs", 0)) for record in members),
        "sum_Cj2_local": sum(int(record.get("objective", 0)) for record in members),
        "sum_Cj2_calendar": sum(
            int(record.get("calendar_objective", record.get("objective", 0))) for record in members
        ),
        "sum_release_time_local": sum(int(record.get("release_sum", 0)) for record in members),
        "runtime_seconds": sum(float(record.get("runtime_seconds", 0.0)) for record in members),
        "process_cpu_runtime_seconds": sum(
            float(record.get("process_cpu_runtime_seconds", 0.0)) for record in members
        ),
        "evaluation_count": sum(int(record.get("evaluations", 0)) for record in members),
        # The paper reports the arithmetic mean of daily active-horizon
        # utilization. Keep the ratio-of-sums variant beside it because that
        # is the literal interpretation of draft Eq. (4-10).
        "stage1_utilization_percent": daily_mean_stage1,
        "stage2_utilization_percent": daily_mean_stage2,
        "stage1_utilization_daily_mean_percent": daily_mean_stage1,
        "stage2_utilization_daily_mean_percent": daily_mean_stage2,
        "stage1_utilization_pooled_active_horizon_percent": pooled_stage1,
        "stage2_utilization_pooled_active_horizon_percent": pooled_stage2,
        "stage1_processing_minutes": processing_stage1,
        "stage2_processing_minutes": processing_stage2,
        "stage1_active_horizon_minutes": horizon_stage1,
        "stage2_active_horizon_minutes": horizon_stage2,
    }
    for metric in _DEVICE_METRICS:
        mean, std, count, total, _ = _pooled_moment(members, metric)
        row[f"{metric}_mean_min"] = mean
        row[f"{metric}_std_min"] = std
        row[f"{metric}_device_count"] = count
        row[f"{metric}_sum_min"] = total
        row[f"{metric}_paper"] = f"{mean:.2f} ({std:.2f})"
    row["stage1_utilization_paper"] = f"{row['stage1_utilization_percent']:.2f}%"
    row["stage2_utilization_paper"] = f"{row['stage2_utilization_percent']:.2f}%"
    return row


def _seed_summary_rows(
    aggregate_rows: Sequence[Mapping[str, object]],
    specs: Sequence[Mapping[str, str]],
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    """Summarize complete three-month outcomes across independent seeds.

    Each input row is already a pooled 91-day result for one algorithm/seed.
    The sample standard deviation below is therefore across independent
    stochastic runs, not across jobs or days.
    """

    expected_seeds = tuple(int(seed) for seed in seeds)
    rows: list[dict[str, object]] = []
    for spec in specs:
        members = sorted(
            (
                row
                for row in aggregate_rows
                if str(row.get("algorithm", "")) == spec["name"]
            ),
            key=lambda row: int(row.get("random_seed", -1)),
        )
        if not members:
            continue
        observed_seeds = tuple(int(row["random_seed"]) for row in members)
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"{spec['label']} seed summary observed {observed_seeds}, "
                f"expected {expected_seeds}"
            )
        row: dict[str, object] = {
            "method": spec["label"],
            "algorithm": spec["name"],
            "information_mode": "DBD",
            "random_seed": "",
            "random_seeds": ";".join(str(seed) for seed in expected_seeds),
            "seed_count": len(members),
            "day_count": int(members[0]["day_count"]),
            "device_count": int(members[0]["device_count"]),
            "standard_deviation_scope": "across_independent_three_month_seeds",
        }
        for metric in _DEVICE_METRICS:
            source = f"{metric}_mean_min"
            values = [float(member[source]) for member in members]
            mean = fmean(values)
            seed_std = stdev(values) if len(values) > 1 else 0.0
            row[source] = mean
            row[f"{metric}_across_seed_std_min"] = seed_std
            row[f"{metric}_paper"] = f"{mean:.2f} ({seed_std:.2f})"
        for stage in (1, 2):
            source = f"stage{stage}_utilization_percent"
            values = [float(member[source]) for member in members]
            mean = fmean(values)
            seed_std = stdev(values) if len(values) > 1 else 0.0
            row[source] = mean
            row[f"stage{stage}_utilization_across_seed_std_percentage_points"] = seed_std
            row[f"stage{stage}_utilization_paper"] = f"{mean:.2f}% ({seed_std:.2f})"
        runtimes = [float(member["runtime_seconds"]) for member in members]
        row["runtime_seconds_mean"] = fmean(runtimes)
        row["runtime_seconds_across_seed_std"] = (
            stdev(runtimes) if len(runtimes) > 1 else 0.0
        )
        rows.append(row)
    return rows


def _hourly_rows(
    records: Sequence[Mapping[str, object]],
    specs: Sequence[Mapping[str, str]],
    seeds: Sequence[int],
    expected_days: int,
    bins: int,
    stage: int,
) -> list[dict[str, object]]:
    if not records:
        return []
    first_name = specs[0]["name"]
    first_seed = seeds[0]
    reference = [
        record
        for record in records
        if str(record.get("algorithm")) == first_name and int(record.get("seed", -1)) == first_seed
    ]
    arrivals = _sum_hourly(reference, "arrivals_hourly", bins)
    totals_by_label: dict[str, list[float]] = {}
    for spec in specs:
        per_seed: list[list[int]] = []
        for seed in seeds:
            members = [
                record
                for record in records
                if str(record.get("algorithm")) == spec["name"]
                and int(record.get("seed", -1)) == seed
            ]
            if members:
                per_seed.append(_sum_hourly(members, f"completed_stage{stage}_hourly", bins))
        if per_seed:
            totals_by_label[spec["label"]] = [
                fmean(values[index] for values in per_seed) for index in range(bins)
            ]

    rows: list[dict[str, object]] = []
    for hour in range(bins):
        row: dict[str, object] = {
            "hour": hour,
            "Arrival_total": arrivals[hour],
            "Arrival_average_per_day": arrivals[hour] / max(1, expected_days),
        }
        for label, totals in totals_by_label.items():
            row[f"{label}_total"] = totals[hour]
            row[f"{label}_average_per_day"] = totals[hour] / max(1, expected_days)
        rows.append(row)
    return rows


def _plot_hourly(rows: Sequence[Mapping[str, object]], stage: int, path: Path) -> None:
    if not rows:
        return
    hours = [int(row["hour"]) for row in rows]
    series = [key for key in rows[0] if key.endswith("_total")]
    colors = {
        "Arrival_total": "#111111",
        "IBS-CABC_total": "#D62728",
        "ALNS_total": "#1F77B4",
        "TS_total": "#2CA02C",
    }
    markers = {"Arrival_total": "o", "IBS-CABC_total": "s", "ALNS_total": "^", "TS_total": "D"}
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    for key in series:
        axis.plot(
            hours,
            [float(row[key]) for row in rows],
            label=key.removesuffix("_total"),
            color=colors.get(key),
            marker=markers.get(key, "o"),
            markersize=3.8,
            linewidth=1.35,
        )
    axis.set_xticks(hours)
    axis.set_xlim(-0.4, max(hours) + 0.4)
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Devices over the case-study window")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.set_title(f"DBD arrivals and stage {stage} completions")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _daily_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fields = (
        "task_id",
        "case_day_index",
        "instance",
        "instance_sha256",
        "algorithm",
        "algorithm_label",
        "seed",
        "jobs",
        "objective",
        "calendar_objective",
        "average_time_in_system",
        "waiting_stage1",
        "waiting_stage2",
        "utilization_stage1",
        "utilization_stage2",
        "stage1_processing_minutes",
        "stage1_active_horizon_minutes",
        "stage2_processing_minutes",
        "stage2_active_horizon_minutes",
        "runtime_seconds",
        "process_cpu_runtime_seconds",
        "evaluations",
        "iterations",
        "feasible",
    )
    rows: list[dict[str, object]] = []
    for record in sorted(
        records,
        key=lambda item: (
            str(item.get("algorithm", "")),
            int(item.get("seed", -1)),
            int(item.get("case_day_index", -1)),
        ),
    ):
        row = {field: record.get(field) for field in fields}
        row["case_day_number"] = int(record.get("case_day_index", -1)) + 1
        rows.append(row)
    return rows


def _markdown(
    status: str,
    manifest: Mapping[str, object],
    aggregate_rows: Sequence[Mapping[str, object]],
    errors: Sequence[str],
    warnings: Sequence[str],
) -> str:
    expected_days, expected_jobs, _, _ = _expected_values(manifest)
    raw_audit = manifest.get("audit")
    audit: Mapping[str, object] = raw_audit if isinstance(raw_audit, Mapping) else {}
    actual_jobs = int(audit.get("total_jobs", expected_jobs))
    lines = [
        "# 三个月 DBD 案例实验汇总",
        "",
        f"执行状态：**{status}**",
        "",
        "本实验目录只运行 IBS-CABC、ALNS 和 TS；信息模式固定为 DBD。"
        "FFCBSPT 结果应由原作者代码独立复现后再合并，Kraul-GA 不在本项目范围内。",
        "",
        f"案例窗口包含 {expected_days} 个连续日实例、{actual_jobs:,} 件器械。"
        "每个随机种子先独立汇总完整三个月窗口，表中括号报告这些种子级指标的样本标准差；"
        "表中利用率采用各 DBD 日活动时间窗利用率的算术平均，以对齐原论文。"
        "CSV 同时给出活动时间窗加权的总加工时间比值，用于核对式（4-10）。",
        "",
        "## 表 4-9 所需的三种算法结果",
        "",
        "| 方法 | 系统停留时间/min | 阶段1等待/min | 阶段2等待/min | 阶段1利用率 | 阶段2利用率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {method} (DBD) | {tis} | {w1} | {w2} | {u1} | {u2} |".format(
                method=row["method"],
                tis=row["time_in_system_paper"],
                w1=row["waiting_stage1_paper"],
                w2=row["waiting_stage2_paper"],
                u1=row["stage1_utilization_paper"],
                u2=row["stage2_utilization_paper"],
            )
        )
    lines.extend(
        [
            "",
            "## 小时曲线口径",
            "",
            "本项目保存每个算法跨随机种子的小时完成量均值；最终论文中的排队曲线由五算法"
            "合并报告基于每个种子的完整三个月排程重建，并报告种子均值及 95% 置信区间。",
        ]
    )
    experiment = manifest.get("experiment")
    contract = experiment.get("contract") if isinstance(experiment, Mapping) else ""
    if contract in {"formal_proxy91", "formal_proxy91_multiseed"}:
        lines.extend(
            [
                "",
                "## 数据边界",
                "",
                "当前公开数据不含原论文三个月案例期的逐日原始文件。正式配置锁定 RealWorldInstances365 "
                "中 seed 233–323 的连续 91 日代理窗口，共 22,927 件器械；"
                "该结果不得表述为原论文 22,934 件案例数据的精确复现。",
            ]
        )
    if warnings:
        lines.extend(["", "## 警告", ""] + [f"- {item}" for item in warnings])
    if errors:
        lines.extend(["", "## 尚未满足的完整性检查", ""] + [f"- {item}" for item in errors])
    return "\n".join(lines) + "\n"


def build_report(experiment_directory: str | Path) -> Path:
    root = Path(experiment_directory).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"experiment has no manifest: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, line_count = _load_records(root / "raw" / "results.jsonl")
    report_dir = root / "report"
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    for directory in (report_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    specs = _algorithm_specs(manifest)
    expected_days, _, seeds, bins = _expected_values(manifest)
    task_contract, task_contract_errors = _load_task_contract(root, manifest)
    errors, warnings = _integrity_checks(
        manifest,
        records,
        line_count,
        task_contract,
        task_contract_errors,
    )
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("algorithm", "")), int(record.get("seed", -1)))].append(record)

    aggregate_rows = [
        _aggregate(spec["name"], spec["label"], seed, grouped[(spec["name"], seed)])
        for spec in specs
        for seed in seeds
        if grouped.get((spec["name"], seed))
    ]
    seed_summary_rows = _seed_summary_rows(aggregate_rows, specs, seeds)
    stage1_rows = (
        _hourly_rows(records, specs, seeds, expected_days, bins, stage=1) if specs and seeds else []
    )
    stage2_rows = (
        _hourly_rows(records, specs, seeds, expected_days, bins, stage=2) if specs and seeds else []
    )

    _write_csv(tables_dir / "table_4_9_dbd_three_algorithms.csv", aggregate_rows)
    _write_csv(
        tables_dir / "table_9_dbd_three_algorithms_seed_summary.csv",
        seed_summary_rows,
    )
    _write_csv(tables_dir / "day_level_runs.csv", _daily_rows(records))
    _write_csv(tables_dir / "hourly_stage1_dbd.csv", stage1_rows)
    _write_csv(tables_dir / "hourly_stage2_dbd.csv", stage2_rows)
    _plot_hourly(stage1_rows, 1, figures_dir / "figure_4_7_dbd_stage1.png")
    _plot_hourly(stage2_rows, 2, figures_dir / "figure_4_8_dbd_stage2.png")

    status = "PASS" if not errors else "INCOMPLETE"
    integrity = {
        "status": status,
        "checked_records": len(records),
        "raw_result_lines": line_count,
        "expected_records": int(manifest.get("task_count_total", 0)),
        "errors": errors,
        "warnings": warnings,
        "scope": {
            "planning_mode": "DBD_only",
            "algorithms": [spec["name"] for spec in specs],
            "seeds": seeds,
            "expected_days": expected_days,
        },
        "aggregation": {
            "time_metrics": "pooled device-level moments with sample standard deviation",
            "utilization_primary": "arithmetic mean of daily active-horizon utilization",
            "utilization_secondary": "ratio of total processing minutes to machine-weighted active horizons",
            "hourly_figure": "total counts over all days; averaged across random seeds",
            "hourly_csv": "both total counts and average-per-day counts",
        },
    }
    _write_json(report_dir / "integrity.json", integrity)
    summary_path = report_dir / "dbd_case_summary_zh.md"
    summary_path.write_text(
        _markdown(status, manifest, seed_summary_rows, errors, warnings),
        encoding="utf-8-sig",
    )
    return summary_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the isolated CSSD DBD case report")
    parser.add_argument("--experiment", required=True, help="completed experiment directory")
    args = parser.parse_args(argv)
    summary = build_report(args.experiment)
    print(summary)
    integrity = json.loads((summary.parent / "integrity.json").read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
