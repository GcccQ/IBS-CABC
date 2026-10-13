from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from .decoder import Batch, DecodedSchedule, decode_permutation, validate_schedule
from .metrics import schedule_metrics
from .model import load_instance


METRICS = ("time_in_system", "waiting_stage1", "waiting_stage2")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load_records(path: Path) -> tuple[list[dict[str, object]], int, list[str]]:
    records: dict[str, dict[str, object]] = {}
    duplicates: list[str] = []
    line_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            line_count += 1
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"result line {line_count} is not a JSON object")
            if record.get("status") == "success":
                task_id = str(record["task_id"])
                if task_id in records:
                    duplicates.append(task_id)
                else:
                    records[task_id] = record
    return list(records.values()), line_count, duplicates


def _decoded_schedule(record: Mapping[str, object], payload: Mapping[str, object]) -> DecodedSchedule:
    def integer(value: object, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} is not an integer")
        numeric = float(value)  # type: ignore[arg-type]
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} is not an integer")
        return int(numeric)

    def vector(name: str, *, integral: bool = False) -> np.ndarray:
        values = payload.get(name)
        if not isinstance(values, list):
            raise ValueError(f"schedule.{name} is not a list")
        array = np.asarray(values, dtype=np.float64)
        if integral:
            if not np.all(np.isfinite(array)) or not np.array_equal(array, np.floor(array)):
                raise ValueError(f"schedule.{name} contains non-integral identifiers")
            return array.astype(np.int64)
        return array

    def batches(name: str) -> tuple[Batch, ...]:
        raw_batches = payload.get(name)
        if not isinstance(raw_batches, list):
            raise ValueError(f"schedule.{name} is not a list")
        output: list[Batch] = []
        for scheduled_index, raw in enumerate(raw_batches, start=1):
            if not isinstance(raw, Mapping):
                raise ValueError(f"schedule.{name} contains a non-object batch")
            job_ids = raw.get("job_ids")
            if not isinstance(job_ids, list):
                raise ValueError(f"schedule.{name} batch has no job_ids list")
            if integer(raw.get("scheduled_batch_index"), "scheduled_batch_index") != scheduled_index:
                raise ValueError(f"schedule.{name} has inconsistent scheduled batch indices")
            output.append(
                Batch(
                    stage=integer(raw["stage"], "stage"),
                    family=integer(raw["family"], "family"),
                    family_batch_index=integer(
                        raw["family_batch_index"], "family_batch_index"
                    ),
                    job_indices=tuple(integer(job_id, "job_id") - 1 for job_id in job_ids),
                    processing_time=float(raw["processing_time"]),
                    release_time=float(raw["release_time"]),
                    load=float(raw["load"]),
                    machine_id=integer(raw["machine_id"], "machine_id"),
                    start_time=float(raw["start_time"]),
                    completion_time=float(raw["completion_time"]),
                )
            )
        return tuple(output)

    permutation = payload.get("best_permutation_zero_based")
    if not isinstance(permutation, list):
        raise ValueError("schedule.best_permutation_zero_based is not a list")
    permutation_array = np.asarray(permutation, dtype=np.float64)
    if not np.all(np.isfinite(permutation_array)) or not np.array_equal(
        permutation_array, np.floor(permutation_array)
    ):
        raise ValueError("schedule.best_permutation_zero_based contains non-integral genes")
    return DecodedSchedule(
        permutation=permutation_array.astype(np.int64),
        stage1_batches=batches("stage1_batches"),
        stage2_batches=batches("stage2_batches"),
        start_stage1=vector("start_stage1"),
        completion_stage1=vector("completion_stage1"),
        machine_stage1=vector("machine_stage1", integral=True),
        batch_stage1=vector("batch_stage1", integral=True),
        start_stage2=vector("start_stage2"),
        completion_stage2=vector("completion_stage2"),
        machine_stage2=vector("machine_stage2", integral=True),
        batch_stage2=vector("batch_stage2", integral=True),
        objective=float(record["objective"]),
    )


def _pooled_moment(records: Sequence[Mapping[str, object]], metric: str) -> dict[str, float | int]:
    count = 0
    total = 0.0
    sum_squares = 0.0
    for record in records:
        raw = record.get(metric)
        if not isinstance(raw, Mapping):
            continue
        count += int(raw.get("count", 0))
        total += float(raw.get("sum", 0.0))
        sum_squares += float(raw.get("sum_squares", 0.0))
    mean = total / count if count else 0.0
    variance = max(0.0, (sum_squares - total * total / count) / (count - 1)) if count > 1 else 0.0
    return {
        "count": count,
        "sum": total,
        "sum_squares": sum_squares,
        "mean": mean,
        "sample_std": math.sqrt(variance),
    }


def _aggregate(
    algorithm: str,
    label: str,
    seed: int | None,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ordered = sorted(records, key=lambda record: int(record["case_day_index"]))
    m1 = int(ordered[0]["machines_stage1"])
    m2 = int(ordered[0]["machines_stage2"])
    p1 = sum(float(record["stage1_processing_minutes"]) for record in ordered)
    p2 = sum(float(record["stage2_processing_minutes"]) for record in ordered)
    h1 = sum(float(record["stage1_active_horizon_minutes"]) for record in ordered)
    h2 = sum(float(record["stage2_active_horizon_minutes"]) for record in ordered)
    row: dict[str, object] = {
        "method": label,
        "algorithm": algorithm,
        "information_mode": "DBD",
        "random_seed": "" if seed is None else seed,
        "day_count": len(ordered),
        "device_count": sum(int(record["jobs"]) for record in ordered),
        "sum_Cj2_local": sum(float(record["objective"]) for record in ordered),
        "sum_release_time_local": sum(float(record["release_sum"]) for record in ordered),
        "runtime_seconds": sum(float(record["runtime_seconds"]) for record in ordered),
        "evaluation_count": sum(int(record["evaluations"]) for record in ordered),
        # This arithmetic mean of daily active-horizon utilization is the
        # definition used to reproduce the original paper's Table 4.
        "stage1_utilization_percent": 100.0
        * fmean(float(record["utilization_stage1"]) for record in ordered),
        "stage2_utilization_percent": 100.0
        * fmean(float(record["utilization_stage2"]) for record in ordered),
        "stage1_utilization_pooled_active_horizon_percent": 100.0 * p1 / (m1 * h1),
        "stage2_utilization_pooled_active_horizon_percent": 100.0 * p2 / (m2 * h2),
        "stage1_utilization_fixed_24h_percent": 100.0
        * fmean(float(record["utilization_stage1_fixed_day"]) for record in ordered),
        "stage2_utilization_fixed_24h_percent": 100.0
        * fmean(float(record["utilization_stage2_fixed_day"]) for record in ordered),
    }
    for metric in METRICS:
        pooled = _pooled_moment(ordered, metric)
        row[f"{metric}_mean_min"] = pooled["mean"]
        row[f"{metric}_std_min"] = pooled["sample_std"]
        row[f"{metric}_paper"] = f"{pooled['mean']:.2f} ({pooled['sample_std']:.2f})"
    row["stage1_utilization_paper"] = f"{row['stage1_utilization_percent']:.2f}%"
    row["stage2_utilization_paper"] = f"{row['stage2_utilization_percent']:.2f}%"
    return row


def _seed_summary_rows(
    aggregate_rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    """Collapse seed-level three-month outcomes to one paper row per method."""

    raw_experiment = manifest.get("experiment")
    experiment = raw_experiment if isinstance(raw_experiment, Mapping) else {}
    raw_algorithms = experiment.get("algorithms")
    algorithms = raw_algorithms if isinstance(raw_algorithms, list) else []
    expected_seeds = tuple(int(seed) for seed in experiment.get("seeds", []))
    rows: list[dict[str, object]] = []
    for spec in algorithms:
        if not isinstance(spec, Mapping):
            continue
        name = str(spec["name"])
        label = str(spec["label"])
        members = [row for row in aggregate_rows if str(row.get("algorithm")) == name]
        if not members:
            continue
        stochastic = name != "ffcbspt"
        if stochastic:
            members.sort(key=lambda row: int(row.get("random_seed", -1)))
            observed = tuple(int(row["random_seed"]) for row in members)
            if observed != expected_seeds:
                raise ValueError(
                    f"{label} seed summary observed {observed}, expected {expected_seeds}"
                )
        elif len(members) != 1:
            raise ValueError("deterministic FFCBSPT must have exactly one aggregate row")

        row: dict[str, object] = {
            "method": label,
            "algorithm": name,
            "information_mode": "DBD",
            "random_seed": "",
            "random_seeds": (
                ";".join(str(seed) for seed in expected_seeds) if stochastic else ""
            ),
            "seed_count": len(members) if stochastic else 0,
            "day_count": int(members[0]["day_count"]),
            "device_count": int(members[0]["device_count"]),
            "standard_deviation_scope": (
                "across_independent_three_month_seeds"
                if stochastic
                else "not_applicable_deterministic_method"
            ),
        }
        for metric in METRICS:
            source = f"{metric}_mean_min"
            values = [float(member[source]) for member in members]
            mean = fmean(values)
            seed_std = stdev(values) if stochastic and len(values) > 1 else None
            row[source] = mean
            row[f"{metric}_across_seed_std_min"] = seed_std
            row[f"{metric}_paper"] = (
                f"{mean:.2f} ({seed_std:.2f})"
                if seed_std is not None
                else f"{mean:.2f} (—)"
            )
        for stage in (1, 2):
            source = f"stage{stage}_utilization_percent"
            values = [float(member[source]) for member in members]
            mean = fmean(values)
            seed_std = stdev(values) if stochastic and len(values) > 1 else None
            row[source] = mean
            row[f"stage{stage}_utilization_across_seed_std_percentage_points"] = seed_std
            row[f"stage{stage}_utilization_paper"] = (
                f"{mean:.2f}% ({seed_std:.2f})"
                if seed_std is not None
                else f"{mean:.2f}% (—)"
            )
        runtimes = [float(member["runtime_seconds"]) for member in members]
        row["runtime_seconds_mean"] = fmean(runtimes)
        row["runtime_seconds_across_seed_std"] = (
            stdev(runtimes) if stochastic and len(runtimes) > 1 else None
        )
        rows.append(row)
    return rows


def _groups(
    records: Sequence[Mapping[str, object]], manifest: Mapping[str, object]
) -> list[tuple[str, str, int | None, list[Mapping[str, object]]]]:
    raw_experiment = manifest.get("experiment")
    experiment = raw_experiment if isinstance(raw_experiment, Mapping) else {}
    raw_algorithms = experiment.get("algorithms")
    algorithms = raw_algorithms if isinstance(raw_algorithms, list) else []
    seeds = [int(seed) for seed in experiment.get("seeds", [])]
    groups: list[tuple[str, str, int | None, list[Mapping[str, object]]]] = []
    for spec in algorithms:
        if not isinstance(spec, Mapping):
            continue
        name = str(spec["name"])
        label = str(spec["label"])
        method_seeds: list[int | None] = [None] if name == "ffcbspt" else seeds
        for seed in method_seeds:
            members = [
                record
                for record in records
                if record.get("algorithm") == name and record.get("seed") == seed
            ]
            groups.append((name, label, seed, members))
    return groups


def _integrity(
    root: Path,
    manifest: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    duplicate_successes: Sequence[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if duplicate_successes:
        errors.append(
            "duplicate successful task records: " + ", ".join(sorted(set(duplicate_successes))[:5])
        )
    task_manifest = json.loads((root / "task_manifest.json").read_text(encoding="utf-8"))
    expected_tasks = {
        str(task["task_id"]): task
        for task in task_manifest.get("tasks", [])
        if isinstance(task, Mapping)
    }
    observed = {str(record.get("task_id")): record for record in records}
    missing = sorted(set(expected_tasks) - set(observed))
    extra = sorted(set(observed) - set(expected_tasks))
    if missing:
        errors.append(f"missing {len(missing)} successful tasks: {', '.join(missing[:5])}")
    if extra:
        errors.append(f"unexpected tasks: {', '.join(extra[:5])}")

    raw_experiment = manifest.get("experiment")
    experiment = raw_experiment if isinstance(raw_experiment, Mapping) else {}
    raw_audit = manifest.get("audit")
    audit = raw_audit if isinstance(raw_audit, Mapping) else {}
    data_root = Path(str(experiment.get("data_root", "")))
    sequence = audit.get("sequence", [])
    paths = {
        int(item["case_day_index"]): data_root / str(item["relative_path"])
        for item in sequence
        if isinstance(item, Mapping)
    }
    instance_cache = {day: load_instance(path) for day, path in paths.items()}

    def close(left: object, right: object) -> bool:
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-8)
        except (TypeError, ValueError, OverflowError):
            return False

    for task_id, record in observed.items():
        task = expected_tasks.get(task_id)
        if task is None:
            continue
        if record.get("task_hash") != task.get("task_hash"):
            errors.append(f"task hash mismatch: {task_id}")
        for field in (
            "case_day_index",
            "case_day_number",
            "instance",
            "instance_sha256",
            "jobs",
            "algorithm",
            "algorithm_label",
            "seed",
            "profile",
            "overrides",
            "time_limit_seconds",
            "max_evaluations",
            "day_length_minutes",
            "run_contract",
        ):
            if record.get(field) != task.get(field):
                errors.append(f"task field mismatch ({field}): {task_id}")
        try:
            jobs = int(record.get("jobs", 0))
            objective = float(record.get("objective", math.nan))
            releases = float(record.get("release_sum", math.nan))
        except (TypeError, ValueError, OverflowError):
            errors.append(f"invalid numeric task metrics: {task_id}")
            continue
        if not record.get("feasible"):
            errors.append(f"infeasible schedule: {task_id}")
        tis = record.get("time_in_system")
        if not isinstance(tis, Mapping) or not math.isclose(
            objective - releases,
            float(tis.get("sum", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            errors.append(f"objective/TIS identity failed: {task_id}")
        schedule = record.get("schedule")
        if not isinstance(schedule, Mapping):
            errors.append(f"missing schedule payload: {task_id}")
            continue
        day = int(record.get("case_day_index", -1))
        instance = instance_cache.get(day)
        if instance is None:
            errors.append(f"no audited instance for case day {day}: {task_id}")
            continue
        if instance.job_count != jobs or instance.source_sha256 != record.get("instance_sha256"):
            errors.append(f"instance identity mismatch: {task_id}")
            continue
        if schedule.get("job_ids") != instance.jobs[:, 0].astype(int).tolist():
            errors.append(f"schedule.job_ids mismatch: {task_id}")
        try:
            decoded = _decoded_schedule(record, schedule)
            validate_schedule(instance, decoded)
            canonical = decode_permutation(instance, decoded.permutation)
            if (
                decoded.stage1_batches != canonical.stage1_batches
                or decoded.stage2_batches != canonical.stage2_batches
                or not np.array_equal(decoded.start_stage1, canonical.start_stage1)
                or not np.array_equal(decoded.completion_stage1, canonical.completion_stage1)
                or not np.array_equal(decoded.machine_stage1, canonical.machine_stage1)
                or not np.array_equal(decoded.batch_stage1, canonical.batch_stage1)
                or not np.array_equal(decoded.start_stage2, canonical.start_stage2)
                or not np.array_equal(decoded.completion_stage2, canonical.completion_stage2)
                or not np.array_equal(decoded.machine_stage2, canonical.machine_stage2)
                or not np.array_equal(decoded.batch_stage2, canonical.batch_stage2)
            ):
                raise ValueError("persisted schedule does not match its decoded permutation")
            recomputed = schedule_metrics(
                instance,
                decoded,
                day_length_minutes=int(record["day_length_minutes"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(f"invalid persisted schedule ({error}): {task_id}")
            continue
        scalar_fields = (
            "objective",
            "release_sum",
            "average_time_in_system",
            "average_waiting_stage1",
            "average_waiting_stage2",
            "stage1_processing_minutes",
            "stage2_processing_minutes",
            "stage1_active_horizon_minutes",
            "stage2_active_horizon_minutes",
            "utilization_stage1",
            "utilization_stage2",
            "utilization_stage1_fixed_day",
            "utilization_stage2_fixed_day",
        )
        for field in scalar_fields:
            if not close(record.get(field), recomputed.get(field)):
                errors.append(f"recomputed metric mismatch ({field}): {task_id}")
        for metric in METRICS:
            actual_moment = record.get(metric)
            expected_moment = recomputed.get(metric)
            if not isinstance(actual_moment, Mapping) or not isinstance(expected_moment, Mapping):
                errors.append(f"invalid metric moment ({metric}): {task_id}")
                continue
            for field in ("count", "sum", "sum_squares", "mean", "sample_std"):
                if not close(actual_moment.get(field), expected_moment.get(field)):
                    errors.append(f"recomputed {metric}.{field} mismatch: {task_id}")
        for field in ("arrivals_hourly", "completed_stage1_hourly", "completed_stage2_hourly"):
            if record.get(field) != recomputed.get(field):
                errors.append(f"recomputed hourly series mismatch ({field}): {task_id}")
        expected_calendar = float(recomputed["objective"]) + (
            day * int(record["day_length_minutes"]) * jobs
        )
        if not close(record.get("calendar_objective"), expected_calendar):
            errors.append(f"calendar objective mismatch: {task_id}")
        try:
            observed_equipment = (
                int(record.get("machines_stage1", 0)),
                int(record.get("machines_stage2", 0)),
                int(record.get("capacity_stage1", 0)),
                int(record.get("capacity_stage2", 0)),
            )
        except (TypeError, ValueError, OverflowError):
            observed_equipment = ()
        if observed_equipment != (
            instance.machines_stage1,
            instance.machines_stage2,
            instance.capacity_stage1,
            instance.capacity_stage2,
        ):
            errors.append(f"machine/capacity metadata mismatch: {task_id}")
    for name, _, seed, members in _groups(records, manifest):
        expected_days = int(audit.get("day_count", 0))
        days = sorted(int(record["case_day_index"]) for record in members)
        if days != list(range(expected_days)):
            errors.append(f"{name}/seed={seed} does not contain every DBD day")
    if manifest.get("status") != "complete":
        errors.append(f"runner manifest status is {manifest.get('status')!r}, not complete")
    raw_differences = manifest.get("known_source_differences")
    if isinstance(raw_differences, Mapping):
        warnings.extend(
            str(value) for value in raw_differences.values() if isinstance(value, str)
        )
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def _sum_hourly(
    records: Sequence[Mapping[str, object]], field: str, hour_bins: int
) -> list[int]:
    output = [0] * hour_bins
    for record in records:
        values = record.get(field)
        if isinstance(values, list):
            output = [left + int(right) for left, right in zip(output, values)]
    return output


def _hourly_rows(
    records: Sequence[Mapping[str, object]],
    groups: Sequence[tuple[str, str, int | None, list[Mapping[str, object]]]],
    stage: int,
    day_count: int,
    hour_bins: int,
) -> list[dict[str, object]]:
    nonempty_groups = [group for group in groups if group[3]]
    if not nonempty_groups:
        return []
    reference = nonempty_groups[0][3]
    arrivals = _sum_hourly(reference, "arrivals_hourly", hour_bins)
    totals_by_label: dict[str, list[list[int]]] = {}
    for _, label, _, members in nonempty_groups:
        totals_by_label.setdefault(label, []).append(
            _sum_hourly(members, f"completed_stage{stage}_hourly", hour_bins)
        )
    series: list[tuple[str, list[int] | list[float]]] = []
    for label, seed_totals in totals_by_label.items():
        if len(seed_totals) == 1:
            values: list[int] | list[float] = seed_totals[0]
        else:
            values = [fmean(items) for items in zip(*seed_totals)]
        series.append((label, values))
    rows: list[dict[str, object]] = []
    for hour in range(hour_bins):
        row: dict[str, object] = {
            "hour": hour,
            "Arrival_total": arrivals[hour],
            "Arrival_average_per_day": arrivals[hour] / day_count,
        }
        for label, values in series:
            row[f"{label}_total"] = values[hour]
            row[f"{label}_average_per_day"] = values[hour] / day_count
        rows.append(row)
    return rows


def _plot_hourly(rows: Sequence[Mapping[str, object]], stage: int, path: Path) -> None:
    if not rows:
        return
    hours = [int(row["hour"]) for row in rows]
    keys = [key for key in rows[0] if key.endswith("_average_per_day")]
    colors = {"Arrival_average_per_day": "#111111", "FFCBSPT_average_per_day": "#777777", "Kraul-GA_average_per_day": "#9467BD"}
    markers = {"Arrival_average_per_day": "o", "FFCBSPT_average_per_day": "v", "Kraul-GA_average_per_day": "P"}
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    for key in keys:
        axis.plot(
            hours,
            [float(row[key]) for row in rows],
            label=key.removesuffix("_average_per_day"),
            color=colors.get(key),
            marker=markers.get(key, "o"),
            linewidth=1.4,
            markersize=4,
        )
    axis.set_xticks(hours)
    axis.set_xlim(-0.4, 23.4)
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Average devices per day")
    axis.set_title(f"DBD arrivals and stage {stage} completions")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _daily_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fields = (
        "task_id",
        "case_day_number",
        "instance",
        "instance_sha256",
        "algorithm",
        "algorithm_label",
        "seed",
        "jobs",
        "objective",
        "average_time_in_system",
        "average_waiting_stage1",
        "average_waiting_stage2",
        "utilization_stage1",
        "utilization_stage2",
        "utilization_stage1_fixed_day",
        "utilization_stage2_fixed_day",
        "runtime_seconds",
        "process_cpu_seconds",
        "evaluations",
        "generations",
        "restart_count",
        "feasible",
    )
    return [
        {field: record.get(field) for field in fields}
        for record in sorted(
            records,
            key=lambda record: (
                str(record["algorithm"]),
                -1 if record.get("seed") is None else int(record["seed"]),
                int(record["case_day_index"]),
            ),
        )
    ]


def _job_rows(records: Sequence[Mapping[str, object]], manifest: Mapping[str, object]) -> list[dict[str, object]]:
    experiment = manifest.get("experiment", {})
    data_root = Path(str(experiment.get("data_root")))
    sequence = manifest.get("audit", {}).get("sequence", [])
    paths = {
        int(item["case_day_index"]): data_root / str(item["relative_path"])
        for item in sequence
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, object]] = []
    instance_cache = {day: load_instance(path) for day, path in paths.items()}
    for record in sorted(records, key=lambda item: str(item["task_id"])):
        day = int(record["case_day_index"])
        instance = instance_cache[day]
        schedule = record["schedule"]
        for index in range(instance.job_count):
            rows.append(
                {
                    "task_id": record["task_id"],
                    "method": record["algorithm_label"],
                    "seed": record.get("seed"),
                    "case_day_number": day + 1,
                    "instance": instance.name,
                    "job_id": index + 1,
                    "family": int(instance.jobs[index, 4]),
                    "release": instance.jobs[index, 3],
                    "processing_stage1": instance.jobs[index, 1],
                    "load_stage1": instance.jobs[index, 5],
                    "start_stage1": schedule["start_stage1"][index],
                    "completion_stage1": schedule["completion_stage1"][index],
                    "machine_stage1": schedule["machine_stage1"][index],
                    "batch_stage1": schedule["batch_stage1"][index],
                    "processing_stage2": instance.jobs[index, 2],
                    "load_stage2": instance.jobs[index, 6],
                    "start_stage2": schedule["start_stage2"][index],
                    "completion_stage2": schedule["completion_stage2"][index],
                    "machine_stage2": schedule["machine_stage2"][index],
                    "batch_stage2": schedule["batch_stage2"][index],
                    "waiting_stage1": schedule["start_stage1"][index]
                    - instance.jobs[index, 3],
                    "waiting_stage2": schedule["start_stage2"][index]
                    - schedule["completion_stage1"][index],
                    "time_in_system": schedule["completion_stage2"][index]
                    - instance.jobs[index, 3],
                }
            )
    return rows


def _convergence_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.get("algorithm") != "kraul_ga":
            continue
        diagnostics = record.get("algorithm_diagnostics")
        history = diagnostics.get("history", []) if isinstance(diagnostics, Mapping) else []
        for generation, objective in enumerate(history, start=1):
            rows.append(
                {
                    "task_id": record["task_id"],
                    "case_day_number": record["case_day_number"],
                    "seed": record["seed"],
                    "generation": generation,
                    "generation_best_objective": objective,
                }
            )
    return rows


def _summary_markdown(
    status: str,
    aggregate_rows: Sequence[Mapping[str, object]],
    errors: Sequence[str],
    warnings: Sequence[str],
) -> str:
    lines = [
        "# FFCBSPT 与 Kraul-GA 的 DBD 复现实验",
        "",
        f"完整性检查：**{status}**",
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
            "每个随机种子先独立汇总完整三个月窗口；随机算法括号中报告种子级指标的样本标准差。"
            "FFCBSPT 为确定性规则，跨种子标准差不适用。主利用率采用原论文表格可复核的"
            "“逐日活动时间窗利用率算术平均”；CSV 另给固定 24 小时口径与活动时间窗加权口径。",
            "",
            "正式 GA 使用论文最终参数（TPX、插入变异、交叉率 0.95），并保留公开源码的搜索控制流与轮盘赌方向。小时曲线使用每天平均作业量，跨午夜完成时刻按一天中的小时取模。",
        ]
    )
    if warnings:
        lines.extend(["", "## 已知复现差异", ""] + [f"- {item}" for item in warnings])
    if errors:
        lines.extend(["", "## 未通过检查", ""] + [f"- {item}" for item in errors])
    return "\n".join(lines) + "\n"


def build_report(experiment_directory: str | Path) -> Path:
    root = Path(experiment_directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records, line_count, duplicate_successes = _load_records(root / "raw" / "results.jsonl")
    errors, warnings = _integrity(root, manifest, records, duplicate_successes)
    groups = _groups(records, manifest)
    aggregate_rows = [
        _aggregate(name, label, seed, members)
        for name, label, seed, members in groups
        if members
    ]
    seed_summary_rows = _seed_summary_rows(aggregate_rows, manifest)
    day_count = int(manifest.get("audit", {}).get("day_count", 1))
    day_length = int(manifest.get("experiment", {}).get("day_length_minutes", 1440))
    hour_bins = day_length // 60
    stage1_rows = _hourly_rows(records, groups, 1, day_count, hour_bins)
    stage2_rows = _hourly_rows(records, groups, 2, day_count, hour_bins)
    tables = root / "tables"
    figures = root / "figures"
    report = root / "report"
    for directory in (tables, figures, report):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "table_4_9_ffcbspt_kraul_ga.csv", aggregate_rows)
    _write_csv(
        tables / "table_9_ffcbspt_kraul_ga_seed_summary.csv",
        seed_summary_rows,
    )
    _write_csv(tables / "day_level_runs.csv", _daily_rows(records))
    _write_csv(tables / "job_level_schedules.csv", _job_rows(records, manifest))
    _write_csv(tables / "hourly_stage1_dbd.csv", stage1_rows)
    _write_csv(tables / "hourly_stage2_dbd.csv", stage2_rows)
    _write_csv(tables / "ga_convergence.csv", _convergence_rows(records))
    _plot_hourly(stage1_rows, 1, figures / "figure_4_7_ffcbspt_kraul_ga_stage1.png")
    _plot_hourly(stage2_rows, 2, figures / "figure_4_8_ffcbspt_kraul_ga_stage2.png")
    status = "PASS" if not errors else "INCOMPLETE"
    _write_json(
        report / "integrity.json",
        {
            "status": status,
            "successful_records": len(records),
            "raw_jsonl_lines": line_count,
            "expected_records": manifest.get("task_count_total"),
            "errors": errors,
            "warnings": warnings,
            "aggregation": {
                "time_metrics_per_seed": "pooled device moments over the complete 91-day window",
                "paper_table_standard_deviation": (
                    "sample standard deviation across independent three-month random seeds"
                ),
                "utilization_primary": "arithmetic mean of daily active-horizon utilization",
                "hourly": "average devices per day; completion hour modulo 24",
            },
        },
    )
    summary = report / "dbd_reproduction_summary_zh.md"
    summary.write_text(
        _summary_markdown(status, seed_summary_rows, errors, warnings),
        encoding="utf-8-sig",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the FFCBSPT/Kraul-GA DBD report")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    summary = build_report(args.experiment)
    print(summary)
    integrity = json.loads((summary.parent / "integrity.json").read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
