from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


METHOD_ORDER = ("FFCBSPT", "ALNS", "TS", "Kraul-GA", "IBS-CABC")
COLORS = {
    "Arrival": "#111111",
    "FFCBSPT": "#7F7F7F",
    "ALNS": "#1F77B4",
    "TS": "#2CA02C",
    "Kraul-GA": "#9467BD",
    "IBS-CABC": "#D62728",
}
MARKERS = {
    "Arrival": "o",
    "FFCBSPT": "v",
    "ALNS": "^",
    "TS": "D",
    "Kraul-GA": "P",
    "IBS-CABC": "s",
}

QUEUE_METHODS_KRAUL = ("FFCBSPT", "Kraul-GA")
QUEUE_METHODS_THREE = ("ALNS", "TS", "IBS-CABC")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260826

QueueInterval = tuple[float, float, float, float]
QueueIntervals = dict[str, dict[int | None, list[QueueInterval]]]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _audit_days(audit: Mapping[str, object]) -> int:
    value = audit.get("day_count", audit.get("days", -1))
    return int(value)


def _audit_sequence(audit: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = audit.get("sequence", audit.get("instance_sequence", []))
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError("experiment manifest has no valid ordered instance sequence")
    return value


def _contract_reference(manifest: Mapping[str, object]) -> str | dict[str, object]:
    """Return provenance for both current and legacy runner manifests."""
    current = manifest.get("run_contract")
    if isinstance(current, str) and current:
        return current
    legacy_fields = (
        "config_sha256",
        "split_sha256",
        "task_manifest_sha256",
        "code_sha256",
    )
    legacy = {field: manifest[field] for field in legacy_fields if field in manifest}
    if not legacy:
        raise ValueError("experiment manifest has no reproducible run-contract provenance")
    return {"schema": "legacy_manifest", **legacy}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def _finite_float(value: object, field: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context} has an invalid {field}") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} has a non-finite {field}")
    return result


def _seed_value(value: object) -> int | None:
    text = str(value if value is not None else "").strip()
    return None if not text else int(text)


def _bootstrap_mean_ci(
    values: Sequence[float],
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Mean, sample SD, and percentile-bootstrap 95% CI across seeds."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty seed series")
    mean = float(array.mean())
    if array.size == 1:
        return mean, 0.0, mean, mean
    sample_std = float(array.std(ddof=1))
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_RESAMPLES, array.size),
    )
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, sample_std, float(low), float(high)


def _interval_minutes_by_hour(
    start: float,
    end: float,
    *,
    day_length_minutes: int,
    bin_minutes: int = 60,
) -> list[float]:
    """Split one half-open waiting interval across clock-hour bins.

    DBD schedules use local daily time and may finish after midnight.  Folding
    elapsed minutes modulo the configured day length keeps this queue profile
    on the same hour-of-day basis as the completion curves.
    """

    if day_length_minutes <= 0 or bin_minutes <= 0:
        raise ValueError("day and queue-bin lengths must be positive")
    if day_length_minutes % bin_minutes:
        raise ValueError("day length must be divisible by the queue-bin length")
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("queue interval endpoints must be finite")
    if end < start - 1e-9:
        raise ValueError(f"negative queue interval: [{start}, {end})")

    bins = day_length_minutes // bin_minutes
    totals = [0.0] * bins
    cursor = start
    while cursor < end - 1e-12:
        minute_of_day = cursor % day_length_minutes
        hour = min(int(minute_of_day // bin_minutes), bins - 1)
        next_boundary = cursor + (hour + 1) * bin_minutes - minute_of_day
        if next_boundary <= cursor + 1e-12:
            next_boundary = cursor + bin_minutes
        segment_end = min(end, next_boundary)
        totals[hour] += segment_end - cursor
        cursor = segment_end
    return totals


def _queue_intervals_from_csv(
    root: Path,
    expected_methods: Sequence[str],
) -> QueueIntervals:
    """Read release/S1-start/C1/S2-start tuples from a report job table."""

    path = root / "tables" / "job_level_schedules.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _read_csv(path)
    required = {
        "method",
        "case_day_number",
        "job_id",
        "release",
        "start_stage1",
        "completion_stage1",
        "start_stage2",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - (set(rows[0]) if rows else set()))
        raise ValueError(f"{path} lacks queue timing columns: {', '.join(missing)}")

    expected = set(expected_methods)
    intervals: QueueIntervals = {method: {} for method in expected_methods}
    observed: set[str] = set()
    identities: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(rows, start=2):
        method = row.get("method", "")
        if method not in expected:
            raise ValueError(f"unexpected method {method!r} in {path}")
        observed.add(method)
        seed = _seed_value(row.get("seed"))
        identity = (
            method,
            "" if seed is None else str(seed),
            row.get("case_day_number", ""),
            row.get("job_id", ""),
        )
        if identity in identities:
            raise ValueError(f"duplicate job timing {identity} in {path}")
        identities.add(identity)
        context = f"{path.name} row {index}"
        release = _finite_float(row.get("release"), "release", context)
        start1 = _finite_float(row.get("start_stage1"), "start_stage1", context)
        completion1 = _finite_float(
            row.get("completion_stage1"), "completion_stage1", context
        )
        start2 = _finite_float(row.get("start_stage2"), "start_stage2", context)
        if start1 < release - 1e-9:
            raise ValueError(f"{context} starts stage 1 before release")
        if completion1 < start1 - 1e-9:
            raise ValueError(f"{context} completes stage 1 before it starts")
        if start2 < completion1 - 1e-9:
            raise ValueError(f"{context} starts stage 2 before stage 1 completion")
        intervals[method].setdefault(seed, []).append(
            (release, start1, completion1, start2)
        )
    if observed != expected:
        raise ValueError(f"{path} contains methods {sorted(observed)}, expected {sorted(expected)}")
    return intervals


def _three_algorithm_queue_intervals(
    root: Path,
) -> QueueIntervals:
    """Replay persisted final batch structures to recover exact queue intervals."""

    article_root = Path(__file__).resolve().parents[3]
    ibscabc_source = article_root / "cssd-kg-edr" / "src"
    if not (ibscabc_source / "cssd_ibscabc").is_dir():
        raise RuntimeError(
            "cannot replay three-algorithm schedules: missing read-only cssd-kg-edr source"
        )
    sys.dont_write_bytecode = True
    source_text = str(ibscabc_source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    from cssd_ibscabc.decoder.integrated import decode_structure
    from cssd_ibscabc.io.loader import load_instance
    from cssd_ibscabc.representation.batch_structure import BatchStructureSolution

    records: dict[str, Mapping[str, object]] = {}
    path = root / "raw" / "results.jsonl"
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"{path} line {line_number} is not an object")
            task_id = str(record.get("task_id", ""))
            if not task_id:
                raise ValueError(f"{path} line {line_number} has no task_id")
            if task_id in records:
                raise ValueError(f"duplicate task_id {task_id} in {path}")
            records[task_id] = record

    intervals: QueueIntervals = {method: {} for method in QUEUE_METHODS_THREE}
    instance_cache: dict[tuple[str, str], object] = {}
    for task_id, record in records.items():
        label = str(record.get("algorithm_label", ""))
        if label not in intervals:
            raise ValueError(f"unexpected method {label!r} in {path}")
        if not bool(record.get("feasible", False)):
            raise ValueError(f"cannot replay infeasible task {task_id}")
        seed = int(record.get("seed", -1))
        diagnostics = record.get("diagnostics")
        solution_payload = diagnostics.get("solution") if isinstance(diagnostics, Mapping) else None
        if not isinstance(solution_payload, Mapping):
            raise ValueError(f"task {task_id} has no persisted final solution")

        instance_path = str(record.get("instance_path", ""))
        instance_sha256 = str(record.get("instance_sha256", ""))
        cache_key = (instance_path, instance_sha256)
        instance = instance_cache.get(cache_key)
        if instance is None:
            instance = load_instance(instance_path)
            if instance.source_sha256 != instance_sha256:
                raise ValueError(f"replayed instance hash mismatch for task {task_id}")
            instance_cache[cache_key] = instance

        realization = str(diagnostics.get("final_schedule_realization", ""))
        if realization == "matheuristic_explicit":
            def explicit_job_times(name: str) -> tuple[dict[int, float], dict[int, float]]:
                raw_batches = diagnostics.get(name)
                if not isinstance(raw_batches, list) or not raw_batches:
                    raise ValueError(f"task {task_id} has no {name}")
                starts: dict[int, float] = {}
                completions: dict[int, float] = {}
                for batch_index, batch in enumerate(raw_batches):
                    if not isinstance(batch, Mapping):
                        raise ValueError(f"task {task_id} has an invalid {name} batch")
                    job_ids = batch.get("job_ids")
                    if not isinstance(job_ids, list):
                        raise ValueError(f"task {task_id} has an invalid {name}.job_ids")
                    context = f"task {task_id} {name}[{batch_index}]"
                    start = _finite_float(batch.get("start"), "start", context)
                    end = _finite_float(batch.get("end"), "end", context)
                    if end < start - 1e-9:
                        raise ValueError(f"{context} ends before it starts")
                    for raw_job_id in job_ids:
                        job_id = int(raw_job_id)
                        if job_id in starts:
                            raise ValueError(f"task {task_id} repeats job {job_id} in {name}")
                        starts[job_id] = start
                        completions[job_id] = end
                return starts, completions

            starts1, completions1 = explicit_job_times("explicit_batches_stage1")
            starts2, completions2 = explicit_job_times("explicit_batches_stage2")
            replayed_objective = sum(completions2.values())
        else:
            solution = BatchStructureSolution.from_dict(solution_payload)
            architecture = diagnostics.get("architecture")
            dispatch_rule = (
                str(architecture.get("stage2_dispatch", ""))
                if isinstance(architecture, Mapping)
                else ""
            )
            strategic_idle = "strategic_idle" in dispatch_rule
            schedule = decode_structure(
                instance,
                solution,
                algorithm=label,
                strategic_idle_best_response=strategic_idle,
            )
            starts1 = {
                job_id: batch.start
                for batch in schedule.batches_stage1
                for job_id in batch.job_ids
            }
            starts2 = {
                job_id: batch.start
                for batch in schedule.batches_stage2
                for job_id in batch.job_ids
            }
            completions1 = {
                job_id: float(schedule.completion_stage1[job_id - 1])
                for job_id in instance.jobs_by_id
            }
            replayed_objective = schedule.objective
        if replayed_objective != int(record.get("objective", -1)):
            raise ValueError(f"replayed objective mismatch for task {task_id}")

        if set(starts1) != set(instance.jobs_by_id) or set(starts2) != set(instance.jobs_by_id):
            raise ValueError(f"replayed schedule does not cover every job for task {task_id}")
        if set(completions1) != set(instance.jobs_by_id):
            raise ValueError(f"replayed stage-1 completions do not cover every job for task {task_id}")

        task_intervals: list[tuple[float, float, float, float]] = []
        for job in instance.jobs:
            completion1 = completions1[job.job_id]
            task_intervals.append(
                (
                    float(job.release),
                    float(starts1[job.job_id]),
                    float(completion1),
                    float(starts2[job.job_id]),
                )
            )
        moments = record.get("metric_moments")
        waiting1 = moments.get("waiting_stage1") if isinstance(moments, Mapping) else None
        waiting2 = moments.get("waiting_stage2") if isinstance(moments, Mapping) else None
        expected1 = (
            _finite_float(waiting1.get("sum"), "waiting_stage1.sum", task_id)
            if isinstance(waiting1, Mapping)
            else math.nan
        )
        expected2 = (
            _finite_float(waiting2.get("sum"), "waiting_stage2.sum", task_id)
            if isinstance(waiting2, Mapping)
            else math.nan
        )
        actual1 = sum(start1 - release for release, start1, _, _ in task_intervals)
        actual2 = sum(start2 - completion1 for _, _, completion1, start2 in task_intervals)
        if not math.isclose(actual1, expected1, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"replayed stage-1 waiting mismatch for task {task_id}")
        if not math.isclose(actual2, expected2, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"replayed stage-2 waiting mismatch for task {task_id}")
        intervals[label].setdefault(seed, []).extend(task_intervals)
    return intervals


def _queue_profiles(
    kraul: Path,
    three: Path,
    seed_table: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int],
    day_count: int,
    job_count: int,
    day_length_minutes: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    left = _queue_intervals_from_csv(kraul, QUEUE_METHODS_KRAUL)
    try:
        right = _queue_intervals_from_csv(three, QUEUE_METHODS_THREE)
        three_source = "tables/job_level_schedules.csv"
    except FileNotFoundError:
        right = _three_algorithm_queue_intervals(three)
        three_source = "raw/results.jsonl final-solution replay"
    intervals = left | right
    if set(intervals) != set(METHOD_ORDER):
        raise ValueError("queue timing sources do not cover exactly the five merged methods")

    bin_minutes = 60
    bins = day_length_minutes // bin_minutes
    table_by_method_seed = {
        (str(row["method"]), _seed_value(row.get("random_seed"))): row
        for row in seed_table
    }
    seed_curves_by_stage: dict[
        int,
        dict[str, dict[int | None, list[float]]],
    ] = {1: {}, 2: {}}
    diagnostics: dict[str, object] = {}
    maximum_conservation_error = 0.0
    maximum_table_error = 0.0
    for method in METHOD_ORDER:
        expected_method_seeds: tuple[int | None, ...] = (
            (None,) if method == "FFCBSPT" else tuple(int(seed) for seed in seeds)
        )
        observed_method_seeds = set(intervals[method])
        if observed_method_seeds != set(expected_method_seeds):
            raise ValueError(
                f"{method} queue timings contain seeds {sorted(map(str, observed_method_seeds))}, "
                f"expected {sorted(map(str, expected_method_seeds))}"
            )
        method_diagnostics: dict[str, object] = {
            "stochastic_seed_count": 0 if method == "FFCBSPT" else len(seeds),
            "timing_rows_per_seed": job_count,
            "seeds": [seed for seed in expected_method_seeds if seed is not None],
        }
        for stage in (1, 2):
            seed_curves_by_stage[stage][method] = {}
            seed_diagnostics: dict[str, object] = {}
            for seed in expected_method_seeds:
                method_intervals = intervals[method][seed]
                if len(method_intervals) != job_count:
                    raise ValueError(
                        f"{method}/seed={seed} has {len(method_intervals)} queue timing rows, "
                        f"expected {job_count}"
                    )
                hourly = [0.0] * bins
                total_waiting = 0.0
                for release, start1, completion1, start2 in method_intervals:
                    entry, exit_time = (
                        (release, start1) if stage == 1 else (completion1, start2)
                    )
                    contributions = _interval_minutes_by_hour(
                        entry,
                        exit_time,
                        day_length_minutes=day_length_minutes,
                        bin_minutes=bin_minutes,
                    )
                    hourly = [
                        left_value + right_value
                        for left_value, right_value in zip(hourly, contributions)
                    ]
                    total_waiting += exit_time - entry
                conservation_error = abs(sum(hourly) - total_waiting)
                maximum_conservation_error = max(
                    maximum_conservation_error,
                    conservation_error,
                )
                if conservation_error > 1e-7:
                    raise ValueError(
                        f"{method}/seed={seed} stage-{stage} queue area does not "
                        "conserve waiting time"
                    )

                aggregate = table_by_method_seed.get((method, seed))
                if aggregate is None:
                    raise ValueError(f"missing seed-level table row for {method}/seed={seed}")
                aggregate_mean = _finite_float(
                    aggregate.get(f"waiting_stage{stage}_mean_min"),
                    f"waiting_stage{stage}_mean_min",
                    f"{method}/seed={seed}",
                )
                table_error = abs(total_waiting - aggregate_mean * job_count)
                maximum_table_error = max(maximum_table_error, table_error)
                if table_error > 1e-6:
                    raise ValueError(
                        f"{method}/seed={seed} stage-{stage} queue area disagrees "
                        "with its seed-level waiting time"
                    )
                averages = [
                    value / (day_count * bin_minutes)
                    for value in hourly
                ]
                seed_curves_by_stage[stage][method][seed] = averages
                peak_hour = max(range(bins), key=lambda hour: averages[hour])
                seed_diagnostics[str(seed)] = {
                    "total_waiting_job_minutes": total_waiting,
                    "mean_waiting_minutes_per_job": total_waiting / job_count,
                    "full_day_average_queue_jobs": total_waiting
                    / (day_count * day_length_minutes),
                    "peak_hour": peak_hour,
                    "peak_average_queue_jobs": averages[peak_hour],
                    "waiting_time_conservation_error_minutes": conservation_error,
                    "table_waiting_sum_error_minutes": table_error,
                }
            method_diagnostics[f"stage{stage}_by_seed"] = seed_diagnostics
        diagnostics[method] = method_diagnostics

    rows_by_stage: dict[int, list[dict[str, object]]] = {1: [], 2: []}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for stage in (1, 2):
        for hour in range(bins):
            row: dict[str, object] = {
                "hour": hour,
                "hour_interval": f"[{hour:02d}:00,{(hour + 1) % bins:02d}:00)",
                "day_count": day_count,
                "bin_minutes": bin_minutes,
            }
            for method in METHOD_ORDER:
                seed_values = [
                    curve[hour]
                    for curve in seed_curves_by_stage[stage][method].values()
                ]
                mean, seed_std, ci_low, ci_high = _bootstrap_mean_ci(seed_values, rng)
                row[f"{method}_average_queue_jobs"] = mean
                row[f"{method}_average_queue_jobs_mean"] = mean
                row[f"{method}_average_queue_jobs_seed_std"] = seed_std
                row[f"{method}_average_queue_jobs_ci95_low"] = ci_low
                row[f"{method}_average_queue_jobs_ci95_high"] = ci_high
                row[f"{method}_queue_job_minutes_total_mean"] = (
                    mean * day_count * bin_minutes
                )
            rows_by_stage[stage].append(row)
    metadata: dict[str, object] = {
        "definition": {
            "stage1": "time average of I[release_j <= t < start_stage1_j]",
            "stage2": "time average of I[completion_stage1_j <= t < start_stage2_j]",
        },
        "aggregation": (
            "for each seed, waiting-interval overlap minutes in each clock-hour bin "
            "are divided by day_count * 60; the plotted line is the arithmetic mean "
            "of complete three-month seed curves; times after midnight are folded "
            "modulo day length"
        ),
        "uncertainty": {
            "unit": "independent complete three-month random-seed run",
            "method": "percentile bootstrap confidence interval of the seed mean",
            "confidence_level": 0.95,
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "deterministic_ffcbspt_interval": "not applicable; bounds equal the point estimate",
        },
        "timing_sources": {
            "ffcbspt_kraul_ga": "tables/job_level_schedules.csv",
            "ibscabc_alns_ts": three_source,
        },
        "maximum_waiting_time_conservation_error_minutes": maximum_conservation_error,
        "maximum_table_waiting_sum_error_minutes": maximum_table_error,
        "methods": diagnostics,
    }
    return rows_by_stage[1], rows_by_stage[2], metadata


def _run_settings(root: Path) -> tuple[int, int, int, int, int]:
    settings: set[tuple[int, int, int, int, int]] = set()
    with (root / "raw" / "results.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping) or record.get("status") == "error":
                continue
            try:
                settings.add(
                    (
                        int(record["machines_stage1"]),
                        int(record["machines_stage2"]),
                        int(record["capacity_stage1"]),
                        int(record["capacity_stage2"]),
                        int(record["day_length_minutes"]),
                    )
                )
            except KeyError as error:
                raise ValueError(f"result file lacks equipment metadata: {root}") from error
    if len(settings) != 1:
        raise ValueError(f"result file has inconsistent equipment settings: {root}")
    return next(iter(settings))


def _validate_pair(kraul: Path, three: Path) -> tuple[dict[str, object], dict[str, object]]:
    kraul_integrity = _read_json(kraul / "report" / "integrity.json")
    three_integrity = _read_json(three / "report" / "integrity.json")
    if kraul_integrity.get("status") != "PASS":
        raise ValueError("FFCBSPT/Kraul-GA report has not passed integrity checks")
    if three_integrity.get("status") != "PASS":
        raise ValueError("IBS-CABC/ALNS/TS report has not passed integrity checks")
    kraul_manifest = _read_json(kraul / "manifest.json")
    three_manifest = _read_json(three / "manifest.json")
    left = kraul_manifest.get("audit", {})
    right = three_manifest.get("audit", {})
    for field in ("dataset_sha256", "total_jobs"):
        if left.get(field) != right.get(field):
            raise ValueError(f"experiments do not share the same {field}")
    if _audit_days(left) != _audit_days(right):
        raise ValueError("experiments do not share the same day count")
    left_sequence = [
        (item.get("relative_path"), item.get("instance_sha256"), item.get("jobs"))
        for item in _audit_sequence(left)
    ]
    right_sequence = [
        (item.get("relative_path"), item.get("instance_sha256"), item.get("jobs"))
        for item in _audit_sequence(right)
    ]
    if left_sequence != right_sequence:
        raise ValueError("experiments do not use the same ordered DBD instance sequence")
    left_experiment = kraul_manifest.get("experiment", {})
    right_experiment = three_manifest.get("experiment", {})
    for field in (
        "contract",
        "expected_days",
        "expected_jobs",
        "first_day_seed",
        "last_day_seed",
        "machines_stage1",
        "machines_stage2",
        "day_length_minutes",
        "seeds",
        "workers",
        "protocol",
    ):
        if left_experiment.get(field) != right_experiment.get(field):
            raise ValueError(f"experiments differ in {field}")
    if _run_settings(kraul) != _run_settings(three):
        raise ValueError("experiments differ in machine, capacity, or day-length settings")
    return kraul_manifest, three_manifest


def _combined_table(
    kraul: Path,
    three: Path,
    manifest: Mapping[str, object],
) -> list[dict[str, str]]:
    left_summary = kraul / "tables" / "table_9_ffcbspt_kraul_ga_seed_summary.csv"
    right_summary = three / "tables" / "table_9_dbd_three_algorithms_seed_summary.csv"
    left = _read_csv(
        left_summary
        if left_summary.is_file()
        else kraul / "tables" / "table_4_9_ffcbspt_kraul_ga.csv"
    )
    right = _read_csv(
        right_summary
        if right_summary.is_file()
        else three / "tables" / "table_4_9_dbd_three_algorithms.csv"
    )
    rows = left + right
    methods = [row.get("method", "") for row in rows]
    duplicates = sorted({method for method in methods if methods.count(method) > 1})
    if duplicates:
        raise ValueError(f"duplicate aggregate method rows: {', '.join(duplicates)}")
    by_method = {row["method"]: row for row in rows}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError(f"expected methods {METHOD_ORDER}, observed {tuple(by_method)}")
    experiment = manifest.get("experiment", {})
    audit = manifest.get("audit", {})
    expected_days = _audit_days(audit)
    expected_jobs = int(audit.get("total_jobs", -1))
    seed_values = experiment.get("seeds", [])
    if not isinstance(seed_values, list) or not seed_values:
        raise ValueError("five-algorithm merge requires a non-empty shared random-seed set")
    expected_seeds = tuple(int(seed) for seed in seed_values)
    expected_seed_text = ";".join(str(seed) for seed in expected_seeds)
    for method, row in by_method.items():
        if row.get("information_mode") != "DBD":
            raise ValueError(f"{method} aggregate row is not DBD")
        if int(row.get("day_count", -1)) != expected_days:
            raise ValueError(f"{method} aggregate row has the wrong day count")
        if int(row.get("device_count", -1)) != expected_jobs:
            raise ValueError(f"{method} aggregate row has the wrong device count")
        if "seed_count" in row:
            expected_count = 0 if method == "FFCBSPT" else len(expected_seeds)
            if int(row.get("seed_count", -1)) != expected_count:
                raise ValueError(f"{method} seed-summary row has the wrong seed count")
            expected_text = "" if method == "FFCBSPT" else expected_seed_text
            if row.get("random_seeds", "") != expected_text:
                raise ValueError(f"{method} seed-summary row has the wrong seed set")
        else:
            if len(expected_seeds) != 1:
                raise ValueError(
                    f"{method} lacks a multiseed summary row for {len(expected_seeds)} seeds"
                )
            expected_method_seed = "" if method == "FFCBSPT" else str(expected_seeds[0])
            if row.get("random_seed", "") != expected_method_seed:
                raise ValueError(f"{method} aggregate row has the wrong random seed")
    return [by_method[method] for method in METHOD_ORDER]


def _combined_seed_table(
    kraul: Path,
    three: Path,
    manifest: Mapping[str, object],
) -> list[dict[str, str]]:
    """Return one complete three-month aggregate row per method and seed."""

    rows = _read_csv(kraul / "tables" / "table_4_9_ffcbspt_kraul_ga.csv") + _read_csv(
        three / "tables" / "table_4_9_dbd_three_algorithms.csv"
    )
    experiment = manifest.get("experiment", {})
    audit = manifest.get("audit", {})
    seed_values = experiment.get("seeds", [])
    if not isinstance(seed_values, list) or not seed_values:
        raise ValueError("experiment has no random seeds")
    seeds = tuple(int(seed) for seed in seed_values)
    expected_days = _audit_days(audit)
    expected_jobs = int(audit.get("total_jobs", -1))
    by_key: dict[tuple[str, int | None], dict[str, str]] = {}
    for row in rows:
        method = row.get("method", "")
        if method not in METHOD_ORDER:
            raise ValueError(f"unexpected method {method!r} in seed-level table")
        seed = _seed_value(row.get("random_seed"))
        key = (method, seed)
        if key in by_key:
            raise ValueError(f"duplicate seed-level aggregate row for {method}/seed={seed}")
        if row.get("information_mode") != "DBD":
            raise ValueError(f"{method}/seed={seed} aggregate row is not DBD")
        if int(row.get("day_count", -1)) != expected_days:
            raise ValueError(f"{method}/seed={seed} aggregate row has the wrong day count")
        if int(row.get("device_count", -1)) != expected_jobs:
            raise ValueError(f"{method}/seed={seed} aggregate row has the wrong device count")
        by_key[key] = row

    expected_keys = {("FFCBSPT", None)} | {
        (method, seed)
        for method in METHOD_ORDER
        if method != "FFCBSPT"
        for seed in seeds
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key), key=str)
        extra = sorted(set(by_key) - expected_keys, key=str)
        raise ValueError(
            f"seed-level aggregate rows do not match the experiment; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    return [
        by_key[(method, seed)]
        for method in METHOD_ORDER
        for seed in ((None,) if method == "FFCBSPT" else seeds)
    ]


def _combined_hourly(
    kraul: Path,
    three: Path,
    stage: int,
    *,
    day_count: int,
    job_count: int,
    hour_bins: int,
) -> list[dict[str, object]]:
    filename = f"hourly_stage{stage}_dbd.csv"
    left = _read_csv(kraul / "tables" / filename)
    right = _read_csv(three / "tables" / filename)
    if len(left) != hour_bins or len(right) != hour_bins:
        raise ValueError(f"both {filename} files must have {hour_bins} rows")
    rows: list[dict[str, object]] = []
    for left_row, right_row in zip(left, right):
        if left_row["hour"] != right_row["hour"]:
            raise ValueError(f"hour mismatch in {filename}")
        if not math.isclose(
            float(left_row["Arrival_total"]),
            float(right_row["Arrival_total"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"arrival series mismatch in {filename}")
        row: dict[str, object] = {
            "hour": int(left_row["hour"]),
            "Arrival_total": float(left_row["Arrival_total"]),
            "Arrival_average_per_day": float(left_row["Arrival_average_per_day"]),
        }
        for method in METHOD_ORDER:
            source = left_row if f"{method}_total" in left_row else right_row
            row[f"{method}_total"] = float(source[f"{method}_total"])
            row[f"{method}_average_per_day"] = float(source[f"{method}_average_per_day"])
        rows.append(row)
    if [int(row["hour"]) for row in rows] != list(range(hour_bins)):
        raise ValueError(f"{filename} does not contain contiguous hourly bins")
    for method in ("Arrival", *METHOD_ORDER):
        total = sum(float(row[f"{method}_total"]) for row in rows)
        if not math.isclose(total, job_count, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"{method} totals do not cover all jobs in {filename}")
        for row in rows:
            if not math.isclose(
                float(row[f"{method}_average_per_day"]),
                float(row[f"{method}_total"]) / day_count,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{method} daily average is inconsistent in {filename}")
    return rows


def _plot(rows: Sequence[Mapping[str, object]], stage: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 5.3))
    hours = [int(row["hour"]) for row in rows]
    for method in ("Arrival", *METHOD_ORDER):
        key = f"{method}_average_per_day"
        axis.plot(
            hours,
            [float(row[key]) for row in rows],
            label=method,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=3.8,
            linewidth=1.35,
        )
    axis.set_xticks(hours)
    axis.set_xlim(-0.4, 23.4)
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Average devices per day")
    axis.set_title(f"Three-month DBD arrivals and stage {stage} completions")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=240)
    plt.close(figure)


def _plot_queue(rows: Sequence[Mapping[str, object]], stage: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 5.3))
    hours = [int(row["hour"]) for row in rows]
    line_styles = {
        "FFCBSPT": "--",
        "ALNS": "-",
        "TS": "-.",
        "Kraul-GA": ":",
        "IBS-CABC": "-",
    }
    for method in METHOD_ORDER:
        mean_key = f"{method}_average_queue_jobs_mean"
        low_key = f"{method}_average_queue_jobs_ci95_low"
        high_key = f"{method}_average_queue_jobs_ci95_high"
        focal = method == "IBS-CABC"
        means = [float(row[mean_key]) for row in rows]
        lower = [float(row[low_key]) for row in rows]
        upper = [float(row[high_key]) for row in rows]
        if method != "FFCBSPT":
            axis.fill_between(
                hours,
                lower,
                upper,
                color=COLORS[method],
                alpha=0.12 if focal else 0.08,
                linewidth=0.0,
                zorder=1,
            )
        axis.plot(
            hours,
            means,
            label=method,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=line_styles[method],
            markersize=4.0 if focal else 3.5,
            linewidth=2.2 if focal else 1.35,
            zorder=5 if focal else 2,
        )
    axis.set_xticks(hours)
    axis.set_xlim(-0.4, 23.4)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Average queued jobs")
    axis.set_title(f"Three-month DBD average queue before stage {stage}")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=240)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def merge_reports(
    kraul_experiment: str | Path,
    three_algorithm_experiment: str | Path,
    output_directory: str | Path,
) -> Path:
    kraul = Path(kraul_experiment).resolve()
    three = Path(three_algorithm_experiment).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    kraul_manifest, three_manifest = _validate_pair(kraul, three)
    table = _combined_table(kraul, three, kraul_manifest)
    seed_table = _combined_seed_table(kraul, three, kraul_manifest)
    days = _audit_days(kraul_manifest["audit"])
    jobs = int(kraul_manifest["audit"]["total_jobs"])
    seeds = tuple(int(seed) for seed in kraul_manifest["experiment"]["seeds"])
    hour_bins = int(kraul_manifest["experiment"]["day_length_minutes"]) // 60
    stage1 = _combined_hourly(
        kraul, three, 1, day_count=days, job_count=jobs, hour_bins=hour_bins
    )
    stage2 = _combined_hourly(
        kraul, three, 2, day_count=days, job_count=jobs, hour_bins=hour_bins
    )
    queue_stage1, queue_stage2, queue_metadata = _queue_profiles(
        kraul,
        three,
        seed_table,
        seeds=seeds,
        day_count=days,
        job_count=jobs,
        day_length_minutes=int(kraul_manifest["experiment"]["day_length_minutes"]),
    )
    _write_csv(output / "table_4_9_dbd_five_algorithms.csv", table)
    _write_csv(output / "table_9_dbd_five_algorithms_10seeds.csv", table)
    _write_csv(output / "hourly_stage1_dbd_five_algorithms.csv", stage1)
    _write_csv(output / "hourly_stage2_dbd_five_algorithms.csv", stage2)
    _write_csv(
        output / "hourly_average_queue_stage1_dbd_five_algorithms.csv",
        queue_stage1,
    )
    _write_csv(
        output / "hourly_average_queue_stage2_dbd_five_algorithms.csv",
        queue_stage2,
    )
    _plot(stage1, 1, output / "figure_4_7_dbd_five_algorithms_stage1.png")
    _plot(stage2, 2, output / "figure_4_8_dbd_five_algorithms_stage2.png")
    _plot_queue(
        queue_stage1,
        1,
        output / "figure_dbd_five_algorithms_average_queue_stage1.png",
    )
    _plot_queue(
        queue_stage2,
        2,
        output / "figure_dbd_five_algorithms_average_queue_stage2.png",
    )
    provenance = {
        "status": "PASS",
        "kraul_experiment": str(kraul),
        "three_algorithm_experiment": str(three),
        "dataset_sha256": kraul_manifest["audit"]["dataset_sha256"],
        "days": days,
        "jobs": jobs,
        "methods": list(METHOD_ORDER),
        "seed_set": list(seeds),
        "stochastic_seed_count": len(seeds),
        "table_standard_deviation_unit": (
            "independent complete three-month random-seed run"
        ),
        "information_mode": "DBD",
        "source_run_contracts": {
            "ffcbspt_kraul_ga": _contract_reference(kraul_manifest),
            "ibscabc_alns_ts": _contract_reference(three_manifest),
        },
        "average_queue_curves": queue_metadata,
    }
    (output / "merge_integrity.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 五算法 DBD 合并结果",
        "",
        "两组实验已通过数据哈希、逐日实例顺序、设备数、作业数、随机种子和完整性状态检查。",
        "表中随机算法的括号内数值为10个完整三个月随机种子运行之间的样本标准差；"
        "确定性规则FFCBSPT不计算跨种子标准差。",
        "",
        "| 方法 | 系统停留时间/min | 阶段1等待/min | 阶段2等待/min | 阶段1利用率 | 阶段2利用率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['method']} (DBD) | {row['time_in_system_paper']} | "
            f"{row['waiting_stage1_paper']} | {row['waiting_stage2_paper']} | "
            f"{row['stage1_utilization_paper']} | {row['stage2_utilization_paper']} |"
        )
    lines.extend(
        [
            "",
            "## 平均排队作业数曲线",
            "",
            "阶段1按 `release <= t < start_stage1` 统计，阶段2按 "
            "`completion_stage1 <= t < start_stage2` 统计。每个点是对应钟点一小时内的"
            "时间平均排队作业数。中心线为10个完整三个月随机种子运行的算术均值，"
            "阴影为对种子均值实施10,000次百分位Bootstrap得到的95%置信区间；"
            "FFCBSPT为确定性规则，因此仅绘制单条曲线。跨午夜等待按一天中的小时取模，"
            "曲线面积已与种子级等待时间总和逐算法、逐阶段核对。",
            "",
            "数据文件：`hourly_average_queue_stage1_dbd_five_algorithms.csv`、"
            "`hourly_average_queue_stage2_dbd_five_algorithms.csv`。",
        ]
    )
    summary = output / "five_algorithm_summary_zh.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge the two isolated DBD experiment reports")
    parser.add_argument("--kraul-experiment", required=True)
    parser.add_argument("--three-experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(merge_reports(args.kraul_experiment, args.three_experiment, args.output))


if __name__ == "__main__":
    main()
