from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import numpy as np

from .decoder import Batch, DecodedSchedule, validate_schedule
from .model import AuthorInstance


def _number(value: float) -> int | float:
    rounded = round(float(value))
    return (
        int(rounded)
        if math.isclose(float(value), rounded, rel_tol=0.0, abs_tol=1e-9)
        else float(value)
    )


def moment(values: np.ndarray) -> dict[str, int | float]:
    data = np.asarray(values, dtype=np.float64)
    count = int(data.size)
    total = float(np.sum(data))
    sum_squares = float(np.dot(data, data))
    mean = total / count if count else 0.0
    variance = max(0.0, (sum_squares - total * total / count) / (count - 1)) if count > 1 else 0.0
    return {
        "count": count,
        "sum": _number(total),
        "sum_squares": _number(sum_squares),
        "mean": mean,
        "sample_std": math.sqrt(variance),
    }


def _hourly(values: np.ndarray, bins: int) -> list[int]:
    hours = (np.floor(np.asarray(values, dtype=np.float64) / 60.0).astype(np.int64) % bins)
    return np.bincount(hours, minlength=bins).astype(int).tolist()


def _batch_payload(batch: Batch, scheduled_index: int) -> dict[str, object]:
    payload = asdict(batch)
    payload["scheduled_batch_index"] = scheduled_index
    payload["job_ids"] = [index + 1 for index in batch.job_indices]
    del payload["job_indices"]
    for key in ("processing_time", "release_time", "load", "start_time", "completion_time"):
        payload[key] = _number(float(payload[key]))
    return payload


def schedule_metrics(
    instance: AuthorInstance,
    schedule: DecodedSchedule,
    *,
    day_length_minutes: int = 1440,
) -> dict[str, object]:
    validate_schedule(instance, schedule)
    tis = schedule.completion_stage2 - instance.release_times
    waiting_stage1 = schedule.start_stage1 - instance.release_times
    waiting_stage2 = schedule.start_stage2 - schedule.completion_stage1
    stage1_processing = float(sum(batch.processing_time for batch in schedule.stage1_batches))
    stage2_processing = float(sum(batch.processing_time for batch in schedule.stage2_batches))
    stage1_horizon = float(np.max(schedule.completion_stage1) - np.min(schedule.start_stage1))
    stage2_horizon = float(np.max(schedule.completion_stage2) - np.min(schedule.start_stage2))
    utilization_stage1 = (
        stage1_processing / (instance.machines_stage1 * stage1_horizon) if stage1_horizon else 0.0
    )
    utilization_stage2 = (
        stage2_processing / (instance.machines_stage2 * stage2_horizon) if stage2_horizon else 0.0
    )
    return {
        "objective": _number(schedule.objective),
        "release_sum": _number(float(np.sum(instance.release_times))),
        "time_in_system": moment(tis),
        "waiting_stage1": moment(waiting_stage1),
        "waiting_stage2": moment(waiting_stage2),
        "average_time_in_system": float(np.mean(tis)),
        "average_waiting_stage1": float(np.mean(waiting_stage1)),
        "average_waiting_stage2": float(np.mean(waiting_stage2)),
        "stage1_processing_minutes": _number(stage1_processing),
        "stage2_processing_minutes": _number(stage2_processing),
        "stage1_active_horizon_minutes": _number(stage1_horizon),
        "stage2_active_horizon_minutes": _number(stage2_horizon),
        "utilization_stage1": utilization_stage1,
        "utilization_stage2": utilization_stage2,
        "utilization_stage1_fixed_day": stage1_processing
        / (instance.machines_stage1 * day_length_minutes),
        "utilization_stage2_fixed_day": stage2_processing
        / (instance.machines_stage2 * day_length_minutes),
        "arrivals_hourly": _hourly(instance.release_times, day_length_minutes // 60),
        "completed_stage1_hourly": _hourly(
            schedule.completion_stage1, day_length_minutes // 60
        ),
        "completed_stage2_hourly": _hourly(
            schedule.completion_stage2, day_length_minutes // 60
        ),
    }


def schedule_payload(instance: AuthorInstance, schedule: DecodedSchedule) -> dict[str, object]:
    def values(array: Iterable[float]) -> list[int | float]:
        return [_number(float(value)) for value in array]

    return {
        "best_permutation_zero_based": schedule.permutation.astype(int).tolist(),
        "job_ids": instance.jobs[:, 0].astype(int).tolist(),
        "start_stage1": values(schedule.start_stage1),
        "completion_stage1": values(schedule.completion_stage1),
        "machine_stage1": schedule.machine_stage1.astype(int).tolist(),
        "batch_stage1": schedule.batch_stage1.astype(int).tolist(),
        "start_stage2": values(schedule.start_stage2),
        "completion_stage2": values(schedule.completion_stage2),
        "machine_stage2": schedule.machine_stage2.astype(int).tolist(),
        "batch_stage2": schedule.batch_stage2.astype(int).tolist(),
        "stage1_batches": [
            _batch_payload(batch, index)
            for index, batch in enumerate(schedule.stage1_batches, start=1)
        ],
        "stage2_batches": [
            _batch_payload(batch, index)
            for index, batch in enumerate(schedule.stage2_batches, start=1)
        ],
    }
