from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ["FFCBSPT", "ALNS", "TS", "Kraul-GA", "IBS-CABC"]
METRICS = ["time_in_system", "waiting_stage1", "waiting_stage2"]


records = []
for source in map(Path, sys.argv[1:]):
    with source.open(encoding="utf-8-sig") as stream:
        records.extend(json.loads(line) for line in stream if line.strip())

task_ids = [row["task_id"] for row in records]
groups = defaultdict(list)
hashes_by_day = defaultdict(set)
for row in records:
    method = row["algorithm_label"]
    seed = None if method == "FFCBSPT" else int(row["seed"])
    groups[(method, seed)].append(row)
    hashes_by_day[int(row["case_day_index"])].add(row["instance_sha256"])

print(
    f"records={len(records)} unique_task_ids={len(set(task_ids))} "
    f"duplicate_task_ids={len(task_ids)-len(set(task_ids))} "
    f"infeasible={sum(not bool(row.get('feasible', False)) for row in records)}"
)
print(
    f"days={len(hashes_by_day)} days_with_single_shared_instance_hash="
    f"{sum(len(value) == 1 for value in hashes_by_day.values())}"
)

seed_rows = []
max_tis_identity_error = 0.0
max_util_identity_error = 0.0
for method in METHODS:
    seeds = [None] if method == "FFCBSPT" else list(range(1, 11))
    for seed in seeds:
        members = groups[(method, seed)]
        assert len(members) == 91
        assert {int(row["case_day_index"]) for row in members} == set(range(91))
        jobs = sum(int(row["jobs"]) for row in members)
        assert jobs == 22_927
        output = {"method": method, "seed": seed}
        for metric in METRICS:
            total = 0.0
            count = 0
            for row in members:
                moment = row.get("metric_moments", row)[metric]
                total += float(moment["sum"])
                count += int(moment["count"])
            assert count == jobs
            output[metric] = total / jobs
        tis_from_objective = sum(float(row["objective"]) - float(row["release_sum"]) for row in members) / jobs
        max_tis_identity_error = max(max_tis_identity_error, abs(output["time_in_system"] - tis_from_objective))
        for stage in (1, 2):
            daily = []
            for row in members:
                denominator = float(row[f"machines_stage{stage}"]) * float(row[f"stage{stage}_active_horizon_minutes"])
                derived = float(row[f"stage{stage}_processing_minutes"]) / denominator if denominator else 0.0
                max_util_identity_error = max(max_util_identity_error, abs(derived - float(row[f"utilization_stage{stage}"])))
                daily.append(float(row[f"utilization_stage{stage}"]) * 100.0)
            output[f"utilization_stage{stage}"] = float(np.mean(daily))
        seed_rows.append(output)

print(f"complete_groups={len(seed_rows)} max_tis_identity_error={max_tis_identity_error:.3g} max_util_identity_error={max_util_identity_error:.3g}")
print("method,n,TIS mean(sd),W1 mean(sd),W2 mean(sd),U1 mean(sd),U2 mean(sd)")
for method in METHODS:
    members = [row for row in seed_rows if row["method"] == method]
    values = []
    for field in ["time_in_system", "waiting_stage1", "waiting_stage2", "utilization_stage1", "utilization_stage2"]:
        array = np.array([row[field] for row in members], dtype=float)
        sd = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
        values.append((float(array.mean()), sd))
    rendered = ",".join(f"{mean:.12f}({sd:.12f})" if np.isfinite(sd) else f"{mean:.12f}(—)" for mean, sd in values)
    print(f"{method},{len(members)},{rendered}")
