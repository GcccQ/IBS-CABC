from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from cssd_ibscabc.algorithms.assignment_repair import (
    family_decomposed_assignment_repair,
)
from cssd_ibscabc.algorithms.common import FoodSource
from cssd_ibscabc.core.feasibility import validate_schedule
from cssd_ibscabc.decoder.integrated import decode_structure
from cssd_ibscabc.io import load_instance
from cssd_ibscabc.representation.batch_structure import BatchStructureSolution


def _repair(
    record: dict[str, object],
    workers: int,
    seconds_per_family: float,
    iterations: int,
) -> dict[str, object]:
    instance = load_instance(str(record["instance_path"]))
    diagnostics = dict(record["diagnostics"])
    if "solution" not in diagnostics:
        raise ValueError("IBS-CABC record has no serialized batch structure")
    solution = BatchStructureSolution.from_dict(dict(diagnostics["solution"]))
    schedule = decode_structure(instance, solution, algorithm="IBS-CABC")
    recorded = int(record["objective"])
    if schedule.objective != recorded:
        raise ValueError(
            f"{instance.name} seed={record['seed']}: decoded objective "
            f"{schedule.objective} != recorded {recorded}"
        )
    result = family_decomposed_assignment_repair(
        instance,
        FoodSource(solution, schedule),
        seconds_per_family=seconds_per_family,
        workers=workers,
        iterations=iterations,
        joint_max_jobs=600,
        seed=int(record["seed"]),
    )
    validate_schedule(instance, result.source.schedule)
    return {
        "instance": instance.name,
        "instance_path": str(instance.source),
        "seed": int(record["seed"]),
        "before": recorded,
        "after": result.source.objective,
        "gain_from_repair_percent": (
            100.0 * (recorded - result.source.objective) / recorded
        ),
        "model_calls": result.model_calls,
        "feasible_models": result.feasible_models,
        "optimal_models": result.optimal_models,
        "accepted_iterations": result.accepted_iterations,
        "runtime_seconds": result.runtime_seconds,
        "method_sequence": list(result.method_sequence),
        "solution": result.source.solution.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--solver-workers", type=int, default=2)
    parser.add_argument("--seconds-per-family", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = set(args.instance)
    if selected:
        records = [
            record for record in records if str(record["instance"]) in selected
        ]
    selected_seeds = set(args.seed)
    if selected_seeds:
        records = [
            record
            for record in records
            if int(record["seed"]) in selected_seeds
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("w", encoding="utf-8") as handle,
        ProcessPoolExecutor(max_workers=max(1, args.processes)) as pool,
    ):
        futures = {
            pool.submit(
                _repair,
                record,
                max(1, args.solver_workers),
                args.seconds_per_family,
                args.iterations,
            ): record
            for record in records
        }
        with tqdm(total=len(futures), desc="修复重放", unit="任务") as bar:
            for future in as_completed(futures):
                repaired = future.result()
                handle.write(
                    json.dumps(repaired, ensure_ascii=False) + "\n"
                )
                handle.flush()
                bar.set_postfix(
                    instance=repaired["instance"],
                    seed=repaired["seed"],
                )
                bar.update(1)


if __name__ == "__main__":
    main()
