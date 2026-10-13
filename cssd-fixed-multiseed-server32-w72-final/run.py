from __future__ import annotations

import bootstrap
import argparse
import json
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from adapters import run_task
from contracts import METHODS, atomic_json, digest, load_checkpoints, load_config, prepare, read_json, save_checkpoint
from kraul_dbd_reproduction.runner import _run_lock, _AsyncPrinter, _terminate_pool


def utc():
    return datetime.now(timezone.utc).isoformat()


def dry_summary(manifest, tasks):
    workers = manifest["core"]["config"]["workers"]
    recycle_tasks = manifest["core"]["config"]["worker_recycle_tasks"]
    by_scale = []
    for scale in manifest["selected_scales"]:
        selected = [t for t in tasks if t["jobs"] == scale]
        by_scale.append({"scale": scale, "instances": len({t["instance_sha256"] for t in selected}), "repeats_per_instance_algorithm": len({t["seed"] for t in selected}), "algorithms": len(manifest["selected_algorithms"]), "tasks": len(selected), "budget_seconds_each": sorted({t["time_limit_seconds"] for t in selected})})
    total = sum(t["time_limit_seconds"] or 0 for t in tasks)
    return {
        "mode": manifest["core"]["config"]["mode"],
        "tasks": len(tasks),
        "by_scale": by_scale,
        "workers": workers,
        "worker_recycle_tasks": recycle_tasks,
        "pool_generation_task_limit": workers * recycle_tasks,
        "nominal_task_hours": total / 3600,
        "ideal_parallel_hours": total / workers / 3600,
        "no_optimization_started": True,
    }


def validate_resume(root, current_manifest, tasks):
    saved = read_json(root / "manifest.json")
    if saved["run_contract"] != current_manifest["run_contract"]:
        raise ValueError("Cannot resume: configuration, seeds, data, solver source, workers, environment or selection changed. Use the original command/settings; do not mix experiments.")
    saved_tasks = read_json(root / "task_manifest.json")
    if digest(saved_tasks) != current_manifest["task_manifest_hash"] or saved_tasks != tasks:
        raise ValueError("Cannot resume: task manifest mismatch")
    return load_checkpoints(root, tasks)


def execute(manifest, tasks, root, resume=False, stop_after=None):
    if resume:
        if not (root / "manifest.json").is_file():
            raise ValueError("resume directory has no manifest")
    else:
        if root.exists():
            raise ValueError(f"Output already exists. Continue with --resume {root}; no data were overwritten.")
        root.mkdir(parents=True)
    with _run_lock(root):
        if resume:
            successes = set(validate_resume(root, manifest, tasks))
        else:
            atomic_json(root / "task_manifest.json", tasks)
            atomic_json(root / "manifest.json", {**manifest, "created_utc": utc()})
            successes = set()
        pending = [t for t in tasks if t["task_id"] not in successes]
        if stop_after is not None:
            pending = pending[:stop_after]
        selected_count = len(pending)
        iterator = iter(pending)
        printer = _AsyncPrinter()
        start = time.perf_counter()
        old_seconds = 0.0
        if resume and (root / "progress.json").exists():
            old_seconds = float(read_json(root / "progress.json").get("elapsed_seconds", 0))
        active, errors = {}, []
        workers = manifest["core"]["config"]["workers"]
        recycle_tasks = manifest["core"]["config"]["worker_recycle_tasks"]
        def progress(status, emit=True):
            completed = len(successes)
            elapsed = old_seconds + time.perf_counter() - start
            payload = {"status": status, "total": len(tasks), "completed": completed, "pending": len(tasks) - completed, "failed_this_invocation": len(errors), "active_tasks": [t["task_id"] for t in active.values()], "elapsed_seconds": elapsed, "updated_utc": utc(), "output": str(root)}
            atomic_json(root / "progress.json", payload)
            if emit:
                done = int(24 * completed / len(tasks))
                bar = "#" * done + "." * (24 - done)
                names = ", ".join(payload["active_tasks"][:2])
                printer.emit(f"[{bar}] {completed}/{len(tasks)} ({completed/len(tasks):.1%}), failed={len(errors)}, elapsed={elapsed/60:.1f} min, active=[{names}{', ...' if len(active)>2 else ''}]")
        printer.emit(f"[run] {root}")
        printer.emit(f"[resume] retained {len(successes)} validated checkpoints; this invocation schedules {selected_count} tasks")
        printer.emit(f"[execution] workers={workers}, one internal thread per task, pool_generation_limit={workers * recycle_tasks} tasks (target {recycle_tasks} tasks/worker)")
        progress("running")
        pool = None
        state = "failed"
        try:
            if pending:
                last_print = time.perf_counter()
                generation = 0
                generation_limit = workers * recycle_tasks
                while True:
                    batch = list(islice(iterator, generation_limit))
                    if not batch:
                        break
                    generation += 1
                    batch_iterator = iter(batch)

                    def submit(pool):
                        task = next(batch_iterator, None)
                        if task is not None:
                            active[pool.submit(run_task, task)] = task

                    pool = ProcessPoolExecutor(max_workers=workers)
                    printer.emit(f"[pool] generation={generation}, tasks={len(batch)}, remaining_after_generation={selected_count - min(selected_count, generation * generation_limit)}")
                    for _ in range(min(workers, len(batch))):
                        submit(pool)
                    while active:
                        finished, _ = wait(active, timeout=1.0, return_when=FIRST_COMPLETED)
                        for future in finished:
                            task = active.pop(future)
                            try:
                                result = future.result()
                                save_checkpoint(root, result, task)
                                successes.add(task["task_id"])
                                printer.emit(f"[{len(successes)}/{len(tasks)}] {task['algorithm_label']} J{task['jobs']} seed={task['seed']} instance={task['instance']} objective={result['objective']}")
                            except Exception as error:
                                errors.append(task["task_id"])
                                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                                atomic_json(root / "failures" / f"{task['task_id']}_{stamp}.json", {"task": task, "error": repr(error), "traceback": traceback.format_exc(), "utc": utc()})
                                printer.emit(f"[failed] {task['task_id']}: {error}")
                            submit(pool)
                        if finished:
                            progress("running", emit=False)
                        if time.perf_counter() - last_print >= manifest["core"]["config"]["heartbeat_seconds"]:
                            progress("running")
                            last_print = time.perf_counter()
                    pool.shutdown(wait=True)
                    pool = None
            state = "complete" if len(successes) == len(tasks) else "failed" if errors else "paused"
        except KeyboardInterrupt:
            state = "interrupted"
            printer.emit("[interrupted] completed checkpoints are safe; unfinished tasks restart on resume.")
        except BaseException:
            state = "aborted"
            raise
        finally:
            if pool is not None:
                _terminate_pool(pool, active)
            active.clear()
            progress(state)
            printer.close()
        print(f"[output] {root}", flush=True)
        print(f"[resume] python run.py --config <same-config> --resume \"{root}\" (retain any --scales/--algorithms filters)", flush=True)
        if state == "complete":
            from report import build_report
            build_report([root], root / "report")
        return state


def main():
    parser = argparse.ArgumentParser(description="Fixed-scale 30/30/20/20 independent-seed experiments with atomic checkpoints")
    parser.add_argument("--config", default=str(bootstrap.ROOT / "configs/fixed_30_20.yaml"))
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--output")
    target.add_argument("--resume")
    parser.add_argument("--scales", nargs="+", type=int, choices=[100, 250, 500, 1000])
    parser.add_argument("--algorithms", nargs="+", choices=list(METHODS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after", type=int, help="run at most this many pending tasks and pause safely (for batching/testing)")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    config = load_config(args.config)
    manifest, tasks = prepare(config, args.scales, args.algorithms)
    if args.dry_run:
        print(json.dumps(dry_summary(manifest, tasks), ensure_ascii=False, indent=2))
        return
    default = bootstrap.ROOT / "outputs" / (config["name"] + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    root = Path(args.resume or args.output or default).resolve()
    state = execute(manifest, tasks, root, resume=bool(args.resume), stop_after=args.stop_after)
    if state in {"failed", "aborted"}:
        raise SystemExit(1)
    if state == "interrupted":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
