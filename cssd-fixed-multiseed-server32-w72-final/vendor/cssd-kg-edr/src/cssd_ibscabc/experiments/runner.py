from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import yaml
from tqdm import tqdm

from ..algorithms import solve_ibscabc
from ..algorithms.common import SearchConfig
from ..bounds import published_jobwise_lower_bound
from ..core.feasibility import validate_schedule
from ..core.objective import hourly_completed_devices, job_level_metric_moments
from ..io.loader import load_instance
from .case_study import (
    audit_case_window,
    combine_daily_instances,
    override_machine_counts,
)
from .protocols import resolve_protocol

_INTERNAL_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_POOL_TASKS_PER_WORKER = 8
_PROGRESS_HEARTBEAT_SECONDS = 15.0


def _cpu_model() -> str:
    model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    cpuinfo = Path("/proc/cpuinfo")
    if not model and cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    return model or "unknown"


def _total_memory_bytes() -> int | None:
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
            return None
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (AttributeError, OSError, ValueError):
        return None


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "PyYAML", "matplotlib", "ortools", "tqdm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _configure_internal_threads() -> None:
    """Prevent hidden nested threading inside each experiment worker."""

    for name in _INTERNAL_THREAD_VARIABLES:
        os.environ[name] = "1"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Antivirus, editors, and progress readers can briefly hold the
            # destination open on Windows. A telemetry race must not abort a
            # multi-hour experiment after every worker has already started.
            if attempt == 19:
                raise
            time.sleep(0.05)


def load_experiment_config(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "experiment" not in payload:
        raise ValueError("configuration must contain an experiment mapping")
    base_path = source.parent / "base.yaml"
    if source.name != "base.yaml" and base_path.is_file():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        defaults = dict(base.get("algorithm_defaults", {})) if isinstance(base, dict) else {}
        experiment = dict(payload["experiment"])
        algorithms = []
        for raw in experiment.get("algorithms", []):
            specification = dict(raw)
            if str(specification.get("name")) == "ibscabc":
                specification["parameters"] = defaults | dict(specification.get("parameters", {}))
            algorithms.append(specification)
        experiment["algorithms"] = algorithms
        payload = dict(payload) | {"experiment": experiment}
    return payload


def apply_selected_parameters(
    config: dict[str, object],
    parameter_file: str | Path,
) -> dict[str, object]:
    """Overlay E1-selected search parameters on later IBS-CABC experiments."""

    source = Path(parameter_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"selected parameter file does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("parameters"), dict):
        raise ValueError("selected parameter file must contain a parameters mapping")
    if str(payload.get("algorithm", "ibscabc")) != "ibscabc":
        raise ValueError("selected parameter file is not for IBS-CABC")
    selected = dict(payload["parameters"])
    reserved = {"seed", "max_evaluations", "time_limit_seconds"} & set(selected)
    if reserved:
        raise ValueError(
            "selected parameters may not override protocol fields: "
            + ", ".join(sorted(reserved))
        )
    experiment = dict(config["experiment"])
    algorithms = []
    for raw in experiment.get("algorithms", []):
        specification = dict(raw)
        if str(specification.get("name")) == "ibscabc":
            specification["parameters"] = (
                dict(specification.get("parameters", {})) | selected
            )
        algorithms.append(specification)
    experiment["algorithms"] = algorithms
    return dict(config) | {
        "experiment": experiment,
        "selected_parameter_source": {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }


def _expand_scenarios(experiment: dict[str, object]) -> list[dict[str, object]]:
    explicit = experiment.get("scenarios")
    grid_raw = experiment.get("scenario_grid")
    if explicit is not None and grid_raw is not None:
        raise ValueError("use either scenarios or scenario_grid, not both")
    if grid_raw is None:
        return [
            dict(item)
            for item in (
                explicit if explicit is not None else [{"name": "base"}]
            )
        ]
    if not isinstance(grid_raw, dict) or not grid_raw:
        raise ValueError("scenario_grid must be a non-empty mapping")
    fields = list(grid_raw)
    values: list[list[object]] = []
    for field in fields:
        levels = list(grid_raw[field])  # type: ignore[arg-type]
        if not levels:
            raise ValueError(f"scenario_grid.{field} must not be empty")
        values.append(levels)
    scenarios = []
    for combination in itertools.product(*values):
        scenario = dict(zip(fields, combination))
        scenario["name"] = "_".join(
            f"{field.replace('machines_stage', 'm')}_{value}"
            for field, value in zip(fields, combination)
        )
        scenarios.append(scenario)
    return scenarios


def _task_id(task: dict[str, object]) -> str:
    encoded = json.dumps(task, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _code_sha256() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob("*.py"), key=lambda item: str(item).lower()):
        digest.update(str(path.relative_to(package)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_one(task: dict[str, object]) -> dict[str, object]:
    planning_mode = str(task.get("planning_mode", "SINGLE")).upper()
    source_files = [str(path) for path in task.get("instance_paths", [])]
    if not source_files:
        source_files = [str(task["instance_path"])]
    day_length_minutes = int(task.get("day_length_minutes", 1440))
    if day_length_minutes <= 0 or day_length_minutes % 60:
        raise ValueError("day_length_minutes must be a positive multiple of 60")
    if planning_mode == "AIO":
        instance = combine_daily_instances(
            source_files,
            day_length_minutes=day_length_minutes,
            machines_stage1=(
                int(task["machines_stage1"])
                if "machines_stage1" in task
                else None
            ),
            machines_stage2=(
                int(task["machines_stage2"])
                if "machines_stage2" in task
                else None
            ),
        )
    else:
        instance = load_instance(str(task["instance_path"]))
        instance = override_machine_counts(
            instance,
            machines_stage1=(
                int(task["machines_stage1"])
                if "machines_stage1" in task
                else None
            ),
            machines_stage2=(
                int(task["machines_stage2"])
                if "machines_stage2" in task
                else None
            ),
        )
    protocol = resolve_protocol(dict(task["protocol"]), instance)
    algorithm = str(task["algorithm"])
    solver_name = str(task.get("solver", algorithm))
    seed = int(task["seed"])
    parameters = dict(task.get("parameters", {}))
    reserved = {"seed", "max_evaluations", "time_limit_seconds"} & set(parameters)
    if reserved:
        raise ValueError(
            "algorithm parameters may not override protocol-controlled fields: "
            + ", ".join(sorted(reserved))
        )
    if solver_name != "ibscabc":
        raise KeyError(f"only the proposed IBS-CABC solver is available, got {solver_name!r}")
    if parameters.get("bottleneck_aware", True) is not True:
        raise ValueError("IBS-CABC must keep bottleneck-aware intensification enabled")
    parameters["bottleneck_aware"] = True
    if parameters.get("intensification_enabled", True) is not True:
        raise ValueError("IBS-CABC must keep critical-batch descent enabled")
    parameters["intensification_enabled"] = True
    config = SearchConfig(
        **parameters,
        seed=seed,
        max_evaluations=protocol.max_evaluations,
        time_limit_seconds=protocol.time_limit_seconds,
    )
    solver_wall_started = time.perf_counter()
    solver_cpu_started = time.process_time()
    result = solve_ibscabc(instance, config)
    solver_wall_seconds = time.perf_counter() - solver_wall_started
    solver_cpu_seconds = time.process_time() - solver_cpu_started
    if (
        protocol.max_evaluations is not None
        and result.evaluations != protocol.max_evaluations
    ):
        raise RuntimeError(
            f"{solver_name} consumed {result.evaluations} evaluations; "
            f"expected exactly {protocol.max_evaluations}"
        )
    validate_schedule(instance, result.schedule)
    metrics = asdict(result.schedule.metrics)
    moments = {
        name: value.to_dict()
        for name, value in job_level_metric_moments(
            instance,
            result.schedule.batches_stage1,
            result.schedule.batches_stage2,
            result.schedule.completion_stage1,
            result.schedule.completion_stage2,
        ).items()
    }
    arrivals_hourly = [0] * (day_length_minutes // 60)
    for job in instance.jobs:
        arrivals_hourly[(job.release % day_length_minutes) // 60] += 1
    day_index = int(task.get("case_day_index", 0))
    day_offset = day_index * day_length_minutes if planning_mode == "DBD" else 0
    calendar_objective = result.schedule.objective + day_offset * instance.job_count
    lower_bound = published_jobwise_lower_bound(instance)
    source_parts = {part.lower() for part in instance.source.parts}
    if planning_mode in {"AIO", "DBD"}:
        dataset_group = "case_study_91d"
    elif "instances 25" in source_parts:
        dataset_group = "small_j25"
    elif "realworldinstances365" in source_parts:
        dataset_group = "real365"
    elif "instances100-1000" in source_parts:
        dataset_group = f"fixed_j{instance.job_count}"
    else:
        raise ValueError(f"instance is outside the published experimental sets: {instance.source}")
    return {
        "task_id": str(task["task_id"]),
        "experiment": str(task["experiment"]),
        "algorithm": algorithm,
        "solver": solver_name,
        "schedule_algorithm": result.schedule.algorithm,
        "protocol": protocol.name,
        "seed": seed,
        "instance": instance.name,
        "instance_path": str(instance.source),
        "instance_sha256": instance.source_sha256,
        "dataset_group": dataset_group,
        "planning_mode": planning_mode,
        "scenario": str(task.get("scenario", "base")),
        "case_day_index": day_index if planning_mode == "DBD" else None,
        "case_day_count": len(source_files) if planning_mode == "AIO" else 1,
        "source_files": source_files,
        "day_length_minutes": day_length_minutes,
        "jobs": instance.job_count,
        "families": len(instance.active_families),
        "machines_stage1": instance.machines_stage1,
        "machines_stage2": instance.machines_stage2,
        "objective": result.schedule.objective,
        "calendar_objective": calendar_objective,
        "lower_bound": lower_bound,
        "gap_lb_percent": 100.0 * (result.schedule.objective - lower_bound) / max(1, lower_bound),
        "paper_gap_lb_percent": 100.0
        * (result.schedule.objective - lower_bound)
        / max(1, result.schedule.objective),
        "evaluations": result.evaluations,
        # Measure the complete solver call, including the final incumbent
        # extraction. This matches the author's solution_time more closely than
        # the internal budget clock sampled before result serialization.
        "runtime_seconds": solver_wall_seconds,
        "process_cpu_runtime_seconds": solver_cpu_seconds,
        "wall_runtime_seconds": solver_wall_seconds,
        "budget_elapsed_seconds": result.runtime_seconds,
        "iterations": result.iterations,
        "history": list(result.history),
        "history_evaluations": list(result.history_evaluations),
        "feasible": True,
        **metrics,
        "total_batch_cycles": result.schedule.metrics.total_batch_cycles,
        "metric_moments": moments,
        "arrivals_hourly": arrivals_hourly,
        "completed_stage1_hourly": list(
            hourly_completed_devices(
                result.schedule.batches_stage1,
                day_length_minutes=day_length_minutes,
            )
        ),
        "completed_stage2_hourly": list(
            hourly_completed_devices(
                result.schedule.batches_stage2,
                day_length_minutes=day_length_minutes,
            )
        ),
        "diagnostics": result.diagnostics,
    }


def _failure_record(task: dict[str, object], error: BaseException) -> dict[str, object]:
    return {
        "task_id": str(task["task_id"]),
        "experiment": str(task["experiment"]),
        "algorithm": str(task["algorithm"]),
        "solver": str(task.get("solver", task["algorithm"])),
        "protocol": str(dict(task["protocol"]).get("name", "unknown")),
        "seed": int(task["seed"]),
        "instance_path": str(task["instance_path"]),
        "planning_mode": str(task.get("planning_mode", "SINGLE")),
        "scenario": str(task.get("scenario", "base")),
        "feasible": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _completed_task_ids(result_file: Path, task_ids: set[str]) -> tuple[set[str], int]:
    """Read resumable JSONL incrementally and discard only a torn final line."""

    completed: set[str] = set()
    if not result_file.exists():
        return completed, 0
    file_size = result_file.stat().st_size
    truncated_bytes = 0
    with result_file.open("rb") as handle:
        while True:
            line_start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if handle.tell() == file_size:
                    truncated_bytes = file_size - line_start
                    break
                raise ValueError(
                    f"invalid result JSON before end of file at byte {line_start}"
                ) from error
            try:
                task_id = str(record["task_id"])
            except KeyError as error:
                raise ValueError(f"result at byte {line_start} has no task_id") from error
            if task_id not in task_ids:
                raise ValueError(f"result at byte {line_start} belongs to another task manifest")
            if bool(record.get("feasible", False)):
                completed.add(task_id)
            else:
                completed.discard(task_id)
    if truncated_bytes:
        with result_file.open("r+b") as handle:
            handle.truncate(file_size - truncated_bytes)
    return completed, truncated_bytes


class ExperimentRunner:
    def __init__(
        self,
        config: dict[str, object],
        data_root: str | Path,
        output_root: str | Path,
        workers: int = 1,
    ) -> None:
        self.config = config
        self.data_root = Path(data_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.workers = max(1, int(workers))

    def _instance_paths(self, experiment: dict[str, object]) -> list[Path]:
        selection = dict(experiment["selection"])
        if "split" in selection:
            split = Path(str(selection["split"]))
            if not split.is_absolute():
                split = Path.cwd() / split
            if not split.is_file():
                raise FileNotFoundError(f"split file does not exist: {split}")
            paths = []
            for raw in split.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                path = (self.data_root / line).resolve()
                if not path.is_relative_to(self.data_root):
                    raise ValueError(f"split entry escapes data root: {line}")
                if not path.is_file() or path.suffix.lower() != ".txt":
                    raise FileNotFoundError(f"split instance does not exist: {path}")
                paths.append(path)
        else:
            paths = sorted(self.data_root.rglob("*.txt"), key=lambda path: str(path).lower())
        if not paths:
            raise ValueError("experiment selection contains no instances")
        if len(paths) != len(set(paths)):
            raise ValueError("experiment selection contains duplicate instance paths")
        return paths

    @staticmethod
    def _validate_specification(experiment: dict[str, object]) -> None:
        required = ("name", "selection", "protocols", "algorithms")
        missing = [field for field in required if field not in experiment]
        if missing:
            raise ValueError(f"experiment is missing fields: {', '.join(missing)}")
        protocols = [dict(item) for item in experiment["protocols"]]  # type: ignore[index]
        if not protocols:
            raise ValueError("experiment must contain at least one protocol")
        protocol_names = [str(item.get("name", item.get("mode", ""))) for item in protocols]
        if len(protocol_names) != len(set(protocol_names)) or any(
            not name for name in protocol_names
        ):
            raise ValueError("protocol names must be non-empty and unique")
        for protocol in protocols:
            mode = str(protocol.get("mode", "evaluations"))
            if mode == "evaluations" and int(protocol.get("max_evaluations", 0)) <= 0:
                raise ValueError("evaluation protocols require max_evaluations > 0")
            if mode == "official_time" and float(protocol.get("tau", 1.0)) <= 0:
                raise ValueError("official-time protocols require tau > 0")
            if mode == "wall_clock" and float(protocol.get("time_limit_seconds", 0)) <= 0:
                raise ValueError("wall-clock protocols require time_limit_seconds > 0")
            if mode not in ("evaluations", "official_time", "wall_clock"):
                raise ValueError(f"unknown protocol mode: {mode}")
        algorithms = [dict(item) for item in experiment["algorithms"]]  # type: ignore[index]
        if not algorithms:
            raise ValueError("experiment must contain at least one algorithm")
        names = [str(item.get("name", "")) for item in algorithms]
        labels = [str(item.get("label", item.get("name", ""))) for item in algorithms]
        if any(not name for name in names):
            raise ValueError("algorithm names must be non-empty")
        if len(labels) != len(set(labels)) or any(not label for label in labels):
            raise ValueError("algorithm labels must be non-empty and unique within one experiment")
        unknown = sorted(set(names) - {"ibscabc"})
        if unknown:
            raise ValueError(f"unknown algorithms: {', '.join(unknown)}")
        for specification in algorithms:
            name = str(specification["name"])
            parameters = dict(specification.get("parameters", {}))
            reserved = {"seed", "max_evaluations", "time_limit_seconds"} & set(parameters)
            if reserved:
                raise ValueError(
                    f"algorithm {name} parameters override protocol-controlled fields: "
                    + ", ".join(sorted(reserved))
                )
            if parameters.get("bottleneck_aware", True) is not True:
                raise ValueError("IBS-CABC must keep bottleneck-aware intensification enabled")
            if (
                parameters.get("intensification_enabled", True) is not True
            ):
                raise ValueError("IBS-CABC must keep critical-batch descent enabled")
        seeds = list(experiment.get("seeds", [0]))
        if not seeds or len(seeds) != len({int(seed) for seed in seeds}):
            raise ValueError("experiment seeds must be non-empty and unique")
        planning_modes = [
            str(mode).upper() for mode in experiment.get("planning_modes", [])
        ]
        if planning_modes:
            if (
                len(planning_modes) != len(set(planning_modes))
                or set(planning_modes) - {"AIO", "DBD"}
            ):
                raise ValueError(
                    "planning_modes must be a unique subset of [DBD, AIO]"
                )
            day_length = int(experiment.get("day_length_minutes", 1440))
            if day_length <= 0 or day_length % 60:
                raise ValueError(
                    "day_length_minutes must be a positive multiple of 60"
                )
        scenarios = _expand_scenarios(experiment)
        scenario_names = [str(item.get("name", "")) for item in scenarios]
        if (
            not scenarios
            or any(not name for name in scenario_names)
            or len(scenario_names) != len(set(scenario_names))
        ):
            raise ValueError("scenario names must be non-empty and unique")
        for scenario in scenarios:
            for field in ("machines_stage1", "machines_stage2"):
                if field in scenario and int(scenario[field]) <= 0:
                    raise ValueError(f"{field} must be positive")
            scenario_modes = [
                str(mode).upper()
                for mode in scenario.get("planning_modes", planning_modes)
            ]
            if scenario_modes and set(scenario_modes) - {"AIO", "DBD"}:
                raise ValueError(f"invalid planning mode in scenario {scenario['name']}")

    def tasks(self) -> list[dict[str, object]]:
        experiment = dict(self.config["experiment"])
        self._validate_specification(experiment)
        paths = self._instance_paths(experiment)
        case_spec = dict(experiment.get("case_study", {}))
        if case_spec:
            audit = audit_case_window(paths)
            observed_fields = {
                "days": audit.days,
                "total_jobs": audit.total_jobs,
                "first_seed": audit.first_seed,
                "last_seed": audit.last_seed,
            }
            mismatched = [
                field
                for field, observed in observed_fields.items()
                if field in case_spec and int(case_spec[field]) != observed
            ]
            if mismatched:
                raise ValueError(
                    "case-study split does not match its locked metadata: "
                    + ", ".join(mismatched)
                )
            if bool(case_spec.get("require_contiguous_seeds", False)) and not (
                audit.contiguous_seeds
            ):
                raise ValueError(
                    "case-study split seeds are not contiguous and ordered"
                )
        tasks = []
        planning_modes = [
            str(mode).upper() for mode in experiment.get("planning_modes", [])
        ]
        scenarios = _expand_scenarios(experiment)
        for algorithm_spec_raw in experiment["algorithms"]:  # type: ignore[index]
            algorithm_spec = dict(algorithm_spec_raw)
            name = str(algorithm_spec["name"])
            label = str(algorithm_spec.get("label", name))
            seeds = list(algorithm_spec.get("seeds", experiment.get("seeds", [0])))
            if not seeds or len(seeds) != len({int(seed) for seed in seeds}):
                raise ValueError(f"algorithm {name} seeds must be non-empty and unique")
            for protocol in experiment["protocols"]:  # type: ignore[index]
                if not planning_modes:
                    task_groups = [
                        (
                            scenario,
                            "SINGLE",
                            index,
                            [path],
                        )
                        for scenario in scenarios
                        for index, path in enumerate(paths)
                    ]
                else:
                    task_groups = []
                    for scenario in scenarios:
                        scenario_modes = [
                            str(mode).upper()
                            for mode in scenario.get("planning_modes", planning_modes)
                        ]
                        for mode in scenario_modes:
                            if mode == "AIO":
                                task_groups.append((scenario, mode, None, paths))
                            else:
                                task_groups.extend(
                                    (scenario, mode, index, [path])
                                    for index, path in enumerate(paths)
                                )
                for scenario, mode, day_index, group_paths in task_groups:
                    for seed in seeds:
                        task = {
                            "experiment": str(experiment["name"]),
                            "algorithm": label,
                            "solver": name,
                            "seed": int(seed),
                            "instance_path": str(group_paths[0].resolve()),
                            "instance_paths": [
                                str(path.resolve()) for path in group_paths
                            ],
                            "protocol": dict(protocol),
                            "parameters": dict(algorithm_spec.get("parameters", {})),
                            "planning_mode": mode,
                            "scenario": str(scenario.get("name", "base")),
                            "day_length_minutes": int(
                                experiment.get("day_length_minutes", 1440)
                            ),
                        }
                        if day_index is not None:
                            task["case_day_index"] = day_index
                        for field in ("machines_stage1", "machines_stage2"):
                            if field in scenario:
                                task[field] = int(scenario[field])
                        task["task_id"] = _task_id(task)
                        tasks.append(task)
        if bool(experiment.get("paired_algorithm_order", False)):
            # Keep algorithm variants for the same instance/seed/protocol
            # adjacent.  This does not alter task identities or budgets, but
            # reduces temporal hardware drift in paired ablation experiments.
            tasks.sort(
                key=lambda task: (
                    tuple(str(path) for path in task.get("instance_paths", [])),
                    str(task.get("planning_mode", "SINGLE")),
                    str(task.get("scenario", "base")),
                    str(dict(task["protocol"]).get("name", "")),
                    int(task["seed"]),
                    str(task["algorithm"]),
                )
            )
        return tasks

    def run(self, resume_directory: str | Path | None = None) -> Path:
        _configure_internal_threads()
        all_tasks = self.tasks()
        task_ids = {str(task["task_id"]) for task in all_tasks}
        task_manifest_sha256 = hashlib.sha256(
            "\n".join(str(task["task_id"]) for task in all_tasks).encode("ascii")
        ).hexdigest()
        config_sha256 = hashlib.sha256(
            json.dumps(
                self.config,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        selected_paths = sorted(
            {
                str(path)
                for task in all_tasks
                for path in task.get("instance_paths", [task["instance_path"]])
            }
        )
        dataset_digest = hashlib.sha256()
        for selected_path in selected_paths:
            path = Path(selected_path)
            dataset_digest.update(str(path.relative_to(self.data_root)).replace("\\", "/").encode())
            dataset_digest.update(path.read_bytes())
        dataset_sha256 = dataset_digest.hexdigest()
        code_sha256 = _code_sha256()
        if resume_directory is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            name = str(dict(self.config["experiment"])["name"])
            run_dir = self.output_root / f"{name}_{stamp}"
            if run_dir.exists():
                raise FileExistsError(f"new experiment directory already exists: {run_dir}")
        else:
            run_dir = Path(resume_directory).resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"resume directory does not exist: {run_dir}")
        print(f"[run] {run_dir}", flush=True)
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        result_file = raw_dir / "results.jsonl"
        manifest_path = run_dir / "manifest.json"
        progress_path = run_dir / "progress.json"
        existing_manifest: dict[str, object] | None = None
        if resume_directory is not None:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"resume manifest does not exist: {manifest_path}")
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks = {
                "task_manifest_sha256": task_manifest_sha256,
                "config_sha256": config_sha256,
                "dataset_sha256": dataset_sha256,
                "code_sha256": code_sha256,
                "workers": self.workers,
            }
            mismatched = [
                key for key, expected in checks.items() if existing_manifest.get(key) != expected
            ]
            if mismatched:
                raise ValueError(
                    "resume refused because the experiment environment changed: "
                    + ", ".join(mismatched)
                    + ". Start a new experiment directory to avoid mixing evidence."
                )
        completed, truncated_result_bytes = _completed_task_ids(result_file, task_ids)
        tasks = [task for task in all_tasks if task["task_id"] not in completed]
        manifest = dict(existing_manifest or {}) | {
            "created_utc": (existing_manifest or {}).get(
                "created_utc", datetime.now(timezone.utc).isoformat()
            ),
            "resumed_utc": (
                datetime.now(timezone.utc).isoformat() if existing_manifest is not None else None
            ),
            "config": self.config,
            "data_root": str(self.data_root),
            "dataset_sha256": dataset_sha256,
            "config_sha256": config_sha256,
            "instance_count": len(selected_paths),
            "python": sys.version,
            "platform": platform.platform(),
            "hardware": {
                "cpu_model": _cpu_model(),
                "logical_cpu_count": os.cpu_count(),
                "memory_bytes": _total_memory_bytes(),
                "machine": platform.machine(),
            },
            "software": _software_versions(),
            "workers": self.workers,
            "worker_recycle_task_limit": self.workers * _POOL_TASKS_PER_WORKER,
            "pid": os.getpid(),
            "code_sha256": code_sha256,
            "task_manifest_sha256": task_manifest_sha256,
            "internal_thread_environment": {
                name: os.environ.get(name) for name in _INTERNAL_THREAD_VARIABLES
            },
            "truncated_torn_result_bytes": truncated_result_bytes,
            "task_count_total": len(all_tasks),
            "task_count_successful": len(completed),
            "task_count_failed": 0,
            "task_count_pending": len(tasks),
        }
        _atomic_write_json(manifest_path, manifest)
        failures = 0
        successes = 0
        progress_write_failures = 0

        def update_progress(
            status: str,
            *,
            last_record: dict[str, object] | None = None,
            waiting_seconds: float = 0.0,
        ) -> None:
            nonlocal progress_write_failures
            successful = len(completed) + successes
            payload: dict[str, object] = {
                "status": status,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "task_count_total": len(all_tasks),
                "task_count_successful": successful,
                "task_count_failed_this_invocation": failures,
                "task_count_pending": len(all_tasks) - successful,
                "waiting_seconds_since_last_completion": round(waiting_seconds, 3),
                "progress_write_failures": progress_write_failures,
            }
            if last_record is not None:
                payload["last_task"] = {
                    key: last_record.get(key)
                    for key in (
                        "task_id",
                        "algorithm",
                        "planning_mode",
                        "scenario",
                        "instance",
                        "instance_path",
                        "seed",
                    )
                }
            try:
                _atomic_write_json(progress_path, payload)
            except OSError:
                # Progress is best-effort telemetry. Raw JSONL rows and the
                # final manifest are the evidence-bearing artifacts.
                progress_write_failures += 1

        update_progress("running")
        with result_file.open("a", encoding="utf-8", buffering=1) as handle:
            if self.workers == 1:
                with tqdm(
                    tasks, total=len(tasks), desc="实验进度", unit="任务", dynamic_ncols=True
                ) as bar:
                    for task in bar:
                        try:
                            record = _run_one(task)
                            successes += 1
                        except Exception as error:  # noqa: BLE001 - persisted for resumability
                            failures += 1
                            record = _failure_record(task, error)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        update_progress("running", last_record=record)
                        bar.set_postfix(algorithm=record["algorithm"], failures=failures)
            else:
                chunk_size = self.workers * _POOL_TASKS_PER_WORKER
                with tqdm(
                    total=len(tasks), desc="实验进度", unit="任务", dynamic_ncols=True
                ) as bar:
                    for chunk_start in range(0, len(tasks), chunk_size):
                        chunk = tasks[chunk_start : chunk_start + chunk_size]
                        pool = ProcessPoolExecutor(
                            max_workers=self.workers,
                            initializer=_configure_internal_threads,
                        )
                        futures = {pool.submit(_run_one, task): task for task in chunk}
                        last_completion = time.monotonic()
                        interrupted = False
                        try:
                            while futures:
                                done, _ = wait(
                                    futures,
                                    timeout=_PROGRESS_HEARTBEAT_SECONDS,
                                    return_when=FIRST_COMPLETED,
                                )
                                if not done:
                                    waiting_seconds = time.monotonic() - last_completion
                                    update_progress(
                                        "waiting_for_worker",
                                        waiting_seconds=waiting_seconds,
                                    )
                                    bar.set_postfix(
                                        waiting_s=int(waiting_seconds),
                                        failures=failures,
                                    )
                                    continue
                                last_completion = time.monotonic()
                                for future in done:
                                    task = futures.pop(future)
                                    try:
                                        record = future.result()
                                        successes += 1
                                    except Exception as error:  # noqa: BLE001
                                        failures += 1
                                        record = _failure_record(task, error)
                                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                    handle.flush()
                                    update_progress("running", last_record=record)
                                    bar.set_postfix(
                                        algorithm=record["algorithm"],
                                        instance=record.get(
                                            "instance",
                                            Path(str(record["instance_path"])).stem,
                                        ),
                                        failures=failures,
                                    )
                                    bar.update(1)
                        except KeyboardInterrupt:
                            interrupted = True
                            for future in futures:
                                future.cancel()
                            update_progress("interrupted")
                            pool.shutdown(wait=False, cancel_futures=True)
                            raise
                        finally:
                            if not interrupted:
                                pool.shutdown(wait=True, cancel_futures=False)
        manifest |= {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "task_count_successful": len(completed) + successes,
            "task_count_failed": failures,
            "task_count_pending": len(all_tasks) - len(completed) - successes,
        }
        _atomic_write_json(manifest_path, manifest)
        update_progress("failed" if failures else "finished")
        if failures:
            raise RuntimeError(
                f"{failures} 个任务失败；错误已写入 {result_file}。"
                "若代码、配置、数据集和 worker 数均未改变，可使用 --resume 重试；"
                "修改代码后必须新建实验，避免混合证据。"
            )
        return run_dir
