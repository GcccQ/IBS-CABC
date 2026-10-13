from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from .bridge import PROJECT_ROOT


FORMAL_ALGORITHMS = ("ibscabc", "alns", "ts")
FORMAL_DATASET_SHA256 = "b3fa396eeaec8e71feaa720b2425ada102f57531cf288c39bd9b6b94c5f94116"
ALGORITHM_LABELS = MappingProxyType({"ibscabc": "IBS-CABC", "alns": "ALNS", "ts": "TS"})
FORMAL_DELTA_SEED_PAIRS = (
    (1.00, 1.00),
    (0.93, 1.00),
    (1.00, 0.95),
    (0.67, 0.78),
    (0.40, 0.52),
    (0.85, 0.85),
    (0.70, 0.70),
)
FORMAL_PARAMETER_CONTRACT = MappingProxyType(
    {
        "ibscabc": {
            "colony_size": 20,
            "abandonment_limit": 10,
            "archive_size": 20,
            "operator_attempts": 2,
            "ejection_depth": 3,
            "intensification_enabled": True,
            "intensification_fraction": 0.90,
            "intensification_max_evaluations": 9000,
            "intensification_critical_batches": 8,
            "assignment_repair_enabled": True,
            "assignment_repair_seconds_per_family": 10.0,
            "assignment_repair_workers": 1,
            "assignment_repair_iterations": 3,
            "assignment_repair_joint_max_jobs": 600,
            "assignment_repair_max_jobs": 600,
            "assignment_repair_allow_empty_batches": True,
            "assignment_repair_time_fraction": 0.30,
            "mathematical_repair_seconds": 45.0,
            "onlooker_quality_weight": 0.60,
            "onlooker_diversity_weight": 0.15,
            "onlooker_bottleneck_weight": 0.25,
            "delta_seed_pairs": FORMAL_DELTA_SEED_PAIRS,
            "bottleneck_aware": True,
        },
        "alns": {
            "population_size": 20,
            "stagnation_limit": 10,
            "candidate_attempts": 2,
            "cache_capacity": 120,
            "delta_seed_pairs": FORMAL_DELTA_SEED_PAIRS,
            "reaction_factor": 0.20,
            "initial_temperature_ratio": 0.01,
            "final_temperature_ratio": 0.00001,
        },
        "ts": {
            "population_size": 20,
            "stagnation_limit": 10,
            "candidate_attempts": 2,
            "cache_capacity": 120,
            "delta_seed_pairs": FORMAL_DELTA_SEED_PAIRS,
            "neighborhood_size": 20,
            "tabu_tenure": 10,
        },
    }
)
_SEED_PATTERN = re.compile(r"_seed_(\d+)\.txt$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    name: str
    mode: str
    tau: float | None = None
    max_evaluations: int | None = None
    time_limit_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    name: str
    label: str
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    source: Path
    source_sha256: str
    name: str
    contract: str
    data_root: Path
    split: Path
    instance_paths: tuple[Path, ...]
    expected_days: int
    expected_jobs: int
    first_day_seed: int
    last_day_seed: int
    machines_stage1: int
    machines_stage2: int
    day_length_minutes: int
    seeds: tuple[int, ...]
    workers: int
    protocol: ProtocolSpec
    algorithms: tuple[AlgorithmSpec, ...]

    @property
    def is_formal(self) -> bool:
        return self.contract in {"formal_proxy91", "formal_proxy91_multiseed"}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _project_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _read_split(path: Path, data_root: Path) -> tuple[Path, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"split file does not exist: {path}")
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError("DBD split is empty")
    paths = tuple((data_root / value).resolve() for value in values)
    outside_root: list[str] = []
    for item in paths:
        try:
            item.relative_to(data_root)
        except ValueError:
            outside_root.append(str(item))
    if outside_root:
        raise ValueError("split paths must stay inside data_root: " + ", ".join(outside_root[:5]))
    missing = [str(item) for item in paths if not item.is_file()]
    if missing:
        raise FileNotFoundError("split references missing instances: " + ", ".join(missing[:5]))
    if len(paths) != len(set(paths)):
        raise ValueError("DBD split contains duplicate instance paths")
    return paths


def _protocol(raw: object) -> ProtocolSpec:
    value = _mapping(raw, "experiment.protocol")
    mode = str(value.get("mode", ""))
    name = str(value.get("name", mode))
    if not name.strip():
        raise ValueError("experiment.protocol.name must be non-empty")
    if mode == "official_time":
        tau = float(value.get("tau", 1.0))
        if tau <= 0:
            raise ValueError("official_time tau must be positive")
        return ProtocolSpec(name, mode, tau=tau)
    if mode == "evaluations":
        maximum = int(value.get("max_evaluations", 0))
        if maximum <= 0:
            raise ValueError("evaluation protocol requires max_evaluations > 0")
        return ProtocolSpec(name, mode, max_evaluations=maximum)
    if mode == "wall_clock":
        seconds = float(value.get("time_limit_seconds", 0.0))
        if seconds <= 0:
            raise ValueError("wall_clock protocol requires time_limit_seconds > 0")
        return ProtocolSpec(name, mode, time_limit_seconds=seconds)
    raise ValueError(f"unsupported protocol mode: {mode!r}")


def _algorithms(raw: object) -> tuple[AlgorithmSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("experiment.algorithms must be a non-empty list")
    algorithms: list[AlgorithmSpec] = []
    for index, item in enumerate(raw):
        value = _mapping(item, f"experiment.algorithms[{index}]")
        name = str(value.get("name", ""))
        if name not in FORMAL_ALGORITHMS:
            raise ValueError(f"unsupported DBD algorithm: {name!r}")
        label = str(value.get("label", ALGORITHM_LABELS[name]))
        if not label.strip():
            raise ValueError(f"algorithm label for {name} must be non-empty")
        parameters = _mapping(value.get("parameters", {}), f"parameters for {name}")
        reserved = {"seed", "max_evaluations", "time_limit_seconds"} & set(parameters)
        if reserved:
            raise ValueError(
                f"{name} parameters override protocol fields: {', '.join(sorted(reserved))}"
            )
        algorithms.append(AlgorithmSpec(name, label, MappingProxyType(parameters)))
    names = [algorithm.name for algorithm in algorithms]
    labels = [algorithm.label for algorithm in algorithms]
    if len(names) != len(set(names)) or len(labels) != len(set(labels)):
        raise ValueError("algorithm names and labels must be unique")
    return tuple(algorithms)


def _filename_seeds(paths: tuple[Path, ...]) -> tuple[int, ...]:
    seeds: list[int] = []
    for path in paths:
        match = _SEED_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"cannot recover day seed from filename: {path.name}")
        seeds.append(int(match.group(1)))
    return tuple(seeds)


def _normalized_parameter_contract(values: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(values)
    pairs = normalized.get("delta_seed_pairs")
    if isinstance(pairs, (list, tuple)):
        normalized_pairs: list[tuple[float, float]] = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("delta_seed_pairs must contain two-number pairs")
            normalized_pairs.append((float(pair[0]), float(pair[1])))
        normalized["delta_seed_pairs"] = tuple(normalized_pairs)
    return normalized


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, "configuration")
    experiment = _mapping(root.get("experiment"), "experiment")
    data_root = _project_path(experiment.get("data_root"), "experiment.data_root")
    split = _project_path(experiment.get("split"), "experiment.split")
    paths = _read_split(split, data_root)
    protocol = _protocol(experiment.get("protocol"))
    algorithms = _algorithms(experiment.get("algorithms"))
    raw_seeds = experiment.get("seeds", [])
    if not isinstance(raw_seeds, list):
        raise ValueError("experiment.seeds must be a list")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("experiment.seeds must be non-empty and unique")
    day_seeds = _filename_seeds(paths)
    spec = ExperimentSpec(
        source=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        name=str(experiment.get("name", "dbd_case")),
        contract=str(experiment.get("contract", "development")),
        data_root=data_root,
        split=split,
        instance_paths=paths,
        expected_days=int(experiment.get("expected_days", len(paths))),
        expected_jobs=int(experiment.get("expected_jobs", 0)),
        first_day_seed=int(experiment.get("first_day_seed", day_seeds[0])),
        last_day_seed=int(experiment.get("last_day_seed", day_seeds[-1])),
        machines_stage1=int(experiment.get("machines_stage1", 12)),
        machines_stage2=int(experiment.get("machines_stage2", 5)),
        day_length_minutes=int(experiment.get("day_length_minutes", 1440)),
        seeds=seeds,
        workers=int(experiment.get("workers", 1)),
        protocol=protocol,
        algorithms=algorithms,
    )
    if not spec.name.strip() or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", spec.name) is None:
        raise ValueError(
            "experiment.name must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )
    if spec.expected_days != len(paths):
        raise ValueError(f"DBD split has {len(paths)} days, expected {spec.expected_days}")
    if spec.workers <= 0:
        raise ValueError("workers must be positive")
    if spec.expected_jobs < 0:
        raise ValueError("expected_jobs cannot be negative")
    if spec.machines_stage1 <= 0 or spec.machines_stage2 <= 0:
        raise ValueError("machine counts must be positive")
    if spec.day_length_minutes <= 0 or spec.day_length_minutes % 60:
        raise ValueError("day_length_minutes must be a positive multiple of 60")
    if day_seeds[0] != spec.first_day_seed or day_seeds[-1] != spec.last_day_seed:
        raise ValueError("DBD split endpoints do not match the configured day seeds")
    if day_seeds != tuple(range(day_seeds[0], day_seeds[0] + len(day_seeds))):
        raise ValueError("DBD split must be ordered by contiguous day seeds")
    if spec.is_formal:
        if tuple(algorithm.name for algorithm in algorithms) != FORMAL_ALGORITHMS:
            raise ValueError("formal DBD algorithm order must be ibscabc, alns, ts")
        expected_seeds = (
            tuple(range(1, 11))
            if spec.contract == "formal_proxy91_multiseed"
            else (101,)
        )
        if spec.seeds != expected_seeds:
            raise ValueError(
                "formal DBD seed set must be [1,...,10] for the multiseed contract "
                "or [101] for the legacy single-seed contract"
            )
        if protocol.mode != "official_time" or protocol.tau != 1.0:
            raise ValueError("formal DBD protocol must be official_time with tau=1")
        for algorithm in algorithms:
            if algorithm.label != ALGORITHM_LABELS[algorithm.name]:
                raise ValueError(f"formal label for {algorithm.name} was altered")
            observed = _normalized_parameter_contract(algorithm.parameters)
            expected = dict(FORMAL_PARAMETER_CONTRACT[algorithm.name])
            if observed != expected:
                raise ValueError(f"formal parameters for {algorithm.name} were altered")
        if (
            spec.expected_days != 91
            or spec.expected_jobs != 22_927
            or spec.first_day_seed != 233
            or spec.last_day_seed != 323
            or spec.machines_stage1 != 12
            or spec.machines_stage2 != 5
            or spec.day_length_minutes != 1440
            or spec.workers != 8
        ):
            raise ValueError("formal proxy91 case-study contract was altered")
    return spec
