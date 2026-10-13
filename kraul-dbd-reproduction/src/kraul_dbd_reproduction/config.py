from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_DATASET_SHA256 = "b3fa396eeaec8e71feaa720b2425ada102f57531cf288c39bd9b6b94c5f94116"
FORMAL_NOTEBOOK_SHA256 = "fa9c5f4d3cd105980def46eb7a8a83445643a70318a7fe2a0f0e56a502693f67"
FORMAL_ALGORITHMS = ("ffcbspt", "kraul_ga")
ALGORITHM_LABELS = MappingProxyType({"ffcbspt": "FFCBSPT", "kraul_ga": "Kraul-GA"})
_SEED_PATTERN = re.compile(r"_seed_(\d+)\.txt$", re.IGNORECASE)
_SAFE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    mode: str
    name: str
    tau: float | None = None
    time_limit_seconds: float | None = None
    max_evaluations: int | None = None


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    name: str
    label: str
    profile: str | None
    overrides: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    source: Path
    source_sha256: str
    name: str
    contract: str
    data_root: Path
    split: Path
    instance_paths: tuple[Path, ...]
    source_notebook: Path
    expected_days: int
    expected_jobs: int
    first_day_seed: int
    last_day_seed: int
    machines_stage1: int
    machines_stage2: int
    capacities: tuple[int, int]
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


def _reject_unknown(value: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        converted = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, float) and (not math.isfinite(value) or value != converted):
        raise ValueError(f"{name} must be an integer")
    return converted


def _read_split(path: Path, data_root: Path) -> tuple[Path, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries:
        raise ValueError("DBD split is empty")
    paths = tuple((data_root / entry).resolve() for entry in entries)
    for item in paths:
        try:
            item.relative_to(data_root)
        except ValueError as error:
            raise ValueError(f"split path escapes data_root: {item}") from error
        if not item.is_file():
            raise FileNotFoundError(item)
    if len(paths) != len(set(paths)):
        raise ValueError("DBD split contains duplicates")
    return paths


def _protocol(raw: object) -> ProtocolSpec:
    value = _mapping(raw, "experiment.protocol")
    mode = str(value.get("mode", ""))
    name = str(value.get("name", mode)).strip()
    if not name:
        raise ValueError("experiment.protocol.name must be non-empty")
    if mode == "official_time":
        _reject_unknown(value, {"mode", "name", "tau"}, "protocol")
        tau = float(value.get("tau", 1.0))
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("official_time tau must be finite and positive")
        return ProtocolSpec(mode=mode, name=name, tau=tau)
    if mode == "wall_clock":
        _reject_unknown(
            value,
            {"mode", "name", "time_limit_seconds", "max_evaluations"},
            "protocol",
        )
        seconds = float(value.get("time_limit_seconds", 0.0))
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("wall_clock time_limit_seconds must be finite and positive")
        maximum = value.get("max_evaluations")
        maximum_value = _integer(maximum, "max_evaluations") if maximum is not None else None
        if maximum_value is not None and maximum_value <= 0:
            raise ValueError("max_evaluations must be positive")
        return ProtocolSpec(
            mode=mode,
            name=name,
            time_limit_seconds=seconds,
            max_evaluations=maximum_value,
        )
    raise ValueError(f"unsupported protocol mode: {mode!r}")


def _algorithms(raw: object) -> tuple[AlgorithmSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("experiment.algorithms must be a non-empty list")
    output: list[AlgorithmSpec] = []
    for index, item in enumerate(raw):
        value = _mapping(item, f"experiment.algorithms[{index}]")
        _reject_unknown(
            value,
            {"name", "label", "profile", "overrides"},
            f"algorithm[{index}]",
        )
        name = str(value.get("name", ""))
        if name not in FORMAL_ALGORITHMS:
            raise ValueError(f"unsupported algorithm: {name!r}")
        label = str(value.get("label", ALGORITHM_LABELS[name])).strip()
        if not label:
            raise ValueError(f"algorithm {name!r} has an empty label")
        profile = str(value["profile"]) if "profile" in value else None
        overrides = MappingProxyType(_mapping(value.get("overrides", {}), f"{name}.overrides"))
        if name == "ffcbspt" and (profile is not None or overrides):
            raise ValueError("FFCBSPT does not accept a GA profile or overrides")
        if name == "kraul_ga" and profile is None:
            raise ValueError("Kraul-GA requires an explicit profile")
        if name == "kraul_ga":
            from .ga import config_from_profile

            config_from_profile(profile, overrides)
        output.append(AlgorithmSpec(name, label, profile, overrides))
    names = [item.name for item in output]
    if len(names) != len(set(names)):
        raise ValueError("algorithm names must be unique")
    labels = [item.label for item in output]
    if len(labels) != len(set(labels)):
        raise ValueError("algorithm labels must be unique")
    return tuple(output)


def _filename_seeds(paths: tuple[Path, ...]) -> tuple[int, ...]:
    output: list[int] = []
    for path in paths:
        match = _SEED_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"cannot extract day seed from {path.name}")
        output.append(int(match.group(1)))
    return tuple(output)


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    source = Path(path).resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "configuration")
    _reject_unknown(root, {"experiment"}, "configuration")
    experiment = _mapping(root.get("experiment"), "experiment")
    _reject_unknown(
        experiment,
        {
            "name",
            "contract",
            "data_root",
            "split",
            "source_notebook",
            "expected_days",
            "expected_jobs",
            "first_day_seed",
            "last_day_seed",
            "machines_stage1",
            "machines_stage2",
            "capacities",
            "day_length_minutes",
            "seeds",
            "workers",
            "protocol",
            "algorithms",
        },
        "experiment",
    )
    data_root = _path(experiment.get("data_root"), "experiment.data_root")
    split = _path(experiment.get("split"), "experiment.split")
    instance_paths = _read_split(split, data_root)
    notebook = _path(experiment.get("source_notebook"), "experiment.source_notebook")
    if not notebook.is_file():
        raise FileNotFoundError(notebook)
    day_seeds = _filename_seeds(instance_paths)
    raw_seeds = experiment.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("experiment.seeds must be a non-empty list")
    seeds = tuple(_integer(value, "experiment.seeds[]") for value in raw_seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("experiment.seeds must be unique")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ValueError("experiment seeds must be in [0, 2^32 - 1]")
    capacities_raw = experiment.get("capacities", [15, 11000])
    if not isinstance(capacities_raw, list) or len(capacities_raw) != 2:
        raise ValueError("experiment.capacities must contain two integers")
    name = str(experiment.get("name", "kraul_dbd")).strip()
    if not _SAFE_NAME_PATTERN.fullmatch(name):
        raise ValueError("experiment.name must be a filesystem-safe identifier")
    contract = str(experiment.get("contract", "development"))
    if contract not in {
        "development",
        "formal_proxy91",
        "formal_proxy91_multiseed",
    }:
        raise ValueError(f"unsupported experiment contract: {contract!r}")
    spec = ExperimentSpec(
        source=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        name=name,
        contract=contract,
        data_root=data_root,
        split=split,
        instance_paths=instance_paths,
        source_notebook=notebook,
        expected_days=_integer(experiment.get("expected_days", len(instance_paths)), "expected_days"),
        expected_jobs=_integer(experiment.get("expected_jobs", 0), "expected_jobs"),
        first_day_seed=_integer(experiment.get("first_day_seed", day_seeds[0]), "first_day_seed"),
        last_day_seed=_integer(experiment.get("last_day_seed", day_seeds[-1]), "last_day_seed"),
        machines_stage1=_integer(experiment.get("machines_stage1", 12), "machines_stage1"),
        machines_stage2=_integer(experiment.get("machines_stage2", 5), "machines_stage2"),
        capacities=(
            _integer(capacities_raw[0], "capacities[0]"),
            _integer(capacities_raw[1], "capacities[1]"),
        ),
        day_length_minutes=_integer(
            experiment.get("day_length_minutes", 1440), "day_length_minutes"
        ),
        seeds=seeds,
        workers=_integer(experiment.get("workers", 1), "workers"),
        protocol=_protocol(experiment.get("protocol")),
        algorithms=_algorithms(experiment.get("algorithms")),
    )
    if spec.expected_days != len(instance_paths):
        raise ValueError(f"split contains {len(instance_paths)} days, expected {spec.expected_days}")
    if spec.expected_jobs <= 0:
        raise ValueError("expected_jobs must be positive")
    if min(spec.first_day_seed, spec.last_day_seed) < 0:
        raise ValueError("first/last day seeds cannot be negative")
    if min(spec.machines_stage1, spec.machines_stage2, *spec.capacities) <= 0:
        raise ValueError("machine counts and capacities must be positive")
    if spec.workers <= 0:
        raise ValueError("workers must be positive")
    if spec.day_length_minutes <= 0 or spec.day_length_minutes % 60:
        raise ValueError("day_length_minutes must be a positive multiple of 60")
    if day_seeds[0] != spec.first_day_seed or day_seeds[-1] != spec.last_day_seed:
        raise ValueError("split endpoints differ from first/last day seeds")
    if day_seeds != tuple(range(day_seeds[0], day_seeds[0] + len(day_seeds))):
        raise ValueError("DBD split must contain ordered contiguous seeds")
    if spec.is_formal:
        if tuple(item.name for item in spec.algorithms) != FORMAL_ALGORITHMS:
            raise ValueError("formal algorithm order must be ffcbspt, kraul_ga")
        ga = spec.algorithms[1]
        if ga.profile != "paper_final" or ga.overrides:
            raise ValueError("formal Kraul-GA must use the unmodified paper_final profile")
        if any(item.label != ALGORITHM_LABELS[item.name] for item in spec.algorithms):
            raise ValueError("formal algorithm labels were altered")
        if spec.protocol.mode != "official_time" or spec.protocol.tau != 1.0:
            raise ValueError("formal protocol must use the paper time formula with tau=1")
        expected_seeds = (
            tuple(range(1, 11))
            if spec.contract == "formal_proxy91_multiseed"
            else (101,)
        )
        if (
            spec.expected_days,
            spec.expected_jobs,
            spec.first_day_seed,
            spec.last_day_seed,
            spec.machines_stage1,
            spec.machines_stage2,
            spec.capacities,
            spec.day_length_minutes,
            spec.seeds,
            spec.workers,
        ) != (
            91,
            22927,
            233,
            323,
            12,
            5,
            (15, 11000),
            1440,
            expected_seeds,
            8,
        ):
            raise ValueError("formal proxy91 experiment contract was altered")
        notebook_sha = hashlib.sha256(spec.source_notebook.read_bytes()).hexdigest()
        if notebook_sha != FORMAL_NOTEBOOK_SHA256:
            raise ValueError("the locked public author Notebook has changed")
    return spec
