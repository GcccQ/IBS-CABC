from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from .config import (
    ALGORITHM_LABELS,
    FORMAL_NOTEBOOK_SHA256,
    ProtocolSpec,
    _integer,
    _mapping,
    _path,
    _protocol,
    _reject_unknown,
    _SAFE_NAME_PATTERN,
)
from .ga import config_from_profile


# Filled from the ordered relative paths and bytes in data/fixed_scale_all_100.txt.
FORMAL_FIXED_DATASET_SHA256 = "df415ae52e38327b1fd5d60724b9c0a027ee2333f7800f1aeb494978c94e308d"
FORMAL_FIXED_COUNTS = MappingProxyType({25: 20, 100: 20, 250: 20, 500: 20, 1000: 20})
FORMAL_FIXED_TOTAL_JOBS = sum(size * count for size, count in FORMAL_FIXED_COUNTS.items())


@dataclass(frozen=True, slots=True)
class FixedScaleSpec:
    source: Path
    source_sha256: str
    name: str
    contract: str
    data_root: Path
    split: Path
    instance_paths: tuple[Path, ...]
    source_notebook: Path
    expected_instances: int
    expected_jobs: int
    expected_by_jobs: Mapping[int, int]
    day_length_minutes: int
    seeds: tuple[int, ...]
    workers: int
    protocol: ProtocolSpec
    algorithm_name: str
    algorithm_label: str
    profile: str
    overrides: Mapping[str, object]

    @property
    def is_formal(self) -> bool:
        return self.contract == "formal_fixed_scale"


def _read_split(path: Path, data_root: Path) -> tuple[Path, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries:
        raise ValueError("fixed-scale split is empty")
    paths = tuple((data_root / entry).resolve() for entry in entries)
    for item in paths:
        try:
            item.relative_to(data_root)
        except ValueError as error:
            raise ValueError(f"split path escapes data_root: {item}") from error
        if not item.is_file():
            raise FileNotFoundError(item)
    if len(paths) != len(set(paths)):
        raise ValueError("fixed-scale split contains duplicates")
    return paths


def _expected_counts(raw: object) -> Mapping[int, int]:
    value = _mapping(raw, "experiment.expected_by_jobs")
    output: dict[int, int] = {}
    for raw_size, raw_count in value.items():
        size = _integer(raw_size, "expected_by_jobs key")
        count = _integer(raw_count, f"expected_by_jobs[{size}]")
        if size <= 0 or count <= 0:
            raise ValueError("expected_by_jobs sizes and counts must be positive")
        if size in output:
            raise ValueError(f"duplicate expected job size: {size}")
        output[size] = count
    if not output:
        raise ValueError("expected_by_jobs cannot be empty")
    return MappingProxyType(dict(sorted(output.items())))


def load_fixed_scale_spec(path: str | Path) -> FixedScaleSpec:
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
            "expected_instances",
            "expected_jobs",
            "expected_by_jobs",
            "day_length_minutes",
            "seeds",
            "workers",
            "protocol",
            "algorithm",
        },
        "experiment",
    )
    name = str(experiment.get("name", "kraul_ga_fixed_scale")).strip()
    if not _SAFE_NAME_PATTERN.fullmatch(name):
        raise ValueError("experiment.name must be a filesystem-safe identifier")
    contract = str(experiment.get("contract", "development"))
    if contract not in {"development", "formal_fixed_scale"}:
        raise ValueError(f"unsupported fixed-scale contract: {contract!r}")

    data_root = _path(experiment.get("data_root"), "experiment.data_root")
    split = _path(experiment.get("split"), "experiment.split")
    instance_paths = _read_split(split, data_root)
    notebook = _path(experiment.get("source_notebook"), "experiment.source_notebook")
    if not notebook.is_file():
        raise FileNotFoundError(notebook)

    raw_seeds = experiment.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("experiment.seeds must be a non-empty list")
    seeds = tuple(_integer(value, "experiment.seeds[]") for value in raw_seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("experiment.seeds must be unique")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ValueError("experiment seeds must be in [0, 2^32 - 1]")

    algorithm = _mapping(experiment.get("algorithm"), "experiment.algorithm")
    _reject_unknown(
        algorithm,
        {"name", "label", "profile", "overrides"},
        "experiment.algorithm",
    )
    algorithm_name = str(algorithm.get("name", ""))
    if algorithm_name != "kraul_ga":
        raise ValueError("fixed-scale runner only accepts kraul_ga")
    algorithm_label = str(algorithm.get("label", ALGORITHM_LABELS[algorithm_name])).strip()
    if not algorithm_label:
        raise ValueError("algorithm label cannot be empty")
    profile = str(algorithm.get("profile", ""))
    overrides = MappingProxyType(_mapping(algorithm.get("overrides", {}), "algorithm.overrides"))
    config_from_profile(profile, overrides)

    spec = FixedScaleSpec(
        source=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        name=name,
        contract=contract,
        data_root=data_root,
        split=split,
        instance_paths=instance_paths,
        source_notebook=notebook,
        expected_instances=_integer(
            experiment.get("expected_instances", len(instance_paths)),
            "expected_instances",
        ),
        expected_jobs=_integer(experiment.get("expected_jobs", 0), "expected_jobs"),
        expected_by_jobs=_expected_counts(experiment.get("expected_by_jobs")),
        day_length_minutes=_integer(
            experiment.get("day_length_minutes", 1440),
            "day_length_minutes",
        ),
        seeds=seeds,
        workers=_integer(experiment.get("workers", 1), "workers"),
        protocol=_protocol(experiment.get("protocol")),
        algorithm_name=algorithm_name,
        algorithm_label=algorithm_label,
        profile=profile,
        overrides=overrides,
    )
    if spec.expected_instances != len(spec.instance_paths):
        raise ValueError(
            f"split contains {len(spec.instance_paths)} instances, expected {spec.expected_instances}"
        )
    if spec.expected_instances != sum(spec.expected_by_jobs.values()):
        raise ValueError("expected_instances differs from expected_by_jobs total")
    if spec.expected_jobs != sum(size * count for size, count in spec.expected_by_jobs.items()):
        raise ValueError("expected_jobs differs from expected_by_jobs weighted total")
    if spec.workers <= 0:
        raise ValueError("workers must be positive")
    if spec.day_length_minutes <= 0 or spec.day_length_minutes % 60:
        raise ValueError("day_length_minutes must be a positive multiple of 60")

    if spec.is_formal:
        if dict(spec.expected_by_jobs) != dict(FORMAL_FIXED_COUNTS):
            raise ValueError("formal fixed-scale group counts were altered")
        if (
            spec.expected_instances,
            spec.expected_jobs,
            spec.seeds,
            spec.workers,
            spec.day_length_minutes,
            spec.algorithm_name,
            spec.algorithm_label,
            spec.profile,
            dict(spec.overrides),
        ) != (
            100,
            FORMAL_FIXED_TOTAL_JOBS,
            (101,),
            8,
            1440,
            "kraul_ga",
            "Kraul-GA",
            "paper_final",
            {},
        ):
            raise ValueError("formal fixed-scale experiment contract was altered")
        if spec.protocol.mode != "official_time" or spec.protocol.tau != 1.0:
            raise ValueError("formal fixed-scale protocol must use paper_time_tau1")
        notebook_sha = hashlib.sha256(spec.source_notebook.read_bytes()).hexdigest()
        if notebook_sha != FORMAL_NOTEBOOK_SHA256:
            raise ValueError("the locked public author Notebook has changed")
    return spec
