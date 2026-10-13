from __future__ import annotations

from pathlib import Path

from ..io.loader import iter_instance_paths


def build_published_splits(
    data_root: str | Path,
    output: str | Path,
) -> dict[str, list[str]]:
    """Build the exact instance groups used in Tables 3--5 of the paper."""

    root = Path(data_root).resolve()
    groups: dict[str, list[Path]] = {
        "paper_small25": sorted((root / "Instances 25").glob("*.txt")),
        "paper_fixed": sorted((root / "Instances100-1000").rglob("*.txt")),
        "paper_real365": sorted((root / "RealWorldInstances365").glob("*.txt")),
    }
    expected = {
        "paper_small25": 20,
        "paper_fixed": 80,
        "paper_real365": 365,
    }
    for name, count in expected.items():
        if len(groups[name]) != count:
            raise RuntimeError(
                f"{name} requires {count} instances, found {len(groups[name])}"
            )
    fixed_counts = {
        size: sum(path.parent.name == f"J_{size}" for path in groups["paper_fixed"])
        for size in (100, 250, 500, 1000)
    }
    if set(fixed_counts.values()) != {20}:
        raise RuntimeError(f"fixed-size groups must contain 20 instances each: {fixed_counts}")
    for size in (100, 250, 500, 1000):
        groups[f"paper_fixed_j{size}"] = [
            path
            for path in groups["paper_fixed"]
            if path.parent.name == f"J_{size}"
        ]

    all_paths = list(iter_instance_paths(root))
    if len(all_paths) != 465:
        raise RuntimeError(f"the published experiment requires 465 instances, found {len(all_paths)}")
    groups["paper_all465"] = all_paths
    groups["smoke"] = [groups["paper_small25"][0]]

    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, list[str]] = {}
    for name, paths in groups.items():
        relative = [str(path.relative_to(root)).replace("\\", "/") for path in paths]
        (destination / f"{name}.txt").write_text(
            "\n".join(relative) + "\n",
            encoding="utf-8",
        )
        rendered[name] = relative

    if set(rendered["paper_small25"]) & set(rendered["paper_fixed"]):
        raise AssertionError("published groups must be disjoint")
    if set(rendered["paper_real365"]) & set(rendered["paper_fixed"]):
        raise AssertionError("published groups must be disjoint")
    if (
        set(rendered["paper_small25"])
        | set(rendered["paper_fixed"])
        | set(rendered["paper_real365"])
    ) != set(rendered["paper_all465"]):
        raise AssertionError("paper_all465 must be the exact union of Tables 3--5")
    return rendered
