from __future__ import annotations

import json
from pathlib import Path

import pytest

from kraul_dbd_reproduction.runner import _load_successes


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_resume_rejects_unknown_successful_task(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [{"status": "success", "task_id": "unknown", "task_hash": "hash"}],
    )
    with pytest.raises(ValueError, match="unknown task ID"):
        _load_successes(path, {"known": {"task_hash": "hash"}})


def test_resume_rejects_duplicate_successful_task(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    record = {"status": "success", "task_id": "known", "task_hash": "hash"}
    _write_jsonl(path, [record, record])
    with pytest.raises(ValueError, match="duplicate successful result"):
        _load_successes(path, {"known": {"task_hash": "hash"}})
