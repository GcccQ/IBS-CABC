from pathlib import Path

import pytest

from cssd_ibscabc.io import load_instance


@pytest.fixture(scope="session")
def data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset"


@pytest.fixture(scope="session")
def small_instance(data_root):
    path = sorted((data_root / "Instances 25").glob("*.txt"))[0]
    return load_instance(path)
