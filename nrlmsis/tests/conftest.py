"""Shared fixtures for NRLMSIS 2.1 tests."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    return DATA_DIR
