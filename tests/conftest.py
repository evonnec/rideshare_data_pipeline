"""Fixtures shared by the test modules (plain helpers live in synthetic.py)."""

from pathlib import Path

import pytest
from synthetic import synthetic_rows, write_csv_text


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """A 3-row synthetic CSV; see synthetic.ALL_NULL and synthetic.CONSTANT."""
    return write_csv_text(tmp_path / "sample.csv", synthetic_rows(3))
