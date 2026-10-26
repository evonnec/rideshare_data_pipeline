"""Fixtures shared by the test modules (plain helpers live in synthetic.py).

``synthetic_csv`` is the 3-row file the I/O and CLI tests use. The seeded ~300-row
file that reproduces the known data issues is ``full_synthetic_csv`` (and its typed
frame, ``full_synthetic_frame``); see fixtures/make_synthetic.py.
"""

from pathlib import Path

import polars as pl
import pytest
from fixtures.make_synthetic import build_synthetic, write_synthetic_csv
from synthetic import synthetic_rows, write_csv_text

from rideshare_data_pipeline.dedupe import supersede_flags
from rideshare_data_pipeline.parse import derived_columns


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """A 3-row synthetic CSV; see synthetic.ALL_NULL and synthetic.CONSTANT."""
    return write_csv_text(tmp_path / "sample.csv", synthetic_rows(3))


@pytest.fixture
def full_synthetic_csv(tmp_path: Path) -> Path:
    """The seeded synthetic CSV with every known issue injected (CRLF, UTF-8)."""
    return write_synthetic_csv(tmp_path / "synthetic.csv")


@pytest.fixture(scope="session")
def full_synthetic_frame() -> pl.DataFrame:
    """The same data as a typed frame (treat it as read-only)."""
    return build_synthetic()


@pytest.fixture(scope="session")
def full_synthetic_derived(full_synthetic_frame: pl.DataFrame) -> pl.DataFrame:
    """The typed frame plus the derived and flag columns, as the checks see it."""
    return supersede_flags(full_synthetic_frame.with_columns(derived_columns()))
