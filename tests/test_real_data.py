"""Opt-in run on the real data, which is never committed.

Set RIDESHARE_QC_DATA to a local CSV or archive to run it; otherwise it is
skipped. It asserts only loose properties, never real values.
"""

import os
from pathlib import Path

import polars as pl
import pytest

from rideshare_data_pipeline.cli import main
from rideshare_data_pipeline.io import input_stem
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

DATA = os.environ.get("RIDESHARE_QC_DATA", "")


@pytest.mark.slow
@pytest.mark.skipif(
    not Path(DATA).is_file(), reason="RIDESHARE_QC_DATA doesn't name a local file"
)
def test_cli_on_real_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = main(["run", "--input", DATA, "--out", str(tmp_path)])
    assert status in {0, 1}, capsys.readouterr().err

    stdout = capsys.readouterr().out
    assert "rows:" in stdout
    written = pl.scan_parquet(tmp_path / f"{input_stem(DATA)}.curated.parquet")
    names = written.collect_schema().names()
    assert names[: len(EXPECTED_COLUMNS)] == list(EXPECTED_COLUMNS)
    assert names[-1] == "qc_flags"
    assert written.select(pl.len()).collect().item() > 0
