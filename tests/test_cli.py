"""The rideshare-qc command: arguments, summary, output and exit codes."""

import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from synthetic import (
    ALL_NULL,
    CONSTANT,
    make_tar,
    make_zip,
    synthetic_rows,
    write_csv_text,
)

from rideshare_data_pipeline import __version__
from rideshare_data_pipeline.cli import main
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS, csv_schema


def child_env() -> dict[str, str]:
    """Environment for a child Python that imports what this process imports."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}


def summary(stdout: str) -> dict[str, str]:
    """Parse the ``label: value`` summary into a dict of stripped strings."""
    pairs = (line.split(":", 1) for line in stdout.splitlines())
    return {label.strip(): value.strip() for label, value in pairs}


@pytest.fixture
def empty_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the test from an empty working directory (to show nothing lands there)."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


@pytest.fixture
def private_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point tempfile at an empty directory, to check what extraction leaves."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    return tmp


# --- help and version ------------------------------------------------------------


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--input PATH" in out
    assert "--drop-columns" in out
    assert "exit status" in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"rideshare-qc {__version__}"


def test_help_does_not_import_polars() -> None:
    # The CLI imports every module of the package; each one defers polars.
    code = (
        "import sys\n"
        "from rideshare_data_pipeline.cli import main\n"
        "try:\n"
        "    main(['--help'])\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0, e.code\n"
        "sys.exit('polars' in sys.modules)\n"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=child_env(), capture_output=True
    )


def test_console_script_help() -> None:
    script = Path(sys.executable).with_name("rideshare-qc")
    if not script.exists():
        pytest.skip("the rideshare-qc entry point isn't installed in this environment")
    result = subprocess.run(
        [script, "--help"], capture_output=True, text=True, env=child_env()
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: rideshare-qc")


def test_python_dash_m_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rideshare_data_pipeline", "--help"],
        capture_output=True,
        text=True,
        env=child_env(),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: rideshare-qc")


@pytest.mark.parametrize(
    "argv",
    [[], ["--input", "x.csv"], ["--out", "out"]],
    ids=["no-args", "no-out", "no-input"],
)
def test_missing_required_arguments_exit_2(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    assert "required" in capsys.readouterr().err


# --- end to end ------------------------------------------------------------------


def test_csv_end_to_end(
    synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert main(["--input", str(synthetic_csv), "--out", str(out)]) == 0

    found = summary(capsys.readouterr().out)
    assert found["rows"] == "3"
    assert found["columns"] == "33"
    assert found["all-null columns"] == f"5 ({', '.join(ALL_NULL)})"
    assert found["constant columns"] == f"4 ({', '.join(CONSTANT)})"
    assert found["duplicate rows"] == "0"
    assert found["wrote"] == f"{out / 'sample.parquet'} (33 columns)"

    # All columns are kept, all-null and constant ones included, with their dtypes.
    written = pl.read_parquet(out / "sample.parquet")
    assert written.schema == csv_schema()
    assert written.height == 3
    assert sorted(p.name for p in out.iterdir()) == ["sample.parquet"]


@pytest.mark.parametrize("name", ["Data.tar.bz2", "Data.zip"])
def test_archive_end_to_end(
    name: str,
    synthetic_csv: Path,
    tmp_path: Path,
    private_tmp: Path,
    empty_cwd: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    members = {"Data.csv": synthetic_csv.read_bytes()}
    archive = tmp_path / name
    if name.endswith(".zip"):
        make_zip(archive, members)
    else:
        make_tar(archive, members)
    out = tmp_path / "out"

    assert main(["--input", str(archive), "--out", str(out)]) == 0

    assert summary(capsys.readouterr().out)["rows"] == "3"
    assert pl.read_parquet(out / "Data.parquet").schema == csv_schema()
    # The extraction directory is gone, and nothing was written to the cwd.
    assert list(private_tmp.iterdir()) == []
    assert list(empty_cwd.iterdir()) == []


def test_relative_paths_write_only_to_the_named_out_dir(
    synthetic_csv: Path, empty_cwd: Path
) -> None:
    (empty_cwd / "in.csv").write_bytes(synthetic_csv.read_bytes())
    assert main(["--input", "in.csv", "--out", "reports"]) == 0
    assert sorted(p.name for p in (empty_cwd / "reports").iterdir()) == ["in.parquet"]
    assert sorted(p.name for p in empty_cwd.iterdir()) == ["in.csv", "reports"]


def test_header_only_csv_writes_an_empty_parquet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(tmp_path / "header.csv", [])
    assert main(["--input", str(path), "--out", str(tmp_path / "out")]) == 0
    found = summary(capsys.readouterr().out)
    assert found["rows"] == "0"
    assert found["all-null columns"] == "0"
    assert found["constant columns"] == "0"
    assert pl.read_parquet(tmp_path / "out" / "header.parquet").height == 0


# --- duplicates: exit 1, rows kept -------------------------------------------------


def test_duplicate_rows_exit_1_and_keep_every_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = synthetic_rows(3)
    path = write_csv_text(tmp_path / "dups.csv", [*rows, rows[0]])
    out = tmp_path / "out"
    assert main(["--input", str(path), "--out", str(out)]) == 1
    assert summary(capsys.readouterr().out)["duplicate rows"] == "1"
    assert pl.read_parquet(out / "dups.parquet").height == 4


# --- dropping columns only on request ----------------------------------------------


def test_drop_columns(synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    argv = ["--input", str(synthetic_csv), "--out", str(out)]
    assert main([*argv, "--drop-columns", "order_points, SKU,file_id"]) == 0
    columns = list(pl.read_parquet_schema(out / "sample.parquet"))
    dropped = {"order_points", "SKU", "file_id"}
    assert columns == [c for c in EXPECTED_COLUMNS if c not in dropped]


def test_drop_columns_rejects_unknown_names(
    synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    argv = ["--input", str(synthetic_csv), "--out", str(out)]
    with pytest.raises(SystemExit) as info:
        main([*argv, "--drop-columns", "SKU,nope,also_nope"])
    assert info.value.code == 2
    assert "not in the schema: nope, also_nope" in capsys.readouterr().err
    assert not out.exists()


# --- input errors: exit 2 ----------------------------------------------------------


def _missing(tmp_path: Path) -> Path:
    return tmp_path / "missing.csv"


def _empty(tmp_path: Path) -> Path:
    path = tmp_path / "empty.csv"
    path.touch()
    return path


def _bad_value(tmp_path: Path) -> Path:
    rows = synthetic_rows(2, order_time={1: "yesterday"})
    return write_csv_text(tmp_path / "bad.csv", rows)


def _bad_header(tmp_path: Path) -> Path:
    return write_csv_text(tmp_path / "hdr.csv", synthetic_rows(1), EXPECTED_COLUMNS[1:])


def _unsupported(tmp_path: Path) -> Path:
    path = tmp_path / "data.json"
    path.write_text("{}")
    return path


def _traversal(tmp_path: Path) -> Path:
    return make_tar(tmp_path / "evil.tar.bz2", {"../evil.csv": b"a\r\n1\r\n"})


def _no_csv(tmp_path: Path) -> Path:
    return make_zip(tmp_path / "nothing.zip", {"README": b"hi"})


def _corrupt(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.tar.bz2"
    path.write_bytes(b"not bz2")
    return path


@pytest.mark.parametrize(
    ("make_input", "message"),
    [
        (_missing, "input not found"),
        (_empty, "empty.csv is empty"),
        (_bad_value, r"don't parse as their dtype: \{'order_time': 1\}"),
        (_bad_header, "header of hdr.csv doesn't match the schema"),
        (_unsupported, "unsupported input 'data.json'"),
        (_traversal, "can't extract evil.tar.bz2"),
        (_no_csv, "expected exactly one CSV in nothing.zip, found 0"),
        (_corrupt, "can't extract corrupt.tar.bz2"),
    ],
    ids=[
        "missing",
        "empty",
        "bad-value",
        "bad-header",
        "unsupported",
        "traversal",
        "no-csv",
        "corrupt",
    ],
)
def test_input_errors_exit_2(
    make_input: Callable[[Path], Path],
    message: str,
    tmp_path: Path,
    private_tmp: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    assert main(["--input", str(make_input(tmp_path)), "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("rideshare-qc: error: ")
    assert re.search(message, captured.err), captured.err
    # Nothing is written on an input error, and no extracted files are left.
    assert not out.exists()
    assert list(private_tmp.iterdir()) == []


def test_unwritable_output_exits_2(
    synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    assert main(["--input", str(synthetic_csv), "--out", str(blocker / "out")]) == 2
    assert capsys.readouterr().err.startswith("rideshare-qc: error: ")
