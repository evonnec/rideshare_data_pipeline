"""The rideshare-qc command: arguments, summary, output and exit codes."""

import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
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
    # The summary ends at the first blank line; the check lines follow it.
    lines = stdout.split("\n\n", 1)[0].splitlines()
    pairs = (line.split(":", 1) for line in lines)
    return {label.strip(): value.strip() for label, value in pairs}


# --- help and version ------------------------------------------------------------


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--version" in out
    assert "run" in out


def test_run_help_documents_options_and_exit_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as info:
        main(["run", "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    for text in (
        "--input PATH",
        "--out DIR",
        "--config FILE",
        "--fail-on",
        "--drop-columns",
    ):
        assert text in out
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
    [[], ["run"], ["run", "--input", "x.csv"], ["run", "--out", "out"]],
    ids=["no-command", "no-args", "no-out", "no-input"],
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
    assert main(["run", "--input", str(synthetic_csv), "--out", str(out)]) == 0

    stdout = capsys.readouterr().out
    check_lines = stdout.split("\n\n")[1].splitlines()
    assert [line.split()[2] for line in check_lines] == [
        "header_valid",
        "utf8_valid",
        "cast_failures",
        "row_count_band",
        "required_fields",
        "id_formats",
        "non_negative_total",
        "item_price_equals_subtotal",
        "zero_fare_rides",
        "discount_within_subtotal",
        "tax_within_total",
        "unexplained_residual",
        "checksum_unique",
        "category_mismatch",
        "merchant_sender_consistent",
        "unknown_sender",
        "ingestion_order",
        "order_before_receipt",
        "late_receipts",
        "order_before_window",
        "email_after_update",
        "all_null_columns",
        "constant_columns",
        "duplicate_rows",
        "ride_completeness",
        "identity_share",
        "receipt_kinds",
        "unclassified_subject_templates",
        "duplicate_receipts",
        "dedupe_key_unique",
        "cross_panelist_duplicates",
        "near_duplicate_rides",
        "description_parse_coverage",
        "currency_suspect_rows",
        "date_only_order_time",
    ]
    assert all(line.split()[1] == "PASS" for line in check_lines)
    found = summary(stdout)
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
    assert sorted(p.name for p in out.iterdir()) == [
        "report.json",
        "report.md",
        "sample.parquet",
    ]


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

    assert main(["run", "--input", str(archive), "--out", str(out)]) == 0

    assert summary(capsys.readouterr().out)["rows"] == "3"
    assert pl.read_parquet(out / "Data.parquet").schema == csv_schema()
    # The extraction directory is gone, and nothing was written to the cwd.
    assert list(private_tmp.iterdir()) == []
    assert list(empty_cwd.iterdir()) == []


def test_relative_paths_write_only_to_the_named_out_dir(
    synthetic_csv: Path, empty_cwd: Path
) -> None:
    (empty_cwd / "in.csv").write_bytes(synthetic_csv.read_bytes())
    assert main(["run", "--input", "in.csv", "--out", "reports"]) == 0
    assert sorted(p.name for p in (empty_cwd / "reports").iterdir()) == [
        "in.parquet",
        "report.json",
        "report.md",
    ]
    assert sorted(p.name for p in empty_cwd.iterdir()) == ["in.csv", "reports"]


def test_header_only_csv_writes_an_empty_parquet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(tmp_path / "header.csv", [])
    assert main(["run", "--input", str(path), "--out", str(tmp_path / "out")]) == 0
    found = summary(capsys.readouterr().out)
    assert found["rows"] == "0"
    assert found["all-null columns"] == "0"
    assert found["constant columns"] == "0"
    assert pl.read_parquet(tmp_path / "out" / "header.parquet").height == 0


def test_header_only_csv_fails_the_row_count_band_unless_min_rows_is_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(tmp_path / "header.csv", [])
    argv = ["run", "--input", str(path), "--out", str(tmp_path / "out")]
    assert main(argv) == 0  # a warn failure doesn't gate by default
    line = next(
        ln for ln in capsys.readouterr().out.splitlines() if "row_count_band" in ln
    )
    assert line.split()[:3] == ["warn", "FAIL", "row_count_band"]
    assert line.split()[-2:] == ["0", "rows"]

    config = tmp_path / "qc.toml"
    config.write_text("min_rows = 0\n", encoding="utf-8")
    assert main([*argv, "--config", str(config)]) == 0
    assert "warn   PASS  row_count_band" in capsys.readouterr().out
    assert main([*argv, "--fail-on", "warn"]) == 1


# --- duplicates: exit 1, rows kept -------------------------------------------------


def test_duplicate_rows_exit_1_and_keep_every_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = synthetic_rows(3)
    path = write_csv_text(tmp_path / "dups.csv", [*rows, rows[0]])
    out = tmp_path / "out"
    assert main(["run", "--input", str(path), "--out", str(out)]) == 1
    assert summary(capsys.readouterr().out)["duplicate rows"] == "1"
    assert pl.read_parquet(out / "dups.parquet").height == 4


# --- dropping columns only on request ----------------------------------------------


def test_drop_columns(synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    argv = ["--input", str(synthetic_csv), "--out", str(out)]
    assert main(["run", *argv, "--drop-columns", "order_points, SKU,file_id"]) == 0
    columns = list(pl.read_parquet_schema(out / "sample.parquet"))
    dropped = {"order_points", "SKU", "file_id"}
    assert columns == [c for c in EXPECTED_COLUMNS if c not in dropped]


def test_drop_columns_rejects_unknown_names(
    synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    argv = ["--input", str(synthetic_csv), "--out", str(out)]
    with pytest.raises(SystemExit) as info:
        main(["run", *argv, "--drop-columns", "SKU,nope,also_nope"])
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
        (_bad_value, r"order_time: 1\n"),
        (_bad_header, "header doesn't match the schema: missing 'merchant_name'"),
        (_unsupported, "unsupported input 'data.json'"),
        (_traversal, "unsafe member '../evil.csv' in evil.tar.bz2"),
        (_no_csv, "expected exactly one .*csv. member in nothing.zip, found none"),
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
    assert main(["run", "--input", str(make_input(tmp_path)), "--out", str(out)]) == 2
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
    assert (
        main(["run", "--input", str(synthetic_csv), "--out", str(blocker / "out")]) == 2
    )
    assert capsys.readouterr().err.startswith("rideshare-qc: error: ")


def test_cast_failures_exit_2_with_per_column_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = synthetic_rows(
        4,
        order_total_amount={0: "12,34", 2: "1,234.00"},
        order_time={1: "2019/01/02 08:00"},
    )
    path = write_csv_text(tmp_path / "bad.csv", rows)
    out = tmp_path / "out"
    assert main(["run", "--input", str(path), "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "  order_total_amount: 2\n" in captured.err
    assert "  order_time: 1" in captured.err
    assert "order_tax" not in captured.err  # clean columns aren't listed
    assert not out.exists()


def test_invalid_utf8_exits_2_with_the_byte_offset(
    synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "latin1.csv"
    data = synthetic_csv.read_bytes()
    path.write_bytes(data + b"caf\xe9\r\n")
    assert main(["run", "--input", str(path), "--out", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert f"byte offset {len(data) + 3:,}" in err
    assert not (tmp_path / "out").exists()


def test_summary_reports_encoding_bom_and_line_endings(
    synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["run", "--input", str(synthetic_csv), "--out", str(tmp_path / "a")]) == 0
    )
    assert summary(capsys.readouterr().out)["encoding"] == (
        "UTF-8, no BOM, CRLF line endings"
    )

    bom_lf = tmp_path / "bom.csv"
    bom_lf.write_bytes(
        b"\xef\xbb\xbf" + synthetic_csv.read_bytes().replace(b"\r\n", b"\n")
    )
    assert main(["run", "--input", str(bom_lf), "--out", str(tmp_path / "b")]) == 0
    assert summary(capsys.readouterr().out)["encoding"] == (
        "UTF-8, BOM, LF line endings"
    )


# --- nothing lands in the working directory ----------------------------------------


@pytest.mark.parametrize("mode", ["no-cache", "cache-flag", "cache-env"])
@pytest.mark.parametrize("name", ["Data.csv", "Data.tar.bz2", "Data.zip"])
def test_nothing_is_written_to_the_cwd_in_any_mode(
    mode: str,
    name: str,
    synthetic_csv: Path,
    tmp_path: Path,
    private_tmp: Path,
    empty_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = {"Data.csv": synthetic_csv.read_bytes()}
    source = tmp_path / "inputs" / name
    source.parent.mkdir()
    if name.endswith(".csv"):
        source.write_bytes(members["Data.csv"])
    elif name.endswith(".zip"):
        make_zip(source, members)
    else:
        make_tar(source, members)
    args = ["run", "--input", str(source), "--out", str(tmp_path / "out")]
    cache = tmp_path / "cache"
    monkeypatch.delenv("RIDESHARE_QC_CACHE", raising=False)
    if mode == "cache-flag":
        args += ["--cache-dir", str(cache)]
    elif mode == "cache-env":
        monkeypatch.setenv("RIDESHARE_QC_CACHE", str(cache))

    for _ in range(2):  # a miss, then a hit
        assert main(args) == 0

    assert list(empty_cwd.iterdir()) == []
    assert list(private_tmp.iterdir()) == []
    assert cache.exists() == (mode != "no-cache")
    assert sorted(p.name for p in source.parent.iterdir()) == [name]


# --- archive provenance and archive failures ------------------------------------------


def test_archive_report_describes_the_archive_and_names_the_member(
    synthetic_csv: Path, tmp_path: Path
) -> None:
    csv_bytes = synthetic_csv.read_bytes()
    archive = make_tar(tmp_path / "Data.tar.gz", {"export/Data.csv": csv_bytes})
    out = tmp_path / "out"
    assert main(["run", "--input", str(archive), "--out", str(out)]) == 0

    run = json.loads((out / "report.json").read_text())["run"]

    assert run["input_name"] == "Data.tar.gz"
    assert run["input_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert run["input_size_bytes"] == archive.stat().st_size
    assert run["member"] == {
        "name": "export/Data.csv",
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "size_bytes": len(csv_bytes),
        "has_bom": False,
        "line_ending": "crlf",
    }
    markdown = (out / "report.md").read_text()
    assert f"member: export/Data.csv ({run['member']['sha256'][:12]})" in markdown


def test_csv_report_has_a_null_member(synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert main(["run", "--input", str(synthetic_csv), "--out", str(out)]) == 0
    assert json.loads((out / "report.json").read_text())["run"]["member"] is None
    assert "member:" not in (out / "report.md").read_text()


def test_stage_errors_name_the_member_not_data_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = make_tar(tmp_path / "in.tar.bz2", {"export/Data.csv": b""})
    assert main(["run", "--input", str(archive), "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "export/Data.csv is empty" in err
    assert "data.csv" not in err


def _truncated(kind: str, tmp_path: Path, csv_bytes: bytes) -> Path:
    payload = os.urandom(200_000)  # incompressible, so cutting it hurts
    path = make_tar(tmp_path / f"cut{kind}", {"Data.csv": csv_bytes + payload})
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    return path


def _corrupt_zip(tmp_path: Path, csv_bytes: bytes) -> Path:
    path = make_zip(tmp_path / "bad.zip", {"Data.csv": csv_bytes})
    data = bytearray(path.read_bytes())
    data[len(data) // 2 :] = b"\xff" * (len(data) - len(data) // 2)
    path.write_bytes(bytes(data))
    return path


def _encrypted_zip(tmp_path: Path, csv_bytes: bytes) -> Path:
    path = make_zip(tmp_path / "locked.zip", {"Data.csv": csv_bytes})
    data = bytearray(path.read_bytes())
    # Set the "encrypted" bit in the local and the central header (the data needn't
    # be real: the member is refused before it is read).
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        data[data.index(signature) + offset] |= 0x1
    path.write_bytes(bytes(data))
    return path


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda p, b: _truncated(".tar.bz2", p, b), "truncated or corrupt"),
        (lambda p, b: _truncated(".tar.gz", p, b), "truncated or corrupt"),
        (lambda p, b: _truncated(".tar.xz", p, b), "truncated or corrupt"),
        (_corrupt_zip, "truncated or corrupt"),
        (_encrypted_zip, "encrypted zip members are not supported"),
    ],
    ids=["tar.bz2", "tar.gz", "tar.xz", "zip-corrupt", "zip-encrypted"],
)
def test_damaged_archives_exit_2_and_write_nothing(
    make: Callable[[Path, bytes], Path],
    message: str,
    synthetic_csv: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = make(tmp_path, synthetic_csv.read_bytes())
    out = tmp_path / "out"
    assert main(["run", "--input", str(archive), "--out", str(out)]) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_a_zip_member_that_raises_runtime_error_is_an_encrypted_member(
    synthetic_csv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = make_zip(tmp_path / "a.zip", {"Data.csv": synthetic_csv.read_bytes()})

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("File 'Data.csv' is encrypted, password required")

    monkeypatch.setattr(zipfile.ZipFile, "open", refuse)
    assert main(["run", "--input", str(archive), "--out", str(tmp_path / "o")]) == 2
    assert "encrypted zip members are not supported" in capsys.readouterr().err
