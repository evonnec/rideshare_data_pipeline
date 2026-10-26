"""Edge-case inputs, end to end through the CLI: exit code, message, what was written.

One discoverable place for the awkward inputs the legacy code crashed or misbehaved
on. The internals have unit tests (test_io.py for the byte scan, header and cast
checks and archive extraction, test_qa_fixes.py for non-finite numbers and years,
test_cache.py, test_checks_completeness.py for S4/C3); these assert the behaviour a
user sees. Every input is built in tmp_path from synthetic rows.
"""

import json
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from _pytest.mark import ParameterSet
from synthetic import (
    ALL_NULL,
    make_tar,
    make_zip,
    synthetic_rows,
    write_csv_text,
)
from test_cli import child_env

from rideshare_data_pipeline.cli import main, run
from rideshare_data_pipeline.io import (
    ArchiveError,
    CastError,
    EncodingError,
    HeaderError,
    InputError,
    RaggedRowError,
    locate_ragged_row,
    read_typed,
)
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

type Builder = Callable[[Path], Path]

# --- helpers -----------------------------------------------------------------------


def run_cli(
    capsys: pytest.CaptureFixture[str], path: Path, out: Path, *extra: str
) -> tuple[int, str, str]:
    """Run ``rideshare-qc run`` and return (exit code, stdout, stderr)."""
    code = main(["run", "--input", str(path), "--out", str(out), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def report_of(out: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((out / "report.json").read_text("utf-8"))
    return loaded


def checks_of(out: Path) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in report_of(out)["checks"]}


def stable(report: dict[str, Any]) -> dict[str, Any]:
    """The report without the fields that legitimately differ between two runs."""
    run_info = {
        k: v
        for k, v in report["run"].items()
        if k not in {"started", "finished", "elapsed_seconds"}
    }
    return {**report, "run": run_info}


def without_byte_facts(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The check results minus the utf8_valid details naming the BOM and line ending."""
    return [
        {**c, "details": {}} if c["name"] == "utf8_valid" else c
        for c in report["checks"]
    ]


def curated_of(out: Path, stem: str) -> pl.DataFrame:
    return pl.read_parquet(out / f"{stem}.curated.parquet")


def lines_of(path: Path) -> list[bytes]:
    """The file's lines without their CRLF (the synthetic rows hold no newline)."""
    return path.read_bytes().split(b"\r\n")[:-1]


def write_lines(path: Path, lines: list[bytes]) -> Path:
    path.write_bytes(b"\r\n".join(lines) + b"\r\n")
    return path


def zip_with(path: Path, members: dict[str, bytes]) -> Path:
    return make_zip(path, members)


def csv_bytes(tmp_path: Path, n: int = 3, name: str = "Data.csv") -> bytes:
    """The bytes of an n-row synthetic CSV, to put inside an archive."""
    return write_csv_text(tmp_path / f"_{name}", synthetic_rows(n)).read_bytes()


# --- the table of failing inputs: exception class, message, exit 2 -------------------
# Each builder returns an input that must be refused. test_error_is_typed asserts the
# exception class and a message naming the column, line or byte; test_error_exits_2
# asserts the same message on stderr, exit 2 and that nothing was written.


def _empty(tmp_path: Path) -> Path:
    path = tmp_path / "Data.csv"
    path.write_bytes(b"")
    return path


def _dup_header(tmp_path: Path) -> Path:
    header = [*EXPECTED_COLUMNS[:-1], EXPECTED_COLUMNS[0]]  # last name replaced
    return write_csv_text(tmp_path / "Data.csv", synthetic_rows(2), header)


def _reordered(tmp_path: Path) -> Path:
    header = [EXPECTED_COLUMNS[1], EXPECTED_COLUMNS[0], *EXPECTED_COLUMNS[2:]]
    return write_csv_text(tmp_path / "Data.csv", synthetic_rows(2), header)


def _missing_column(tmp_path: Path) -> Path:
    return write_csv_text(
        tmp_path / "Data.csv", synthetic_rows(2), EXPECTED_COLUMNS[:-1]
    )


def _short_row(tmp_path: Path) -> Path:
    path = write_csv_text(tmp_path / "Data.csv", synthetic_rows(4))
    lines = lines_of(path)
    lines[4] = b",".join(lines[4].split(b",")[:-2])  # line 5: 31 fields
    return write_lines(path, lines)


def _long_row(tmp_path: Path) -> Path:
    path = write_csv_text(tmp_path / "Data.csv", synthetic_rows(4))
    lines = lines_of(path)
    lines[2] += b",extra"  # line 3: 34 fields
    return write_lines(path, lines)


def _bad_utf8(tmp_path: Path) -> Path:
    path = write_csv_text(tmp_path / "Data.csv", synthetic_rows(2))
    path.write_bytes(path.read_bytes() + b"caf\xe9\r\n")
    return path


def _cast_failure(tmp_path: Path) -> Path:
    rows = synthetic_rows(3, order_total_amount={1: "1,234.00"})
    return write_csv_text(tmp_path / "Data.csv", rows)


def _zip_no_csv(tmp_path: Path) -> Path:
    return zip_with(tmp_path / "Data.zip", {"readme.txt": b"hello"})


def _zip_traversal(tmp_path: Path) -> Path:
    return zip_with(tmp_path / "Data.zip", {"../evil.csv": csv_bytes(tmp_path)})


def _zip_sidecar_only(tmp_path: Path) -> Path:
    return zip_with(tmp_path / "Data.zip", {"._Data.csv": b"\x00\x05\x16\x07"})


def _tar_traversal(tmp_path: Path) -> Path:
    return make_tar(tmp_path / "Data.tar", {"/abs/evil.csv": csv_bytes(tmp_path)})


def _missing_file(tmp_path: Path) -> Path:
    return tmp_path / "nope.csv"


def _unsupported_suffix(tmp_path: Path) -> Path:
    path = tmp_path / "Data.txt"
    path.write_text("a,b\n", encoding="utf-8")
    return path


def _case(
    case_id: str, build: Builder, error: type[InputError], text: str
) -> ParameterSet:
    return pytest.param(build, error, text, id=case_id)


ERROR_CASES = [
    _case("empty-file", _empty, InputError, "Data.csv is empty"),
    _case("duplicate-header", _dup_header, HeaderError, "duplicated 'merchant_name'"),
    _case("reordered-header", _reordered, HeaderError, "column 1 is 'user_id'"),
    _case("missing-column", _missing_column, HeaderError, "missing 'dttimestamp'"),
    _case(
        "ragged-too-few",
        _short_row,
        RaggedRowError,
        "Data.csv line 5: expected 33 fields, found 31",
    ),
    _case(
        "ragged-too-many",
        _long_row,
        RaggedRowError,
        "Data.csv line 3: expected 33 fields, found 34",
    ),
    _case("invalid-utf8", _bad_utf8, EncodingError, "byte offset"),
    _case("cast-failure", _cast_failure, CastError, "order_total_amount: 1"),
    _case("archive-no-csv", _zip_no_csv, ArchiveError, "members: readme.txt"),
    _case("archive-zip-traversal", _zip_traversal, ArchiveError, "'..' component"),
    _case("archive-tar-absolute", _tar_traversal, ArchiveError, "absolute path"),
    _case("archive-sidecar-only", _zip_sidecar_only, ArchiveError, "._Data.csv"),
    _case("missing-file", _missing_file, InputError, "input not found"),
    _case("unsupported-suffix", _unsupported_suffix, InputError, "unsupported input"),
]


@pytest.mark.parametrize(("build", "error", "text"), ERROR_CASES)
def test_error_is_typed(
    tmp_path: Path, build: Builder, error: type[InputError], text: str
) -> None:
    path = build(tmp_path)
    with pytest.raises(InputError) as info:
        run(path, tmp_path / "out")
    assert type(info.value) is error  # the specific subclass, not just InputError
    assert text in str(info.value)


@pytest.mark.parametrize(("build", "error", "text"), ERROR_CASES)
def test_error_exits_2_with_the_message_and_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    build: Builder,
    error: type[InputError],
    text: str,
) -> None:
    path = build(tmp_path)
    out = tmp_path / "out"
    code, stdout, stderr = run_cli(capsys, path, out)
    assert code == 2
    assert stdout == ""
    assert stderr.startswith("rideshare-qc: error: ")
    assert text in stderr
    assert not out.exists()


# --- empty and header-only ------------------------------------------------------------


def test_empty_file_is_an_input_error_in_an_archive_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = zip_with(tmp_path / "Data.zip", {"Data.csv": b""})
    code, _, err = run_cli(capsys, path, tmp_path / "out")
    assert code == 2
    assert "Data.csv is empty" in err  # the member's name, not the temp "data.csv"
    assert not (tmp_path / "out").exists()


def test_header_only_writes_the_report_and_the_row_count_band_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(tmp_path / "Data.csv", [])
    out = tmp_path / "out"
    code, stdout, _ = run_cli(capsys, path, out)
    assert code == 0  # a warn failure doesn't gate by default
    assert "warn   FAIL  row_count_band" in stdout
    band = checks_of(out)["row_count_band"]
    assert (band["passed"], band["severity"], band["n_failed"]) == (False, "warn", 1)
    assert report_of(out)["run"]["rows"] == 0
    assert (out / "report.md").is_file()
    curated = curated_of(out, "Data")
    assert curated.height == 0
    assert curated.columns[: len(EXPECTED_COLUMNS)] == list(EXPECTED_COLUMNS)
    assert curated.width == report_of(out)["outputs"]["curated"]["columns"]
    # No other check is a verdict on zero rows.
    assert [n for n, c in checks_of(out).items() if not c["passed"]] == [
        "row_count_band"
    ]
    code, _, _ = run_cli(capsys, path, tmp_path / "strict", "--fail-on", "warn")
    assert code == 1
    assert (tmp_path / "strict" / "report.json").is_file()  # still written


# --- one row, an all-null column ------------------------------------------------------


def test_single_row_completes_and_every_non_null_column_is_constant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(tmp_path / "Data.csv", synthetic_rows(1))
    out = tmp_path / "out"
    code, _, _ = run_cli(capsys, path, out)
    assert code == 0
    assert report_of(out)["run"]["rows"] == 1
    # Documented: with one row nothing varies, so every non-null column is constant.
    constant = checks_of(out)["constant_columns"]["details"]["columns"]
    assert sorted(constant) == sorted(set(EXPECTED_COLUMNS) - set(ALL_NULL))
    assert curated_of(out, "Data").height == 1
    # The outlier and dedupe checks ran on a group of one without failing.
    names = checks_of(out)
    for name in ("fare_outlier_conditional", "fare_outlier_univariate"):
        assert names[name]["passed"] is True
    assert names["dedupe_key_unique"]["passed"] is True


def test_all_null_columns_are_reported_and_never_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(tmp_path / "Data.csv", synthetic_rows(5))
    out = tmp_path / "out"
    assert run_cli(capsys, path, out)[0] == 0
    found = checks_of(out)["all_null_columns"]
    assert sorted(found["details"]["columns"]) == sorted(ALL_NULL)
    assert found["n_failed"] == len(ALL_NULL)
    curated = curated_of(out, "Data")
    for column in ALL_NULL:
        assert column in curated.columns
        assert curated[column].null_count() == 5


# --- BOM, line endings ----------------------------------------------------------------


def test_bom_and_line_endings_change_nothing_but_their_own_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    crlf = write_csv_text(tmp_path / "crlf.csv", synthetic_rows(6)).read_bytes()
    lf = crlf.replace(b"\r\n", b"\n")
    bom = b"\xef\xbb\xbf"
    variants = {
        "crlf": (crlf, False, "crlf"),
        "lf": (lf, False, "lf"),
        "bom-crlf": (bom + crlf, True, "crlf"),
        "bom-lf": (bom + lf, True, "lf"),
    }
    results = {}
    for stem, (data, has_bom, ending) in variants.items():
        path = tmp_path / f"{stem}.csv"
        path.write_bytes(data)
        out = tmp_path / f"out-{stem}"
        assert run_cli(capsys, path, out)[0] == 0
        run_info = report_of(out)["run"]
        assert (run_info["bom"], run_info["line_ending"]) == (has_bom, ending)
        assert run_info["input_sha256"] != ""
        results[stem] = (without_byte_facts(report_of(out)), curated_of(out, stem))
    baseline_checks, baseline_frame = results["crlf"]
    for stem, (checks, frame) in results.items():
        assert checks == baseline_checks, stem
        assert frame.equals(baseline_frame), stem


# --- quoted values --------------------------------------------------------------------


def test_quoted_commas_quotes_and_newlines_stay_in_their_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    description = 'Ride, with "quotes", and\na line break'
    subject = "Your trip, receipt"
    rows = synthetic_rows(
        3, product_description={1: description}, email_subject={0: subject}
    )
    path = write_csv_text(tmp_path / "Data.csv", rows)
    out = tmp_path / "out"
    assert run_cli(capsys, path, out)[0] == 0
    curated = curated_of(out, "Data")
    assert curated.height == 3
    assert curated["product_description"][1] == description
    assert curated["email_subject"][0] == subject
    assert curated["order_total_amount"].to_list() == [10.5, 11.5, 12.5]
    # The quoted line break also means physical lines != records; the file is
    # still valid, so the slow locator clears it.
    locate_ragged_row(path)


# --- ragged rows ----------------------------------------------------------------------


def _quoted_comma_then_short(tmp_path: Path) -> Path:
    # Row 0 loses one field but holds a quoted comma, so a naive comma count of the
    # line still says 32: only a quote-aware count sees 32 fields' worth minus one.
    rows = synthetic_rows(3, product_description={0: "Ride, one"})
    path = write_csv_text(tmp_path / "Data.csv", rows)
    lines = lines_of(path)
    lines[1] = lines[1].rsplit(b",", 1)[0]
    return write_lines(path, lines)


def _short_after_multiline_field(tmp_path: Path) -> Path:
    # Row 0 spans physical lines 2-3, so row 2 starts on line 5, not line 4.
    rows = synthetic_rows(3, product_description={0: "line one\nline two"})
    path = write_csv_text(tmp_path / "Data.csv", rows)
    lines = lines_of(path)
    assert len(lines) == 4  # the embedded "\n" isn't a CRLF, so it doesn't split
    lines[3] = b",".join(lines[3].split(b",")[:-2])
    return write_lines(path, lines)


def _short_in_the_last_line_without_newline(tmp_path: Path) -> Path:
    path = write_csv_text(tmp_path / "Data.csv", synthetic_rows(2))
    lines = lines_of(path)
    lines[2] = b",".join(lines[2].split(b",")[:-1])
    path.write_bytes(b"\r\n".join(lines))
    return path


@pytest.mark.parametrize(
    ("build", "line", "found"),
    [
        pytest.param(_short_row, 5, 31, id="too-few"),
        pytest.param(_long_row, 3, 34, id="too-many"),
        pytest.param(_quoted_comma_then_short, 2, 32, id="quoted-comma-hides-short"),
        pytest.param(
            _short_after_multiline_field, 5, 31, id="physical-line-after-multiline"
        ),
        pytest.param(
            _short_in_the_last_line_without_newline, 3, 32, id="last-line-no-newline"
        ),
    ],
)
def test_ragged_row_names_the_physical_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    build: Builder,
    line: int,
    found: int,
) -> None:
    path = build(tmp_path)
    with pytest.raises(RaggedRowError) as info:
        read_typed(path)
    assert (info.value.line, info.value.expected, info.value.found) == (line, 33, found)
    assert str(info.value) == (
        f"Data.csv line {line}: expected 33 fields, found {found}"
    )
    code, _, err = run_cli(capsys, path, tmp_path / "out")
    assert code == 2
    assert f"Data.csv line {line}: expected 33 fields, found {found}" in err
    assert not (tmp_path / "out").exists()


def test_ragged_row_in_an_archive_names_the_member(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = lines_of(write_csv_text(tmp_path / "x.csv", synthetic_rows(3)))
    bad[2] = bad[2].rsplit(b",", 1)[0]
    data = b"\r\n".join(bad) + b"\r\n"
    path = make_tar(tmp_path / "Data.tar.gz", {"extract/Rides.csv": data})
    code, _, err = run_cli(capsys, path, tmp_path / "out")
    assert code == 2
    assert "extract/Rides.csv line 3: expected 33 fields, found 32" in err


def test_blank_lines_are_not_ragged_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(tmp_path / "Data.csv", synthetic_rows(3))
    lines = lines_of(path)
    write_lines(path, [lines[0], b"", lines[1], lines[2], b"", lines[3]])
    out = tmp_path / "out"
    assert run_cli(capsys, path, out)[0] == 0
    assert report_of(out)["run"]["blank_lines"] == 2
    assert report_of(out)["run"]["rows"] == 3


# --- money values ---------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1,234.00", "$5", "n/a", "nan", "inf", "12,34"])
def test_unparseable_money_is_a_cast_failure_counted_by_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    rows = synthetic_rows(4, order_total_amount={0: value}, item_price={2: value})
    path = write_csv_text(tmp_path / "Data.csv", rows)
    out = tmp_path / "out"
    code, stdout, err = run_cli(capsys, path, out)
    assert code == 2
    assert stdout == ""
    assert "  order_total_amount: 1\n" in err
    assert "  item_price: 1" in err
    assert "order_tax" not in err  # clean columns aren't listed
    assert not out.exists()


def test_scientific_notation_is_accepted_as_the_number_it_means(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = synthetic_rows(3, order_total_amount={0: "1e3"}, order_tax={1: "1E-2"})
    path = write_csv_text(tmp_path / "Data.csv", rows)
    out = tmp_path / "out"
    assert run_cli(capsys, path, out)[0] in (0, 1)  # run completed, whatever it found
    curated = curated_of(out, "Data")
    assert curated["order_total_amount"][0] == 1000.0
    assert curated["order_tax"][1] == pytest.approx(0.01)
    assert checks_of(out)["cast_failures"]["passed"] is True


# --- invalid UTF-8 --------------------------------------------------------------------


def test_invalid_utf8_message_gives_the_exact_byte_offset(tmp_path: Path) -> None:
    path = write_csv_text(tmp_path / "Data.csv", synthetic_rows(3))
    clean = path.read_bytes()
    path.write_bytes(clean + b"caf\xe9\r\n")
    with pytest.raises(EncodingError) as info:
        read_typed(path)
    assert info.value.offset == len(clean) + 3
    assert f"byte offset {len(clean) + 3:,}" in str(info.value)


# --- archives -------------------------------------------------------------------------


def test_archive_with_a_sidecar_next_to_the_csv_ignores_the_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    members = {
        "._Data.csv": b"\x00\x05\x16\x07 not a csv",
        "dir/._Data.csv": b"\x00\x05\x16\x07",
        "dir/Data.csv": csv_bytes(tmp_path, 4),
    }
    out = tmp_path / "out"
    for suffix, path in (
        ("zip", zip_with(tmp_path / "Data.zip", members)),
        ("tar", make_tar(tmp_path / "Data.tar.bz2", members)),
    ):
        assert run_cli(capsys, path, out / suffix)[0] == 0
        assert report_of(out / suffix)["run"]["member"]["name"] == "dir/Data.csv"
        assert report_of(out / suffix)["run"]["rows"] == 4


def test_archive_without_a_csv_lists_its_members(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = zip_with(tmp_path / "Data.zip", {"a.txt": b"1", "sub/b.json": b"{}"})
    code, _, err = run_cli(capsys, path, tmp_path / "out")
    assert code == 2
    assert "a.txt, sub/b.json" in err


def test_a_traversal_member_is_rejected_and_nothing_is_extracted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], private_tmp: Path
) -> None:
    path = _zip_traversal(tmp_path)
    code, _, err = run_cli(capsys, path, tmp_path / "out")
    assert code == 2
    assert "unsafe member '../evil.csv'" in err
    assert not (tmp_path / "evil.csv").exists()
    assert not (tmp_path.parent / "evil.csv").exists()
    assert list(private_tmp.iterdir()) == []  # the extraction dir was removed
    with zipfile.ZipFile(path) as archive:  # the archive itself is untouched
        assert archive.namelist() == ["../evil.csv"]


# --- re-runs and concurrency ----------------------------------------------------------


def test_rerun_into_the_same_out_dir_is_idempotent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], full_synthetic_csv: Path
) -> None:
    out = tmp_path / "out"
    first_code, _, _ = run_cli(capsys, full_synthetic_csv, out)
    first_report = stable(report_of(out))
    first_frame = curated_of(out, "synthetic")
    first_files = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    assert first_report["outputs"]["samples"]  # a dirty file: samples were written
    second_code, _, _ = run_cli(capsys, full_synthetic_csv, out)
    assert second_code == first_code
    assert stable(report_of(out)) == first_report
    assert curated_of(out, "synthetic").equals(first_frame)
    assert (
        sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file()) == first_files
    )


def test_rerun_with_a_clean_input_leaves_no_stale_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], full_synthetic_csv: Path
) -> None:
    out = tmp_path / "out"
    run_cli(capsys, full_synthetic_csv, out)
    assert list((out / "samples").glob("*.parquet"))
    clean = write_csv_text(tmp_path / "clean.csv", synthetic_rows(3))
    assert run_cli(capsys, clean, out)[0] == 0
    assert list((out / "samples").glob("*.parquet")) == []
    assert report_of(out)["outputs"]["samples"] == []
    assert not list(out.rglob(".*tmp*"))  # no temporary files left behind


def _spawn(path: Path, out: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rideshare_data_pipeline",
            "run",
            "--input",
            str(path),
            "--out",
            str(out),
        ],
        env=child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(proc: subprocess.Popen[str]) -> int:
    _, err = proc.communicate(timeout=300)
    assert proc.returncode == 0, err
    return proc.returncode


def test_concurrent_runs_into_different_out_dirs_do_not_collide(
    tmp_path: Path,
) -> None:
    three = write_csv_text(tmp_path / "three.csv", synthetic_rows(3))
    seven = write_csv_text(tmp_path / "seven.csv", synthetic_rows(7))
    procs = [_spawn(three, tmp_path / "a"), _spawn(seven, tmp_path / "b")]
    for proc in procs:
        _finish(proc)
    assert report_of(tmp_path / "a")["run"]["rows"] == 3
    assert report_of(tmp_path / "b")["run"]["rows"] == 7
    assert curated_of(tmp_path / "a", "three").height == 3
    assert curated_of(tmp_path / "b", "seven").height == 7


def test_concurrent_runs_into_the_same_out_dir_leave_valid_files(
    tmp_path: Path,
) -> None:
    # Same stem, so both want the same file names; writes are atomic, so what is
    # left is one run's complete file, never a mix or a partial.
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    three = write_csv_text(tmp_path / "x" / "Data.csv", synthetic_rows(3))
    seven = write_csv_text(tmp_path / "y" / "Data.csv", synthetic_rows(7))
    out = tmp_path / "out"
    procs = [_spawn(three, out), _spawn(seven, out)]
    for proc in procs:
        _finish(proc)
    report = report_of(out)  # valid JSON
    assert report["run"]["rows"] in (3, 7)
    assert report["run"]["input_size_bytes"] in (
        three.stat().st_size,
        seven.stat().st_size,
    )
    assert curated_of(out, "Data").height in (3, 7)
    assert not list(out.glob(".*tmp*"))
