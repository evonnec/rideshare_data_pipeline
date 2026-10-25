"""Archive extraction, the typed scan, strict parsing and the writers."""

import datetime as dt
import re
import tarfile
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from synthetic import make_tar, make_zip, synthetic_rows, write_csv_text

from rideshare_data_pipeline.io import (
    InputError,
    count_parse_failures,
    extract_archive,
    extract_tar,
    extract_zip,
    input_csv,
    input_stem,
    read_typed_csv,
    scan_typed_csv,
    write_csv,
    write_parquet,
)
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS, csv_schema

CSV_BYTES = b"a,b\r\n1,2\r\n"

# --- the typed scan -------------------------------------------------------------


def test_scan_is_lazy_and_typed(synthetic_csv: Path) -> None:
    lf = scan_typed_csv(synthetic_csv)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect_schema() == csv_schema()


def test_read_parses_every_column_to_its_dtype(synthetic_csv: Path) -> None:
    df = read_typed_csv(synthetic_csv)
    assert df.schema == csv_schema()
    assert df.height == 3
    first = df.row(0, named=True)
    assert first["order_time"] == dt.datetime(2019, 1, 1, 8)
    assert first["start_source_folder_date"] == dt.date(2019, 2, 1)
    assert first["order_total_amount"] == 10.5
    assert first["order_pickup"] == 0
    # empty fields are null, whatever the dtype
    assert first["order_points"] is None
    assert first["SKU"] is None


def test_read_accepts_lf_line_endings(tmp_path: Path, synthetic_csv: Path) -> None:
    lf_csv = tmp_path / "lf.csv"
    lf_csv.write_bytes(synthetic_csv.read_bytes().replace(b"\r\n", b"\n"))
    assert_frame_equal(read_typed_csv(lf_csv), read_typed_csv(synthetic_csv))


def test_header_only_file_reads_as_zero_rows(tmp_path: Path) -> None:
    df = read_typed_csv(write_csv_text(tmp_path / "empty.csv", []))
    assert df.height == 0
    assert df.schema == csv_schema()


# --- strict parsing ---------------------------------------------------------------


def test_clean_file_has_no_parse_failures(synthetic_csv: Path) -> None:
    assert count_parse_failures(synthetic_csv) == {}


def bad_values_csv(tmp_path: Path) -> Path:
    rows = synthetic_rows(
        4,
        order_time={1: "2019/01/02 08:00"},  # wrong timestamp format
        start_source_folder_date={2: "02-03-2019"},  # wrong date format
        order_total_amount={0: "12,34"},  # decimal comma (quoted by the writer)
        digital_transaction={1: "x", 3: "1.5"},  # not integers
    )
    return write_csv_text(tmp_path / "bad.csv", rows)


EXPECTED_FAILURES = {
    "order_time": 1,
    "order_total_amount": 1,
    "digital_transaction": 2,
    "start_source_folder_date": 1,
}


def test_count_parse_failures_per_column(tmp_path: Path) -> None:
    assert count_parse_failures(bad_values_csv(tmp_path)) == EXPECTED_FAILURES


def test_unparseable_value_raises_value_error_with_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="don't parse as their dtype") as info:
        read_typed_csv(bad_values_csv(tmp_path))
    assert isinstance(info.value, InputError)
    assert str(EXPECTED_FAILURES) in str(info.value)


def test_unparseable_value_is_never_silently_null(tmp_path: Path) -> None:
    rows = synthetic_rows(2, order_tax={1: "n/a"})
    with pytest.raises(InputError, match=r"\{'order_tax': 1\}"):
        read_typed_csv(write_csv_text(tmp_path / "bad.csv", rows))


@pytest.mark.parametrize(
    ("header", "case"),
    [
        (EXPECTED_COLUMNS[:-1], "missing column"),
        ((*EXPECTED_COLUMNS, "surprise"), "extra column"),
        (("merchant", *EXPECTED_COLUMNS[1:]), "renamed column"),
    ],
)
def test_header_mismatch_raises(
    tmp_path: Path, header: tuple[str, ...], case: str
) -> None:
    path = write_csv_text(tmp_path / "header.csv", synthetic_rows(1), header=header)
    with pytest.raises(InputError, match=r"header .* doesn't match the schema") as info:
        read_typed_csv(path)
    assert isinstance(info.value.__cause__, pl.exceptions.SchemaError), case


def test_empty_file_raises_input_error(tmp_path: Path) -> None:
    path = tmp_path / "zero.csv"
    path.touch()
    with pytest.raises(InputError, match="is empty"):
        read_typed_csv(path)


def test_row_with_too_many_fields_raises_input_error(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    text = synthetic_csv.read_bytes()
    path = tmp_path / "ragged.csv"
    path.write_bytes(text + text.splitlines()[-1] + b",one too many\r\n")
    with pytest.raises(InputError, match=re.escape("can't parse ragged.csv")):
        read_typed_csv(path)


# --- writers ----------------------------------------------------------------------


def test_write_csv_round_trips_with_the_input_formats(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    df = read_typed_csv(synthetic_csv)
    out = tmp_path / "out.csv"
    write_csv(df, out)
    assert out.read_text().splitlines()[1].count("2019-01-01 08:00:00.000") == 1
    assert_frame_equal(read_typed_csv(out), df)


def test_write_parquet_keeps_every_column_and_dtype(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    df = read_typed_csv(synthetic_csv)
    out = tmp_path / "out.parquet"
    write_parquet(df, out)
    assert_frame_equal(pl.read_parquet(out), df)


# --- archives ---------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".tar.bz2", ".tar.gz", ".tar.xz", ".tar"])
def test_extract_tar(tmp_path: Path, suffix: str) -> None:
    archive = make_tar(tmp_path / f"data{suffix}", {"dir/data.csv": CSV_BYTES})
    extract_tar(archive, tmp_path / "dest")
    assert (tmp_path / "dest" / "dir" / "data.csv").read_bytes() == CSV_BYTES


@pytest.mark.parametrize(
    ("members", "symlinks"),
    [
        ({"../evil.csv": CSV_BYTES}, {}),
        ({"ok/../../evil.csv": CSV_BYTES}, {}),
        ({}, {"evil.csv": "../outside.csv"}),
        ({}, {"evil.csv": "/etc/passwd"}),
    ],
    ids=["dotdot", "nested-dotdot", "symlink-up", "symlink-absolute"],
)
def test_extract_tar_rejects_members_outside_the_destination(
    tmp_path: Path, members: dict[str, bytes], symlinks: dict[str, str]
) -> None:
    archive = make_tar(tmp_path / "evil.tar.bz2", members, symlinks)
    dest = tmp_path / "a" / "dest"
    with pytest.raises(tarfile.FilterError):
        extract_tar(archive, dest)
    assert not (tmp_path / "a" / "evil.csv").exists()
    assert not (dest / "evil.csv").exists()


def test_extract_tar_strips_absolute_member_paths(tmp_path: Path) -> None:
    archive = make_tar(tmp_path / "abs.tar.bz2", {"/abs/data.csv": CSV_BYTES})
    extract_tar(archive, tmp_path / "dest")
    assert (tmp_path / "dest" / "abs" / "data.csv").read_bytes() == CSV_BYTES


def test_extract_zip(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "data.zip", {"dir/data.csv": CSV_BYTES})
    extract_zip(archive, tmp_path / "dest")
    assert (tmp_path / "dest" / "dir" / "data.csv").read_bytes() == CSV_BYTES


def test_extract_zip_keeps_dotdot_members_inside_the_destination(
    tmp_path: Path,
) -> None:
    archive = make_zip(tmp_path / "evil.zip", {"../evil.csv": CSV_BYTES})
    dest = tmp_path / "a" / "dest"
    extract_zip(archive, dest)
    assert not (tmp_path / "a" / "evil.csv").exists()
    assert (dest / "evil.csv").read_bytes() == CSV_BYTES


def test_extract_archive_returns_the_single_csv(tmp_path: Path) -> None:
    archive = make_tar(
        tmp_path / "data.tar.bz2", {"nested/data.csv": CSV_BYTES, "README": b"hi"}
    )
    assert extract_archive(archive, tmp_path / "dest") == (
        tmp_path / "dest" / "nested" / "data.csv"
    )


def test_extract_archive_ignores_macos_sidecars(tmp_path: Path) -> None:
    members = {"._data.csv": b"\x00\x05\x16\x07", "data.csv": CSV_BYTES}
    archive = make_tar(tmp_path / "data.tar.bz2", members)
    assert extract_archive(archive, tmp_path / "dest") == tmp_path / "dest" / "data.csv"


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ({"README": b"hi"}, "found 0: none"),
        ({"._data.csv": b"\x00\x05\x16\x07"}, "found 0: none"),
        ({"a.csv": CSV_BYTES, "b.csv": CSV_BYTES}, "found 2: a.csv, b.csv"),
    ],
)
def test_extract_archive_needs_exactly_one_csv(
    tmp_path: Path, members: dict[str, bytes], message: str
) -> None:
    archive = make_zip(tmp_path / "data.zip", members)
    with pytest.raises(InputError, match=message):
        extract_archive(archive, tmp_path / "dest")


def test_extract_archive_wraps_extraction_errors(tmp_path: Path) -> None:
    archive = make_tar(tmp_path / "evil.tar.bz2", {"../evil.csv": CSV_BYTES})
    with pytest.raises(
        InputError, match=re.escape("can't extract evil.tar.bz2")
    ) as info:
        extract_archive(archive, tmp_path / "dest")
    assert isinstance(info.value.__cause__, tarfile.OutsideDestinationError)


@pytest.mark.parametrize("name", ["data.tar.bz2", "data.zip"])
def test_extract_archive_rejects_corrupt_archives(tmp_path: Path, name: str) -> None:
    archive = tmp_path / name
    archive.write_bytes(b"not an archive")
    with pytest.raises(InputError, match=re.escape(f"can't extract {name}")):
        extract_archive(archive, tmp_path / "dest")


def test_extract_archive_rejects_a_csv(synthetic_csv: Path, tmp_path: Path) -> None:
    with pytest.raises(InputError, match="is a CSV, not an archive"):
        extract_archive(synthetic_csv, tmp_path / "dest")


# --- input resolution -------------------------------------------------------------


def test_input_csv_yields_a_csv_as_is(synthetic_csv: Path) -> None:
    with input_csv(synthetic_csv) as path:
        assert path == synthetic_csv
    assert synthetic_csv.exists()


@pytest.mark.parametrize("suffix", [".tar.bz2", ".zip"])
def test_input_csv_extracts_archives_to_a_temporary_directory(
    tmp_path: Path, synthetic_csv: Path, suffix: str
) -> None:
    members = {"Data.csv": synthetic_csv.read_bytes()}
    archive = tmp_path / f"in{suffix}"
    if suffix == ".zip":
        make_zip(archive, members)
    else:
        make_tar(archive, members)
    with input_csv(archive) as path:
        assert path.name == "Data.csv"
        assert not path.is_relative_to(tmp_path)
        assert_frame_equal(read_typed_csv(path), read_typed_csv(synthetic_csv))
    assert not path.exists()
    assert not path.parent.exists()


def test_input_csv_removes_the_temporary_directory_on_error(
    tmp_path: Path,
) -> None:
    archive = make_tar(tmp_path / "in.tar.bz2", {"Data.csv": b""})
    with pytest.raises(InputError, match="is empty"), input_csv(archive) as path:
        read_typed_csv(path)
    assert not path.exists()


def test_input_csv_rejects_a_missing_file(tmp_path: Path) -> None:
    with (
        pytest.raises(InputError, match="input not found"),
        input_csv(tmp_path / "nope.csv"),
    ):
        pass


def test_input_csv_rejects_unsupported_names(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("{}")
    with (
        pytest.raises(InputError, match=re.escape("unsupported input 'data.json'")),
        input_csv(path),
    ):
        pass


@pytest.mark.parametrize(
    ("name", "stem"),
    [
        ("Data.csv", "Data"),
        ("Data.CSV", "Data"),
        ("Data.tar.bz2", "Data"),
        ("Data.tgz", "Data"),
        ("Data.tar", "Data"),
        ("Data.v2.zip", "Data.v2"),
        ("dir/Data.tar.xz", "Data"),
    ],
)
def test_input_stem_strips_the_whole_suffix(name: str, stem: str) -> None:
    assert input_stem(name) == stem
