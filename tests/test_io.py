"""Archive extraction, header and encoding validation, typed reads and the writers."""

import datetime as dt
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from synthetic import (
    make_tar,
    make_zip,
    read_typed_frame,
    synthetic_rows,
    write_csv_text,
)

import rideshare_data_pipeline.io as io_module
from rideshare_data_pipeline.io import (
    CastError,
    EncodingError,
    FileFacts,
    HeaderError,
    InputError,
    cast_failures,
    extract_archive,
    extract_tar,
    extract_zip,
    input_csv,
    input_stem,
    inspect_csv,
    read_raw,
    read_typed,
    typed,
    write_csv,
    write_parquet,
)
from rideshare_data_pipeline.schema import (
    EXPECTED_COLUMNS,
    casts,
    csv_schema,
    raw_schema,
)

CSV_BYTES = b"a,b\r\n1,2\r\n"

# --- raw and typed reads ----------------------------------------------------------


def test_read_raw_is_lazy_and_all_string(synthetic_csv: Path) -> None:
    lf = read_raw(synthetic_csv)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect_schema() == raw_schema()
    assert set(lf.collect_schema().values()) == {pl.String()}


def test_typed_yields_exactly_the_cast_dtypes(synthetic_csv: Path) -> None:
    df = typed(read_raw(synthetic_csv)).collect()
    assert df.schema == csv_schema()
    for col, dtype in casts().items():
        assert df.schema[col] == dtype
    first = df.row(0, named=True)
    assert first["order_time"] == dt.datetime(2019, 1, 1, 8)
    assert first["start_source_folder_date"] == dt.date(2019, 2, 1)
    assert first["order_total_amount"] == 10.5
    assert first["order_total_qty"] == 1.0
    assert first["order_pickup"] == 0
    # empty fields are null, whatever the dtype
    assert first["order_points"] is None
    assert first["SKU"] is None


def test_read_typed_returns_the_frame_and_facts(synthetic_csv: Path) -> None:
    df, facts = read_typed(synthetic_csv)
    assert df.schema == csv_schema()
    assert df.height == 3
    assert facts == FileFacts(
        size_bytes=synthetic_csv.stat().st_size,
        has_bom=False,
        line_ending="crlf",
        header=EXPECTED_COLUMNS,
    )


def test_header_only_file_reads_as_zero_rows(tmp_path: Path) -> None:
    df, _ = read_typed(write_csv_text(tmp_path / "empty.csv", []))
    assert df.height == 0
    assert df.schema == csv_schema()


def test_empty_file_raises_input_error(tmp_path: Path) -> None:
    path = tmp_path / "zero.csv"
    path.touch()
    with pytest.raises(InputError, match="is empty"):
        read_typed(path)


def test_row_with_too_many_fields_raises_input_error(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    text = synthetic_csv.read_bytes()
    path = tmp_path / "ragged.csv"
    path.write_bytes(text + text.splitlines()[-1] + b",one too many\r\n")
    with pytest.raises(InputError, match=re.escape("can't parse ragged.csv")):
        read_typed(path)


def test_row_with_too_few_fields_is_padded_with_nulls(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    # Known gap, left to the edge-case suite (#23): polars pads the short row, so
    # nothing flags it. This test pins the behaviour so a change is noticed.
    lines = synthetic_csv.read_bytes().splitlines(keepends=True)
    path = tmp_path / "short.csv"
    path.write_bytes(b"".join(lines[:2]) + b"Uber,only-two\r\n")
    df, _ = read_typed(path)
    assert df.height == 2
    assert df["dttimestamp"][1] is None


# --- cast failures ----------------------------------------------------------------


def failures(path: Path) -> dict[str, int]:
    return cast_failures(read_raw(path))


def test_clean_file_has_zero_cast_failures_for_every_cast_column(
    synthetic_csv: Path,
) -> None:
    assert failures(synthetic_csv) == dict.fromkeys(casts(), 0)


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


def test_cast_failures_per_column(tmp_path: Path) -> None:
    counts = failures(bad_values_csv(tmp_path))
    assert {col: n for col, n in counts.items() if n} == EXPECTED_FAILURES
    assert counts.keys() == casts().keys()


def test_typed_nulls_failures_and_failures_count_them(tmp_path: Path) -> None:
    raw = read_raw(bad_values_csv(tmp_path))
    df = typed(raw).collect()
    assert df["order_total_amount"][0] is None  # "12,34" became null...
    assert cast_failures(raw)["order_total_amount"] == 1  # ...and was counted


def test_empty_fields_are_not_failures(tmp_path: Path) -> None:
    rows = synthetic_rows(2, order_tax={0: ""}, order_time={1: ""})
    assert not any(failures(write_csv_text(tmp_path / "e.csv", rows)).values())


@pytest.mark.parametrize("value", ["1,234.00", "12,34", "$5.00", "n/a", "1.2.3", " 5"])
def test_malformed_money_is_a_failure(tmp_path: Path, value: str) -> None:
    rows = synthetic_rows(2, item_price={1: value})
    counts = failures(write_csv_text(tmp_path / "m.csv", rows))
    assert counts["item_price"] == 1


def test_scientific_notation_parses_and_is_accepted(tmp_path: Path) -> None:
    # polars' Float64 cast reads "1e3" as 1000.0. That is the exact value the
    # writer meant, so it is accepted rather than flagged (as are "nan" and "inf",
    # which later value checks must watch for).
    path = write_csv_text(
        tmp_path / "sci.csv", synthetic_rows(2, order_tax={0: "1e3", 1: "1E-2"})
    )
    assert failures(path)["order_tax"] == 0
    assert typed(read_raw(path)).collect()["order_tax"].to_list() == [1000.0, 0.01]


@pytest.mark.parametrize(
    "value",
    [
        "2019-01-02T08:00:00.000",  # ISO "T"
        "2019/01/02 08:00:00.000",
        "2019-01-02 08:00:00.5",  # not three fractional digits
        "2019-01-02 08:00",
        "yesterday",
    ],
)
def test_timestamp_in_another_format_is_a_failure(tmp_path: Path, value: str) -> None:
    rows = synthetic_rows(3, update_time={2: value})
    assert failures(write_csv_text(tmp_path / "t.csv", rows))["update_time"] == 1


def test_timestamp_without_fractional_seconds_is_accepted(tmp_path: Path) -> None:
    # chrono's "%.3f" treats the fraction as optional, so whole seconds parse.
    path = write_csv_text(
        tmp_path / "t.csv", synthetic_rows(2, update_time={1: "2019-01-02 08:00:00"})
    )
    assert failures(path)["update_time"] == 0
    assert typed(read_raw(path)).collect()["update_time"][1] == dt.datetime(
        2019, 1, 2, 8
    )


def test_cast_failures_runs_one_query(
    monkeypatch: pytest.MonkeyPatch, synthetic_csv: Path
) -> None:
    raw = read_raw(synthetic_csv)
    collect = pl.LazyFrame.collect
    calls: list[pl.LazyFrame] = []

    def counting_collect(self: pl.LazyFrame, *args: Any, **kwargs: Any) -> Any:
        calls.append(self)
        return collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", counting_collect)
    cast_failures(raw)
    assert len(calls) == 1


def test_read_typed_raises_cast_error_with_counts(tmp_path: Path) -> None:
    with pytest.raises(CastError) as info:
        read_typed(bad_values_csv(tmp_path))
    assert isinstance(info.value, InputError)
    assert info.value.counts == EXPECTED_FAILURES
    for col, n in EXPECTED_FAILURES.items():
        assert f"{col}: {n}" in str(info.value)


# --- header validation ------------------------------------------------------------


def header_error(tmp_path: Path, header: Sequence[str]) -> HeaderError:
    path = write_csv_text(tmp_path / "header.csv", synthetic_rows(1), header=header)
    with pytest.raises(HeaderError) as info:
        read_raw(path)
    assert isinstance(info.value, InputError)
    return info.value


def test_missing_column(tmp_path: Path) -> None:
    err = header_error(tmp_path, [c for c in EXPECTED_COLUMNS if c != "order_tax"])
    assert err.missing == ["order_tax"]
    assert (err.extra, err.duplicated, err.reordered) == ([], [], False)
    assert "missing 'order_tax'" in str(err)


def test_extra_column(tmp_path: Path) -> None:
    err = header_error(tmp_path, (*EXPECTED_COLUMNS, "surprise", "another"))
    assert err.extra == ["surprise", "another"]
    assert err.missing == []
    assert "extra 'surprise', 'another'" in str(err)


def test_duplicated_column(tmp_path: Path) -> None:
    err = header_error(tmp_path, (*EXPECTED_COLUMNS, "order_tax"))
    assert err.duplicated == ["order_tax"]
    assert "duplicated 'order_tax'" in str(err)


def test_renamed_column_is_missing_and_extra(tmp_path: Path) -> None:
    err = header_error(tmp_path, ("merchant", *EXPECTED_COLUMNS[1:]))
    assert (err.missing, err.extra) == (["merchant_name"], ["merchant"])


def test_reordered_columns(tmp_path: Path) -> None:
    swapped = list(EXPECTED_COLUMNS)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    err = header_error(tmp_path, swapped)
    assert err.reordered
    assert (err.missing, err.extra, err.duplicated) == ([], [], [])
    assert "column 4 is 'email_time', expected 'order_time'" in str(err)


def test_header_with_a_trailing_comma_has_an_unnamed_extra_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trailing.csv"
    path.write_bytes((",".join(EXPECTED_COLUMNS) + ",\r\n").encode())
    with pytest.raises(HeaderError) as info:
        read_raw(path)
    assert info.value.extra == [""]


def test_header_is_parsed_as_csv_so_quoted_commas_stay_in_the_name(
    tmp_path: Path,
) -> None:
    # "a,b" is one (unknown) column, not two: a naive split would report two extras
    # and lose the real name.
    path = tmp_path / "quoted.csv"
    header = ",".join(EXPECTED_COLUMNS[:-1]) + ',"odd,name"\r\n'
    path.write_bytes(header.encode())
    with pytest.raises(HeaderError) as info:
        read_raw(path)
    assert info.value.extra == ["odd,name"]
    assert info.value.missing == [EXPECTED_COLUMNS[-1]]


def test_quoted_expected_names_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "quoted_ok.csv"
    header = ",".join(f'"{c}"' for c in EXPECTED_COLUMNS) + "\r\n"
    path.write_bytes(header.encode())
    assert inspect_csv(path).header == EXPECTED_COLUMNS


def test_header_error_precedes_any_scan(tmp_path: Path) -> None:
    # read_raw raises before returning a frame, not when it is collected.
    path = write_csv_text(tmp_path / "h.csv", [], header=EXPECTED_COLUMNS[:-1])
    with pytest.raises(HeaderError):
        read_raw(path)


# --- encoding, BOM and line endings -------------------------------------------------


def test_invalid_byte_at_the_start(synthetic_csv: Path, tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(b"\xff" + synthetic_csv.read_bytes())
    with pytest.raises(EncodingError) as info:
        read_raw(path)
    assert isinstance(info.value, InputError)
    assert info.value.offset == 0
    assert "byte offset 0" in str(info.value)


def test_invalid_byte_after_a_multibyte_character(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes("caf\u00e9,".encode() + b"\xc3\x28")  # 0xc3 needs a continuation
    with pytest.raises(EncodingError) as info:
        inspect_csv(path)
    assert info.value.offset == 6  # c a f (2 bytes) , -> the 0xc3 is the 7th byte


@pytest.mark.parametrize("delta", [-1, 0, 1, 2])
def test_invalid_byte_offset_is_exact_across_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta: int
) -> None:
    monkeypatch.setattr(io_module, "_CHUNK_BYTES", 1024)
    header = (",".join(EXPECTED_COLUMNS) + "\r\n").encode()
    offset = 3 * 1024 + delta  # in or next to the fourth chunk's first bytes
    filler = b"x" * (offset - len(header))
    path = tmp_path / "deep.csv"
    path.write_bytes(header + filler + b"\xfe" + b"y" * 3000)
    with pytest.raises(EncodingError) as info:
        inspect_csv(path)
    assert info.value.offset == offset


def test_invalid_byte_deep_in_a_file_larger_than_one_chunk(tmp_path: Path) -> None:
    header = (",".join(EXPECTED_COLUMNS) + "\r\n").encode()
    offset = io_module._CHUNK_BYTES + 12_345
    path = tmp_path / "deep.csv"
    path.write_bytes(header + b"x" * (offset - len(header)) + b"\x80tail")
    with pytest.raises(EncodingError) as info:
        inspect_csv(path)
    assert info.value.offset == offset
    assert f"{offset:,}" in str(info.value)


def test_multibyte_character_split_across_chunks_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(io_module, "_CHUNK_BYTES", 1024)
    header = (",".join(EXPECTED_COLUMNS) + "\r\n").encode()
    # Place a 3-byte character so that it straddles the 1024-byte boundary.
    pad = b"x" * (1023 - len(header))
    path = tmp_path / "split.csv"
    path.write_bytes(header + pad + "\u20ac".encode() * 5 + b"\r\n")
    assert inspect_csv(path).line_ending == "crlf"


def test_truncated_final_sequence_is_an_encoding_error(tmp_path: Path) -> None:
    path = tmp_path / "cut.csv"
    path.write_bytes((",".join(EXPECTED_COLUMNS) + "\r\n").encode() + b"ab\xe2\x82")
    with pytest.raises(EncodingError, match="truncated") as info:
        inspect_csv(path)
    assert info.value.offset == len(",".join(EXPECTED_COLUMNS)) + 2 + 2


def test_bom_is_recorded_and_stripped(tmp_path: Path, synthetic_csv: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + synthetic_csv.read_bytes())
    facts = inspect_csv(path)
    assert facts.has_bom
    assert facts.header[0] == "merchant_name"
    assert_frame_equal(read_typed(path)[0], read_typed(synthetic_csv)[0])


def test_bom_with_crlf_together(tmp_path: Path, synthetic_csv: Path) -> None:
    path = tmp_path / "bom_crlf.csv"
    path.write_bytes(b"\xef\xbb\xbf" + synthetic_csv.read_bytes())
    df, facts = read_typed(path)
    assert (facts.has_bom, facts.line_ending) == (True, "crlf")
    assert df.columns == list(EXPECTED_COLUMNS)


def test_crlf_and_lf_parse_identically(tmp_path: Path, synthetic_csv: Path) -> None:
    lf_csv = tmp_path / "lf.csv"
    lf_csv.write_bytes(synthetic_csv.read_bytes().replace(b"\r\n", b"\n"))
    lf_df, lf_facts = read_typed(lf_csv)
    crlf_df, crlf_facts = read_typed(synthetic_csv)
    assert_frame_equal(lf_df, crlf_df)
    assert (lf_facts.line_ending, crlf_facts.line_ending) == ("lf", "crlf")


def test_mixed_line_endings(tmp_path: Path, synthetic_csv: Path) -> None:
    lines = synthetic_csv.read_bytes().split(b"\r\n")
    path = tmp_path / "mixed.csv"
    path.write_bytes(b"\r\n".join(lines[:2]) + b"\n" + b"\r\n".join(lines[2:]))
    df, facts = read_typed(path)
    assert facts.line_ending == "mixed"
    assert df.height == 3


def test_crlf_split_across_a_chunk_boundary_is_counted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(io_module, "_CHUNK_BYTES", 1024)
    # The "\r" is the last byte of the first chunk and the "\n" the first of the
    # second; missing the pair would leave a lone "\n" and report "mixed".
    path = tmp_path / "boundary.csv"
    path.write_bytes(b"x" * 1023 + b"\r\n" + b"a\r\n")
    assert io_module._scan_bytes(path) == (1023 + 2 + 3, False, "crlf")


def test_file_without_a_line_break_is_reported_as_lf(tmp_path: Path) -> None:
    path = tmp_path / "one_line.csv"
    path.write_text(",".join(EXPECTED_COLUMNS))
    assert inspect_csv(path).line_ending == "lf"


def test_encoding_check_reports_the_size(synthetic_csv: Path) -> None:
    assert inspect_csv(synthetic_csv).size_bytes == synthetic_csv.stat().st_size


def test_importing_the_schema_and_io_modules_leaves_polars_unimported() -> None:
    code = (
        "import sys\n"
        "import rideshare_data_pipeline.schema, rideshare_data_pipeline.io\n"
        "sys.exit('polars' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    assert (
        subprocess.run([sys.executable, "-c", code], env=env, check=False).returncode
        == 0
    )


# --- writers ----------------------------------------------------------------------


def test_write_csv_round_trips_with_the_input_formats(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    df = read_typed_frame(synthetic_csv)
    out = tmp_path / "out.csv"
    write_csv(df, out)
    assert out.read_text().splitlines()[1].count("2019-01-01 08:00:00.000") == 1
    assert_frame_equal(read_typed_frame(out), df)


def test_write_parquet_keeps_every_column_and_dtype(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    df = read_typed_frame(synthetic_csv)
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
        assert_frame_equal(read_typed_frame(path), read_typed_frame(synthetic_csv))
    assert not path.exists()
    assert not path.parent.exists()


def test_input_csv_removes_the_temporary_directory_on_error(
    tmp_path: Path,
) -> None:
    archive = make_tar(tmp_path / "in.tar.bz2", {"Data.csv": b""})
    with pytest.raises(InputError, match="is empty"), input_csv(archive) as path:
        read_typed(path)
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
