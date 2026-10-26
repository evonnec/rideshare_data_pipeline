"""Fixes for the second QA pass (#56): row conservation around blank lines, outputs
written only after the report, config validation and the report wording.

Every value is made up. N-numbers refer to the findings of the second QA report.
"""

from pathlib import Path

import polars as pl
import pytest
from synthetic import synthetic_rows, write_csv_text

import rideshare_data_pipeline.io as io_module
from rideshare_data_pipeline.io import blank_rows, read_raw, read_typed

COMMAS = "," * 32 + "\r\n"


def _with_lines(tmp_path: Path, n: int, inserts: dict[int, bytes]) -> Path:
    """A synthetic CSV whose physical line k (0 = header) is followed by inserts[k]."""
    path = write_csv_text(tmp_path / "in.csv", synthetic_rows(n))
    lines = path.read_bytes().splitlines(keepends=True)
    out = bytearray()
    for k, line in enumerate(lines):
        out += line + inserts.get(k, b"")
    path.write_bytes(bytes(out))
    return path


# --- N1: an all-empty row survives blank lines --------------------------------------


@pytest.mark.parametrize(
    ("inserts", "blanks", "rows"),
    [
        ({}, 0, 5),
        ({2: COMMAS.encode()}, 0, 6),
        ({2: COMMAS.encode() + b"\r\n"}, 1, 6),
        ({2: b"\r\n" + COMMAS.encode()}, 1, 6),
        ({1: b"\r\n", 3: COMMAS.encode(), 5: b"\r\n\r\n"}, 3, 6),
        ({5: COMMAS.encode() + b"\r\n"}, 1, 6),  # trailing blank after the commas
        ({0: b"\r\n"}, 1, 5),  # blank right after the header
    ],
    ids=["clean", "commas", "commas-blank", "blank-commas", "mixed", "tail", "head"],
)
def test_only_blank_lines_are_dropped(
    tmp_path: Path, inserts: dict[int, bytes], blanks: int, rows: int
) -> None:
    path = _with_lines(tmp_path, 5, inserts)
    df, facts = read_typed(path)
    assert (facts.blank_lines, df.height) == (blanks, rows)
    expected_empty = sum(COMMAS.encode() in v for v in inserts.values())
    all_empty = df.filter(pl.col("checksum").is_null()).height
    assert all_empty == expected_empty
    # The lazy reader agrees with the eager one.
    assert typed_height(path) == rows


def typed_height(path: Path) -> int:
    return read_raw(path).collect().height


def test_blank_rows_are_found_by_the_line_pass_without_quotes(tmp_path: Path) -> None:
    path = _with_lines(tmp_path, 4, {1: b"\r\n", 3: COMMAS.encode() + b"\r\n"})
    assert io_module._blank_rows_by_lines(path) == [1, 5]
    assert blank_rows(path, 2) == [1, 5]


def test_a_quoted_multiline_field_falls_back_to_the_exact_locator(
    tmp_path: Path,
) -> None:
    # Row 0's subtitle holds a line break and a blank line; row 1 is followed by a
    # real blank line, then a row of commas. Physical lines no longer match records.
    rows = synthetic_rows(4, product_subtitle={0: "first\n\nsecond"})
    path = write_csv_text(tmp_path / "in.csv", rows)
    lines = path.read_bytes().split(b"\r\n")
    # lines: header, row0 (3 physical lines), row1, row2, row3, ""
    data = b"\r\n".join(lines[:3]) + b"\r\n\r\n" + COMMAS.encode()
    data += b"\r\n".join(lines[3:])
    path.write_bytes(data)
    assert io_module._blank_rows_by_lines(path) is None
    df, facts = read_typed(path)
    assert facts.blank_lines == 2  # the one inside the quotes counts, from the bytes
    assert df.height == 4 + 1
    assert df.filter(pl.col("checksum").is_null()).height == 1
    assert df["product_subtitle"][0] == "first\n\nsecond"
    assert blank_rows(path, facts.blank_lines) == [2]
