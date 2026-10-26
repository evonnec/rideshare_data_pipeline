"""Fixes for the second QA pass (#56): row conservation around blank lines, outputs
written only after the report, config validation and the report wording.

Every value is made up. N-numbers refer to the findings of the second QA report.
"""

import json
from pathlib import Path

import polars as pl
import pytest
from synthetic import synthetic_rows, write_csv_text

import rideshare_data_pipeline.cli as cli_module
import rideshare_data_pipeline.io as io_module
from rideshare_data_pipeline.cli import EXIT_INPUT_ERROR, EXIT_INTERNAL_ERROR, main
from rideshare_data_pipeline.io import (
    TotalOverflowError,
    blank_rows,
    read_raw,
    read_typed,
)
from rideshare_data_pipeline.money import _STEP_LABELS, spend_waterfall

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


def _checks(out: Path) -> dict[str, dict[str, object]]:
    document = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return {c["name"]: c for c in document["checks"]}


def test_required_fields_flags_the_all_empty_row_beside_a_blank_line(
    tmp_path: Path,
) -> None:
    path = _with_lines(tmp_path, 5, {2: COMMAS.encode() + b"\r\n"})
    out = tmp_path / "out"
    assert main(["run", "--input", str(path), "--out", str(out)]) in (0, 1)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["run"]["rows"] == 6
    assert report["run"]["blank_lines"] == 1
    assert _checks(out)["required_fields"]["n_failed"] == 1


# --- N2: overflow is an input error; outputs follow the report -----------------------


def _overflow_csv(tmp_path: Path) -> Path:
    rows = synthetic_rows(5, order_total_amount={0: "1e308", 1: "1e308", 2: "1e308"})
    return write_csv_text(tmp_path / "big.csv", rows)


def _files(out: Path) -> list[str]:
    if not out.exists():
        return []
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def test_an_overflowing_total_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    code = main(["run", "--input", str(_overflow_csv(tmp_path)), "--out", str(out)])
    err = capsys.readouterr().err
    assert code == EXIT_INPUT_ERROR
    assert "order_total_amount" in err
    assert "overflows" in err
    assert "internal error" not in err
    assert _files(out) == []


def test_a_single_huge_amount_still_runs(tmp_path: Path) -> None:
    rows = synthetic_rows(3, order_total_amount={0: "1.7e308"})
    path = write_csv_text(tmp_path / "one.csv", rows)
    out = tmp_path / "out"
    assert main(["run", "--input", str(path), "--out", str(out)]) in (0, 1)
    assert "one.curated.parquet" in _files(out)


def test_spend_waterfall_names_the_overflowing_step() -> None:
    row = {f"{n}_rows": 1 for n in _STEP_LABELS} | {
        f"{n}_amount": 1.0 for n in _STEP_LABELS
    }
    row["suspect_amount"] = float("inf")
    with pytest.raises(TotalOverflowError, match="currency-suspect total"):
        spend_waterfall(row)


def test_a_bug_while_building_the_report_leaves_no_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Rows that fail checks, so a sample file would be written in a normal run.
    rows = synthetic_rows(6, order_total_amount={0: "-5.00"})
    path = write_csv_text(tmp_path / "in.csv", rows)
    normal = tmp_path / "normal"
    main(["run", "--input", str(path), "--out", str(normal)])
    assert any(f.startswith("samples/") for f in _files(normal))

    def boom(*_args: object, **_kwargs: object) -> object:
        message = "boom"
        raise RuntimeError(message)

    monkeypatch.setattr(cli_module, "render_report", boom)
    out = tmp_path / "out"
    assert main(["run", "--input", str(path), "--out", str(out)]) == EXIT_INTERNAL_ERROR
    assert "internal error" in capsys.readouterr().err
    assert _files(out) == []


def test_a_failed_run_does_not_touch_an_earlier_runs_outputs(tmp_path: Path) -> None:
    good = write_csv_text(tmp_path / "in.csv", synthetic_rows(5))
    out = tmp_path / "out"
    main(["run", "--input", str(good), "--out", str(out)])
    before = {f: (out / f).read_bytes() for f in _files(out)}
    bad = _overflow_csv(tmp_path)
    assert main(["run", "--input", str(bad), "--out", str(out)]) == EXIT_INPUT_ERROR
    assert {f: (out / f).read_bytes() for f in _files(out)} == before
