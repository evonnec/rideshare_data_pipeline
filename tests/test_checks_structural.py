"""All-null, constant and duplicate-row detection (issue #7's regression tests)."""

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from synthetic import ALL_NULL, CONSTANT, synthetic_rows, write_csv_text

from rideshare_data_pipeline.checks.structural import (
    DegenerateColumns,
    all_null_columns,
    constant_columns,
    degenerate_columns,
    duplicate_row_count,
)
from rideshare_data_pipeline.io import read_typed_csv, scan_typed_csv
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS


def lazy(data: dict[str, list[Any]], **schema: pl.DataType) -> pl.LazyFrame:
    return pl.LazyFrame(data, schema_overrides=schema)


# --- constant columns -------------------------------------------------------------


def test_two_value_string_column_with_first_equal_to_last_is_not_constant() -> None:
    # The legacy check compared the first value with the and-reduction of the rest,
    # so a column whose first and last values matched looked constant; that is how
    # merchant_name and from_domain were dropped from the output.
    lf = lazy({"merchant_name": ["Uber", "Lyft", "Lyft", "Uber"]})
    assert constant_columns(lf) == []


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 0],  # starts with 0 and holds another 0: legacy said 0 == False
        [0.0, 2.5, 0.0],
        [1, 2, 3],  # starts with 1 and holds no 0: legacy said 1 == True
        [1.0, 7.0, 9.0],
    ],
)
def test_numeric_columns_with_several_values_are_not_constant(
    values: list[float],
) -> None:
    assert constant_columns(lazy({"n": values})) == []


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (["Travel", "Travel", "Travel"], pl.String()),
        ([1.0, 1.0, 1.0], pl.Float64()),
        ([0, 0, 0], pl.Int64()),
        ([True, True, True], pl.Boolean()),
        ([dt.date(2020, 2, 18)] * 3, pl.Date()),
        ([dt.datetime(2019, 1, 1, 9)] * 3, pl.Datetime("ms")),
    ],
)
def test_column_with_one_value_is_constant(
    values: list[Any], dtype: pl.DataType
) -> None:
    assert constant_columns(lazy({"c": values}, c=dtype)) == ["c"]


def test_nulls_are_ignored_when_judging_constant() -> None:
    lf = lazy({"flag": ["x", None, "x", None], "mixed": ["x", None, "y", None]})
    assert constant_columns(lf) == ["flag"]


def test_constant_columns_come_back_in_frame_order() -> None:
    lf = lazy({"b": [1, 1], "a": [2, 2], "varies": [1, 2], "c": ["z", "z"]})
    assert constant_columns(lf) == ["b", "a", "c"]


# --- all-null columns -------------------------------------------------------------


@pytest.mark.parametrize("dtype", [pl.String(), pl.Float64(), pl.Null()])
def test_all_null_column_is_reported_as_all_null_not_constant(
    dtype: pl.DataType,
) -> None:
    lf = lazy({"empty": [None, None, None], "ok": [1, 2, 3]}, empty=dtype)
    assert degenerate_columns(lf) == DegenerateColumns(all_null=["empty"], constant=[])


def test_partly_null_column_is_not_all_null() -> None:
    assert all_null_columns(lazy({"c": [None, "x", None]}, c=pl.String())) == []


# --- small frames -----------------------------------------------------------------


def test_single_row_frame_every_non_null_column_is_constant() -> None:
    lf = lazy(
        {"s": ["x"], "n": [0], "f": [1.5], "missing": [None]},
        missing=pl.String(),
    )
    assert degenerate_columns(lf) == DegenerateColumns(
        all_null=["missing"], constant=["s", "n", "f"]
    )


def test_header_only_frame_reports_nothing_and_does_not_raise() -> None:
    # No rows means no values to judge: nothing is all-null or constant. A row
    # count check is the place to flag an empty input.
    lf = pl.LazyFrame(schema={"s": pl.String(), "n": pl.Int64()})
    assert degenerate_columns(lf) == DegenerateColumns(all_null=[], constant=[])
    assert duplicate_row_count(lf) == 0


def test_header_only_csv_reports_nothing(tmp_path: Path) -> None:
    path = tmp_path / "header.csv"
    path.write_text(",".join(EXPECTED_COLUMNS) + "\r\n")
    lf = scan_typed_csv(path)
    assert degenerate_columns(lf) == DegenerateColumns(all_null=[], constant=[])
    assert duplicate_row_count(lf) == 0


def test_frame_without_columns() -> None:
    assert degenerate_columns(pl.LazyFrame()) == DegenerateColumns([], [])
    assert duplicate_row_count(pl.LazyFrame()) == 0


# --- on the real column layout ----------------------------------------------------


def test_synthetic_csv_columns(synthetic_csv: Path) -> None:
    # 3 rows: merchant_name is Uber, Lyft, Uber (first == last, not constant).
    found = degenerate_columns(scan_typed_csv(synthetic_csv))
    assert found.all_null == list(ALL_NULL)
    assert found.constant == list(CONSTANT)
    assert "merchant_name" not in found.constant


def test_wrappers_match_degenerate_columns(synthetic_csv: Path) -> None:
    lf = read_typed_csv(synthetic_csv).lazy()
    assert all_null_columns(lf) == degenerate_columns(lf).all_null
    assert constant_columns(lf) == degenerate_columns(lf).constant


def test_both_are_found_in_a_single_query(
    monkeypatch: pytest.MonkeyPatch, synthetic_csv: Path
) -> None:
    lf = scan_typed_csv(synthetic_csv)
    collect = pl.LazyFrame.collect
    calls: list[pl.LazyFrame] = []

    def counting_collect(self: pl.LazyFrame, *args: Any, **kwargs: Any) -> Any:
        calls.append(self)
        return collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", counting_collect)
    degenerate_columns(lf)
    assert len(calls) == 1


# --- detection never drops ----------------------------------------------------------


def test_detection_leaves_the_frame_unchanged(synthetic_csv: Path) -> None:
    df = read_typed_csv(synthetic_csv)
    before = df.clone()
    degenerate_columns(df.lazy())
    duplicate_row_count(df.lazy())
    assert df.columns == list(EXPECTED_COLUMNS)
    assert df.equals(before)


# --- duplicate rows ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": [1, 2, 3], "b": ["x", "y", "z"]}, 0),
        ({"a": [1, 1], "b": ["x", "x"]}, 1),
        ({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]}, 2),  # copies beyond the first
        ({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]}, 2),
        ({"a": [1, 1], "b": ["x", "y"]}, 0),  # every column takes part
        ({"a": [None, None], "b": [None, None]}, 1),  # nulls compare equal
        ({"a": [None, 1], "b": ["x", "x"]}, 0),
        ({"a": [7], "b": ["x"]}, 0),
    ],
)
def test_duplicate_row_count(data: dict[str, list[Any]], expected: int) -> None:
    lf = lazy(data, a=pl.Int64(), b=pl.String())
    assert duplicate_row_count(lf) == expected


def test_duplicate_row_count_on_the_real_layout(tmp_path: Path) -> None:
    rows = synthetic_rows(3)
    path = write_csv_text(tmp_path / "dups.csv", [*rows, rows[1], rows[1]])
    assert duplicate_row_count(scan_typed_csv(path)) == 2
