"""Structural checks: properties of whole columns and whole rows.

Each function takes a LazyFrame, runs a single query and reports what it found.
Nothing here drops columns or rows; a caller that wants a column gone must name
it explicitly (e.g. ``rideshare-qc --drop-columns``).
"""

from typing import NamedTuple

lazy import polars as pl


class DegenerateColumns(NamedTuple):
    """Columns that carry no information, in frame order."""

    all_null: list[str]
    """The frame has at least one row, and every value in the column is null."""

    constant: list[str]
    """The column's non-null values are all equal (and at least one is present)."""


def degenerate_columns(lf: pl.LazyFrame) -> DegenerateColumns:
    """Find the all-null and the constant columns, in one lazy select.

    A column is constant when its non-null values have exactly one distinct
    value: nulls are ignored, so ``[x, null, x]`` is constant. An all-null column
    has no non-null value, so it is reported only as all-null, never as constant.

    Edge cases: in a single-row frame every non-null column is constant and every
    null one is all-null. A frame with no rows (e.g. a header-only CSV) has
    neither, because there are no values to judge; a row-count check is the place
    to flag it.
    """
    names = lf.collect_schema().names()
    if not names:
        return DegenerateColumns(all_null=[], constant=[])
    flags = (
        lf.select(
            pl.struct(
                all_null=(pl.len() > 0) & (pl.col(c).null_count() == pl.len()),
                constant=pl.col(c).drop_nulls().n_unique() == 1,
            ).alias(c)
            for c in names
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    return DegenerateColumns(
        all_null=[c for c in names if flags[c]["all_null"]],
        constant=[c for c in names if flags[c]["constant"]],
    )


def all_null_columns(lf: pl.LazyFrame) -> list[str]:
    """Return the columns whose every value is null (see degenerate_columns)."""
    return degenerate_columns(lf).all_null


def constant_columns(lf: pl.LazyFrame) -> list[str]:
    """Return the columns with one distinct non-null value (see degenerate_columns)."""
    return degenerate_columns(lf).constant


def duplicate_row_count(lf: pl.LazyFrame) -> int:
    """Count the rows that exactly repeat an earlier row; nulls compare equal.

    This is the number of copies beyond the first of each repeated row, so three
    identical rows count as 2. Every column takes part in the comparison.
    """
    if not lf.collect_schema().names():
        return 0
    # Identical rows hash alike, so only rows whose 64-bit row hash repeats can be
    # duplicates; comparing just those exactly avoids building a full-width key
    # for every row (and a hash collision can't produce a false count).
    candidates = lf.filter(pl.struct(pl.all()).hash().is_duplicated()).collect(
        engine="streaming"
    )
    return candidates.height - candidates.n_unique()
