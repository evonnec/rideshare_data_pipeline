"""Structural checks: properties of whole columns and whole rows.

Each function takes a LazyFrame, runs a single query and reports what it found.
Nothing here drops columns or rows; a caller that wants a column gone must name
it explicitly (e.g. ``rideshare-qc --drop-columns``).
"""

from collections.abc import Mapping
from typing import NamedTuple

lazy import polars as pl

from rideshare_data_pipeline.checks.model import CheckResult, DatasetCheck, Severity


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


def _info(
    name: str, check_id: str, rule: str, columns: list[str], n_evaluated: int
) -> CheckResult:
    """An info finding: reports the columns and always passes."""
    return CheckResult(
        name=name,
        check_id=check_id,
        category="structural",
        severity=Severity.INFO,
        passed=True,
        metric=len(columns) / n_evaluated if n_evaluated else None,
        threshold=None,
        n_failed=len(columns),
        n_evaluated=n_evaluated,
        predicate=rule,
        details={"columns": columns},
    )


def all_null_columns_check(lf: pl.LazyFrame) -> CheckResult:
    """S4: the all-null columns, as an info result (see degenerate_columns)."""
    found = degenerate_columns(lf).all_null
    return _info(
        "all_null_columns",
        "S4",
        "every column has a non-null value",
        found,
        len(lf.collect_schema()),
    )


def constant_columns_check(lf: pl.LazyFrame) -> CheckResult:
    """S5: the constant columns, as an info result (see degenerate_columns)."""
    found = degenerate_columns(lf).constant
    return _info(
        "constant_columns",
        "S5",
        "every column has more than one distinct non-null value",
        found,
        len(lf.collect_schema()),
    )


def duplicate_rows_check(lf: pl.LazyFrame) -> CheckResult:
    """An error result counting exact duplicate rows (see duplicate_row_count).

    n_evaluated is the row count, which costs a second (cheap) pass.
    """
    n_failed = duplicate_row_count(lf)
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    return CheckResult(
        name="duplicate_rows",
        check_id="U0",
        category="structural",
        severity=Severity.ERROR,
        passed=n_failed == 0,
        metric=n_failed / n_rows if n_rows else None,
        threshold=0.0,
        n_failed=n_failed,
        n_evaluated=n_rows,
        predicate="no row repeats an earlier row exactly",
    )


def cast_failures_check(counts: Mapping[str, int]) -> CheckResult:
    """S2: wrap io.cast_failures() counts (per cast column) as a result.

    That count needs the raw String frame, which a typed LazyFrame no longer has,
    so it isn't a DatasetCheck: the caller computes the counts and wraps them
    here. n_failed is the number of columns with failures, n_evaluated the number
    of cast columns; the per-column counts are in `details`.
    """
    bad = {c: n for c, n in counts.items() if n}
    return CheckResult(
        name="cast_failures",
        check_id="S2",
        category="structural",
        severity=Severity.ERROR,
        passed=not bad,
        metric=len(bad) / len(counts) if counts else None,
        threshold=0.0,
        n_failed=len(bad),
        n_evaluated=len(counts),
        predicate="every non-null raw value converts to its column's type",
        details={"failures": bad},
    )


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The structural checks that run on a typed frame, for the registry."""
    return (all_null_columns_check, constant_columns_check, duplicate_rows_check)
