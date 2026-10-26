"""Structural and completeness checks: properties of whole columns and whole rows.

Each function takes a LazyFrame, runs a single query and reports what it found.
Nothing here drops columns or rows; a caller that wants a column gone must name
it explicitly (e.g. ``rideshare-qc --drop-columns``).

The catalog rows: S1 header and S3 encoding (from the byte-level facts), S2 cast
failures, C1 required fields, S4 all-null and S5 constant columns, S6 ID formats,
C2 ride completeness, and a row-count band. S1, S2 and S3 can't fail on a run that
got as far as a frame (a failure is an input error, exit 2, and nothing is
written), so the CLI records them as passing results built from what the reader
learned.
"""

from collections.abc import Mapping
from typing import NamedTuple

lazy import polars as pl

from rideshare_data_pipeline.checks.model import (
    POSITION,
    CheckResult,
    DatasetCheck,
    FlaggedCheck,
    RowRule,
    Severity,
    rows_at,
)
from rideshare_data_pipeline.io import FileFacts
from rideshare_data_pipeline.parse import input_columns
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

REQUIRED_COLUMNS: tuple[str, ...] = (
    "merchant_name",
    "user_id",
    "email_time",
    "checksum",
    "from_domain",
    "email_subject",
)
"""C1: columns that must have a value in every row."""

# C2: the most a ride's field may be null, as a rate among rides that are not
# superseded. A ride without a total is unusable for spend, so its limit is a
# rounding error (0.01%, about 1 ride in 10,000), not 0: the real file has 2 such
# rides among ~1.6M and a hard 0 would fail every run on a known blemish. The order
# number and time may be missing a little more often (0.11% and 0.26% on the
# real file), so they get 0.5%.
RIDE_NULL_LIMITS: dict[str, float] = {
    "order_total_amount": 0.0001,
    "order_number": 0.005,
    "order_time": 0.005,
}

_USER_ID = r"^[0-9a-f]{64}$"
_CHECKSUM = r"^[0-9a-f]{32}$"
_LYFT_ORDER = r"^[0-9]{18,19}$"
_UBER_ORDER = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


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


def duplicate_row_positions(lf: pl.LazyFrame) -> pl.Series:
    """The positions (``POSITION``) of the rows that exactly repeat an earlier row.

    Nulls compare equal. These are the copies beyond the first of each repeated
    row, so three identical rows give 2 positions. Every column of `lf` takes part
    in the comparison.
    """
    columns = lf.collect_schema().names()
    if not columns:
        return pl.Series(POSITION, [], dtype=pl.UInt32)
    key = pl.struct(columns)
    # Identical rows hash alike, so only rows whose 64-bit row hash repeats can be
    # duplicates; comparing just those exactly avoids building a full-width key
    # for every row (and a hash collision can't produce a false count).
    candidates = (
        lf.with_row_index(POSITION)
        .filter(key.hash().is_duplicated())
        .collect(engine="streaming")
    )
    return candidates.filter(~key.is_first_distinct())[POSITION]


def duplicate_row_count(lf: pl.LazyFrame) -> int:
    """Count the rows that exactly repeat an earlier row (see above)."""
    return duplicate_row_positions(lf).len()


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


def _all_null_result(found: DegenerateColumns, n_columns: int) -> CheckResult:
    return _info(
        "all_null_columns",
        "S4",
        "every column has a non-null value",
        found.all_null,
        n_columns,
    )


def _constant_result(found: DegenerateColumns, n_columns: int) -> CheckResult:
    return _info(
        "constant_columns",
        "S5",
        "every column has more than one distinct non-null value",
        found.constant,
        n_columns,
    )


def all_null_columns_check(lf: pl.LazyFrame) -> CheckResult:
    """S4: the all-null columns, as an info result (see degenerate_columns)."""
    lf = input_columns(lf)
    return _all_null_result(degenerate_columns(lf), len(lf.collect_schema()))


def constant_columns_check(lf: pl.LazyFrame) -> CheckResult:
    """S5: the constant columns, as an info result (see degenerate_columns)."""
    lf = input_columns(lf)
    return _constant_result(degenerate_columns(lf), len(lf.collect_schema()))


def column_profile_checks() -> tuple[DatasetCheck, DatasetCheck]:
    """S4 and S5 as a pair of checks that share one aggregation.

    Both read the same null/constant profile, so the first one called on a frame
    computes it and the second reuses it (the pair remembers the last frame it was
    given, by identity). Each call builds a fresh pair, so nothing outlives a run.
    """
    last: list[tuple[pl.LazyFrame, DegenerateColumns, int]] = []

    def profile(lf: pl.LazyFrame) -> tuple[DegenerateColumns, int]:
        if last and last[0][0] is lf:
            return last[0][1], last[0][2]
        inputs = input_columns(lf)
        n_columns = len(inputs.collect_schema())
        found = degenerate_columns(inputs)
        last[:] = [(lf, found, n_columns)]
        return found, n_columns

    def all_null(lf: pl.LazyFrame) -> CheckResult:
        return _all_null_result(*profile(lf))

    def constant(lf: pl.LazyFrame) -> CheckResult:
        return _constant_result(*profile(lf))

    return all_null, constant


def _duplicate_rows_result(lf: pl.LazyFrame, n_failed: int) -> CheckResult:
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


def duplicate_rows_check(lf: pl.LazyFrame) -> CheckResult:
    """An error result counting exact duplicate rows (see duplicate_row_count).

    n_evaluated is the row count, which costs a second (cheap) pass. Derived
    columns (parse.derived_columns) don't take part in the comparison.
    """
    lf = input_columns(lf)
    return _duplicate_rows_result(lf, duplicate_row_count(lf))


def duplicate_rows() -> FlaggedCheck:
    """U0 with its row flag: every copy after the first of an exactly repeated row.

    The flag sums to the check's ``n_failed``. The pair finds the rows once (the
    check and the flag share the scan, remembered for the last frame by identity).
    """
    last: list[tuple[pl.LazyFrame, pl.Series]] = []

    def later(lf: pl.LazyFrame) -> pl.Series:
        if not (last and last[0][0] is lf):
            last[:] = [(lf, duplicate_row_positions(input_columns(lf)))]
        return last[0][1]

    return FlaggedCheck(
        "duplicate_rows",
        lambda lf: _duplicate_rows_result(input_columns(lf), later(lf).len()),
        lambda lf: rows_at(later(lf)),
    )


def cast_failures_check(counts: Mapping[str, int]) -> CheckResult:
    """S2: wrap io.cast_failures() counts (per cast column) as a result.

    That count needs the raw String frame, which a typed LazyFrame no longer has,
    so it isn't a DatasetCheck: the caller computes the counts and wraps them
    here. Counted as failures: a value that doesn't parse, a Float64 that isn't
    finite (``nan``, ``inf``, ``1e400``) and a timestamp or date outside the years
    1 to 9999.

    A run that reports this check had a strict parse that passed: any failure is a
    CastError (exit 2, nothing written) before the checks run, so on a written
    report it is always a pass, and the config can't disable it (see config.py).

    ``n_failed`` is the number of columns with failures, ``n_evaluated`` the number
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


def header_valid_check(facts: FileFacts) -> CheckResult:
    """S1: the header is the 33 expected columns, in order.

    A wrong header already raised HeaderError before a frame existed, so on a run
    that got here this passes; it is built from the facts anyway, not assumed.
    ``details``: ``columns`` (the count) and ``in_order``.
    """
    in_order = facts.header == EXPECTED_COLUMNS
    return CheckResult(
        name="header_valid",
        check_id="S1",
        category="structural",
        severity=Severity.ERROR,
        passed=in_order,
        metric=0.0 if in_order else 1.0,
        threshold=0.0,
        n_failed=0 if in_order else 1,
        n_evaluated=1,
        predicate=f"the header is the {len(EXPECTED_COLUMNS)} columns, in order",
        details={"columns": len(facts.header), "in_order": in_order},
    )


def utf8_valid_check(facts: FileFacts) -> CheckResult:
    """S3: the file is valid UTF-8.

    Invalid bytes already raised EncodingError, so this passes on any run with a
    frame. ``details``: ``has_bom``, ``line_ending`` and ``blank_lines`` (empty
    lines, dropped on reading and not an error).
    """
    return CheckResult(
        name="utf8_valid",
        check_id="S3",
        category="structural",
        severity=Severity.ERROR,
        passed=True,
        metric=0.0,
        threshold=0.0,
        n_failed=0,
        n_evaluated=1,
        predicate="the file is valid UTF-8",
        details={
            "has_bom": facts.has_bom,
            "line_ending": facts.line_ending,
            "blank_lines": facts.blank_lines,
        },
    )


def row_count_band_check(
    n_rows: int, min_rows: int | None, max_rows: int | None
) -> CheckResult:
    """A warn result: the row count lies in [min_rows, max_rows] (None: no bound).

    ``metric`` is the row count (not a rate). ``threshold`` is the bound that was
    broken, else the lower bound, else the upper one. A header-only file (0 rows)
    fails under the default ``min_rows`` of 1. ``n_failed`` is 1 when the count is
    out of band, ``n_evaluated`` 1.
    """
    below = min_rows is not None and n_rows < min_rows
    above = max_rows is not None and n_rows > max_rows
    if below:
        threshold = min_rows
    elif above:
        threshold = max_rows
    else:
        threshold = min_rows if min_rows is not None else max_rows
    low = "0" if min_rows is None else f"{min_rows:,}"
    high = "no limit" if max_rows is None else f"{max_rows:,}"
    return CheckResult(
        name="row_count_band",
        check_id="C3",
        category="completeness",
        severity=Severity.WARN,
        passed=not (below or above),
        metric=float(n_rows),
        threshold=None if threshold is None else float(threshold),
        n_failed=int(below or above),
        n_evaluated=1,
        predicate=f"the row count is between {low} and {high}",
        details={"rows": n_rows, "min_rows": min_rows, "max_rows": max_rows},
    )


def rules() -> tuple[RowRule, ...]:
    """C1 and S6, the row rules (built on call, so import stays cheap)."""
    missing = {c: pl.col(c).is_null() for c in REQUIRED_COLUMNS}
    order = pl.col("order_number")
    merchant = pl.col("merchant_name")
    # A null never fails a format (it is C1's, or C2's for the order number): each
    # test is "has a value and it is malformed", and `str.contains` of a null is null.
    bad_ids = {
        "user_id": ~pl.col("user_id").str.contains(_USER_ID),
        "checksum": ~pl.col("checksum").str.contains(_CHECKSUM),
        "lyft_order_number": (merchant == "Lyft") & ~order.str.contains(_LYFT_ORDER),
        "uber_order_number": (merchant == "Uber") & ~order.str.contains(_UBER_ORDER),
    }
    return (
        RowRule(
            name="required_fields",
            check_id="C1",
            category="completeness",
            severity=Severity.ERROR,
            rule=f"{', '.join(REQUIRED_COLUMNS)} are never null",
            predicate=pl.any_horizontal(*missing.values()),
            breakdown=missing,
        ),
        RowRule(
            name="id_formats",
            check_id="S6",
            category="structural",
            severity=Severity.WARN,
            rule=(
                "user_id is 64 hex digits, checksum 32; a Lyft order number is "
                "18-19 digits, an Uber one a lowercase UUID (nulls are not judged)"
            ),
            predicate=pl.any_horizontal(*bad_ids.values()),
            breakdown=bad_ids,
        ),
    )


def ride_completeness_check(lf: pl.LazyFrame) -> CheckResult:
    """C2 (warn): rides carry a total, an order number and an order time.

    Looks at rides (kind ``ride``) that are not superseded. Each field has a limit
    on its null rate (RIDE_NULL_LIMITS); the check fails when any is over. All three
    counts come from one select. ``n_failed`` is the rides missing at least one of
    the three, ``n_evaluated`` the rides. ``metric`` and ``threshold`` are the rate
    and the limit of the field closest to (or furthest over) its limit; ``details``
    holds ``rates`` and ``limits`` for all three and the ``worst`` field.
    """
    fields = tuple(RIDE_NULL_LIMITS)
    names = lf.collect_schema().names()
    ride = pl.col("receipt_kind").cast(pl.String) == "ride"
    if "is_superseded" in names:
        ride = ride & ~pl.col("is_superseded").fill_null(False)
    counts = (
        lf.select(
            ride.sum().alias("rides"),
            (ride & pl.any_horizontal(pl.col(c).is_null() for c in fields))
            .sum()
            .alias("any"),
            *((ride & pl.col(c).is_null()).sum().alias(c) for c in fields),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    n_rides = counts["rides"]
    rates = {c: counts[c] / n_rides if n_rides else 0.0 for c in fields}
    worst = max(fields, key=lambda c: rates[c] / RIDE_NULL_LIMITS[c])
    passed = all(rates[c] <= RIDE_NULL_LIMITS[c] for c in fields)
    return CheckResult(
        name="ride_completeness",
        check_id="C2",
        category="completeness",
        severity=Severity.WARN,
        passed=passed,
        metric=rates[worst] if n_rides else None,
        threshold=RIDE_NULL_LIMITS[worst],
        n_failed=counts["any"],
        n_evaluated=n_rides,
        predicate=(
            "rides have an order_total_amount, order_number and order_time: each "
            "null rate within its limit"
        ),
        details={"rates": rates, "limits": dict(RIDE_NULL_LIMITS), "worst": worst},
    )


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The structural checks that run on a typed frame, for the registry.

    S4 and S5 share one aggregation (see column_profile_checks).
    """
    all_null, constant = column_profile_checks()
    return (
        all_null,
        constant,
        duplicate_rows(),
        ride_completeness_check,
    )
