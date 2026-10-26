"""Arithmetic checks (A1-A7): relations between a receipt's money columns.

They run on the frame with the derived columns of ``parse.derived_columns()`` and
the dedupe flag ``is_superseded`` (``parse.with_derived`` and ``supersede_flags``).

``total = subtotal + tax - discount`` does NOT hold row by row: tips, tolls and fees
are not itemized, and many credits are missing from ``order_discount``. On the real
file only about a third of ride rows match, so the identity is reported (A7, info)
and only the gross violation of it is a warning (A6). The narrower relations that do
hold are rules: A1 (a rider-spend total is not negative), A2 (item price equals the
subtotal), A3 (zero-fare rides), A4 (discount within subtotal) and A5 (tax within
total).

Money is compared in whole cents (``cents``): a float difference such as
``10.02 - 10.01`` is 0.00999... or 0.01000...1 by luck, and a tolerance of 0.01 must
not depend on that. Nulls: A2, A4 and A5 skip a row with a null operand. A6 and A7
treat a null tax or discount as 0 (the receipt did not list one) but a null total
or subtotal makes the row "not computable".

Zero-total rides stay in spend sums but are not fares: the outlier baselines (#20)
use ``money.positive_usd_rides``. A3 only flags them; A1 owns the negatives.
"""

lazy import polars as pl

from rideshare_data_pipeline.checks.model import (
    KEY_COLS,
    CheckResult,
    DatasetCheck,
    RowRule,
    Severity,
)
from rideshare_data_pipeline.dedupe import supersede_flags

RIDE_KINDS: tuple[str, ...] = ("ride", "tip_update")
"""The ride rows: a ride and a tip update (which supersedes the ride it amends)."""

TOLERANCE_CENTS = 1
"""A difference of up to one cent is rounding, not an error (0.01 in dollars)."""

RESIDUAL_FLOOR_CENTS = 2000
"""A6 tolerates an unexplained residual up to max($20, subtotal)."""

_MONEY_SAMPLE_COLS = (
    *KEY_COLS,
    "receipt_kind",
    "item_price",
    "order_subtotal",
    "order_tax",
    "order_discount",
)


def cents(column: str) -> pl.Expr:
    """A money column in whole cents (a float that is integral), null stays null."""
    return (pl.col(column) * 100).round(0)


def is_ride_kind() -> pl.Expr:
    """True for ride and tip-update rows."""
    return pl.col("receipt_kind").cast(pl.String).is_in(RIDE_KINDS)


def _both(a: str, b: str) -> pl.Expr:
    return pl.col(a).is_not_null() & pl.col(b).is_not_null()


def residual_cents() -> pl.Expr:
    """``total - (subtotal + tax - discount)`` in cents; a null tax or discount is 0.

    Null when the total or the subtotal is null (not computable). Positive means
    the total is above what the itemized lines explain (a tip, a fee), negative
    that it is below (an unlisted credit).
    """
    tax = cents("order_tax").fill_null(0.0)
    discount = cents("order_discount").fill_null(0.0)
    return cents("order_total_amount") - (cents("order_subtotal") + tax - discount)


def residual_population() -> pl.Expr:
    """The rows A6 and A7 judge: USD rider-spend ride rows that are not superseded.

    Unlike ``money.is_usd_rider_spend`` this keeps rows with a null total, which
    A7 reports as not computable rather than leaving out.
    """
    return (
        pl.col("is_rider_spend")
        & ~pl.col("currency_suspect")
        & ~pl.col("is_superseded")
        & is_ride_kind()
    )


def unexplained_residual() -> pl.Expr:
    """True where the residual exceeds max($20, subtotal); null if not computable.

    Positive-only, not ``abs``: a residual far BELOW zero is an unlisted credit or
    refund (the total is lower than the itemized lines), which is the benign side;
    the warning is for a total far above what is itemized. On the real file the
    positive-only count is 2,011 against 2,179 for ``abs`` (the 168 extra are
    negative); the review's figure was 2,024, which only the positive-only
    reading approximates (the gap is its slightly different population).
    """
    limit = pl.max_horizontal(
        pl.lit(float(RESIDUAL_FLOOR_CENTS)), cents("order_subtotal")
    )
    return residual_cents() > limit


def _zero_fare_by_merchant(zero: pl.Expr) -> dict[str, pl.Expr]:
    merchant = pl.col("merchant_name")
    named = merchant.is_in(["Uber", "Lyft"]).fill_null(False)
    return {
        "Uber": zero & (merchant == "Uber"),
        "Lyft": zero & (merchant == "Lyft"),
        "other": zero & ~named,
    }


def rules() -> tuple[RowRule, ...]:
    """Return this category's row rules (built on call, so import stays cheap)."""
    zero_fare = pl.col("order_total_amount") == 0
    return (
        RowRule(
            name="non_negative_total",
            check_id="A1",
            category="arithmetic",
            severity=Severity.ERROR,
            rule="order_total_amount >= 0 on rider-spend receipts",
            predicate=pl.col("order_total_amount") < 0,
            # A refund on a bike, pass or other non-ride receipt is not a rider-spend
            # error (on the real file every negative total is a bike or scooter).
            applies_to=pl.col("is_rider_spend"),
        ),
        RowRule(
            name="item_price_equals_subtotal",
            check_id="A2",
            category="arithmetic",
            severity=Severity.ERROR,
            rule="|item_price - order_subtotal| <= 0.01 where both are present",
            predicate=(cents("item_price") - cents("order_subtotal")).abs()
            > TOLERANCE_CENTS,
            applies_to=_both("item_price", "order_subtotal"),
            sample_cols=_MONEY_SAMPLE_COLS,
        ),
        RowRule(
            name="zero_fare_rides",
            check_id="A3",
            category="arithmetic",
            severity=Severity.WARN,
            rule=(
                "order_total_amount is not exactly 0 on ride and tip-update "
                "receipts (superseded versions excluded)"
            ),
            predicate=zero_fare,
            applies_to=is_ride_kind() & ~pl.col("is_superseded"),
            sample_cols=(*KEY_COLS, "receipt_kind"),
            breakdown=_zero_fare_by_merchant(zero_fare),
        ),
        RowRule(
            name="discount_within_subtotal",
            check_id="A4",
            category="arithmetic",
            severity=Severity.WARN,
            rule="order_discount <= order_subtotal + 0.01 where both are present",
            predicate=cents("order_discount") - cents("order_subtotal")
            > TOLERANCE_CENTS,
            applies_to=_both("order_discount", "order_subtotal"),
            sample_cols=_MONEY_SAMPLE_COLS,
        ),
        RowRule(
            name="tax_within_total",
            check_id="A5",
            category="arithmetic",
            severity=Severity.WARN,
            rule="order_tax <= order_total_amount + 0.01 where both are present",
            predicate=cents("order_tax") - cents("order_total_amount")
            > TOLERANCE_CENTS,
            applies_to=_both("order_tax", "order_total_amount"),
            sample_cols=_MONEY_SAMPLE_COLS,
        ),
        RowRule(
            name="unexplained_residual",
            check_id="A6",
            category="arithmetic",
            severity=Severity.WARN,
            rule=(
                "total - (subtotal + tax - discount) <= max(20, subtotal) on "
                "USD rider-spend ride receipts (a null tax or discount is 0)"
            ),
            predicate=unexplained_residual(),
            applies_to=residual_population(),
            sample_cols=_MONEY_SAMPLE_COLS,
        ),
    )


def identity_share_check(lf: pl.LazyFrame) -> CheckResult:
    """A7 (info): the share of ride rows where ``total = subtotal + tax - discount``.

    Over the A6 population (USD rider-spend ride rows, not superseded). A row
    "holds" when ``|residual| <= 0.01``; ``above`` and ``below`` are the rest (the
    total over or under the itemized lines) and ``not_computable`` the rows with a
    null total or subtotal. ``metric`` is holds / population (rows that are not
    computable count as not holding) and the check always passes: the identity is
    not a rule because tips, tolls and fees are not itemized.

    ``details``: ``share``, ``holds``, ``above``, ``below``, ``not_computable``,
    ``median_gap_above`` (the median positive residual in dollars, None if no row
    is above) and ``population``.
    """
    if "is_superseded" not in lf.collect_schema().names():
        lf = supersede_flags(lf)
    gap = residual_cents()
    row = (
        lf.filter(residual_population())
        .select(
            pl.len().alias("n"),
            (gap.abs() <= TOLERANCE_CENTS).sum().alias("holds"),
            (gap > TOLERANCE_CENTS).sum().alias("above"),
            (gap < -TOLERANCE_CENTS).sum().alias("below"),
            gap.is_null().sum().alias("not_computable"),
            gap.filter(gap > TOLERANCE_CENTS).median().alias("median_above"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    n, holds = int(row["n"]), int(row["holds"])
    median = row["median_above"]
    share = holds / n if n else None
    return CheckResult(
        name="identity_share",
        check_id="A7",
        category="arithmetic",
        severity=Severity.INFO,
        passed=True,
        metric=share,
        threshold=None,
        n_failed=holds,
        n_evaluated=n,
        predicate="total = subtotal + tax - discount (within 0.01), reported only",
        details={
            "share": share,
            "holds": holds,
            "above": int(row["above"]),
            "below": int(row["below"]),
            "not_computable": int(row["not_computable"]),
            "median_gap_above": None if median is None else round(median / 100, 2),
            "population": n,
        },
    )


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The arithmetic checks that need the whole dataset, for the registry."""
    return (identity_share_check,)
