"""Temporal checks (T1-T6): the receipt's timestamps against each other.

The timezone assumption (owner decision, from the evidence below)
=================================================================

``order_time`` is the trip's **local** time (the rider's wall clock) and
``email_time`` is **UTC**. The evidence, in the sample: the Lyft lag
``email_time - order_time`` is never negative and clusters at whole-hour US
offsets plus a few minutes for the receipt, and every row whose ``order_time``
is after its ``email_time`` comes from a sender east of UTC (Asia-Pacific). So the
two columns are not on one clock, and a plain ``order_time <= email_time`` is wrong.

Consequence: T2 compares **dates**, with one day of tolerance. A trip at 08:00 on
the 12th in UTC+10 is at 22:00 on the 11th UTC, so the receipt's UTC date can be
a day *before* the local order date, never more (the offsets span at most 14 h).
Without the tolerance some real rows would fail; with it none do.

The upsert assumption
=====================

The vendor upserts receipts: an updated receipt (a tip added to a ride, say) is a new
row with the original ``insert_time``, so ``insert_time`` can precede
``email_time``. That is normal, not an error: T1 orders the vendor's own stamps
(insert, update, source, batch) and never involves ``email_time``; T5 flags the
opposite oddity (an email *after* the vendor's last update).

Date-only ``order_time`` (T6)
=============================

Uber ``order_time`` is a date at midnight on nearly every row: the time of day was
not recorded. T6 reports the share per merchant. "Midnight" here means "time unknown":
a polars Datetime cannot tell it from a real 00:00:00.000, and no check compares times
of day for that reason (T2 to T4 use dates only).

The checks
==========

All run on every row, superseded rows included (they describe the delivered data,
not spend), and a null operand is never a failure: a rule that needs two columns
applies only to the rows that have both.

- T1 ``ingestion_order`` (error): ``insert_time <= update_time <= source_dttimestamp
  <= dttimestamp``; ``by_field`` counts each broken link.
- T2 ``order_before_receipt`` (error): ``date(order_time) <= date(email_time) + 1 day``;
  ``by_field.order_after_email_no_tolerance`` counts ``order_time > email_time``.
- T3 ``late_receipts`` (warn): the email date is more than 7 days after the order
  date; ``by_field.over_30_days`` counts the rows more than 30 days late.
- T4 ``order_before_window`` (warn): ``order_time`` is more than 7 days before the
  window start (the earliest ``email_time`` date of the frame).
- T5 ``email_after_update`` (warn): ``email_time > update_time``.
- T6 ``date_only_order_time`` (info): per merchant, the share of ``order_time``
  values at exactly 00:00:00.000.

T4 is a row rule whose predicate holds the aggregate ``email_time.min()``. Inside one
select that is a scalar broadcast against the rows (not the length-broadcast literal
``pl.repeat`` that made the streaming engine panic in a ``filter`` plus ``head``), so
T4 shares the runner's single scan, and the sample query (a filter with the same
aggregate) is a second pass that runs only if T4 fails. The window is the frame's, so
a run on a cached or sliced frame measures against its own minimum.
"""

import datetime

lazy import polars as pl

from rideshare_data_pipeline.checks.model import (
    KEY_COLS,
    CheckResult,
    DatasetCheck,
    NamedCheck,
    RowRule,
    Severity,
)

LATE_DAYS = 7
"""T3: a receipt this many days after the trip is late (strictly more than)."""

VERY_LATE_DAYS = 30
"""T3 reports separately the receipts more than this many days late."""

WINDOW_SLACK_DAYS = 7
"""T4: an order this many days before the window start is still plausible."""

TOLERANCE_DAYS = 1
"""T2: the UTC date of the receipt may be this many days before the local order date."""

_TIME_COLS = (
    *KEY_COLS,
    "order_time",
    "insert_time",
    "update_time",
    "source_dttimestamp",
    "dttimestamp",
)


def _both(a: str, b: str) -> pl.Expr:
    return pl.col(a).is_not_null() & pl.col(b).is_not_null()


def _order_date() -> pl.Expr:
    return pl.col("order_time").dt.date()


def _email_date() -> pl.Expr:
    return pl.col("email_time").dt.date()


def days_late() -> pl.Expr:
    """Whole days from the order date to the email date (negative: email earlier)."""
    return (_email_date() - _order_date()).dt.total_days()


def window_start() -> pl.Expr:
    """The earliest email date of the frame (a scalar; null on an empty frame)."""
    return _email_date().min()


def days_before_window() -> pl.Expr:
    """Whole days the order date is before the window start (negative: inside)."""
    return (window_start() - _order_date()).dt.total_days()


def ingestion_breaks() -> dict[str, pl.Expr]:
    """T1's three links, each True where the later stamp is earlier than the prior."""
    insert, update = pl.col("insert_time"), pl.col("update_time")
    source, batch = pl.col("source_dttimestamp"), pl.col("dttimestamp")
    return {
        "insert_after_update": insert > update,
        "update_after_source": update > source,
        "source_after_dttimestamp": source > batch,
    }


def rules() -> tuple[RowRule, ...]:
    """Return this category's row rules (built on call, so import stays cheap)."""
    links = ingestion_breaks()
    first, second, third = links.values()
    return (
        RowRule(
            name="ingestion_order",
            check_id="T1",
            category="temporal",
            severity=Severity.ERROR,
            rule=(
                "insert_time <= update_time <= source_dttimestamp <= dttimestamp "
                "where present"
            ),
            predicate=first | second | third,
            sample_cols=_TIME_COLS,
            breakdown=links,
        ),
        RowRule(
            name="order_before_receipt",
            check_id="T2",
            category="temporal",
            severity=Severity.ERROR,
            rule=(
                f"date(order_time) <= date(email_time) + {TOLERANCE_DAYS} day "
                "(order_time is local, email_time UTC)"
            ),
            predicate=days_late() < -TOLERANCE_DAYS,
            applies_to=_both("order_time", "email_time"),
            sample_cols=(*KEY_COLS, "order_time"),
            breakdown={
                "order_after_email_no_tolerance": pl.col("order_time")
                > pl.col("email_time")
            },
        ),
        RowRule(
            name="late_receipts",
            check_id="T3",
            category="temporal",
            severity=Severity.WARN,
            rule=(
                f"date(email_time) - date(order_time) <= {LATE_DAYS} days "
                "where both are present"
            ),
            predicate=days_late() > LATE_DAYS,
            applies_to=_both("order_time", "email_time"),
            sample_cols=(*KEY_COLS, "order_time"),
            breakdown={"over_30_days": days_late() > VERY_LATE_DAYS},
        ),
        RowRule(
            name="order_before_window",
            check_id="T4",
            category="temporal",
            severity=Severity.WARN,
            rule=(
                f"date(order_time) >= window start - {WINDOW_SLACK_DAYS} days "
                "(window start: the earliest email date)"
            ),
            predicate=days_before_window() > WINDOW_SLACK_DAYS,
            applies_to=_both("order_time", "email_time"),
            sample_cols=(*KEY_COLS, "order_time"),
        ),
        RowRule(
            name="email_after_update",
            check_id="T5",
            category="temporal",
            severity=Severity.WARN,
            rule="email_time <= update_time where both are present",
            predicate=pl.col("email_time") > pl.col("update_time"),
            applies_to=_both("email_time", "update_time"),
            sample_cols=(*KEY_COLS, "update_time"),
        ),
    )


def date_only_check(lf: pl.LazyFrame) -> CheckResult:
    """T6 (info): per merchant, the share of ``order_time`` values at midnight.

    Over the rows with a non-null ``order_time``. Midnight (00:00:00.000) is read as
    "time unknown", which a Datetime cannot tell from a real midnight (see the
    module docstring). ``metric`` is the overall share, ``n_failed`` the date-only
    rows and ``n_evaluated`` the non-null ``order_time`` rows. The check always
    passes.

    ``details["by_merchant"]``: ``{merchant: {"rows", "date_only", "share"}}``, sorted
    by name (a null merchant is ``"(null)"``).
    """
    order = pl.col("order_time")
    frame = (
        lf.filter(order.is_not_null())
        .group_by("merchant_name")
        .agg(
            pl.len().alias("rows"),
            (order.dt.time() == datetime.time(0)).sum().alias("date_only"),
        )
        .sort("merchant_name", nulls_last=True)
        .collect(engine="streaming")
    )
    by_merchant: dict[str, dict[str, object]] = {}
    n_rows = n_date = 0
    for merchant, n, k in frame.iter_rows():
        by_merchant[merchant if merchant is not None else "(null)"] = {
            "rows": n,
            "date_only": k,
            "share": k / n if n else None,
        }
        n_rows += n
        n_date += k
    return CheckResult(
        name="date_only_order_time",
        check_id="T6",
        category="temporal",
        severity=Severity.INFO,
        passed=True,
        metric=n_date / n_rows if n_rows else None,
        threshold=None,
        n_failed=n_date,
        n_evaluated=n_rows,
        predicate="order_time at exactly 00:00:00.000 (time unknown), reported only",
        details={"by_merchant": by_merchant},
    )


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The temporal checks that need the whole dataset, for the registry."""
    return (NamedCheck("date_only_order_time", date_only_check),)
