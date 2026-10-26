"""Temporal checks (#19): T1-T6 on tiny frames (the fixture is in test_synthetic)."""

from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from rideshare_data_pipeline.checks import run_checks
from rideshare_data_pipeline.checks.model import CheckResult, Severity
from rideshare_data_pipeline.checks.temporal import (
    dataset_checks,
    date_only_check,
    rules,
)

STAMPS = (
    "email_time",
    "order_time",
    "insert_time",
    "update_time",
    "source_dttimestamp",
    "dttimestamp",
)
EMAIL = datetime(2019, 4, 10, 15, 0)


def row(**changes: Any) -> dict[str, Any]:
    """A clean Lyft ride: order 5 h before the email, vendor stamps in order."""
    base: dict[str, Any] = {
        "merchant_name": "Lyft",
        "user_id": "u",
        "order_number": "o",
        "checksum": "c",
        "order_total_amount": 10.0,
        "email_time": EMAIL,
        "order_time": EMAIL - timedelta(hours=5),
        "insert_time": EMAIL + timedelta(minutes=5),
        "update_time": EMAIL + timedelta(minutes=5),
        "source_dttimestamp": EMAIL + timedelta(hours=1),
        "dttimestamp": EMAIL + timedelta(hours=1, minutes=5),
    }
    return base | changes


def frame(*rows: dict[str, Any]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema_overrides=dict.fromkeys(STAMPS[:], pl.Datetime("ms"))
        | {"order_total_amount": pl.Float64},
    )


def run(name: str, *rows: dict[str, Any]) -> CheckResult:
    results = run_checks(frame(*rows), rules=rules(), dataset_checks=[])
    return next(r for r in results if r.name == name)


def counts(name: str, *rows: dict[str, Any]) -> tuple[int, int]:
    res = run(name, *rows)
    return res.n_failed, res.n_evaluated


def on_day(email_day: datetime, order_day: datetime, **changes: Any) -> dict[str, Any]:
    """A row whose email and order fall on the given dates (times kept distinct)."""
    return row(
        email_time=email_day.replace(hour=3),
        order_time=order_day.replace(hour=22),
        insert_time=email_day.replace(hour=4),
        update_time=email_day.replace(hour=4),
        source_dttimestamp=email_day.replace(hour=5),
        dttimestamp=email_day.replace(hour=6),
        **changes,
    )


DAY = datetime(2019, 4, 10)


def test_registry_ids_names_and_severities() -> None:
    found = {r.check_id: (r.name, r.severity) for r in rules()}
    assert found == {
        "T1": ("ingestion_order", Severity.ERROR),
        "T2": ("order_before_receipt", Severity.ERROR),
        "T3": ("late_receipts", Severity.WARN),
        "T4": ("order_before_window", Severity.WARN),
        "T5": ("email_after_update", Severity.WARN),
    }
    assert {r.category for r in rules()} == {"temporal"}
    (t6,) = dataset_checks()
    res = t6(frame(row()))
    assert (res.check_id, res.name, res.severity) == (
        "T6",
        "date_only_order_time",
        Severity.INFO,
    )


# --- T1 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "failed", "by_field"),
    [
        ({}, 0, (0, 0, 0)),
        # equal stamps are in order
        ({"insert_time": EMAIL, "update_time": EMAIL}, 0, (0, 0, 0)),
        ({"insert_time": EMAIL + timedelta(hours=1)}, 1, (1, 0, 0)),
        ({"update_time": EMAIL + timedelta(hours=2)}, 1, (0, 1, 0)),
        ({"source_dttimestamp": EMAIL + timedelta(hours=3)}, 1, (0, 0, 1)),
        # a null stamp breaks no link it is part of
        ({"insert_time": None}, 0, (0, 0, 0)),
        ({"update_time": None, "source_dttimestamp": None}, 0, (0, 0, 0)),
        ({"dttimestamp": None}, 0, (0, 0, 0)),
        # a break elsewhere is still found beside a null
        (
            {"insert_time": EMAIL + timedelta(hours=1), "dttimestamp": None},
            1,
            (1, 0, 0),
        ),
    ],
)
def test_t1_ingestion_order(
    changes: dict[str, Any], failed: int, by_field: tuple[int, int, int]
) -> None:
    res = run("ingestion_order", row(**changes))
    assert (res.n_failed, res.n_evaluated) == (failed, 1)
    assert res.details["by_field"] == dict(
        zip(
            ("insert_after_update", "update_after_source", "source_after_dttimestamp"),
            by_field,
            strict=True,
        )
    )


def test_t1_an_upserted_tip_update_with_insert_before_email_is_fine() -> None:
    # The vendor keeps the first version's insert_time, so it precedes the update's
    # email_time by a day; update_time still follows the email. Not a T1 or T5 failure.
    update = row(
        merchant_name="Uber",
        email_time=EMAIL + timedelta(days=1),
        insert_time=EMAIL + timedelta(minutes=5),
        update_time=EMAIL + timedelta(days=1, minutes=3),
        source_dttimestamp=EMAIL + timedelta(days=1, hours=1),
        dttimestamp=EMAIL + timedelta(days=1, hours=2),
    )
    assert update["insert_time"] < update["email_time"]
    assert counts("ingestion_order", update) == (0, 1)
    assert counts("email_after_update", update) == (0, 1)


# --- T2 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("order_minus_email_days", "failed"),
    [
        (-3, 0),
        (0, 0),
        (1, 0),  # the tolerance: order date = email date + 1 passes
        (2, 1),  # + 2 fails
        (30, 1),
    ],
)
def test_t2_one_day_tolerance_boundary(
    order_minus_email_days: int, failed: int
) -> None:
    order = DAY + timedelta(days=order_minus_email_days)
    assert counts("order_before_receipt", on_day(DAY, order)) == (failed, 1)


def test_t2_compares_dates_not_instants() -> None:
    # Local 23:50 on the 11th, UTC email at 00:10 on the 11th: order > email as
    # instants, same date, so it passes; the no-tolerance count shows the row.
    late = row(
        email_time=datetime(2019, 4, 11, 0, 10),
        order_time=datetime(2019, 4, 11, 23, 50),
    )
    res = run("order_before_receipt", late)
    assert (res.n_failed, res.n_evaluated) == (0, 1)
    assert res.details["by_field"] == {"order_after_email_no_tolerance": 1}


def test_t2_a_date_only_uber_row_passes() -> None:
    uber = row(
        merchant_name="Uber",
        order_time=datetime(2019, 4, 10),  # midnight: the time was not recorded
        email_time=datetime(2019, 4, 10, 23, 30),
    )
    assert counts("order_before_receipt", uber) == (0, 1)
    assert counts("late_receipts", uber) == (0, 1)
    # a UTC+10 trip dated the 12th whose UTC receipt is still on the 11th
    apac = row(
        merchant_name="Uber",
        order_time=datetime(2019, 4, 12),
        email_time=datetime(2019, 4, 11, 14, 20),
    )
    res = run("order_before_receipt", apac)
    assert (res.n_failed, res.details["by_field"]) == (
        0,
        {"order_after_email_no_tolerance": 1},
    )


def test_t2_nulls_are_out_of_the_denominator() -> None:
    assert counts("order_before_receipt", row(order_time=None)) == (0, 0)
    assert counts("order_before_receipt", row(email_time=None)) == (0, 0)


# --- T3 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "late", "very_late"),
    [
        (0, 0, 0),
        (7, 0, 0),
        (8, 1, 0),
        (30, 1, 0),
        (31, 1, 1),
        (-1, 0, 0),  # a receipt dated before the order is T2's business
    ],
)
def test_t3_thresholds(days: int, late: int, very_late: int) -> None:
    res = run("late_receipts", on_day(DAY, DAY - timedelta(days=days)))
    assert (res.n_failed, res.n_evaluated) == (late, 1)
    assert res.details["by_field"] == {"over_30_days": very_late}


def test_t3_nulls_are_out_of_the_denominator() -> None:
    assert counts("late_receipts", row(order_time=None)) == (0, 0)


# --- T4 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days_before", "failed"),
    [(0, 0), (6, 0), (7, 0), (8, 1), (4000, 1)],
)
def test_t4_boundary_against_the_window_start(days_before: int, failed: int) -> None:
    # The first row sets the window start (its email date is the earliest).
    anchor = on_day(DAY, DAY)
    probe = on_day(DAY + timedelta(days=20), DAY - timedelta(days=days_before))
    res = run("order_before_window", anchor, probe)
    assert (res.n_failed, res.n_evaluated) == (failed, 2)


def test_t4_window_start_ignores_null_email_and_order_times() -> None:
    anchor = on_day(DAY, DAY)
    nulls = row(email_time=None, order_time=DAY - timedelta(days=99))
    assert counts("order_before_window", anchor, nulls) == (0, 1)
    assert counts("order_before_window", row(order_time=None)) == (0, 0)


def test_t4_fails_in_the_sample_query_too() -> None:
    # The sample of a failing rule filters on the aggregate predicate (streaming).
    old = on_day(DAY, DAY - timedelta(days=100), checksum="old")
    res = run("order_before_window", on_day(DAY, DAY), old)
    assert not res.passed
    assert res.sample is not None
    assert res.sample["checksum"].to_list() == ["old"]


# --- T5 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, (0, 1)),
        ({"update_time": EMAIL}, (0, 1)),  # equal passes
        ({"update_time": EMAIL - timedelta(seconds=1)}, (1, 1)),
        ({"update_time": None}, (0, 0)),
        ({"email_time": None}, (0, 0)),
    ],
)
def test_t5_email_after_update(
    changes: dict[str, Any], expected: tuple[int, int]
) -> None:
    assert counts("email_after_update", row(**changes)) == expected


# --- T6 -------------------------------------------------------------------------


def test_t6_per_merchant_shares_and_info_verdict() -> None:
    res = date_only_check(
        frame(
            row(merchant_name="Uber", order_time=datetime(2019, 4, 1)),
            row(merchant_name="Uber", order_time=datetime(2019, 4, 2)),
            row(merchant_name="Uber", order_time=datetime(2019, 4, 3, 0, 0, 0, 1000)),
            row(merchant_name="Uber", order_time=None),
            row(merchant_name="Lyft", order_time=datetime(2019, 4, 1, 12, 30)),
            row(merchant_name="Lyft", order_time=datetime(2019, 4, 1, 0, 0)),
            row(merchant_name=None, order_time=datetime(2019, 4, 1)),
        )
    )
    assert res.passed
    assert res.severity is Severity.INFO
    assert (res.n_failed, res.n_evaluated) == (4, 6)
    assert res.metric == pytest.approx(4 / 6)
    assert res.threshold is None
    assert res.details["by_merchant"] == {
        "Lyft": {"rows": 2, "date_only": 1, "share": 0.5},
        "Uber": {"rows": 3, "date_only": 2, "share": 2 / 3},
        "(null)": {"rows": 1, "date_only": 1, "share": 1.0},
    }


def test_t6_no_order_times_has_no_metric() -> None:
    res = date_only_check(frame(row(order_time=None)))
    assert (res.passed, res.metric, res.n_evaluated) == (True, None, 0)
    assert res.details["by_merchant"] == {}
