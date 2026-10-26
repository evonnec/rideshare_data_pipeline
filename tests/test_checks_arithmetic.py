"""Arithmetic checks (#18): A1-A7 on tiny frames (the fixture is in test_synthetic)."""

from typing import Any

import polars as pl
import pytest

from rideshare_data_pipeline.checks import run_checks
from rideshare_data_pipeline.checks.arithmetic import (
    dataset_checks,
    identity_share_check,
    rules,
)
from rideshare_data_pipeline.checks.model import CheckResult, Severity
from rideshare_data_pipeline.money import positive_usd_rides, usd_rides

MONEY = (
    "order_total_amount",
    "order_subtotal",
    "item_price",
    "order_tax",
    "order_discount",
)


def row(**changes: Any) -> dict[str, Any]:
    """A clean USD ride: total == subtotal == item price, no tax or discount."""
    base: dict[str, Any] = {
        "merchant_name": "Uber",
        "user_id": "u",
        "order_number": "o",
        "email_time": "t",
        "checksum": "c",
        "receipt_kind": "ride",
        "is_rider_spend": True,
        "is_superseded": False,
        "currency_suspect": False,
        "order_total_amount": 10.0,
        "order_subtotal": 10.0,
        "item_price": 10.0,
        "order_tax": None,
        "order_discount": None,
    }
    return base | changes


def frame(*rows: dict[str, Any]) -> pl.LazyFrame:
    schema = dict.fromkeys(MONEY, pl.Float64) | {
        "receipt_kind": pl.Enum(["ride", "tip_update", "pass", "micromobility"]),
    }
    return pl.LazyFrame(
        [{k: r[k] for k in rows[0]} for r in rows],
        schema_overrides=schema,
    )


def run(name: str, *rows: dict[str, Any]) -> CheckResult:
    results = run_checks(frame(*rows), rules=rules(), dataset_checks=[])
    return next(r for r in results if r.name == name)


def counts(name: str, *rows: dict[str, Any]) -> tuple[int, int]:
    res = run(name, *rows)
    return res.n_failed, res.n_evaluated


def test_registry_ids_names_and_severities() -> None:
    found = {r.check_id: (r.name, r.severity) for r in rules()}
    assert found == {
        "A1": ("non_negative_total", Severity.ERROR),
        "A2": ("item_price_equals_subtotal", Severity.ERROR),
        "A3": ("zero_fare_rides", Severity.WARN),
        "A4": ("discount_within_subtotal", Severity.WARN),
        "A5": ("tax_within_total", Severity.WARN),
        "A6": ("unexplained_residual", Severity.WARN),
    }
    (a7,) = dataset_checks()
    res = a7(frame(row()))
    assert (res.check_id, res.name, res.severity) == (
        "A7",
        "identity_share",
        Severity.INFO,
    )


# --- A1 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"order_total_amount": -0.01}, (1, 1)),
        ({"order_total_amount": 0.0}, (0, 1)),
        ({"order_total_amount": None}, (0, 1)),
        # a bike refund is not rider spend: out of the denominator
        ({"order_total_amount": -4.5, "is_rider_spend": False}, (0, 0)),
    ],
)
def test_a1_non_negative_total(
    changes: dict[str, Any], expected: tuple[int, int]
) -> None:
    assert counts("non_negative_total", row(**changes)) == expected


# --- A2, A4, A5: the cent tolerance and null skipping ---------------------------


@pytest.mark.parametrize(
    ("item", "subtotal", "failed"),
    [
        (10.0, 10.0, 0),
        (10.01, 10.0, 0),  # exactly the tolerance
        (10.02, 10.01, 0),  # a float difference that may exceed 0.01 by luck
        (10.0, 10.01, 0),
        (10.02, 10.0, 1),
        (10.0, 10.03, 1),
        (-5.75, -5.75, 0),
    ],
)
def test_a2_item_price_equals_subtotal(
    item: float, subtotal: float, failed: int
) -> None:
    assert counts(
        "item_price_equals_subtotal", row(item_price=item, order_subtotal=subtotal)
    ) == (failed, 1)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"item_price": None}, (0, 0)),
        ({"order_subtotal": None}, (0, 0)),
        ({"item_price": None, "order_subtotal": None}, (0, 0)),
    ],
)
def test_a2_skips_rows_with_a_null_operand(
    changes: dict[str, Any], expected: tuple[int, int]
) -> None:
    assert counts("item_price_equals_subtotal", row(**changes)) == expected


@pytest.mark.parametrize(
    ("discount", "subtotal", "failed"),
    [
        (5.0, 10.0, 0),
        (10.0, 10.0, 0),
        (10.01, 10.0, 0),  # exactly the tolerance
        (10.02, 10.0, 1),
        (11.0, 10.0, 1),
        (1.0, 0.0, 1),
        (0.0, 0.0, 0),
    ],
)
def test_a4_discount_within_subtotal(
    discount: float, subtotal: float, failed: int
) -> None:
    got = counts(
        "discount_within_subtotal",
        row(order_discount=discount, order_subtotal=subtotal),
    )
    assert got == (failed, 1)


@pytest.mark.parametrize(
    ("tax", "total", "failed"),
    [
        (1.0, 10.0, 0),
        (10.0, 10.0, 0),
        (10.01, 10.0, 0),
        (10.02, 10.0, 1),
        (9.25, 1.0, 1),
        (0.01, 0.0, 0),
        (0.02, 0.0, 1),
    ],
)
def test_a5_tax_within_total(tax: float, total: float, failed: int) -> None:
    got = counts("tax_within_total", row(order_tax=tax, order_total_amount=total))
    assert got == (failed, 1)


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("discount_within_subtotal", {"order_discount": None}),
        ("discount_within_subtotal", {"order_discount": 99.0, "order_subtotal": None}),
        ("tax_within_total", {"order_tax": None}),
        ("tax_within_total", {"order_tax": 99.0, "order_total_amount": None}),
    ],
)
def test_a4_a5_skip_rows_with_a_null_operand(
    name: str, changes: dict[str, Any]
) -> None:
    """A null tax or discount is not treated as 0 here: the row is not evaluated."""
    assert counts(name, row(**changes)) == (0, 0)


# --- A3 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"order_total_amount": 0.0}, (1, 1)),
        ({"order_total_amount": -0.0}, (1, 1)),
        ({"order_total_amount": 0.01}, (0, 1)),
        ({"order_total_amount": -0.01}, (0, 1)),  # negatives are A1's
        ({"order_total_amount": None}, (0, 1)),
        ({"order_total_amount": 0.0, "receipt_kind": "tip_update"}, (1, 1)),
        ({"order_total_amount": 0.0, "receipt_kind": "pass"}, (0, 0)),
        ({"order_total_amount": 0.0, "receipt_kind": "micromobility"}, (0, 0)),
        ({"order_total_amount": 0.0, "is_superseded": True}, (0, 0)),
    ],
)
def test_a3_zero_fare_rides(changes: dict[str, Any], expected: tuple[int, int]) -> None:
    assert counts("zero_fare_rides", row(**changes)) == expected


def test_a3_breaks_down_by_merchant() -> None:
    res = run(
        "zero_fare_rides",
        row(order_total_amount=0.0),
        row(order_total_amount=0.0),
        row(order_total_amount=0.0, merchant_name="Lyft"),
        row(order_total_amount=0.0, merchant_name="Other"),
        row(order_total_amount=0.0, merchant_name=None),
        row(),
    )
    assert res.details["by_field"] == {"Uber": 2, "Lyft": 1, "other": 2}
    assert (res.n_failed, res.n_evaluated) == (5, 6)
    assert res.sample is not None
    assert res.sample.height == 5


# --- A6 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "failed"),
    [
        ({}, 0),  # total == subtotal
        ({"order_total_amount": 30.0}, 0),  # residual 20 = the floor, not above it
        ({"order_total_amount": 30.01}, 1),  # residual 20.01 > max(20, 10)
        ({"order_total_amount": 29.0, "order_subtotal": 5.0}, 1),  # 24 > 20
        # subtotal above the floor: the limit is the subtotal
        ({"order_subtotal": 50.0, "order_total_amount": 100.0}, 0),  # 50 <= 50
        ({"order_subtotal": 50.0, "order_total_amount": 100.01}, 1),
        # tax and discount are part of the explained total
        ({"order_tax": 25.0, "order_total_amount": 35.0}, 0),
        ({"order_discount": 25.0, "order_total_amount": -15.0}, 0),
        ({"order_tax": 2.0, "order_discount": 1.0, "order_total_amount": 31.01}, 1),
        # a large NEGATIVE residual (an unlisted credit) is not flagged
        ({"order_total_amount": -40.0}, 0),
        # a negative subtotal falls back to the $20 floor
        ({"order_subtotal": -5.0, "order_total_amount": 16.0}, 1),
    ],
)
def test_a6_unexplained_residual(changes: dict[str, Any], failed: int) -> None:
    assert counts("unexplained_residual", row(**changes)) == (failed, 1)


def test_a6_null_tax_and_discount_count_as_zero() -> None:
    res = run(
        "unexplained_residual",
        row(order_total_amount=40.0, order_tax=None, order_discount=None),
        row(order_total_amount=40.0, order_tax=0.0, order_discount=0.0),
    )
    assert (res.n_failed, res.n_evaluated) == (2, 2)


@pytest.mark.parametrize("field", ["order_total_amount", "order_subtotal"])
def test_a6_null_total_or_subtotal_is_not_a_failure(field: str) -> None:
    assert counts("unexplained_residual", row(**{field: None})) == (0, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"currency_suspect": True},
        {"is_superseded": True},
        {"is_rider_spend": False, "receipt_kind": "pass"},
        {"receipt_kind": "micromobility", "is_rider_spend": False},
    ],
)
def test_a6_excludes_suspect_superseded_and_non_spend_rows(
    changes: dict[str, Any],
) -> None:
    assert counts("unexplained_residual", row(order_total_amount=99.0, **changes)) == (
        0,
        0,
    )


def test_a6_includes_tip_updates() -> None:
    got = counts(
        "unexplained_residual", row(receipt_kind="tip_update", order_total_amount=99.0)
    )
    assert got == (1, 1)


# --- A7 -------------------------------------------------------------------------


def identity(*rows: dict[str, Any]) -> CheckResult:
    return identity_share_check(frame(*rows))


def test_a7_breakdown_and_share() -> None:
    res = identity(
        row(),  # holds
        row(order_total_amount=10.01),  # holds: within a cent
        row(order_tax=1.0, order_total_amount=11.0),  # holds with tax
        row(order_discount=2.0, order_total_amount=8.0),  # holds with discount
        row(order_total_amount=12.5),  # above by 2.50
        row(order_total_amount=16.0),  # above by 6.00
        row(order_total_amount=9.0),  # below by 1.00
        row(order_total_amount=None),  # not computable
        row(order_subtotal=None),  # not computable
        row(currency_suspect=True),  # excluded
        row(is_superseded=True),  # excluded
        row(receipt_kind="pass", is_rider_spend=False),  # excluded
    )
    assert res.passed
    assert res.severity is Severity.INFO
    assert res.threshold is None
    assert res.details == {
        "share": 4 / 9,
        "holds": 4,
        "above": 2,
        "below": 1,
        "not_computable": 2,
        "median_gap_above": 4.25,
        "population": 9,
    }
    assert res.metric == 4 / 9
    assert res.n_failed == 4  # info: the rows where the identity holds
    assert res.n_evaluated == 9


@pytest.mark.parametrize(
    ("total", "bucket"),
    [
        (10.0, "holds"),
        (10.01, "holds"),
        (9.99, "holds"),
        (10.02, "above"),
        (9.98, "below"),
    ],
)
def test_a7_tolerance_boundaries(total: float, bucket: str) -> None:
    res = identity(row(order_total_amount=total))
    assert res.details[bucket] == 1
    assert res.details["population"] == 1


def test_a7_with_no_population_has_no_share() -> None:
    res = identity(row(currency_suspect=True))
    assert res.passed
    assert res.metric is None
    assert res.details["share"] is None
    assert res.details["median_gap_above"] is None
    assert res.details["population"] == 0


# --- money.py: the zero-fare policy ---------------------------------------------


def test_zero_totals_are_in_usd_rides_but_not_in_the_positive_baseline() -> None:
    lf = frame(
        row(order_total_amount=0.0),
        row(order_total_amount=-5.0),
        row(order_total_amount=7.0),
    )
    assert usd_rides(lf).collect().height == 3
    assert positive_usd_rides(lf).collect()["order_total_amount"].to_list() == [7.0]
