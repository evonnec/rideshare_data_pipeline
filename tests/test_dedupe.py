"""The dedupe policy and the uniqueness checks U2, U2b, U3 and U4, on tiny frames."""

from datetime import datetime
from typing import Any

import polars as pl
import pytest

from rideshare_data_pipeline.checks.uniqueness import (
    cross_panelist_check,
    dedupe_key_unique_check,
    duplicate_receipts_check,
    near_duplicate_rides_check,
)
from rideshare_data_pipeline.dedupe import deduplicated, supersede_flags
from rideshare_data_pipeline.parse import KINDS

type Spec = dict[str, Any]


def t(day: int, hour: int = 0) -> datetime:
    return datetime(2019, 3, day, hour)


def receipts(*rows: Spec) -> pl.DataFrame:
    """A frame of the columns dedupe and checks read; ``c`` is the checksum."""
    base: Spec = {
        "merchant_name": "Uber",
        "order_number": None,
        "user_id": "u1",
        "suspect": False,
        "kind": "ride",
        "email_time": t(1),
        "update_time": t(1),
        "order_time": t(1),
        "order_total_amount": 10.0,
        "product_description": "UberX 2 mi",
    }
    full = [{**base, **r} for r in rows]
    return pl.DataFrame(
        {
            "merchant_name": [r["merchant_name"] for r in full],
            "order_number": [r["order_number"] for r in full],
            "user_id": [r["user_id"] for r in full],
            "receipt_kind": pl.Series([r["kind"] for r in full], dtype=pl.Enum(KINDS)),
            "email_time": pl.Series(
                [r["email_time"] for r in full], dtype=pl.Datetime("ms")
            ),
            "update_time": pl.Series(
                [r["update_time"] for r in full], dtype=pl.Datetime("ms")
            ),
            "order_time": pl.Series(
                [r["order_time"] for r in full], dtype=pl.Datetime("ms")
            ),
            "order_total_amount": pl.Series(
                [r["order_total_amount"] for r in full], dtype=pl.Float64
            ),
            "product_description": [r["product_description"] for r in full],
            "checksum": [r["c"] for r in full],
            "currency_suspect": [r["suspect"] for r in full],
            "is_rider_spend": [
                r["kind"] in ("ride", "tip_update", "fare_correction") for r in full
            ],
        }
    )


def superseded(*rows: Spec) -> list[str]:
    """Checksums of the superseded rows, sorted."""
    df = supersede_flags(receipts(*rows))
    return sorted(df.filter("is_superseded")["checksum"])


CASES: list[tuple[str, list[Spec], list[str]]] = [
    (
        "null keys are not duplicates",
        [
            {"c": "a", "order_number": None},
            {"c": "b", "order_number": None},
            {"c": "c", "order_number": None, "merchant_name": None},
        ],
        [],
    ),
    (
        "a null merchant is no key even with equal order numbers",
        [
            {"c": "a", "order_number": "1", "merchant_name": None},
            {"c": "b", "order_number": "1", "merchant_name": None},
        ],
        [],
    ),
    (
        "recurring identical pass charges are untouched",
        [
            {
                "c": f"p{m}",
                "kind": "pass_or_credit",
                "email_time": t(m),
                "order_total_amount": 9.99,
            }
            for m in (1, 2, 3)
        ],
        [],
    ),
    (
        "the later email wins between two rides",
        [
            {"c": "old", "order_number": "1", "email_time": t(1)},
            {"c": "new", "order_number": "1", "email_time": t(2)},
        ],
        ["old"],
    ),
    (
        "a tip update supersedes its ride",
        [
            {"c": "ride", "order_number": "1", "email_time": t(1)},
            {"c": "tip", "order_number": "1", "kind": "tip_update", "email_time": t(2)},
        ],
        ["ride"],
    ),
    (
        "kind priority beats time: a tip update wins although its email is earlier",
        [
            {"c": "ride", "order_number": "1", "email_time": t(5)},
            {"c": "tip", "order_number": "1", "kind": "tip_update", "email_time": t(2)},
        ],
        ["ride"],
    ),
    (
        "a fare correction supersedes the ride",
        [
            {"c": "ride", "order_number": "1", "email_time": t(3)},
            {
                "c": "fix",
                "order_number": "1",
                "kind": "fare_correction",
                "email_time": t(1),
            },
        ],
        ["ride"],
    ),
    (
        "two corrections: the later one wins",
        [
            {
                "c": "fix1",
                "order_number": "1",
                "kind": "fare_correction",
                "email_time": t(1),
            },
            {"c": "tip", "order_number": "1", "kind": "tip_update", "email_time": t(2)},
        ],
        ["fix1"],
    ),
    (
        "equal email_time: the later update_time wins",
        [
            {"c": "a", "order_number": "1", "update_time": t(1)},
            {"c": "b", "order_number": "1", "update_time": t(2)},
        ],
        ["a"],
    ),
    (
        "a full tie breaks by the greater checksum",
        [
            {"c": "a", "order_number": "1"},
            {"c": "b", "order_number": "1"},
        ],
        ["a"],
    ),
    (
        "a null email_time loses",
        [
            {"c": "a", "order_number": "1", "email_time": None},
            {"c": "b", "order_number": "1", "email_time": t(1)},
        ],
        ["a"],
    ),
    (
        "the same order number at two merchants is two keys",
        [
            {"c": "a", "order_number": "1", "merchant_name": "Uber"},
            {"c": "b", "order_number": "1", "merchant_name": "Lyft"},
        ],
        [],
    ),
    (
        "a group of three keeps one",
        [
            {"c": "a", "order_number": "1", "email_time": t(1)},
            {"c": "b", "order_number": "1", "email_time": t(3)},
            {"c": "c", "order_number": "1", "email_time": t(2)},
        ],
        ["a", "c"],
    ),
    (
        "across panelists the later email wins too",
        [
            {"c": "a", "order_number": "1", "user_id": "u1", "email_time": t(1)},
            {"c": "b", "order_number": "1", "user_id": "u2", "email_time": t(2)},
        ],
        ["a"],
    ),
]


@pytest.mark.parametrize(
    ("label", "rows", "expected"), CASES, ids=[c[0] for c in CASES]
)
def test_supersede_flags(label: str, rows: list[Spec], expected: list[str]) -> None:
    assert superseded(*rows) == expected


def test_order_and_columns_are_kept_and_nothing_is_dropped() -> None:
    df = receipts(
        {"c": "a", "order_number": "1", "email_time": t(2)},
        {"c": "b", "order_number": "2"},
        {"c": "c", "order_number": "1", "email_time": t(1)},
    )
    out = supersede_flags(df)
    assert out.columns == [*df.columns, "is_superseded"]
    assert out["checksum"].to_list() == ["a", "b", "c"]
    assert out["is_superseded"].to_list() == [False, False, True]
    assert out.drop("is_superseded").equals(df)


def test_lazy_frames_give_the_same_flags() -> None:
    df = receipts(
        {"c": "a", "order_number": "1", "email_time": t(2)},
        {"c": "c", "order_number": "1", "email_time": t(1)},
    )
    assert supersede_flags(df.lazy()).collect().equals(supersede_flags(df))


def test_dedupe_is_idempotent() -> None:
    df = receipts(
        *(
            {"c": f"{n}{i}", "order_number": str(n), "email_time": t(1 + i), "kind": k}
            for n in (1, 2, 3)
            for i, k in enumerate(("ride", "tip_update", "ride"))
        ),
        {"c": "p1", "email_time": t(1)},
        {"c": "p2", "email_time": t(2)},
    )
    once = deduplicated(supersede_flags(df))
    twice = deduplicated(supersede_flags(once.drop("is_superseded")))
    assert twice.equals(once)
    assert once.height == 3 + 2  # one per key, and both null-key rows
    assert not supersede_flags(once.drop("is_superseded"))["is_superseded"].any()


# --- U2, U2b ------------------------------------------------------------------------


def test_u2_counts_rows_and_groups_and_ignores_nulls() -> None:
    df = supersede_flags(
        receipts(
            {"c": "a", "order_number": "1", "email_time": t(1)},
            {"c": "b", "order_number": "1", "email_time": t(2), "kind": "tip_update"},
            {"c": "c", "order_number": "2", "order_total_amount": 4.0},
            {
                "c": "d",
                "order_number": "2",
                "order_total_amount": 4.0,
                "email_time": t(2),
            },
            {"c": "e", "order_number": "3"},
            {"c": "f"},
            {"c": "g"},
        )
    )
    res = duplicate_receipts_check(df.lazy())
    assert (res.check_id, res.severity.value, res.passed) == ("U2", "warn", False)
    assert (res.n_failed, res.n_evaluated) == (4, 5)
    assert res.details["groups"] == 2
    assert res.details["excess_rows"] == 2
    assert res.details["excess_spend"] == 14.0
    assert res.details["kind_pairs"] == [
        {"pair": "ride+ride", "groups": 1, "excess_spend": 4.0},
        {"pair": "ride+tip_update", "groups": 1, "excess_spend": 10.0},
    ]
    assert res.sample is not None
    assert res.sample.height == 4


def test_u2_excess_spend_counts_rider_spend_rows_only() -> None:
    df = receipts(
        {
            "c": "a",
            "order_number": "1",
            "kind": "food_order",
            "order_total_amount": 7.0,
        },
        {"c": "b", "order_number": "1", "kind": "food_order", "email_time": t(2)},
    )
    res = duplicate_receipts_check(df.lazy())
    assert res.details["excess_rows"] == 1
    assert res.details["excess_spend"] == 0.0


def test_u2_excess_spend_is_usd_only_and_suspect_spend_is_separate() -> None:
    df = receipts(
        # a USD pair: the earlier ride is superseded (4.0)
        {"c": "a", "order_number": "1", "order_total_amount": 4.0, "email_time": t(1)},
        {"c": "b", "order_number": "1", "order_total_amount": 5.0, "email_time": t(2)},
        # a suspect pair (local currency): its superseded ride is not USD spend
        {
            "c": "c",
            "order_number": "2",
            "order_total_amount": 9000.0,
            "suspect": True,
            "email_time": t(1),
        },
        {
            "c": "d",
            "order_number": "2",
            "order_total_amount": 9500.0,
            "suspect": True,
            "email_time": t(2),
        },
    )
    res = duplicate_receipts_check(df.lazy())
    assert res.details["excess_rows"] == 2
    assert res.details["excess_spend"] == 4.0
    assert res.details["excess_spend_suspect"] == 9000.0
    assert res.details["kind_pairs"] == [
        {"pair": "ride+ride", "groups": 2, "excess_spend": 4.0}
    ]
    assert "USD" in str(res.details["currency"])


def test_u2_passes_without_duplicates_and_adds_missing_flags() -> None:
    res = duplicate_receipts_check(receipts({"c": "a", "order_number": "1"}).lazy())
    assert res.passed
    assert res.details["groups"] == 0
    assert res.details["kind_pairs"] == []


def test_u2b_passes_after_dedupe_and_fails_on_a_bad_flag() -> None:
    df = receipts(
        {"c": "a", "order_number": "1", "email_time": t(1)},
        {"c": "b", "order_number": "1", "email_time": t(2)},
    )
    ok = dedupe_key_unique_check(supersede_flags(df).lazy())
    assert (ok.check_id, ok.severity.value, ok.passed) == ("U2b", "error", True)
    assert (ok.n_failed, ok.n_evaluated) == (0, 1)

    broken = df.with_columns(is_superseded=pl.lit(False))
    bad = dedupe_key_unique_check(broken.lazy())
    assert not bad.passed
    assert bad.n_failed == 2
    assert bad.sample is not None


# --- U3 -----------------------------------------------------------------------------


def test_u3_counts_rows_of_keys_under_several_panelists() -> None:
    df = receipts(
        {"c": "a", "order_number": "1", "user_id": "u1"},
        {"c": "b", "order_number": "1", "user_id": "u2"},
        {"c": "c", "order_number": "2", "user_id": "u1"},
        {"c": "d", "order_number": "2", "user_id": "u1", "email_time": t(2)},
        {"c": "e", "order_number": None, "user_id": "u1"},
        {"c": "f", "order_number": None, "user_id": "u2"},
    )
    res = cross_panelist_check(supersede_flags(df).lazy())
    assert (res.check_id, res.severity.value, res.passed) == ("U3", "warn", False)
    assert (res.n_failed, res.n_evaluated) == (2, 4)
    assert res.details["groups"] == 1


# --- U4 -----------------------------------------------------------------------------


NEAR: list[tuple[str, list[Spec], int, int]] = [
    # label, rows, excess rows, rows in groups
    (
        "two distinct order numbers",
        [{"c": "a", "order_number": "1"}, {"c": "b", "order_number": "2"}],
        1,
        2,
    ),
    (
        "one null order number",
        [{"c": "a", "order_number": "1"}, {"c": "b", "order_number": None}],
        1,
        2,
    ),
    (
        "the same order number twice is U2's, not U4's",
        [{"c": "a", "order_number": "1"}, {"c": "b", "order_number": "1"}],
        0,
        0,
    ),
    (
        "recurring ride-kind rows with null order numbers still count (a null one)",
        [{"c": "a", "order_number": None}, {"c": "b", "order_number": None}],
        1,
        2,
    ),
    (
        "another total is no near duplicate",
        [
            {"c": "a", "order_number": "1"},
            {"c": "b", "order_number": "2", "order_total_amount": 11.0},
        ],
        0,
        0,
    ),
    (
        "another day is none",
        [
            {"c": "a", "order_number": "1"},
            {"c": "b", "order_number": "2", "order_time": t(2)},
        ],
        0,
        0,
    ),
    (
        "another panelist is none",
        [
            {"c": "a", "order_number": "1"},
            {"c": "b", "order_number": "2", "user_id": "u2"},
        ],
        0,
        0,
    ),
    (
        "only rides count",
        [
            {"c": "a", "order_number": "1", "kind": "pass_or_credit"},
            {"c": "b", "order_number": "2", "kind": "pass_or_credit"},
        ],
        0,
        0,
    ),
    (
        "three rows are two excess",
        [{"c": x, "order_number": x} for x in "123"],
        2,
        3,
    ),
    (
        "a tip update beside its ride is no near duplicate",
        [
            {"c": "a", "order_number": "1"},
            {"c": "b", "order_number": "2", "kind": "tip_update"},
        ],
        0,
        0,
    ),
]


@pytest.mark.parametrize(
    ("label", "rows", "excess", "in_groups"), NEAR, ids=[c[0] for c in NEAR]
)
def test_u4(label: str, rows: list[Spec], excess: int, in_groups: int) -> None:
    res = near_duplicate_rides_check(receipts(*rows).lazy())
    assert (res.check_id, res.severity.value) == ("U4", "warn")
    assert res.n_failed == excess
    assert res.details["rows"] == in_groups
    assert res.passed == (excess == 0)
