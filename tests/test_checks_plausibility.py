"""Trip plausibility (#21): O1-O3 on tiny frames (the fixture is in test_synthetic)."""

import math
from typing import Any

import polars as pl
import pytest
from fixtures.make_synthetic import INJECTIONS, MANIFEST

from rideshare_data_pipeline.checks import default_rules, run_checks
from rideshare_data_pipeline.checks.model import CheckResult, Severity
from rideshare_data_pipeline.checks.plausibility import implied_mph, rules

KINDS = pl.Enum(["ride", "tip_update", "pass", "micromobility"])


def row(**changes: Any) -> dict[str, Any]:
    """A clean ride: 5 miles in 15 minutes (20 mph), $10."""
    base: dict[str, Any] = {
        "merchant_name": "Uber",
        "user_id": "u",
        "order_number": "o",
        "email_time": "t",
        "checksum": "c",
        "receipt_kind": "ride",
        "is_superseded": False,
        "order_total_amount": 10.0,
        "distance_mi": 5.0,
        "duration_s": 900,
        "product_family": "uberx",
    }
    return base | changes


def frame(*rows: dict[str, Any]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema_overrides={
            "receipt_kind": KINDS,
            "order_total_amount": pl.Float64,
            "distance_mi": pl.Float64,
            "duration_s": pl.Int64,
        },
    )


def run(
    name: str,
    *rows: dict[str, Any],
    params: dict[str, dict[str, float]] | None = None,
) -> CheckResult:
    results = run_checks(frame(*rows), rules=rules(params), dataset_checks=[])
    return next(r for r in results if r.name == name)


def counts(name: str, *rows: dict[str, Any], **kwargs: Any) -> tuple[int, int]:
    res = run(name, *rows, **kwargs)
    return res.n_failed, res.n_evaluated


def test_registry() -> None:
    found = {r.check_id: (r.name, r.severity) for r in rules()}
    assert found == {
        "O1": ("implausible_speed", Severity.WARN),
        "O2": ("zero_trip_metrics", Severity.WARN),
        "O3": ("extreme_trip", Severity.WARN),
    }
    assert {r.name for r in rules()} <= {r.name for r in default_rules()}


def test_a_clean_ride_fails_nothing() -> None:
    for name in ("implausible_speed", "zero_trip_metrics", "extreme_trip"):
        assert counts(name, row()) == (0, 1)


# --- O1 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "flagged"),
    [
        ({"distance_mi": 20.0, "duration_s": 900}, False),  # exactly 80 mph: strict
        ({"distance_mi": 20.01, "duration_s": 900}, True),  # just above
        ({"distance_mi": 79.0, "duration_s": 3600}, False),
        ({"distance_mi": 60.0, "duration_s": 1200}, True),  # 180 mph
        ({"distance_mi": 5.0, "duration_s": 119}, False),  # fast (151 mph) but short
        ({"distance_mi": 5.0, "duration_s": 120}, True),  # the same speed at 120 s
        ({"distance_mi": 3.0, "duration_s": 120}, True),  # 90 mph
        ({"distance_mi": 2.0, "duration_s": 120}, False),  # 60 mph
        ({"distance_mi": 5.0, "duration_s": 0}, False),  # no speed, not an inf
        ({"distance_mi": 0.0, "duration_s": 0}, False),  # 0 / 0 is not NaN-flagged
        ({"distance_mi": None, "duration_s": 900}, False),
        ({"distance_mi": 5.0, "duration_s": None}, False),
        ({"distance_mi": None, "duration_s": None}, False),
    ],
)
def test_o1_boundaries(changes: dict[str, Any], flagged: bool) -> None:
    assert counts("implausible_speed", row(**changes)) == (int(flagged), 1)


def test_o1_never_produces_inf_or_nan() -> None:
    lf = frame(
        row(duration_s=0),
        row(duration_s=0, distance_mi=0.0),
        row(duration_s=None),
        row(duration_s=900),
    )
    speeds = lf.select(implied_mph().alias("mph")).collect()["mph"]
    assert speeds.to_list() == [None, None, None, 20.0]
    assert not any(math.isinf(v) or math.isnan(v) for v in speeds.drop_nulls())
    res = run_checks(lf, rules=rules(), dataset_checks=[])
    assert all(r.metric is None or math.isfinite(r.metric) for r in res)


def test_o1_sample_shows_the_trip_columns() -> None:
    res = run("implausible_speed", row(distance_mi=60.0, duration_s=1200))
    assert res.sample is not None
    assert res.sample.columns == [
        "merchant_name",
        "user_id",
        "order_number",
        "email_time",
        "order_total_amount",
        "checksum",
        "distance_mi",
        "duration_s",
        "product_family",
    ]


# --- O2 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "flagged"),
    [
        ({"distance_mi": 0.0}, True),
        ({"duration_s": 0}, True),
        ({"distance_mi": 0.0, "duration_s": 0}, True),
        ({"distance_mi": 0.0, "order_total_amount": 0.0}, False),  # A3's
        ({"duration_s": 0, "order_total_amount": 0.0}, False),
        ({"distance_mi": 0.0, "order_total_amount": None}, False),
        ({"distance_mi": None}, False),  # null is unparsed, PC1's
        ({"duration_s": None}, False),
        ({"distance_mi": None, "duration_s": None}, False),
        ({"distance_mi": 0.01, "duration_s": 1}, False),
        ({"distance_mi": 0.0, "duration_s": None}, True),  # one zero is enough
    ],
)
def test_o2_cases(changes: dict[str, Any], flagged: bool) -> None:
    assert counts("zero_trip_metrics", row(**changes)) == (int(flagged), 1)


def test_o2_breakdown_splits_the_flagged_rows() -> None:
    res = run(
        "zero_trip_metrics",
        row(distance_mi=0.0),
        row(distance_mi=0.0, checksum="d"),
        row(duration_s=0),
        row(distance_mi=0.0, duration_s=0),
        row(distance_mi=0.0, order_total_amount=0.0),
        row(),
    )
    assert res.n_failed == 4
    assert res.details["by_field"] == {
        "zero_distance": 2,
        "zero_duration": 1,
        "both": 1,
    }
    assert sum(res.details["by_field"].values()) == res.n_failed


# --- O3 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "over_distance", "over_duration"),
    [
        ({"distance_mi": 100.0}, 0, 0),
        ({"distance_mi": 100.01}, 1, 0),
        ({"duration_s": 10_800}, 0, 0),
        ({"duration_s": 10_801}, 0, 1),
        ({"distance_mi": 150.0, "duration_s": 20_000}, 1, 1),
        ({"distance_mi": None, "duration_s": None}, 0, 0),
    ],
)
def test_o3_boundaries(
    changes: dict[str, Any], over_distance: int, over_duration: int
) -> None:
    res = run("extreme_trip", row(**changes))
    assert res.n_failed == int(bool(over_distance or over_duration))
    assert res.details["by_field"] == {
        "over_distance": over_distance,
        "over_duration": over_duration,
    }


# --- population -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("implausible_speed", {"distance_mi": 60.0, "duration_s": 1200}),
        ("zero_trip_metrics", {"duration_s": 0}),
        ("extreme_trip", {"distance_mi": 500.0}),
    ],
)
def test_population_is_rides_that_are_not_superseded(
    name: str, bad: dict[str, Any]
) -> None:
    res = run(
        name,
        row(**bad),  # counted
        row(**bad, receipt_kind="tip_update"),  # counted
        row(**bad, is_superseded=True),  # excluded
        row(**bad, receipt_kind="pass"),  # excluded
        row(**bad, receipt_kind="micromobility"),  # excluded
        row(),
    )
    assert (res.n_failed, res.n_evaluated) == (2, 3)


def test_population_ignores_the_currency() -> None:
    # a km-converted or non-USD ride is still a trip
    res = run(
        "extreme_trip",
        row(distance_mi=500.0, currency_suspect=True, merchant_name=None),
    )
    assert res.n_failed == 1


# --- params ---------------------------------------------------------------------


def test_o1_params_override_the_thresholds() -> None:
    r = row(distance_mi=10.0, duration_s=600)  # 60 mph
    assert counts("implausible_speed", r) == (0, 1)
    strict = {"implausible_speed": {"max_mph": 50.0}}
    assert counts("implausible_speed", r, params=strict) == (1, 1)
    short = row(distance_mi=5.0, duration_s=100)  # 180 mph, under 120 s
    assert counts("implausible_speed", short) == (0, 1)
    loose_min = {"implausible_speed": {"min_seconds": 60.0}}
    assert counts("implausible_speed", short, params=loose_min) == (1, 1)
    res = run("implausible_speed", r, params=strict)
    assert "50 mph" in res.predicate
    assert "120 s" in res.predicate


def test_o3_params_override_the_thresholds() -> None:
    r = row(distance_mi=60.0, duration_s=4000)
    assert counts("extreme_trip", r) == (0, 1)
    params = {"extreme_trip": {"max_miles": 50.0}}
    assert counts("extreme_trip", r, params=params) == (1, 1)
    params = {"extreme_trip": {"max_seconds": 3600.0}}
    assert counts("extreme_trip", r, params=params) == (1, 1)
    assert "3,600 s" in run("extreme_trip", r, params=params).predicate


def test_default_rules_take_params() -> None:
    by_name = {r.name: r for r in default_rules({"extreme_trip": {"max_miles": 7.0}})}
    assert "7 mi" in by_name["extreme_trip"].rule


# --- the synthetic fixture --------------------------------------------------------


def test_fixture_counts_match_the_manifest(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    by_id = {r.check_id: r for r in run_checks(full_synthetic_derived.lazy())}
    d = full_synthetic_derived
    o1, o2, o3 = by_id["O1"], by_id["O2"], by_id["O3"]
    assert (o1.n_failed, o2.n_failed, o3.n_failed) == (
        MANIFEST["O1"],
        MANIFEST["O2"],
        MANIFEST["O3"],
    )
    (fast,) = INJECTIONS["implausible_speed"]
    (zero,) = INJECTIONS["zero_duration_trip"]
    assert o1.sample is not None
    assert o1.sample["checksum"].to_list() == [d["checksum"][fast]]
    assert o2.sample is not None
    assert o2.sample["checksum"].to_list() == [d["checksum"][zero]]
    assert o2.details["by_field"] == {
        "zero_distance": 0,
        "zero_duration": 1,
        "both": 0,
    }
    assert o3.sample is None
    assert o3.details["by_field"] == {"over_distance": 0, "over_duration": 0}
    # the 10x fare outlier is a normal trip: none of the three flags it
    (outlier,) = INJECTIONS["fare_outlier_10x"]
    flagged = d.with_row_index("_row").filter(
        pl.col("checksum").is_in(
            [
                c
                for r in (o1, o2, o3)
                if r.sample is not None
                for c in r.sample["checksum"].to_list()
            ]
        )
    )["_row"]
    assert outlier not in flagged.to_list()
