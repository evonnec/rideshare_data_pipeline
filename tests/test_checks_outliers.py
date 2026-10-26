"""Fare outliers (#20): robust_z, O4 and O5 on tiny frames (the fixture is in
test_synthetic)."""

from collections.abc import Sequence
from typing import Any

import polars as pl
import pytest

from rideshare_data_pipeline.checks import outliers
from rideshare_data_pipeline.checks.model import CheckResult, Severity
from rideshare_data_pipeline.checks.outliers import (
    DEFAULT_K,
    K_GRID,
    MIN_GROUP,
    dataset_checks,
    log_positive,
    outlier_checks,
    robust_z,
    score,
)

N = 60  # baseline rides, comfortably above MIN_GROUP


def ride(i: int, **changes: Any) -> dict[str, Any]:
    """A baseline Uber ride: $/mile in a +-6% band, a steady $/minute."""
    miles = 1.0 + (i % 10) * 0.5
    per_mile = 2.0 * (1 + 0.03 * ((i % 5) - 2))
    row: dict[str, Any] = {
        "merchant_name": "Uber",
        "user_id": f"u{i}",
        "order_number": f"o{i}",
        "email_time": None,
        "checksum": f"c{i}",
        "receipt_kind": "ride",
        "is_rider_spend": True,
        "currency_suspect": False,
        "is_superseded": False,
        "order_total_amount": round(per_mile * miles, 4),
        "distance_mi": miles,
        "duration_s": int(180 * miles),
        "product_family": "uberx",
    }
    return row | changes


def frame(*rows: dict[str, Any]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema_overrides={
            "order_total_amount": pl.Float64,
            "distance_mi": pl.Float64,
            "duration_s": pl.Int64,
            "email_time": pl.Datetime("ms"),
        },
    )


def base(n: int = N) -> list[dict[str, Any]]:
    return [ride(i) for i in range(n)]


def with_extra(*extra: dict[str, Any], n: int = N) -> pl.LazyFrame:
    return frame(*base(n), *extra)


def run(
    lf: pl.LazyFrame, k: float = DEFAULT_K, k5: float = 3.5, limit: int = 20
) -> tuple[CheckResult, CheckResult]:
    o4, o5 = outlier_checks(k, k5, limit)
    return o4(lf), o5(lf)


def expensive(**changes: Any) -> dict[str, Any]:
    """A 5-mile ride at 10x the baseline fare (both prices ~10x)."""
    return (
        ride(1000, order_total_amount=100.0, distance_mi=5.0, duration_s=900) | changes
    )


def detail(result: CheckResult, key: str) -> Any:
    return result.details[key]


# --- robust_z -----------------------------------------------------------------------


def z(
    values: Sequence[float | None], by: list[str] | None = None, **kw: int
) -> list[Any]:
    frame_ = pl.DataFrame({"x": values, "g": by or ["a"] * len(values)})
    return frame_.select(robust_z(pl.col("x"), "g", **kw)).to_series().to_list()


def test_robust_z_known_vector() -> None:
    # median 3, |x - 3| = [2, 1, 0, 1, 97] with median 1, so scale = 1.4826
    got = z([1, 2, 3, 4, 100], min_group=1)
    want = [(x - 3) / 1.4826 for x in (1, 2, 3, 4, 100)]
    assert got == pytest.approx(want)
    assert got[2] == 0.0
    assert got[4] == pytest.approx(65.4, abs=0.05)


def test_robust_z_is_computed_per_group() -> None:
    # group a: median 3, MAD 1; group b: median 14, MAD 2
    got = z(
        [1, 2, 3, 4, 100, 10, 12, 14, 16, 18],
        ["a"] * 5 + ["b"] * 5,
        min_group=1,
    )
    assert got[:5] == pytest.approx([(x - 3) / 1.4826 for x in (1, 2, 3, 4, 100)])
    assert got[5:] == pytest.approx(
        [(x - 14) / (1.4826 * 2) for x in (10, 12, 14, 16, 18)]
    )


def test_robust_z_is_null_when_the_mad_is_zero() -> None:
    # more than half the values equal: MAD 0, so no score rather than inf
    assert z([5, 5, 5, 5, 9], min_group=1) == [None] * 5
    assert z([7.0] * 4, min_group=1) == [None] * 4
    # a zero-MAD group does not spoil another group
    got = z([5, 5, 5, 1, 2, 3, 4, 100], ["a"] * 3 + ["b"] * 5, min_group=1)
    assert got[:3] == [None] * 3
    assert all(v is not None for v in got[3:])


def test_robust_z_is_null_below_the_minimum_group_size() -> None:
    values = [float(i) for i in range(1, 7)]
    assert z(values, min_group=7) == [None] * 6
    assert all(v is not None for v in z(values, min_group=6))
    # the default minimum
    assert MIN_GROUP == 30
    assert z(values) == [None] * 6
    assert all(v is not None for v in z([float(i) for i in range(30)]))


def test_robust_z_ignores_nulls_and_keeps_them_null() -> None:
    got = z([1, 2, 3, None, 4, 100], min_group=5)
    assert got[3] is None
    assert got[0] == pytest.approx(-2 / 1.4826)
    # the null does not count towards the group size
    assert z([1, 2, 3, None, 4, 100], min_group=6) == [None] * 6


def test_log_positive_is_null_for_non_positive_values() -> None:
    values = pl.DataFrame({"x": [-1.0, 0.0, 1.0, 10.0, None]})
    got = values.select(log_positive(pl.col("x"))).to_series().to_list()
    assert got[:2] == [None, None]
    assert got[2] == 0.0
    assert got[3] == pytest.approx(2.302585)
    assert got[4] is None


def test_robust_z_of_a_log_scale_skips_non_positive_inputs() -> None:
    xs = [1.0, 2.0, 4.0, 8.0, 16.0, 0.0, -3.0]
    got = z_log(xs)
    assert got[5:] == [None, None]
    # log2-spaced values are evenly spaced on the log scale: median 4, MAD log(2)
    assert got[2] == 0.0
    assert got[0] == pytest.approx(-2 / 1.4826)


def z_log(xs: list[float]) -> list[Any]:
    df = pl.DataFrame({"x": xs, "g": ["a"] * len(xs)})
    return (
        df.select(robust_z(log_positive(pl.col("x")), "g", min_group=1))
        .to_series()
        .to_list()
    )


# --- O4 -----------------------------------------------------------------------------


def test_registry_ids_names_and_severities() -> None:
    o4, o5 = dataset_checks()
    a, b = o4(with_extra()), o5(with_extra())
    assert (a.check_id, a.name, a.severity, a.category) == (
        "O4",
        "fare_outlier_conditional",
        Severity.WARN,
        "outliers",
    )
    assert (b.check_id, b.name, b.severity) == (
        "O5",
        "fare_outlier_univariate",
        Severity.INFO,
    )


def test_o4_passes_on_clean_rides() -> None:
    o4, _ = run(with_extra())
    assert (o4.passed, o4.n_failed, o4.n_evaluated) == (True, 0, N)
    assert o4.metric == 0.0
    assert o4.sample is None
    assert detail(o4, "k") == DEFAULT_K


def test_o4_flags_a_ride_with_both_prices_extreme() -> None:
    o4, _ = run(with_extra(expensive()))
    assert (o4.passed, o4.n_failed, o4.n_evaluated) == (False, 1, N + 1)
    assert o4.metric == pytest.approx(1 / (N + 1))
    assert detail(o4, "flag_rate") == o4.metric
    assert detail(o4, "by_direction") == {"high": 1, "low": 0, "mixed": 0}
    assert o4.sample is not None
    assert o4.sample["order_number"].to_list() == ["o1000"]


def test_o4_flags_underpriced_rides_as_low() -> None:
    o4, _ = run(with_extra(expensive(order_total_amount=0.05)))
    assert o4.n_failed == 1
    assert detail(o4, "by_direction") == {"high": 0, "low": 1, "mixed": 0}


def test_o4_mixed_direction() -> None:
    # a 10x fare over a duration 1000x as long: $/mile up, $/minute far down
    row = expensive(duration_s=900_000)
    o4, _ = run(with_extra(row))
    assert o4.n_failed == 1
    assert detail(o4, "by_direction") == {"high": 0, "low": 0, "mixed": 1}


def test_a_ride_extreme_on_one_price_only_is_not_flagged() -> None:
    # the typical fare for the miles, over a tenth of the time: only $/minute moves
    fast = expensive(order_total_amount=10.0, duration_s=90)
    o4, _ = run(with_extra(fast))
    assert o4.n_failed == 0
    # a long trip at a normal price per mile and minute is no outlier either
    long = ride(1000, order_total_amount=2.0 * 60, distance_mi=60.0, duration_s=10800)
    o4, _ = run(with_extra(long))
    assert o4.n_failed == 0


def test_k_boundary_is_strict_and_configurable() -> None:
    lf = with_extra(expensive())
    frame_ = score(lf).frame.filter(pl.col("o4"))
    hot = frame_.sort(pl.col("rz_mile").abs(), descending=True).row(0, named=True)
    smaller = min(abs(hot["rz_mile"]), abs(hot["rz_minute"]))
    assert smaller > DEFAULT_K
    at, _ = run(lf, k=smaller)  # |rz| > k is strict: equal is not flagged
    below, _ = run(lf, k=smaller * 0.999)
    above, _ = run(lf, k=smaller * 1.001)
    assert (below.n_failed, at.n_failed, above.n_failed) == (1, 0, 0)
    assert detail(below, "k") == pytest.approx(smaller * 0.999)
    assert f"{smaller * 0.999:g}" in below.predicate
    # the sensitivity table does not depend on the configured k
    assert detail(at, "sensitivity") == detail(above, "sensitivity")


def test_sensitivity_table_at_each_k() -> None:
    o4, _ = run(with_extra(expensive()))
    table = detail(o4, "sensitivity")
    assert isinstance(table, list)
    assert [row["k"] for row in table] == list(K_GRID)
    flagged = [row["flagged"] for row in table]
    assert flagged == sorted(flagged, reverse=True)
    assert flagged[K_GRID.index(4.0)] == o4.n_failed
    assert all(row["rate"] == row["flagged"] / (N + 1) for row in table)


def test_k_grid_separates_a_ride_between_the_thresholds() -> None:
    # scan fares for one whose smaller |rz| lies between k = 3 and k = 5
    for step in range(1, 100):
        row = ride(1000, order_total_amount=10.0 * (1 + step / 100), distance_mi=5.0)
        row["duration_s"] = 900
        found = score(with_extra(row)).frame.sort("_row").row(-1, named=True)
        if 3.0 < min(abs(found["rz_mile"]), abs(found["rz_minute"])) < 5.0:
            o4, _ = run(with_extra(row))
            by_k = {r["k"]: r["flagged"] for r in detail(o4, "sensitivity")}
            assert (by_k[3.0], by_k[3.5], by_k[5.0]) == (1, by_k[3.5], 0)
            return
    pytest.fail("no fare landed between k = 3 and k = 5")


# --- the population -----------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"currency_suspect": True},
        {"is_superseded": True},
        {"receipt_kind": "food_order", "is_rider_spend": False},
        {"is_rider_spend": False},
        {"receipt_kind": "fare_correction"},
        {"receipt_kind": "cancel_noshow"},
        {"distance_mi": 0.4},
        {"duration_s": 59},
        {"order_total_amount": 0.0},
        {"order_total_amount": -100.0},
        {"order_total_amount": None},
        {"product_family": None},
        {"distance_mi": None},
        {"duration_s": None},
    ],
    ids=str,
)
def test_o4_population_filters(changes: dict[str, Any]) -> None:
    o4, _ = run(with_extra(expensive(**changes)))
    assert (o4.n_failed, o4.n_evaluated) == (0, N)


def test_o4_population_includes_tip_updates_and_the_minimum_trip() -> None:
    row = expensive(receipt_kind="tip_update", distance_mi=0.5, duration_s=60)
    o4, _ = run(with_extra(row))
    assert (o4.n_failed, o4.n_evaluated) == (1, N + 1)


def test_a_segment_below_the_minimum_size_is_not_scored() -> None:
    # 5 Lyft rides in their own segment: one is extreme, but 5 < MIN_GROUP
    lyft = [
        ride(2000 + i, merchant_name="Lyft", product_family="lyft_standard")
        for i in range(4)
    ]
    odd = ride(
        2010,
        merchant_name="Lyft",
        product_family="lyft_standard",
        order_total_amount=100.0,
        distance_mi=5.0,
        duration_s=900,
    )
    o4, _ = run(with_extra(*lyft, odd))
    assert (o4.n_failed, o4.n_evaluated) == (0, N + 5)
    assert detail(o4, "scored") == N
    # the same segment is scored once it is big enough
    o4, _ = run(frame(*base(), *lyft, odd), limit=20)
    assert detail(o4, "min_group") == MIN_GROUP


def test_segments_are_scored_apart() -> None:
    # fares of a luxury family are 5x higher on every ride: nothing is an outlier
    # within it, though it would be against the other family
    lux = [
        ride(
            3000 + i,
            product_family="black_lux",
            order_total_amount=5 * ride(i)["order_total_amount"],
        )
        for i in range(N)
    ]
    o4, _ = run(frame(*base(), *lux))
    assert o4.n_failed == 0
    segments = detail(o4, "by_segment")
    assert {(s["merchant"], s["family"]) for s in segments} == {
        ("Uber", "uberx"),
        ("Uber", "black_lux"),
    }
    medians = {f["family"]: f["median_per_mile"] for f in detail(o4, "by_family")}
    assert medians["black_lux"] == pytest.approx(5 * medians["uberx"])


def test_o4_details_by_segment_and_family() -> None:
    o4, _ = run(with_extra(expensive()))
    (segment,) = detail(o4, "by_segment")
    assert segment["merchant"] == "Uber"
    assert segment["family"] == "uberx"
    assert (segment["rows"], segment["scored"], segment["flagged"]) == (N + 1, N + 1, 1)
    assert segment["rate"] == pytest.approx(1 / (N + 1))
    assert 1.5 < segment["median_per_mile"] < 2.5
    assert 0.3 < segment["median_per_minute"] < 1.0
    (family,) = detail(o4, "by_family")
    assert family["family"] == "uberx"


def test_o4_sample_is_the_most_extreme_first_and_limited() -> None:
    rows = [
        expensive(order_number="o_a", order_total_amount=60.0),
        expensive(order_number="o_b", order_total_amount=200.0),
        expensive(order_number="o_c", order_total_amount=100.0),
    ]
    o4, _ = run(with_extra(*rows))
    assert o4.n_failed == 3
    assert o4.sample is not None
    assert o4.sample["order_number"].to_list() == ["o_b", "o_c", "o_a"]
    assert o4.sample.columns == [
        "merchant_name",
        "user_id",
        "order_number",
        "email_time",
        "order_total_amount",
        "checksum",
        "distance_mi",
        "duration_s",
        "product_family",
        "price_per_mile",
        "price_per_minute",
        "rz_mile",
        "rz_minute",
    ]
    first = o4.sample.row(0, named=True)
    assert first["price_per_mile"] == pytest.approx(200.0 / 5.0)
    assert first["price_per_minute"] == pytest.approx(200.0 / 15.0)
    limited, _ = run(with_extra(*rows), limit=2)
    assert limited.sample is not None
    assert limited.sample["order_number"].to_list() == ["o_b", "o_c"]


# --- O5 -----------------------------------------------------------------------------


def test_o5_always_passes_and_reports_high_and_low() -> None:
    long = expensive(order_number="o_long", order_total_amount=400.0)
    cheap = expensive(order_number="o_cheap", order_total_amount=0.02)
    _, o5 = run(with_extra(long, cheap))
    assert (o5.passed, o5.severity, o5.threshold) == (True, Severity.INFO, None)
    assert (o5.n_failed, o5.n_evaluated) == (2, N + 2)
    assert o5.metric == pytest.approx(2 / (N + 2))
    assert detail(o5, "k") == 3.5
    assert (detail(o5, "high"), detail(o5, "low")) == (1, 1)
    assert detail(o5, "note") == "context only: mostly long legitimate trips"
    assert detail(o5, "by_merchant")["Uber"] == {
        "rows": N + 2,
        "flagged": 2,
        "high": 1,
        "low": 1,
        "rate": pytest.approx(2 / (N + 2)),
    }
    assert o5.sample is None


def test_o5_flags_a_long_trip_that_o4_does_not() -> None:
    long = ride(1000, order_total_amount=2.0 * 60, distance_mi=60.0, duration_s=10800)
    o4, o5 = run(with_extra(long))
    assert o4.n_failed == 0
    assert o5.n_failed == 1


def test_o5_scores_each_merchant_apart() -> None:
    # Lyft fares 10x higher throughout: nothing is an outlier within its merchant
    lyft = [
        ride(
            4000 + i,
            merchant_name="Lyft",
            product_family="lyft_standard",
            order_total_amount=10 * ride(i)["order_total_amount"],
        )
        for i in range(N)
    ]
    _, o5 = run(frame(*base(), *lyft))
    assert o5.n_failed == 0
    assert set(detail(o5, "by_merchant")) == {"Uber", "Lyft"}


@pytest.mark.parametrize(
    "changes",
    [
        {"currency_suspect": True},
        {"is_superseded": True},
        {"receipt_kind": "food_order", "is_rider_spend": False},
        {"order_total_amount": 0.0},
        {"order_total_amount": -400.0},
    ],
    ids=str,
)
def test_o5_population_filters(changes: dict[str, Any]) -> None:
    _, o5 = run(with_extra(expensive(**({"order_total_amount": 400.0} | changes))))
    assert (o5.n_failed, o5.n_evaluated) == (0, N)


def test_o5_does_not_need_distance_duration_or_family() -> None:
    row = expensive(
        order_total_amount=400.0, distance_mi=None, duration_s=None, product_family=None
    )
    o4, o5 = run(with_extra(row))
    assert (o4.n_failed, o4.n_evaluated) == (0, N)
    assert (o5.n_failed, o5.n_evaluated) == (1, N + 1)


def test_o5_k_is_configurable() -> None:
    long = expensive(order_total_amount=400.0)
    _, loose = run(with_extra(long), k5=1000.0)
    _, tight = run(with_extra(long), k5=1.0)
    assert (loose.n_failed, detail(loose, "k")) == (0, 1000.0)
    assert tight.n_failed > 1
    assert "1" in tight.predicate


# --- the pair and the registry --------------------------------------------------------


def test_the_pair_shares_one_scoring_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = outliers.score

    def counting(lf: pl.LazyFrame, min_group: int = MIN_GROUP) -> outliers.Scores:
        calls.append(1)
        return real(lf, min_group)

    monkeypatch.setattr(outliers, "score", counting)
    lf = with_extra(expensive())
    o4, o5 = outlier_checks()
    o4(lf)
    o5(lf)
    assert len(calls) == 1
    o5(with_extra())  # another frame is scored afresh
    assert len(calls) == 2


def test_dataset_checks_take_k_from_params() -> None:
    lf = with_extra(expensive())
    default = [c(lf) for c in dataset_checks()]
    assert (detail(default[0], "k"), detail(default[1], "k")) == (4.0, 3.5)
    params = {
        "fare_outlier_conditional": {"k": 1000.0},
        "fare_outlier_univariate": {"k": 2.0},
    }
    o4, o5 = (c(lf) for c in dataset_checks(params))
    assert (detail(o4, "k"), o4.n_failed) == (1000.0, 0)
    assert detail(o5, "k") == 2.0


def test_the_registry_has_both_checks() -> None:
    from rideshare_data_pipeline.checks import default_dataset_checks

    checks = default_dataset_checks({"fare_outlier_conditional": {"k": 9.0}}, 5)
    assert len(checks) == 16
    names = [c(with_extra()).name for c in checks[-2:]]
    assert names == ["fare_outlier_conditional", "fare_outlier_univariate"]


def test_an_empty_population_passes_with_no_rate() -> None:
    lf = frame(ride(0, currency_suspect=True))
    o4, o5 = run(lf)
    assert (o4.passed, o4.metric, o4.n_evaluated) == (True, None, 0)
    assert (o5.passed, o5.metric, o5.n_evaluated) == (True, None, 0)
    assert detail(o4, "by_direction") == {"high": 0, "low": 0, "mixed": 0}
