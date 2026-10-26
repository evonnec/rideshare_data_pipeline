"""Distance, duration, unit and product family from descriptions; PC1 (issue #14)."""

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
import pytest

from rideshare_data_pipeline.checks import default_dataset_checks
from rideshare_data_pipeline.checks.parsing import (
    SHAPE_LIMIT,
    TOP_SHAPES,
    description_parse_coverage_check,
)
from rideshare_data_pipeline.config import KNOWN_CHECKS
from rideshare_data_pipeline.parse import (
    DERIVED_NAMES,
    FAMILIES,
    MI_PER_KM,
    distance_mi,
    distance_unit,
    duration_s,
    product_family,
    with_derived,
)


@dataclass(frozen=True, slots=True)
class Trip:
    """A description and what it parses to (None means null)."""

    description: str | None
    miles: float | None
    seconds: int | None
    unit: str | None
    family: str | None


KM = MI_PER_KM
TRIPS = [
    # Lyft: "<product> fare (<d>mi, <m>m <s>s)", with sub-minute and hour variants.
    Trip("Lyft fare (1.63mi, 12m 23s)", 1.63, 743, "mi", "lyft_standard"),
    Trip("Lyft fare (0.4mi, 45s)", 0.4, 45, "mi", "lyft_standard"),
    Trip("Lyft fare (12.1mi, 1h 2m 3s)", 12.1, 3723, "mi", "lyft_standard"),
    Trip("Lyft fare (5mi, 1h 0m 0s)", 5.0, 3600, "mi", "lyft_standard"),
    # Uber: "<product> <d> mi | <m> min", hours, km, "min(s)".
    Trip("UberX 9.93 mi | 32 min", 9.93, 1920, "mi", "uberx"),
    Trip("UberXL 15.2 mi | 1 h 22 min", 15.2, 4920, "mi", "xl"),
    Trip("UberX 7.5 km | 18 min", 7.5 * KM, 1080, "km", "uberx"),
    Trip("UberX 3.1 mi | 2 h", 3.1, 7200, "mi", "uberx"),
    Trip("UberX 4.00 mi | 9 min(s)", 4.0, 540, "mi", "uberx"),
    # Older Uber: "<d> miles   HH:MM:SS Trip time   <product> Car".
    Trip("2.00 miles   00:15:14 Trip time   uberX Car", 2.0, 914, "mi", "uberx"),
    Trip(
        "3.5 kilometers   01:02:03 Trip time   uberX Car", 3.5 * KM, 3723, "km", "uberx"
    ),
    Trip(
        "3.5 kilometres   00:10:00 Trip time   uberX Car", 3.5 * KM, 600, "km", "uberx"
    ),
    Trip("UberX 6 kilómetros | 20 min", 6 * KM, 1200, "km", "uberx"),
    # Lyft "Base fare" lines: a distance and a duration, no product label.
    Trip("Base fare $1.00 19m 30s $3.90 14.78 mi $15.67", 14.78, 1170, "mi", "other"),
    Trip("Base fare $0.00 1h 5m 8s $2.25 7.06 mi $8.19", 7.06, 3908, "mi", "other"),
    # A thousands separator in the distance.
    Trip("UberX 1,234.5 mi | 20 min", 1234.5, 1200, "mi", "uberx"),
    # Zero is a value, not a null.
    Trip("Lyft fare (0.00mi, 0m 0s)", 0.0, 0, "mi", "lyft_standard"),
    Trip("UberX 0.00 mi | 0 min", 0.0, 0, "mi", "uberx"),
    Trip("0.00 miles   00:00:00 Trip time   uberX Car", 0.0, 0, "mi", "uberx"),
    Trip("Lyft fare (2.00mi, 0s)", 2.0, 0, "mi", "lyft_standard"),
    # Not a trip line: every column is null, never 0.
    Trip("Increase Tip", None, None, None, None),
    Trip("Tip Increase", None, None, None, None),
    Trip("Earnings", None, None, None, None),
    Trip("Subscription", None, None, None, None),
    Trip("UberX Ride cancelled", None, None, None, None),
    Trip("Ride Pass", None, None, None, None),
    Trip("", None, None, None, None),
    Trip(None, None, None, None, None),
]


def frame(descriptions: list[str | None]) -> pl.DataFrame:
    """The columns the derived expressions read (the rest are null)."""
    n = len(descriptions)
    nulls = pl.Series([None] * n, dtype=pl.String)
    return pl.DataFrame(
        {
            "product_description": pl.Series(descriptions, dtype=pl.String),
            "merchant_name": ["Uber"] * n,
            "email_subject": nulls,
            "product_reseller": nulls,
            "product_subtitle": nulls,
            "order_number": nulls,
            "order_pickup": pl.Series([1] * n, dtype=pl.Int8),
            "product_category": ["Travel"] * n,
        }
    )


@pytest.fixture(scope="module")
def parsed() -> pl.DataFrame:
    return with_derived(frame([t.description for t in TRIPS]))


def test_every_template(parsed: pl.DataFrame) -> None:
    for trip, row in zip(TRIPS, parsed.iter_rows(named=True), strict=True):
        got = (
            row["distance_mi"],
            row["duration_s"],
            row["distance_unit"],
            row["product_family"],
        )
        want = (trip.miles, trip.seconds, trip.unit, trip.family)
        assert got[1:] == want[1:], trip.description
        assert got[0] == pytest.approx(want[0]), trip.description


def test_nulls_and_zeros_keep_their_types(parsed: pl.DataFrame) -> None:
    assert parsed.schema["distance_mi"] == pl.Float64
    assert parsed.schema["duration_s"] == pl.Int64
    assert parsed.schema["distance_unit"] == pl.Enum(["mi", "km"])
    assert parsed.schema["product_family"] == pl.Enum(FAMILIES)
    non_trip = parsed.filter(pl.col("product_description").is_null())
    assert (
        non_trip.select(pl.col(c).is_null().all() for c in DERIVED_NAMES[-4:]).row(0)
        == (True,) * 4
    )
    zeros = parsed.filter(
        pl.col("product_description").str.contains(r"^(?:Lyft fare \(|UberX )?0\.00")
    )
    assert zeros["distance_mi"].null_count() == 0
    assert (zeros["distance_mi"] == 0).all()


def test_standalone_factories_equal_with_derived(parsed: pl.DataFrame) -> None:
    df = frame([t.description for t in TRIPS])
    alone = df.with_columns(
        distance_mi(), distance_unit(), duration_s(), product_family()
    )
    assert alone.select(DERIVED_NAMES[-4:]).equals(parsed.select(DERIVED_NAMES[-4:]))


@pytest.mark.parametrize(
    ("label", "family"),
    [
        ("UberX", "uberx"),
        ("uberX", "uberx"),
        ("UberX Diamond", "uberx"),
        ("Lyft", "lyft_standard"),
        ("Pool", "shared"),
        ("POOL", "shared"),
        ("uberPOOL", "shared"),
        ("Shared", "shared"),
        ("Shared Saver", "shared"),
        ("Line", "shared"),
        ("Express Pool", "express_pool"),
        ("EXPRESS POOL", "express_pool"),
        ("UberXL", "xl"),
        ("Lyft XL", "xl"),
        ("XL", "xl"),
        ("Black", "black_lux"),
        ("Black SUV", "black_lux"),
        ("UberBLACK", "black_lux"),
        ("Lux", "black_lux"),
        ("Lux Black", "black_lux"),
        ("Premier", "black_lux"),
        ("SUV", "black_lux"),
        ("Select", "comfort"),
        ("Plus", "comfort"),
        ("Comfort", "comfort"),
        ("UberX VIP", "vip"),
        ("VIP", "vip"),
        ("Black VIP", "vip"),
        ("Wheelchair", "other"),
    ],
)
def test_product_family_mapping(label: str, family: str) -> None:
    uber = frame([f"{label} 3.00 mi | 9 min"])
    old = frame([f"3.00 miles   00:09:00 Trip time   {label} Car"])
    lyft = frame([f"{label} fare (3.00mi, 9m 0s)"])
    for df in (uber, old, lyft):
        assert df.select(product_family())["product_family"].to_list() == [family]


def test_family_values_are_the_documented_short_list() -> None:
    assert FAMILIES == (
        "uberx",
        "lyft_standard",
        "shared",
        "express_pool",
        "xl",
        "black_lux",
        "comfort",
        "vip",
        "other",
    )


# --- PC1 -----------------------------------------------------------------------


def pc1_frame(rows: Sequence[tuple[str, str, str | None, str | None]]) -> pl.LazyFrame:
    """(merchant, kind, description, order number) rows with the derived columns."""
    df = pl.DataFrame(
        {
            "merchant_name": [r[0] for r in rows],
            "receipt_kind": pl.Series([r[1] for r in rows]).cast(
                pl.Enum(["ride", "tip_update", "driver_earnings"])
            ),
            "product_description": pl.Series([r[2] for r in rows], dtype=pl.String),
            "order_number": pl.Series([r[3] for r in rows], dtype=pl.String),
            "is_superseded": [False] * len(rows),
        }
    )
    d = pl.col("product_description")
    return df.with_columns(
        pl.when(d.str.contains("mi")).then(1.0).alias("distance_mi"),
        pl.when(d.str.contains("min")).then(60).alias("duration_s"),
        pl.when(d.str.contains("mi")).then(pl.lit("mi")).alias("distance_unit"),
        pl.when(d.str.contains("mi")).then(pl.lit("uberx")).alias("product_family"),
    ).lazy()


OK = "UberX 1.00 mi | 5 min"


def test_pc1_passes_at_the_coverage_threshold() -> None:
    # 49 parsed + 1 unparsed per merchant is exactly 98%.
    rows = [("Uber", "ride", OK, None)] * 49 + [("Uber", "ride", "Weird 7", None)]
    res = description_parse_coverage_check(pc1_frame(rows))
    assert (res.check_id, res.name) == ("PC1", "description_parse_coverage")
    assert res.severity.value == "warn"
    assert res.passed
    assert res.metric == pytest.approx(0.02)
    assert res.threshold == 0.02
    assert (res.n_failed, res.n_evaluated) == (1, 50)


def test_pc1_fails_when_one_merchant_is_below_it() -> None:
    rows = (
        [("Uber", "ride", OK, None)] * 100
        + [("Lyft", "ride", OK, None)] * 9
        + [("Lyft", "ride", "Weird 7", None)]
    )
    res = description_parse_coverage_check(pc1_frame(rows))
    assert not res.passed
    assert res.metric == pytest.approx(0.1)  # the worst merchant's unparsed rate
    assert (res.n_failed, res.n_evaluated) == (1, 110)
    assert res.details["merchants"] == {
        "Lyft": {"rows": 10, "parsed": 9, "coverage": 0.9},
        "Uber": {"rows": 100, "parsed": 100, "coverage": 1.0},
    }


def test_pc1_population_is_unsuperseded_rides_and_tip_updates() -> None:
    rows = [
        ("Uber", "ride", OK, None),
        ("Uber", "tip_update", OK, None),
        ("Uber", "driver_earnings", "Earnings", None),  # not in the population
        ("Uber", "ride", "Weird 7", None),  # will be superseded below
        ("Uber", "tip_update", None, None),  # a null description is unparsed
    ]
    lf = pc1_frame(rows).with_columns(
        pl.Series([False, False, False, True, False]).alias("is_superseded")
    )
    res = description_parse_coverage_check(lf)
    assert (res.n_failed, res.n_evaluated) == (1, 3)
    assert res.details["merchants"] == {
        "Uber": {"rows": 3, "parsed": 2, "coverage": pytest.approx(2 / 3)}
    }


def test_pc1_details_report_units_families_and_shapes_only() -> None:
    rows = [
        ("Uber", "ride", OK, None),
        ("Uber", "ride", "Secret Name 12 and 3.5 here", None),
        ("Uber", "ride", None, None),
    ]
    res = description_parse_coverage_check(pc1_frame(rows))
    assert res.details["units"] == {"mi": 1, "(none)": 2}
    families = res.details["families"]
    assert isinstance(families, dict)
    assert families["Uber"]["uberx"] == 1
    assert families["Uber"]["(none)"] == 2
    assert set(families["Uber"]) == {*FAMILIES, "(none)"}
    shapes = res.details["unparsed_shapes"]
    assert shapes == [
        {"shape": "(null)", "count": 1},
        {"shape": "Secret Name # and # here", "count": 1},
    ]


def test_pc1_shapes_are_capped_and_ordered() -> None:
    rows = [("Uber", "ride", f"Shape{'x' * i} {i}", None) for i in range(15)]
    rows += [("Uber", "ride", "Common 1", None)] * 3
    shapes = description_parse_coverage_check(pc1_frame(rows)).details[
        "unparsed_shapes"
    ]
    assert isinstance(shapes, list)
    assert len(shapes) == TOP_SHAPES
    assert shapes[0] == {"shape": "Common #", "count": 3}
    long = [("Uber", "ride", "y" * 500, None)]
    longest = description_parse_coverage_check(pc1_frame(long)).details[
        "unparsed_shapes"
    ]
    assert isinstance(longest, list)
    only = longest[0]
    assert len(only["shape"]) == SHAPE_LIMIT


def test_pc1_adds_the_dedupe_flag_when_missing() -> None:
    lf = pc1_frame(
        [
            ("Uber", "ride", OK, "A1"),
            ("Uber", "ride", "Weird 7", "A1"),
        ]
    ).drop("is_superseded")
    # Same key and kind: the later email would win, but there is no email_time here,
    # so the checksum-less tie still leaves exactly one of the two rows.
    lf = lf.with_columns(
        pl.lit(None, dtype=pl.Datetime).alias("email_time"),
        pl.lit(None, dtype=pl.Datetime).alias("update_time"),
        pl.lit(None, dtype=pl.String).alias("checksum"),
    )
    res = description_parse_coverage_check(lf)
    assert res.n_evaluated == 1


def test_pc1_empty_population_passes() -> None:
    res = description_parse_coverage_check(
        pc1_frame([("Uber", "driver_earnings", "Earnings", None)])
    )
    assert res.passed
    assert res.metric is None
    assert (res.n_failed, res.n_evaluated) == (0, 0)


def test_pc1_is_registered_and_known() -> None:
    assert description_parse_coverage_check in default_dataset_checks()
    assert "description_parse_coverage" in KNOWN_CHECKS
