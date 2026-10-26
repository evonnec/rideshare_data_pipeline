"""Trip plausibility (O1-O3): cheap physical checks on the parsed trip metrics.

A receipt that states 60 miles in 20 minutes is not a trip anyone took at 180 mph;
one with a zero distance or duration and a positive fare is a parse or vendor error;
a 100-mile or 3-hour trip is rare enough to look at. These need no baseline, only
``distance_mi`` (km already converted) and ``duration_s`` from ``parse.py``.

Population (all three): ride and tip-update receipts that are not superseded. The
question is about the trip, not the money, so currency does not matter. A null metric
is never a failure: null means the description did not parse, which is PC1's job.

- O1 ``implausible_speed`` (warn): ``duration_s >= min_seconds`` and
  ``distance_mi / (duration_s / 3600) > max_mph`` (defaults 120 s and 80 mph). The
  speed is computed only where ``duration_s > 0``, so it is never inf or NaN. Short
  trips are skipped because a few seconds of rounding makes their speed meaningless.
- O2 ``zero_trip_metrics`` (warn): ``(distance_mi == 0 or duration_s == 0)`` with
  ``order_total_amount > 0``. A zero fare is A3's. ``by_field`` splits the rows into
  ``zero_distance`` (only), ``zero_duration`` (only) and ``both``.
- O3 ``extreme_trip`` (warn): ``distance_mi > max_miles`` or ``duration_s >
  max_seconds`` (defaults 100 mi and 10,800 s). ``by_field`` counts ``over_distance``
  and ``over_duration`` (a row over both counts in both).

All three are row rules in the runner's single select. The implied speed is not a
sample column (the sample mechanism selects existing columns); it is the sample's
``distance_mi / duration_s * 3600``.
"""

from collections.abc import Mapping

lazy import polars as pl

from rideshare_data_pipeline.checks.arithmetic import is_ride_kind
from rideshare_data_pipeline.checks.model import KEY_COLS, RowRule, Severity

DEFAULT_MAX_MPH = 80.0
DEFAULT_MIN_SECONDS = 120.0
DEFAULT_MAX_MILES = 100.0
DEFAULT_MAX_SECONDS = 10_800.0

IMPLAUSIBLE_SPEED = "implausible_speed"
ZERO_METRICS = "zero_trip_metrics"
EXTREME_TRIP = "extreme_trip"

PARAM_NAMES: Mapping[str, tuple[str, ...]] = {
    IMPLAUSIBLE_SPEED: ("max_mph", "min_seconds"),
    ZERO_METRICS: (),
    EXTREME_TRIP: ("max_miles", "max_seconds"),
}
"""The params each check accepts (every one a number greater than 0)."""

_SAMPLE_COLS = (*KEY_COLS, "distance_mi", "duration_s", "product_family")


def _g(x: float) -> str:
    return f"{x:,g}"


def trip_population() -> pl.Expr:
    """Ride and tip-update rows that are not superseded."""
    return is_ride_kind() & ~pl.col("is_superseded")


def implied_mph() -> pl.Expr:
    """Distance over time in mph; null where the duration is not above 0 (never inf)."""
    duration = pl.col("duration_s")
    return (
        pl.when(duration > 0)
        .then(pl.col("distance_mi") / (duration.cast(pl.Float64) / 3600))
        .otherwise(None)
    )


def implausible_speed(max_mph: float, min_seconds: float) -> pl.Expr:
    """True where the trip lasts ``min_seconds`` or more and its speed beats the cap."""
    return (pl.col("duration_s") >= min_seconds) & (implied_mph() > max_mph)


def _is_zero(column: str) -> pl.Expr:
    return (pl.col(column) == 0).fill_null(False)


def zero_metrics_breakdown() -> dict[str, pl.Expr]:
    paid = (pl.col("order_total_amount") > 0).fill_null(False)
    d, t = _is_zero("distance_mi"), _is_zero("duration_s")
    return {
        "zero_distance": paid & d & ~t,
        "zero_duration": paid & t & ~d,
        "both": paid & d & t,
    }


def rules(
    params: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[RowRule, ...]:
    """O1-O3 as row rules; `params` is ``{check name: {param: value}}``."""
    params = params or {}
    speed = params.get(IMPLAUSIBLE_SPEED, {})
    max_mph = speed.get("max_mph", DEFAULT_MAX_MPH)
    min_seconds = speed.get("min_seconds", DEFAULT_MIN_SECONDS)
    extreme = params.get(EXTREME_TRIP, {})
    max_miles = extreme.get("max_miles", DEFAULT_MAX_MILES)
    max_seconds = extreme.get("max_seconds", DEFAULT_MAX_SECONDS)
    return (
        RowRule(
            name=IMPLAUSIBLE_SPEED,
            check_id="O1",
            category="outliers",
            severity=Severity.WARN,
            rule=(
                f"implied speed is at most {_g(max_mph)} mph on trips of "
                f"{_g(min_seconds)} s or more (rides, not superseded)"
            ),
            predicate=implausible_speed(max_mph, min_seconds),
            applies_to=trip_population(),
            sample_cols=_SAMPLE_COLS,
        ),
        RowRule(
            name=ZERO_METRICS,
            check_id="O2",
            category="outliers",
            severity=Severity.WARN,
            rule=(
                "distance and duration are not 0 when order_total_amount > 0 "
                "(rides, not superseded)"
            ),
            predicate=(_is_zero("distance_mi") | _is_zero("duration_s"))
            & (pl.col("order_total_amount") > 0),
            applies_to=trip_population(),
            sample_cols=_SAMPLE_COLS,
            breakdown=zero_metrics_breakdown(),
        ),
        RowRule(
            name=EXTREME_TRIP,
            check_id="O3",
            category="outliers",
            severity=Severity.WARN,
            rule=(
                f"distance is at most {_g(max_miles)} mi and duration at most "
                f"{_g(max_seconds)} s (rides, not superseded)"
            ),
            predicate=(pl.col("distance_mi") > max_miles)
            | (pl.col("duration_s") > max_seconds),
            applies_to=trip_population(),
            sample_cols=_SAMPLE_COLS,
            breakdown={
                "over_distance": pl.col("distance_mi") > max_miles,
                "over_duration": pl.col("duration_s") > max_seconds,
            },
        ),
    )


__all__ = [
    "EXTREME_TRIP",
    "IMPLAUSIBLE_SPEED",
    "PARAM_NAMES",
    "ZERO_METRICS",
    "implausible_speed",
    "implied_mph",
    "rules",
    "trip_population",
]
