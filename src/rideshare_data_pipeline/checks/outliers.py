"""Fare outliers (O4, O5): robust, log-scale, segment-wise, conditional on drivers.

Why not a z-score or a Tukey fence
==================================

Fares are heavy-tailed (roughly log-normal) and depend on distance and product. A raw
z-score or a Tukey fence on the fare flags a few percent of rows, and nearly all of the
high ones have a normal price per mile: they are long trips. So the checks here

1. **score within a segment** (merchant x product family): an UberX and a Black car
   have different baselines;
2. **use a robust scale on the log**: ``rz = (x - median) / (1.4826 * MAD)`` of
   ``log(x)`` (1.4826 makes the MAD match a standard deviation under a normal), so the
   heavy tail neither inflates the scale nor needs a transform per product;
3. **condition on the price drivers**: O4 flags a ride only when its price per mile
   AND its price per minute are both extreme, which a long trip (high fare, ordinary
   price per mile) never is.

A threshold ``k`` is a review budget (how many rows you are willing to look at), not
a false-positive rate: the report shows the flag rate at k in {3, 3.5, 4, 5}. There is
no machine learning on purpose: an isolation forest was tried and flagged mostly
different rows from these, without a reason to read for any of them; a z-score
on a log scale says why a row was flagged.

The checks
==========

Population (both): ``money.positive_usd_rides`` (USD rider spend, not superseded, not
currency-suspect, total > 0: zero totals are A3's and negative ones A1's), restricted
to the kinds ride and tip_update.

- O4 ``fare_outlier_conditional`` (warn): on those rows with ``distance_mi >= 0.5``,
  ``duration_s >= 60`` and a ``product_family``, fails when ``|rz(log $/mile)| > k``
  and ``|rz(log $/minute)| > k`` within merchant x product family (k defaults to 4).
  A trip that is only fast or slow (implausible speed) moves one of the two; it is
  O1's, not this check's.
- O5 ``fare_outlier_univariate`` (info): ``|rz(log total)| > k`` (default 3.5) within
  each merchant, reported as context only (it mostly finds long, legitimate trips)
  and never failing.

A segment with fewer than ``MIN_GROUP`` scored rows, or whose MAD is 0, is not scored
(its rows are never flagged) rather than scored against an unstable scale.

O4 and O5 share one pass: the scores are computed once into a narrow in-memory frame,
and both checks (and the failing-row sample of O4) read it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

lazy import polars as pl

from rideshare_data_pipeline.checks.model import (
    KEY_COLS,
    SAMPLE_LIMIT,
    CheckResult,
    DatasetCheck,
    FlaggedCheck,
    Frame,
    Lazy,
    NamedCheck,
    Severity,
    rows_at,
)
from rideshare_data_pipeline.dedupe import supersede_flags
from rideshare_data_pipeline.money import positive_usd_rides

MAD_SCALE = 1.4826
"""Makes the MAD a consistent estimator of the standard deviation of a normal."""

MIN_GROUP = 30
"""The fewest scored rows a group needs to be scored. A median absolute deviation of
fewer values is too noisy to flag against, so those rows get no score. 30 is the usual
rule of thumb for a stable spread estimate; the synthetic baseline has about 120 rows
per merchant segment."""

DEFAULT_K = 4.0
"""O4's threshold: both |rz| must exceed it."""

DEFAULT_K_UNIVARIATE = 3.5
"""O5's threshold."""

K_GRID: tuple[float, ...] = (3.0, 3.5, 4.0, 5.0)
"""The thresholds of the sensitivity table (flags and rate at each)."""

MIN_MILES = 0.5
MIN_SECONDS = 60
"""O4 skips trips shorter or briefer than these: $/mile and $/minute of a stub trip
are dominated by the base fare."""

OUTLIER_KINDS: tuple[str, ...] = ("ride", "tip_update")
NO_MERCHANT = "(none)"
TOP_SEGMENTS = 10
UNIVARIATE_NOTE = "context only: mostly long legitimate trips"

_SEGMENT = ["merchant", "family"]


def log_positive(x: pl.Expr) -> pl.Expr:
    """``log(x)`` where x > 0, else null (a zero or negative value has no log)."""
    return pl.when(x > 0).then(x.log()).otherwise(None)


def _z(
    x: pl.Expr, median: pl.Expr, scale: pl.Expr, n: pl.Expr, min_group: int
) -> pl.Expr:
    """The score from its parts; null for a group that is too small or has no spread."""
    return pl.when((n >= min_group) & (scale > 0)).then((x - median) / scale)


def robust_z(
    x: pl.Expr,
    by: str | pl.Expr | Sequence[str | pl.Expr],
    *,
    min_group: int = MIN_GROUP,
) -> pl.Expr:
    """Robust z-score of `x` within each group of `by`: (x - median) / (1.4826 * MAD).

    ``MAD = median(|x - median|)`` of the group, both taken with ``.over(by)``; nulls
    are ignored by the medians and stay null. The score is null (not inf, so the row
    is never flagged) when the group's MAD is 0, or when the group has fewer than
    `min_group` non-null values. For a log scale pass ``log_positive(x)``.

    As one expression it computes each group median twice; ``score`` stages the same
    parts as columns instead (about 2.6x faster at a ~2M-row scale, where it matters).
    """
    median = x.median().over(by)
    scale = MAD_SCALE * (x - median).abs().median().over(by)
    return _z(x, median, scale, x.count().over(by), min_group)


@dataclass(frozen=True, slots=True)
class Scores:
    """The scored population: one narrow row per ride in it (see ``score``)."""

    frame: Frame
    """Columns: _row, merchant, family, o4 (eligible for O4), per_mile, per_minute,
    rz_mile, rz_minute (O4's scores), rz_total (O5's), per merchant."""

    min_group: int


def _eligible() -> pl.Expr:
    return (
        (pl.col("distance_mi") >= MIN_MILES)
        & (pl.col("duration_s") >= MIN_SECONDS)
        & pl.col("product_family").is_not_null()
    ).fill_null(False)


def score(lf: pl.LazyFrame, min_group: int = MIN_GROUP) -> Scores:
    """Compute every robust score once, in one lazy query (the shared pass)."""
    if "is_superseded" not in lf.collect_schema().names():
        lf = supersede_flags(lf)
    rides = (
        positive_usd_rides(lf.with_row_index("_row"))
        .filter(pl.col("receipt_kind").cast(pl.String).is_in(OUTLIER_KINDS))
        .select(
            "_row",
            pl.col("merchant_name").fill_null(NO_MERCHANT).alias("merchant"),
            pl.col("product_family").cast(pl.String).alias("family"),
            _eligible().alias("o4"),
            pl.col("order_total_amount").alias("total"),
            pl.col("distance_mi"),
            pl.col("duration_s"),
        )
    )
    # Prices exist only on O4's rows, so every window below sees only those.
    per = rides.with_columns(
        pl.when(pl.col("o4"))
        .then(pl.col("total") / pl.col("distance_mi"))
        .alias("per_mile"),
        pl.when(pl.col("o4"))
        .then(pl.col("total") / (pl.col("duration_s") / 60))
        .alias("per_minute"),
    ).with_columns(
        log_positive(pl.col("per_mile")).alias("log_mile"),
        log_positive(pl.col("per_minute")).alias("log_minute"),
        log_positive(pl.col("total")).alias("log_total"),
    )
    # The same parts as robust_z, staged as columns so no median is computed twice.
    measures = {
        "log_mile": _SEGMENT,
        "log_minute": _SEGMENT,
        "log_total": ["merchant"],
    }
    staged = per.with_columns(
        pl.col(m).median().over(by).alias(f"med_{m}") for m, by in measures.items()
    ).with_columns(
        expr
        for m, by in measures.items()
        for expr in (
            (
                MAD_SCALE * (pl.col(m) - pl.col(f"med_{m}")).abs().median().over(by)
            ).alias(f"scale_{m}"),
            pl.col(m).count().over(by).alias(f"n_{m}"),
        )
    )
    frame = staged.select(
        "_row",
        "merchant",
        "family",
        "o4",
        "per_mile",
        "per_minute",
        *(
            _z(
                pl.col(m),
                pl.col(f"med_{m}"),
                pl.col(f"scale_{m}"),
                pl.col(f"n_{m}"),
                min_group,
            ).alias(name)
            for m, name in (
                ("log_mile", "rz_mile"),
                ("log_minute", "rz_minute"),
                ("log_total", "rz_total"),
            )
        ),
    ).collect()
    return Scores(frame, min_group)


def _pair_flags(k: float) -> pl.Expr:
    return ((pl.col("rz_mile").abs() > k) & (pl.col("rz_minute").abs() > k)).fill_null(
        False
    )


def _direction() -> pl.Expr:
    """``high`` (both scores above), ``low`` (both below) or ``mixed``."""
    up = pl.col("rz_mile") > 0
    down = pl.col("rz_mile") < 0
    both_up = up & (pl.col("rz_minute") > 0)
    both_down = down & (pl.col("rz_minute") < 0)
    return (
        pl.when(both_up)
        .then(pl.lit("high"))
        .when(both_down)
        .then(pl.lit("low"))
        .otherwise(pl.lit("mixed"))
    )


def _rate(n: int, total: int) -> float | None:
    return n / total if total else None


def conditional_result(scores: Scores, k: float, sample: Frame | None) -> CheckResult:
    """O4 from the shared scores (see the module docstring); `sample` is its sample."""
    eligible = scores.frame.filter(pl.col("o4"))
    scored = eligible.filter(
        pl.col("rz_mile").is_not_null() & pl.col("rz_minute").is_not_null()
    )
    flagged = eligible.filter(_pair_flags(k))
    n_eligible, n_flagged = eligible.height, flagged.height
    grid = eligible.select(
        _pair_flags(g).sum().alias(f"k{i}") for i, g in enumerate(K_GRID)
    ).row(0, named=True)
    directions = {"high": 0, "low": 0, "mixed": 0} | dict(
        flagged.group_by(_direction().alias("direction"))
        .len()
        .iter_rows()  # (direction, count)
    )
    segments = (
        eligible.group_by(_SEGMENT)
        .agg(
            pl.len().alias("rows"),
            pl.col("rz_mile").is_not_null().sum().alias("scored"),
            _pair_flags(k).sum().alias("flagged"),
            pl.col("per_mile").median().alias("median_per_mile"),
            pl.col("per_minute").median().alias("median_per_minute"),
        )
        .sort(["flagged", "rows", *_SEGMENT], descending=[True, True, False, False])
    )
    families = (
        eligible.group_by("family")
        .agg(
            pl.len().alias("rows"),
            pl.col("per_mile").median().alias("median_per_mile"),
            pl.col("per_minute").median().alias("median_per_minute"),
        )
        .sort(["rows", "family"], descending=[True, False])
    )
    return CheckResult(
        name="fare_outlier_conditional",
        check_id="O4",
        category="outliers",
        severity=Severity.WARN,
        passed=n_flagged == 0,
        metric=_rate(n_flagged, n_eligible),
        threshold=0.0,
        n_failed=n_flagged,
        n_evaluated=n_eligible,
        predicate=(
            f"not (|rz(log $/mile)| > {k:g} and |rz(log $/minute)| > {k:g}) within "
            "merchant x product family (USD rides, >= 0.5 mi and >= 60 s)"
        ),
        sample=sample,
        details={
            "k": k,
            "flag_rate": _rate(n_flagged, n_eligible),
            "scored": scored.height,
            "min_group": scores.min_group,
            "by_direction": directions,
            "sensitivity": [
                {
                    "k": g,
                    "flagged": grid[f"k{i}"],
                    "rate": _rate(grid[f"k{i}"], n_eligible),
                }
                for i, g in enumerate(K_GRID)
            ],
            "by_segment": _segment_rows(segments),
            "by_family": families.to_dicts(),
        },
    )


def _segment_rows(segments: Frame) -> list[dict[str, object]]:
    return [
        {**row, "rate": _rate(row["flagged"], row["rows"])}
        for row in segments.to_dicts()
    ]


def univariate_result(scores: Scores, k: float) -> CheckResult:
    """O5 from the shared scores: |rz(log total)| > k per merchant, info only."""
    frame = scores.frame
    flag = (pl.col("rz_total").abs() > k).fill_null(False)
    n, flagged = frame.height, int(frame.select(flag.sum()).item())
    merchants = (
        frame.group_by("merchant")
        .agg(
            pl.len().alias("rows"),
            flag.sum().alias("flagged"),
            (flag & (pl.col("rz_total") > 0)).sum().alias("high"),
            (flag & (pl.col("rz_total") < 0)).sum().alias("low"),
        )
        .sort("merchant")
    )
    return CheckResult(
        name="fare_outlier_univariate",
        check_id="O5",
        category="outliers",
        severity=Severity.INFO,
        passed=True,
        metric=_rate(flagged, n),
        threshold=None,
        n_failed=flagged,
        n_evaluated=n,
        predicate=(
            f"|rz(log total)| > {k:g} within merchant, reported only "
            f"({UNIVARIATE_NOTE})"
        ),
        details={
            "k": k,
            "flag_rate": _rate(flagged, n),
            "note": UNIVARIATE_NOTE,
            "min_group": scores.min_group,
            "high": int(merchants["high"].sum()),
            "low": int(merchants["low"].sum()),
            "by_merchant": {
                row["merchant"]: {
                    **{key: row[key] for key in ("rows", "flagged", "high", "low")},
                    "rate": _rate(row["flagged"], row["rows"]),
                }
                for row in merchants.to_dicts()
            },
        },
    )


def conditional_sample(
    lf: pl.LazyFrame, scores: Scores, k: float, limit: int
) -> Frame | None:
    """The `limit` most extreme O4 rows (by the smaller of the two |rz|), or None.

    One more (cheap) pass over `lf` to fetch the rows' receipt columns by position.
    """
    top = (
        scores.frame.filter(pl.col("o4") & _pair_flags(k))
        .with_columns(
            pl.min_horizontal(pl.col("rz_mile").abs(), pl.col("rz_minute").abs()).alias(
                "extremity"
            )
        )
        .sort("extremity", descending=True)
        .head(limit)
    )
    if top.height == 0:
        return None
    present = set(lf.collect_schema().names())
    wanted = (*KEY_COLS, "distance_mi", "duration_s", "product_family")
    rows = (
        lf.with_row_index("_row")
        .filter(pl.col("_row").is_in(top["_row"].implode()))
        .select("_row", *(c for c in wanted if c in present))
        .collect()
    )
    return top.join(rows, on="_row", how="left", maintain_order="left").select(
        *(c for c in wanted if c in present),
        pl.col("per_mile").alias("price_per_mile"),
        pl.col("per_minute").alias("price_per_minute"),
        pl.col("rz_mile"),
        pl.col("rz_minute"),
    )


type Params = Mapping[str, Mapping[str, float]]
"""``{check name: {param: value}}``, as ``QCConfig.check_params`` returns it."""


def outlier_checks(
    conditional_k: float = DEFAULT_K,
    univariate_k: float = DEFAULT_K_UNIVARIATE,
    sample_limit: int = SAMPLE_LIMIT,
    min_group: int = MIN_GROUP,
) -> tuple[FlaggedCheck, NamedCheck]:
    """O4 and O5 as a pair of checks that share one scoring pass.

    O4 is a FlaggedCheck: its ``row_flag`` marks the rows it counted, from the same
    cached scores (call it with the frame the check ran on; it never rescores).

    The first one called on a frame computes the scores and the second reuses them
    (the pair remembers the last frame it was given, by identity), as
    ``structural.column_profile_checks`` does. Each call builds a fresh pair, so
    nothing outlives a run.
    """
    last: list[tuple[pl.LazyFrame, Scores]] = []

    def scores_of(lf: pl.LazyFrame) -> Scores:
        if last and last[0][0] is lf:
            return last[0][1]
        found = score(lf, min_group)
        last[:] = [(lf, found)]
        return found

    def conditional(lf: pl.LazyFrame) -> CheckResult:
        scores = scores_of(lf)
        sample = conditional_sample(lf, scores, conditional_k, sample_limit)
        return conditional_result(scores, conditional_k, sample)

    def conditional_flag(lf: Lazy) -> pl.Expr:
        frame = scores_of(lf).frame
        flagged = frame.filter(pl.col("o4") & _pair_flags(conditional_k))
        return rows_at(flagged["_row"])

    def univariate(lf: pl.LazyFrame) -> CheckResult:
        return univariate_result(scores_of(lf), univariate_k)

    return (
        FlaggedCheck(CONDITIONAL, conditional, conditional_flag),
        NamedCheck(UNIVARIATE, univariate),
    )


CONDITIONAL = "fare_outlier_conditional"
UNIVARIATE = "fare_outlier_univariate"

PARAM_NAMES: Mapping[str, tuple[str, ...]] = {
    CONDITIONAL: ("k",),
    UNIVARIATE: ("k",),
}
"""The params each check accepts (every one a number greater than 0); config.py
validates ``[checks.<name>.params]`` against this."""


def dataset_checks(
    params: Params | None = None, sample_limit: int = SAMPLE_LIMIT
) -> tuple[DatasetCheck, ...]:
    """The outlier checks for the registry; `params` carries a configured k."""
    params = params or {}
    return outlier_checks(
        conditional_k=params.get(CONDITIONAL, {}).get("k", DEFAULT_K),
        univariate_k=params.get(UNIVARIATE, {}).get("k", DEFAULT_K_UNIVARIATE),
        sample_limit=sample_limit,
    )


__all__ = [
    "CONDITIONAL",
    "DEFAULT_K",
    "DEFAULT_K_UNIVARIATE",
    "K_GRID",
    "MIN_GROUP",
    "PARAM_NAMES",
    "UNIVARIATE",
    "Params",
    "Scores",
    "conditional_result",
    "dataset_checks",
    "log_positive",
    "outlier_checks",
    "robust_z",
    "score",
    "univariate_result",
]
