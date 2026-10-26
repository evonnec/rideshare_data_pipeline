"""Description-parsing checks: how much of ``product_description`` could be read.

PC1 runs on the frame that carries the derived columns (``receipt_kind``,
``distance_mi``, ``duration_s``, ``distance_unit``, ``product_family``) and, like
the uniqueness checks, the dedupe flag ``is_superseded`` (added here if missing).
"""

lazy import polars as pl

from rideshare_data_pipeline.checks.model import (
    CheckResult,
    DatasetCheck,
    FlaggedCheck,
    Lazy,
    Severity,
)
from rideshare_data_pipeline.dedupe import supersede_flags
from rideshare_data_pipeline.parse import FAMILIES

COVERAGE_KINDS: tuple[str, ...] = ("ride", "tip_update")
"""Kinds whose description is expected to state a trip."""

TIP_STUBS: tuple[str, ...] = ("increase tip", "tip increase")
"""Descriptions (trimmed, lower case) of a tip update's stub line. They name no trip,
so they can never carry a distance or a duration and are left out of PC1."""

METRIC_LABEL = "worst merchant unparsed rate"

MIN_COVERAGE = 0.98
MAX_UNPARSED = 0.02
"""PC1 passes while every merchant's unparsed rate is at most this."""

TOP_SHAPES = 10
SHAPE_LIMIT = 100
"""Characters of an unparsed description shape kept in a report."""

NO_MERCHANT = "(none)"
NO_VALUE = "(none)"


def _in_population() -> pl.Expr:
    """True on the rows PC1 evaluates (see ``_population``)."""
    stub = (
        pl.col("product_description")
        .str.strip_chars()
        .str.to_lowercase()
        .is_in(TIP_STUBS)
        .fill_null(False)
    )
    return (
        pl.col("receipt_kind").cast(pl.String).is_in(COVERAGE_KINDS)
        & ~pl.col("is_superseded")
        & ~stub
    )


def _parsed() -> pl.Expr:
    return pl.col("distance_mi").is_not_null() & pl.col("duration_s").is_not_null()


def unparsed_flag(_lf: Lazy) -> pl.Expr:
    """The unparsed rows of PC1's population: the rows counted in its ``n_failed``."""
    return (_in_population() & ~_parsed()).fill_null(False)


def _population(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Ride and tip-update rows that are not superseded, with ``parsed`` and shape.

    The tip-update stubs (``Increase Tip``, ``Tip Increase``) are not in it.
    """
    if "is_superseded" not in lf.collect_schema().names():
        lf = supersede_flags(lf)
    return lf.filter(_in_population()).with_columns(
        pl.col("merchant_name").fill_null(NO_MERCHANT).alias("merchant"),
        _parsed().alias("parsed"),
    )


def _shape() -> pl.Expr:
    """The description with every number replaced by ``#``; null is ``(null)``."""
    return (
        pl.col("product_description")
        .str.replace_all(r"\d+(?:[.,]\d+)*", "#")
        .str.slice(0, SHAPE_LIMIT)
        .fill_null("(null)")
    )


def description_parse_coverage_check(lf: pl.LazyFrame) -> CheckResult:
    """PC1 (warn): every merchant parses at least 98% of its ride rows.

    Population: rows of kind ride or tip_update that are not superseded, without
    the tip-update stubs (TIP_STUBS), which can't carry a distance. A row is
    parsed when both ``distance_mi`` and ``duration_s`` are non-null. ``metric`` is
    the **worst merchant's** unparsed rate (``details["metric_label"]``), not
    ``n_failed / n_evaluated``, which pools the merchants; ``threshold`` is 0.02,
    ``n_failed`` the unparsed rows and ``n_evaluated`` the population.

    ``details``: ``merchants`` (``{merchant: {rows, parsed, coverage}}``), ``units``
    (rows per ``distance_unit``), ``families`` (``{merchant: {family: rows}}``, all
    families listed) and ``unparsed_shapes`` (at most TOP_SHAPES ``{shape, count}``
    with digits replaced by ``#``: shapes only, never the text of a receipt).
    """
    pop = _population(lf)
    per_merchant, units, families, shapes = pl.collect_all(
        [
            pop.group_by("merchant").agg(
                pl.len().alias("rows"), pl.col("parsed").sum().alias("parsed")
            ),
            pop.group_by(
                pl.col("distance_unit").cast(pl.String).fill_null(NO_VALUE)
            ).len(),
            pop.group_by(
                "merchant", pl.col("product_family").cast(pl.String).fill_null(NO_VALUE)
            ).len(),
            pop.filter(~pl.col("parsed"))
            .group_by(_shape().alias("shape"))
            .len()
            .sort(["len", "shape"], descending=[True, False])
            .head(TOP_SHAPES),
        ],
        engine="streaming",
    )
    merchants = {
        m: {"rows": n, "parsed": p, "coverage": p / n}
        for m, n, p in per_merchant.sort("merchant").iter_rows()
    }
    n_rows = sum(v["rows"] for v in merchants.values())
    n_unparsed = n_rows - sum(v["parsed"] for v in merchants.values())
    # Compare the rates of unparsed rows, not 1 - coverage, which rounds up.
    worst = max(
        ((v["rows"] - v["parsed"]) / v["rows"] for v in merchants.values()),
        default=None,
    )
    cells = {(m, f): n for m, f, n in families.iter_rows()}
    return CheckResult(
        name="description_parse_coverage",
        check_id="PC1",
        category="semantic",
        severity=Severity.WARN,
        passed=worst is None or worst <= MAX_UNPARSED,
        metric=worst,
        threshold=MAX_UNPARSED,
        n_failed=n_unparsed,
        n_evaluated=n_rows,
        predicate=(
            "every merchant's ride rows have a parsed distance and duration "
            f"(coverage >= {MIN_COVERAGE:.0%})"
        ),
        details={
            "metric_label": METRIC_LABEL,
            "merchants": merchants,
            "units": dict(
                units.sort(
                    ["len", "distance_unit"], descending=[True, False]
                ).iter_rows()
            ),
            "families": {
                m: {f: cells.get((m, f), 0) for f in (*FAMILIES, NO_VALUE)}
                for m in merchants
            },
            "unparsed_shapes": [
                {"shape": s, "count": n} for s, n in shapes.iter_rows()
            ],
        },
    )


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The parsing checks, for the registry."""
    return (
        FlaggedCheck(
            "description_parse_coverage",
            description_parse_coverage_check,
            unparsed_flag,
        ),
    )
