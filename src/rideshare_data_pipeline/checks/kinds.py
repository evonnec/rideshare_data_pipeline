"""Receipt-kind checks: the kind composition, category mismatches, unknown subjects.

These run on the frame that carries the columns of ``parse.derived_columns()``
(receipt_kind, subject_unclassified, is_rider_spend); the CLI adds them once
before running the checks.
"""

lazy import polars as pl

from rideshare_data_pipeline.checks.model import (
    KEY_COLS,
    CheckResult,
    DatasetCheck,
    RowRule,
    Severity,
)
from rideshare_data_pipeline.parse import (
    KINDS,
    MISMATCHED_CATEGORY_KINDS,
    RIDER_SPEND_KINDS,
    subject_template,
)

TOP_TEMPLATES = 20
"""At most this many unclassified templates are listed in a result's details."""

NO_MERCHANT = "(none)"


def rules() -> tuple[RowRule, ...]:
    """V2: the category ``Travel`` on a kind that isn't travel."""
    return (
        RowRule(
            name="category_mismatch",
            check_id="V2",
            category="semantic",
            severity=Severity.WARN,
            rule=(
                "product_category is not 'Travel' on food orders, freight, "
                "rentals and driver earnings"
            ),
            predicate=(pl.col("product_category") == "Travel")
            & pl.col("receipt_kind")
            .cast(pl.String)
            .is_in(sorted(MISMATCHED_CATEGORY_KINDS)),
            sample_cols=(*KEY_COLS, "receipt_kind", "product_category"),
        ),
    )


def kind_matrix(lf: pl.LazyFrame) -> dict[str, dict[str, int]]:
    """Receipts per merchant and kind: ``{merchant: {kind: n}}``, kinds in KINDS order.

    Every merchant lists every kind (0 when absent); merchants are sorted by name,
    and a null merchant is ``(none)``.
    """
    rows = (
        lf.group_by(
            pl.col("merchant_name").fill_null(NO_MERCHANT),
            pl.col("receipt_kind").cast(pl.String),
        )
        .len()
        .collect(engine="streaming")
        .iter_rows()
    )
    cells = {(merchant, kind): n for merchant, kind, n in rows}
    merchants = sorted({merchant for merchant, _ in cells})
    return {m: {k: cells.get((m, k), 0) for k in KINDS} for m in merchants}


def receipt_kinds_check(lf: pl.LazyFrame) -> CheckResult:
    """K1: the receipt kinds, as an info result.

    ``n_evaluated`` is every row and ``n_failed`` the rows that are not rider
    spend. ``details`` holds ``counts`` (rows per kind), ``by_merchant`` (the
    merchant x kind matrix) and the ``rider_spend_kinds``.
    """
    matrix = kind_matrix(lf)
    counts = {k: sum(row[k] for row in matrix.values()) for k in KINDS}
    n_rows = sum(counts.values())
    non_spend = sum(n for k, n in counts.items() if k not in RIDER_SPEND_KINDS)
    return CheckResult(
        name="receipt_kinds",
        check_id="K1",
        category="semantic",
        severity=Severity.INFO,
        passed=True,
        metric=non_spend / n_rows if n_rows else None,
        threshold=None,
        n_failed=non_spend,
        n_evaluated=n_rows,
        predicate=(
            "every receipt is rider spend "
            "(a ride, tip update, fare correction or cancellation fee)"
        ),
        details={
            "rider_spend_kinds": [k for k in KINDS if k in RIDER_SPEND_KINDS],
            "counts": counts,
            "by_merchant": matrix,
        },
    )


def unclassified_templates_check(lf: pl.LazyFrame) -> CheckResult:
    """An info result counting receipts no kind rule matched (they default to ride).

    ``n_failed`` is that count. ``details["templates"]`` lists the most common
    normalised subject templates among them (at most TOP_TEMPLATES, as
    ``{"template", "count"}`` by count then text) and ``distinct`` how many
    templates there are. Only the normalised template is reported: masks, digits,
    dates and weekdays are replaced (see ``parse.subject_template``).
    """
    unknown = (
        lf.filter(pl.col("subject_unclassified"))
        .select(subject_template().alias("template"))
        .collect(engine="streaming")
    )
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    top = (
        unknown.group_by("template")
        .len()
        .sort(["len", "template"], descending=[True, False])
    )
    return CheckResult(
        name="unclassified_subject_templates",
        check_id="K2",
        category="semantic",
        severity=Severity.INFO,
        passed=True,
        metric=unknown.height / n_rows if n_rows else None,
        threshold=None,
        n_failed=unknown.height,
        n_evaluated=n_rows,
        predicate="every receipt matches a known subject or description template",
        details={
            "distinct": top.height,
            "templates": [
                {"template": t, "count": n}
                for t, n in top.head(TOP_TEMPLATES).iter_rows()
            ],
        },
    )


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The kind checks that need the whole dataset, for the registry."""
    return (receipt_kinds_check, unclassified_templates_check)
