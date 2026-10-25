"""Arithmetic checks: relations between a receipt's money columns."""

lazy import polars as pl

from rideshare_data_pipeline.checks.model import RowRule, Severity


def rules() -> tuple[RowRule, ...]:
    """Return this category's row rules (built on call, so import stays cheap)."""
    return (
        RowRule(
            name="non_negative_total",
            check_id="A1",
            category="arithmetic",
            severity=Severity.ERROR,
            rule="order_total_amount >= 0",
            predicate=pl.col("order_total_amount") < 0,
        ),
    )
