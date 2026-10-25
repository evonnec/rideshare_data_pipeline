"""Uniqueness checks: keys that must not repeat."""

lazy import polars as pl

from rideshare_data_pipeline.checks.model import RowRule, Severity


def rules() -> tuple[RowRule, ...]:
    """Return this category's row rules (built on call, so import stays cheap)."""
    return (
        RowRule(
            name="checksum_unique",
            check_id="U1",
            category="uniqueness",
            severity=Severity.ERROR,
            rule="checksum is unique",
            # Every row of a repeated checksum fails, not only the later ones; a
            # null is not a duplicate of another null.
            predicate=pl.col("checksum").is_duplicated()
            & pl.col("checksum").is_not_null(),
        ),
    )
