"""The population every money statistic is computed on, and the spend waterfall.

``order_total_amount`` has no currency column, so a plain sum mixes dollars with
pesos (about 0.5% of rows from non-US merchant senders are about 37% of a naive
total). ``usd_rides`` is THE population for every money statistic and outlier
baseline: totals, quantiles, $/mile and $/minute baselines, per-panelist spend and
volume checks (#18, #20, #27...). A check that sums or ranks ``order_total_amount``
must start from it, or state why it doesn't.

A row is in it when it:

* is rider spend (``is_rider_spend``: a ride, tip update, fare correction or
  cancellation fee);
* is not superseded by a later version of the receipt (``is_superseded``);
* is not currency-suspect (``currency_suspect``: a non-US sender, a km description
  or a Spanish or Portuguese subject; see ``parse.py``);
* has a non-null ``order_total_amount``.

Rows outside it are flagged, never dropped; no exchange rate is applied (FX
conversion is future work).
"""

from collections.abc import Mapping

lazy import polars as pl

type Waterfall = dict[str, dict[str, float | int]]
"""``{step: {"rows": n, "amount": total}}``; see ``spend_waterfall``."""

STEPS: tuple[str, ...] = (
    "naive_total",
    "currency_suspect",
    "non_rider_spend",
    "superseded",
    "clean_usd_rider_spend",
)

AMOUNT = "order_total_amount"


def is_usd_rider_spend() -> pl.Expr:
    """True for rider-spend rows that are not currency-suspect (superseded or not)."""
    return pl.col("is_rider_spend") & ~pl.col("currency_suspect")


def usd_rides[F: (pl.DataFrame, pl.LazyFrame)](frame: F) -> F:
    """The rows every money statistic is computed on (see the module docstring)."""
    return frame.filter(
        is_usd_rider_spend() & ~pl.col("is_superseded") & pl.col(AMOUNT).is_not_null()
    )


def _step(mask: pl.Expr | None, name: str) -> list[pl.Expr]:
    """Row count and amount sum of the rows with an amount where `mask` holds.

    None means every row (no length-broadcast literal: see checks/__init__.py).
    """
    amount = pl.col(AMOUNT)
    if mask is None:
        return [
            amount.count().alias(f"{name}_rows"),
            amount.sum().alias(f"{name}_amount"),
        ]
    return [
        (mask & amount.is_not_null()).sum().alias(f"{name}_rows"),
        amount.filter(mask).sum().alias(f"{name}_amount"),
    ]


def waterfall_exprs() -> list[pl.Expr]:
    """Aggregations for ``spend_waterfall``, computed in one pass.

    Every row with an amount is in exactly one of the four groups, taken in this
    order: currency-suspect (any kind), non-rider-spend, superseded rider spend,
    and the rest (the clean USD rider spend).
    """
    suspect = pl.col("currency_suspect")
    spend = pl.col("is_rider_spend")
    superseded = pl.col("is_superseded")
    return [
        *_step(None, "naive"),
        *_step(suspect, "suspect"),
        *_step(~suspect & ~spend, "non_spend"),
        *_step(~suspect & spend & superseded, "superseded"),
        *_step(~suspect & spend & ~superseded, "clean"),
        *_step(suspect & spend & ~superseded, "suspect_spend"),
    ]


def spend_waterfall(row: Mapping[str, float | int]) -> Waterfall:
    """The waterfall from the one-row result of ``waterfall_exprs``.

    ``naive_total`` is the sum of ``order_total_amount`` over every row with an
    amount; the next three are what is taken off it, and
    ``clean_usd_rider_spend = naive_total - currency_suspect - non_rider_spend -
    superseded``. ``suspect_rider_spend`` (not a step) is the part of the suspect
    rows that would otherwise be rider spend. Amounts are rounded to cents.
    """

    def cell(name: str) -> dict[str, float | int]:
        return {
            "rows": int(row[f"{name}_rows"]),
            "amount": round(float(row[f"{name}_amount"] or 0.0), 2),
        }

    return {
        "naive_total": cell("naive"),
        "currency_suspect": cell("suspect"),
        "non_rider_spend": cell("non_spend"),
        "superseded": cell("superseded"),
        "clean_usd_rider_spend": cell("clean"),
        "suspect_rider_spend": cell("suspect_spend"),
    }
