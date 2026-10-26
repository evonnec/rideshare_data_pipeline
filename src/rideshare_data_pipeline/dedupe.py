"""The dedupe policy: flag the receipts that a later version supersedes.

Rows are never dropped here. ``supersede_flags`` adds the boolean column
``is_superseded``; ``deduplicated`` is the view that spend metrics use (the rows
that are not superseded). The curated Parquet with its flag columns is a later
step; the output Parquet of a run is unchanged.

Why
---
The vendor upserts: a receipt that is updated (a tip added, a fare corrected) or
re-sent arrives again under the same ``(merchant_name, order_number)``, with a new
``email_time`` and ``checksum``. The same receipt can also reach two panelists'
mailboxes. Counting every version would double-count the spend.

Policy
------
* The key is ``(merchant_name, order_number)``, and only rows where both are
  non-null have one. A row without a key is never superseded: recurring pass
  charges have no order number and are identical except for ``email_time``, and
  they are legitimate.
* Within a key group exactly one row wins; every other row is superseded. The
  winner is the first row of this order:

  1. kind priority: a ``tip_update`` or a ``fare_correction`` beats every other
     kind. It is the final version of the ride it amends, even when the original
     ride's email arrived later (kind priority beats time);
  2. the latest ``email_time``;
  3. the latest ``update_time``;
  4. the greatest ``checksum``, so that a full tie is still deterministic.

  A null in 2 to 4 loses to any value.
* Idempotent: every key group of ``deduplicated(...)`` has one row, so flagging it
  again supersedes nothing.
* Rows near-identical but under different order numbers (check U4) are only
  reported, never superseded.
"""

lazy import polars as pl

KEY: tuple[str, str] = ("merchant_name", "order_number")

PRIORITY_KINDS: tuple[str, ...] = ("tip_update", "fare_correction")
"""Kinds that supersede the other versions of a receipt."""

FLAG_NAMES: tuple[str, ...] = ("is_superseded",)
"""The columns supersede_flags() adds."""

_ROW = "_row"
_WINNER = "_winner"


def has_key() -> pl.Expr:
    """True where the row has a ``(merchant_name, order_number)`` key."""
    return pl.col("merchant_name").is_not_null() & pl.col("order_number").is_not_null()


def in_repeated_key() -> pl.Expr:
    """True where the row has a key that another row shares (the key repeats)."""
    return has_key() & pl.struct(KEY).is_duplicated()


def winner_order() -> list[pl.Expr]:
    """The sort keys of a group, best first when sorted descending (see module)."""
    return [
        pl.col("receipt_kind").cast(pl.String).is_in(PRIORITY_KINDS),
        pl.col("email_time"),
        pl.col("update_time"),
        pl.col("checksum"),
    ]


def supersede_flags[F: (pl.DataFrame, pl.LazyFrame)](frame: F) -> F:
    """Add ``is_superseded`` to a frame that has the derived ``receipt_kind``.

    Only the rows of repeated keys are sorted, so the cost follows the (small)
    number of duplicates rather than the frame. Row order and all other columns
    are unchanged.
    """
    indexed = frame.with_row_index(_ROW)
    repeated = in_repeated_key()
    winners = (
        indexed.filter(repeated)
        .sort(winner_order(), descending=True, nulls_last=True, maintain_order=True)
        .unique(subset=KEY, keep="first")
        .select(pl.col(_ROW), pl.lit(True).alias(_WINNER))
    )
    return (
        indexed.join(winners, on=_ROW, how="left", maintain_order="left")
        .with_columns(
            (repeated & pl.col(_WINNER).is_null()).alias("is_superseded"),
        )
        .drop(_ROW, _WINNER)
    )


def deduplicated[F: (pl.DataFrame, pl.LazyFrame)](frame: F) -> F:
    """The rows that are not superseded: the view that spend metrics use."""
    return frame.filter(~pl.col("is_superseded"))
