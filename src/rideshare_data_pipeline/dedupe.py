"""The dedupe policy: flag the receipts that a later version supersedes.

Rows are never dropped here. ``supersede_flags`` adds the boolean columns
``is_superseded`` and ``is_cross_panelist_copy``; ``deduplicated`` is the view that
spend metrics use (the rows that are not superseded). The curated Parquet with
its flag columns is a later step; the output Parquet of a run is unchanged.

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

Cross-panelist copies
---------------------
``is_cross_panelist_copy`` is true for every row of a non-null key group that holds
more than one distinct non-null ``user_id``: the same receipt in two panelists'
mailboxes. The dedupe keeps one row per key for panel totals (a receipt is one
purchase), so the losing copy is superseded. A per-panelist statistic (checks from
#26 on: a panelist's spend, trip count or tenure) is about what each panelist
received, so it must not start from the deduplicated rows alone: it takes the
non-superseded rows plus the superseded rows that have the flag, and counts a shared
receipt once for each panelist it reached. The flag is not an input column, so the
structural checks ignore it (``parse.input_columns``). U3 counts the rows that have
it.
"""

lazy import polars as pl

KEY: tuple[str, str] = ("merchant_name", "order_number")

PRIORITY_KINDS: tuple[str, ...] = ("tip_update", "fare_correction")
"""Kinds that supersede the other versions of a receipt."""

FLAG_NAMES: tuple[str, ...] = ("is_superseded", "is_cross_panelist_copy")
"""The columns supersede_flags() adds."""

_ROW = "_row"
_WINNER = "_winner"
_CROSS = "_cross"


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
    """Add ``is_superseded`` and ``is_cross_panelist_copy`` to a frame that has the
    derived ``receipt_kind``.

    Only the rows of repeated keys are sorted or grouped, so the cost follows the
    (small) number of duplicates rather than the frame. Row order and all other
    columns are unchanged.
    """
    indexed = frame.with_row_index(_ROW)
    repeated = in_repeated_key()
    shared = indexed.filter(repeated)
    winners = (
        shared.sort(
            winner_order(), descending=True, nulls_last=True, maintain_order=True
        )
        .unique(subset=KEY, keep="first")
        .select(pl.col(_ROW), pl.lit(True).alias(_WINNER))
    )
    # Keys that reached more than one panelist; nulls are not a panelist.
    cross_keys = (
        shared.group_by(KEY)
        .agg(pl.col("user_id").drop_nulls().n_unique().alias("_users"))
        .filter(pl.col("_users") > 1)
        .select(KEY)
    )
    cross = shared.join(cross_keys, on=KEY, how="semi").select(
        pl.col(_ROW), pl.lit(True).alias(_CROSS)
    )
    return (
        indexed.join(winners, on=_ROW, how="left", maintain_order="left")
        .join(cross, on=_ROW, how="left", maintain_order="left")
        .with_columns(
            (repeated & pl.col(_WINNER).is_null()).alias("is_superseded"),
            pl.col(_CROSS).is_not_null().alias("is_cross_panelist_copy"),
        )
        .drop(_ROW, _WINNER, _CROSS)
    )


def deduplicated[F: (pl.DataFrame, pl.LazyFrame)](frame: F) -> F:
    """The rows that are not superseded: the view that spend metrics use."""
    return frame.filter(~pl.col("is_superseded"))
