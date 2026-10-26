"""Uniqueness checks: keys that must not repeat, and how the dedupe resolves them.

U1 is a row rule. The rest are dataset checks on the frame that carries the derived
columns (receipt_kind, is_rider_spend) and the dedupe flag ``is_superseded``; the
CLI adds both once. A frame without the flag gets it here.

The legacy whole-row duplicate check could never fire (``checksum`` is unique per
row and the ingestion metadata differs between versions of a receipt), so these
checks work on the order key instead. A repeated key is expected vendor behaviour
(an updated receipt is upserted, the same receipt can reach two mailboxes), so
U2 and U3 are warnings: the dedupe (``dedupe.py``) flags the superseded rows and
U2b, an error, guards that it did.
"""

from collections.abc import Mapping

lazy import polars as pl

from rideshare_data_pipeline.checks.model import (
    KEY_COLS,
    SAMPLE_LIMIT,
    CheckResult,
    DatasetCheck,
    Frame,
    RowRule,
    Severity,
)
from rideshare_data_pipeline.dedupe import (
    KEY,
    has_key,
    in_repeated_key,
    supersede_flags,
)
from rideshare_data_pipeline.parse import KINDS

CURRENCY_NOTE = "mixed currencies (the file has no currency column); USD only from #15"

_SAMPLE_COLS = (*KEY_COLS, "receipt_kind", "is_superseded")
_NEAR_KEY = (
    "user_id",
    "merchant_name",
    "order_date",
    "order_total_amount",
    "product_description",
)


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


def _flagged(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The frame with ``is_superseded``, adding it if the caller hasn't."""
    if "is_superseded" in lf.collect_schema().names():
        return lf
    return supersede_flags(lf)


def _result(
    name: str,
    check_id: str,
    severity: Severity,
    predicate: str,
    n_failed: int,
    n_evaluated: int,
    sample: Frame | None,
    details: Mapping[str, object],
) -> CheckResult:
    """A result that passes only with no failures (threshold 0)."""
    metric = n_failed / n_evaluated if n_evaluated else None
    return CheckResult(
        name=name,
        check_id=check_id,
        category="uniqueness",
        severity=severity,
        passed=n_failed == 0,
        metric=metric,
        threshold=0.0,
        n_failed=n_failed,
        n_evaluated=n_evaluated,
        predicate=predicate,
        sample=sample if n_failed else None,
        details=details,
    )


def _repeated(lf: pl.LazyFrame) -> tuple[Frame, int]:
    """The rows whose key repeats (columns the checks need) and the keyed-row count."""
    rows, keyed = pl.collect_all(
        [
            lf.filter(in_repeated_key()).select(
                *KEY,
                "user_id",
                "email_time",
                "order_total_amount",
                "checksum",
                pl.col("receipt_kind").cast(pl.String),
                "is_rider_spend",
                "is_superseded",
            ),
            lf.select(has_key().sum()),
        ],
        engine="streaming",
    )
    return rows, keyed.item()


def _pair_label(kinds: list[str]) -> str:
    return "+".join(sorted(kinds, key=KINDS.index))


def duplicate_receipts_check(lf: pl.LazyFrame) -> CheckResult:
    """U2 (warn): ``(merchant_name, order_number)`` repeats among keyed rows.

    ``n_failed`` is every row of a repeated key (the nulls are dropped first: a null
    order number is no key); ``n_evaluated`` the rows with a key. ``details``:
    ``groups``; ``excess_rows`` (the superseded rows, one fewer than each group);
    ``excess_spend`` (``order_total_amount`` summed over superseded rider-spend rows;
    see ``currency``); ``kind_pairs``, a list of ``{pair, groups, excess_spend}``
    by the kinds in a group (``ride+tip_update``), biggest first.
    """
    rows, keyed = _repeated(_flagged(lf))
    superseded = pl.col("is_superseded") & pl.col("is_rider_spend")
    groups = (
        rows.group_by(KEY)
        .agg(
            pl.col("receipt_kind"),
            pl.col("order_total_amount").filter(superseded).sum().alias("spend"),
        )
        .with_columns(pl.col("receipt_kind").map_elements(_pair_label, pl.String))
    )
    pairs = (
        groups.group_by("receipt_kind")
        .agg(pl.len().alias("groups"), pl.col("spend").sum())
        .sort(["groups", "receipt_kind"], descending=[True, False])
    )
    sample = rows.sort([*KEY, "email_time"]).head(SAMPLE_LIMIT).select(_SAMPLE_COLS)
    return _result(
        "duplicate_receipts",
        "U2",
        Severity.WARN,
        "(merchant_name, order_number) is unique among receipts with an order number",
        rows.height,
        keyed,
        sample,
        {
            "groups": groups.height,
            "excess_rows": int(rows["is_superseded"].sum()),
            "excess_spend": round(
                float(rows.filter(superseded)["order_total_amount"].sum()), 2
            ),
            "currency": CURRENCY_NOTE,
            "kind_pairs": [
                {"pair": p, "groups": n, "excess_spend": round(float(s), 2)}
                for p, n, s in pairs.iter_rows()
            ],
        },
    )


def dedupe_key_unique_check(lf: pl.LazyFrame) -> CheckResult:
    """U2b (error): after dedupe, the key is unique among the kept rows.

    It holds by construction (see ``dedupe.supersede_flags``) and guards against a
    bug in it. ``n_failed`` is the kept rows that share a key.
    """
    lf = _flagged(lf)
    kept = lf.filter(~pl.col("is_superseded") & has_key())
    failed, evaluated = (
        kept.select(pl.struct(KEY).is_duplicated().sum(), pl.len())
        .collect(engine="streaming")
        .row(0)
    )
    sample = None
    if failed:
        dup = kept.filter(pl.struct(KEY).is_duplicated())
        sample = dup.select(_SAMPLE_COLS).head(SAMPLE_LIMIT).collect()
    return _result(
        "dedupe_key_unique",
        "U2b",
        Severity.ERROR,
        "(merchant_name, order_number) is unique among receipts not superseded",
        failed,
        evaluated,
        sample,
        {},
    )


def cross_panelist_check(lf: pl.LazyFrame) -> CheckResult:
    """U3 (warn): one key under more than one ``user_id`` (a receipt in two mailboxes).

    ``n_failed`` is the rows of those keys, ``n_evaluated`` the rows with a key;
    ``details["groups"]`` the keys.
    """
    rows, keyed = _repeated(_flagged(lf))
    shared = (
        rows.group_by(KEY)
        .agg(pl.col("user_id").drop_nulls().n_unique().alias("users"))
        .filter(pl.col("users") > 1)
    )
    failing = rows.join(shared, on=KEY, how="semi")
    sample = failing.sort([*KEY, "email_time"]).head(SAMPLE_LIMIT).select(_SAMPLE_COLS)
    return _result(
        "cross_panelist_duplicates",
        "U3",
        Severity.WARN,
        "an order key appears under one user_id only",
        failing.height,
        keyed,
        sample,
        {"groups": shared.height},
    )


def near_duplicate_rides_check(lf: pl.LazyFrame) -> CheckResult:
    """U4 (warn): ride rows alike but for the order number (a re-sent receipt?).

    Among rides, rows are grouped by user, merchant, order date, total and
    description. A group counts when it has more than one row and either more than
    one distinct order number or a null one (the same order number twice is U2's).
    ``n_failed`` is the excess rows (group size - 1, summed); ``n_evaluated`` the
    rides. ``details``: ``groups`` and ``rows`` (all rows in those groups). Only
    flagged: the dedupe never acts on it.
    """
    rides = lf.filter(pl.col("receipt_kind").cast(pl.String) == "ride")
    near = rides.with_columns(pl.col("order_time").dt.date().alias("order_date"))
    key = pl.struct(_NEAR_KEY)
    candidates, n_rides = pl.collect_all(
        [
            near.filter(
                key.is_duplicated()
                & pl.col("user_id").is_not_null()
                & pl.col("merchant_name").is_not_null()
                & pl.col("order_date").is_not_null()
                & pl.col("order_total_amount").is_not_null()
            ).select(*_NEAR_KEY, *(c for c in KEY_COLS if c not in _NEAR_KEY)),
            rides.select(pl.len()),
        ],
        engine="streaming",
    )
    groups = (
        candidates.group_by(_NEAR_KEY)
        .agg(
            pl.len().alias("n"),
            pl.col("order_number").drop_nulls().n_unique().alias("distinct"),
            pl.col("order_number").is_null().any().alias("has_null"),
        )
        .filter((pl.col("distinct") > 1) | pl.col("has_null"))
    )
    failing = candidates.join(
        groups.select(_NEAR_KEY), on=_NEAR_KEY, how="semi", nulls_equal=True
    )
    sample = failing.sort([*_NEAR_KEY, "email_time"], nulls_last=True).head(
        SAMPLE_LIMIT
    )
    return _result(
        "near_duplicate_rides",
        "U4",
        Severity.WARN,
        (
            "no two rides share user, merchant, order date, total and description "
            "under different order numbers"
        ),
        failing.height - groups.height,
        n_rides.item(),
        sample.select(KEY_COLS),
        {"groups": groups.height, "rows": failing.height},
    )


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The uniqueness checks that need the whole dataset, for the registry."""
    return (
        duplicate_receipts_check,
        dedupe_key_unique_check,
        cross_panelist_check,
        near_duplicate_rides_check,
    )
