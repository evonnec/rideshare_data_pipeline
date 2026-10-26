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
    POSITION,
    SAMPLE_LIMIT,
    CheckResult,
    DatasetCheck,
    FlaggedCheck,
    Frame,
    RowRule,
    Severity,
    rows_at,
)
from rideshare_data_pipeline.dedupe import (
    KEY,
    has_key,
    in_repeated_key,
    supersede_flags,
)
from rideshare_data_pipeline.parse import KINDS

CURRENCY_NOTE = (
    "USD rider spend only: currency-suspect rows are in excess_spend_suspect "
    "(local currency, not summable as USD)"
)

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
                "currency_suspect",
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
    ``excess_spend`` (``order_total_amount`` summed over superseded rows that would
    otherwise be USD rider spend: rider-spend kinds, not currency-suspect);
    ``excess_spend_suspect`` (the same over currency-suspect rows, in local
    currency, not summable as USD; see ``currency``); ``kind_pairs``, a list of
    ``{pair, groups, excess_spend}`` (USD) by the kinds in a group
    (``ride+tip_update``), biggest first.
    """
    rows, keyed = _repeated(_flagged(lf))
    superseded = pl.col("is_superseded") & pl.col("is_rider_spend")
    usd = superseded & ~pl.col("currency_suspect")
    groups = (
        rows.group_by(KEY)
        .agg(
            pl.col("receipt_kind"),
            pl.col("order_total_amount").filter(usd).sum().alias("spend"),
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
                float(rows.filter(usd)["order_total_amount"].sum()), 2
            ),
            "excess_spend_suspect": round(
                float(
                    rows.filter(superseded & pl.col("currency_suspect"))[
                        "order_total_amount"
                    ].sum()
                ),
                2,
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


def _rides(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("receipt_kind").cast(pl.String) == "ride")


def _near_candidates(rides: pl.LazyFrame) -> pl.LazyFrame:
    """Rides whose (user, merchant, order date, total, description) repeats."""
    near = rides.with_columns(pl.col("order_time").dt.date().alias("order_date"))
    return near.filter(
        pl.struct(_NEAR_KEY).is_duplicated()
        & pl.col("user_id").is_not_null()
        & pl.col("merchant_name").is_not_null()
        & pl.col("order_date").is_not_null()
        & pl.col("order_total_amount").is_not_null()
    )


def _near_groups(candidates: Frame) -> Frame:
    """The candidate groups that count: other order numbers, or a null one."""
    return (
        candidates.group_by(_NEAR_KEY)
        .agg(
            pl.len().alias("n"),
            pl.col("order_number").drop_nulls().n_unique().alias("distinct"),
            pl.col("order_number").is_null().any().alias("has_null"),
        )
        .filter((pl.col("distinct") > 1) | pl.col("has_null"))
    )


def duplicate_receipts_flag(_lf: pl.LazyFrame) -> pl.Expr:
    """Every row of a repeated (merchant_name, order_number) key, as U2 counts."""
    return in_repeated_key()


def cross_panelist_flag(_lf: pl.LazyFrame) -> pl.Expr:
    """Rows of a key shared by more than one user_id (``is_cross_panelist_copy``)."""
    return pl.col("is_cross_panelist_copy")


def _near_scan(lf: pl.LazyFrame) -> tuple[CheckResult, Frame]:
    """U4's result, and the positions (``POSITION``) of the rows it flags.

    The flagged rows are the excess rows: the first row of a near-duplicate group,
    in file order, is not flagged and every later row is, so they number U4's
    ``n_failed`` (group size - 1, summed). A reader filtering on the flag keeps one
    ride of each group.

    U4 (warn): ride rows alike but for the order number (a re-sent receipt?).

    Among rides, rows are grouped by user, merchant, order date, total and
    description. A group counts when it has more than one row and either more than
    one distinct order number or a null one (the same order number twice is U2's).
    ``n_failed`` is the excess rows (group size - 1, summed); ``n_evaluated`` the
    rides. ``details``: ``groups`` and ``rows`` (all rows in those groups). Only
    flagged: the dedupe never acts on it.
    """
    rides = _rides(lf.with_row_index(POSITION))
    candidates, n_rides = pl.collect_all(
        [
            _near_candidates(rides).select(
                POSITION, *_NEAR_KEY, *(c for c in KEY_COLS if c not in _NEAR_KEY)
            ),
            rides.select(pl.len()),
        ],
        engine="streaming",
    )
    groups = _near_groups(candidates)
    failing = candidates.join(
        groups.select(_NEAR_KEY), on=_NEAR_KEY, how="semi", nulls_equal=True
    )
    sample = failing.sort([*_NEAR_KEY, "email_time"], nulls_last=True).head(
        SAMPLE_LIMIT
    )
    later = (
        failing.select(POSITION, *_NEAR_KEY)
        .sort(POSITION)
        .filter(pl.int_range(pl.len()).over(_NEAR_KEY) > 0)
        .select(POSITION)
    )
    result = _result(
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
    return result, later


def near_duplicate_rides_check(lf: pl.LazyFrame) -> CheckResult:
    """U4 (warn); see ``_near_scan`` for the rule and ``near_duplicate_rides``."""
    return _near_scan(lf)[0]


def near_duplicate_rides() -> FlaggedCheck:
    """U4 with its row flag, which reuses the check's scan (one pass for both).

    Call the flag with the frame the check ran on; the pair remembers the last
    frame it was given, by identity (as ``outliers.outlier_checks`` does).
    """
    last: list[tuple[pl.LazyFrame, CheckResult, Frame]] = []

    def scan(lf: pl.LazyFrame) -> tuple[CheckResult, Frame]:
        if not (last and last[0][0] is lf):
            last[:] = [(lf, *_near_scan(lf))]
        return last[0][1], last[0][2]

    return FlaggedCheck(
        "near_duplicate_rides",
        lambda lf: scan(lf)[0],
        lambda lf: rows_at(scan(lf)[1][POSITION]),
    )


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The uniqueness checks that need the whole dataset, for the registry."""
    return (
        FlaggedCheck(
            "duplicate_receipts", duplicate_receipts_check, duplicate_receipts_flag
        ),
        dedupe_key_unique_check,
        FlaggedCheck(
            "cross_panelist_duplicates", cross_panelist_check, cross_panelist_flag
        ),
        near_duplicate_rides(),
    )
