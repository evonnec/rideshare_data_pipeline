"""Currency and sender checks: CUR1 (currency risk), V1 and V3 (the sender).

They run on the frame that carries the derived columns ``sender_country``,
``currency_reason`` and ``currency_suspect`` (see ``parse.with_derived``) and the dedupe
flag ``is_superseded`` (added here if missing, for the spend waterfall).

The gate itself is not a failure: the real file has non-US senders by design, so
CUR1 is a warning that reports the size of the problem, and the exclusion is
``money.usd_rides``, which every money statistic starts from. The sample columns of
V1 and V3 never include ``from_domain``: its local part can be a person's address.
"""

lazy import polars as pl

from rideshare_data_pipeline.checks.model import (
    KEY_COLS,
    CheckResult,
    DatasetCheck,
    FlaggedCheck,
    Lazy,
    RowRule,
    Severity,
)
from rideshare_data_pipeline.dedupe import supersede_flags
from rideshare_data_pipeline.money import (
    spend_waterfall,
    waterfall_exprs,
)
from rideshare_data_pipeline.parse import CURRENCY_REASONS
from rideshare_data_pipeline.sender import (
    is_lyft_sender,
    is_uber_sender,
    known_sender,
)

NO_COUNTRY = "(none)"
LOCAL_CURRENCY_NOTE = "local currency, not summable as USD"
FOREIGN_SUBJECT_NOTE = (
    "Spanish or Portuguese subject from a US sender: tagged, not currency-suspect"
)

_SAMPLE_COLS = (*KEY_COLS, "receipt_kind", "sender_country")


def rules() -> tuple[RowRule, ...]:
    """V1 and V3, the sender rules."""
    uber = pl.col("merchant_name") == "Uber"
    lyft = pl.col("merchant_name") == "Lyft"
    return (
        RowRule(
            name="merchant_sender_consistent",
            check_id="V1",
            category="semantic",
            severity=Severity.ERROR,
            rule=(
                "Uber receipts come from an Uber sender domain, Lyft receipts "
                "from a Lyft one (lyft.com, lyftmail.com and subdomains)"
            ),
            predicate=(uber & ~is_uber_sender()) | (lyft & ~is_lyft_sender()),
            applies_to=(uber | lyft) & pl.col("from_domain").is_not_null(),
            sample_cols=_SAMPLE_COLS,
        ),
        RowRule(
            name="unknown_sender",
            check_id="V3",
            category="semantic",
            severity=Severity.WARN,
            rule=(
                "the sender matches a known receipt-sender pattern "
                "(a country or city sender, or a generic merchant sender)"
            ),
            predicate=~known_sender(pl.col("sender_country")),
            applies_to=pl.col("from_domain").is_not_null(),
            sample_cols=_SAMPLE_COLS,
        ),
    )


def currency_suspect_check(lf: pl.LazyFrame) -> CheckResult:
    """CUR1 (warn): receipts whose amount may not be in USD.

    ``n_failed`` is the rows with ``currency_suspect``; ``n_evaluated`` every row.
    It is a warning, not an error: the gate is the ``usd_rides`` exclusion, and an
    error would fail every real run for an expected property of the data.

    ``details``: ``by_reason`` and ``by_country`` (``{key: {rows, amount}}``, biggest
    first; ``amount`` is the naive sum of ``order_total_amount``, ``note`` says it is
    in local currency and not summable as USD; a sender with no country is
    ``(none)``), ``suspect_sum`` (rows and amount over every suspect row),
    ``suspect_rider_spend`` (the suspect rows that would be rider spend) and
    ``spend``, the waterfall from the naive total to the clean USD rider spend (see
    ``money.spend_waterfall``; ``micromobility`` is a sub-line of its non-rider
    step) and ``foreign_subject`` (rows with a Spanish or Portuguese subject that
    are NOT currency-suspect, because the sender is a US one: an info tag, not a
    reason).
    """
    if "is_superseded" not in lf.collect_schema().names():
        lf = supersede_flags(lf)
    suspect = lf.filter(pl.col("currency_suspect"))
    amount = pl.col("order_total_amount")
    names = lf.collect_schema().names()
    tagged = (
        lf.filter(pl.col("foreign_subject") & ~pl.col("currency_suspect"))
        if "foreign_subject" in names
        else lf.clear()
    )
    by_reason, by_country, totals, n_rows, tag = pl.collect_all(
        [
            suspect.group_by(pl.col("currency_reason").cast(pl.String)).agg(
                pl.len().alias("rows"), amount.sum().alias("amount")
            ),
            suspect.group_by(pl.col("sender_country").fill_null(NO_COUNTRY)).agg(
                pl.len().alias("rows"), amount.sum().alias("amount")
            ),
            lf.select(waterfall_exprs()),
            lf.select(pl.len()),
            tagged.select(pl.len().alias("rows"), amount.sum().alias("amount")),
        ],
        engine="streaming",
    )
    spend = spend_waterfall(totals.row(0, named=True))

    def table(frame: pl.DataFrame, key: str) -> dict[str, dict[str, float | int]]:
        ordered = frame.sort(["rows", key], descending=[True, False])
        return {
            k: {"rows": n, "amount": round(float(a or 0.0), 2)}
            for k, n, a in ordered.iter_rows()
        }

    reasons = table(by_reason, "currency_reason")
    n_suspect = int(by_reason["rows"].sum())
    total = n_rows.item()
    return CheckResult(
        name="currency_suspect_rows",
        check_id="CUR1",
        category="semantic",
        severity=Severity.WARN,
        passed=n_suspect == 0,
        metric=n_suspect / total if total else None,
        threshold=0.0,
        n_failed=n_suspect,
        n_evaluated=total,
        predicate=(
            "no receipt is currency-suspect (a non-US sender, a km distance or a "
            "Spanish or Portuguese subject from a sender that is not a US one)"
        ),
        sample=None,
        details={
            "reasons": list(CURRENCY_REASONS),
            "by_reason": reasons,
            "by_country": table(by_country, "sender_country"),
            "suspect_sum": {**spend["currency_suspect"], "note": LOCAL_CURRENCY_NOTE},
            "suspect_rider_spend": spend["suspect_rider_spend"],
            "spend": {k: v for k, v in spend.items() if k != "suspect_rider_spend"},
            "foreign_subject": {
                "rows": tag["rows"].item(),
                "amount": round(float(tag["amount"].item() or 0.0), 2),
                "note": FOREIGN_SUBJECT_NOTE,
            },
        },
    )


def currency_suspect_flag(_lf: Lazy) -> pl.Expr:
    """CUR1's failed rows: ``currency_suspect``."""
    return pl.col("currency_suspect")


def dataset_checks() -> tuple[DatasetCheck, ...]:
    """The currency checks that need the whole dataset, for the registry."""
    return (
        FlaggedCheck(
            "currency_suspect_rows", currency_suspect_check, currency_suspect_flag
        ),
    )
