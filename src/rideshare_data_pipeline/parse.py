"""Derived columns: receipt kind, subject profile and the rider-spend flag.

Everything here is a pure ``pl.Expr`` factory: nothing reads or writes data. The
CLI evaluates ``typed.with_columns(derived_columns())`` once, and the checks run on
that frame.

Receipt kinds
-------------
A row is one receipt, and not every receipt is a rider's spend on a ride: the file
also holds driver payouts, cancellation and failed-payment notices, passes and
credits, bikes and scooters, and a few food, freight and rental receipts, all
tagged with the category ``Travel``. ``receipt_kind()`` names which one a row is,
from the cleaned email subject (the vendor's templates), the description line, the
reseller, the order number and the vendor's ``order_pickup`` code.

**Rider spend.** ``is_rider_spend`` is true for the kinds ride, tip_update,
fare_correction and cancel_noshow. Other kinds stay in the data with their flag
but are left out of spend: they are payouts to a driver, notices without a
charge, products that aren't a trip, or another service's receipts. A
cancellation fee counts as rider spend under its own kind. A tip update or a fare
correction supersedes the receipt of the ride it amends; removing the superseded
receipt is the dedupe's job, not this module's. Currency gating (the file has no
currency column) narrows this to USD spend separately.

Precedence
----------
The rules form one ordered chain of ``pl.when`` expressions, most specific
first: the first rule that matches names the kind (see ``_rules``). A row that
matches no rule defaults to ``ride`` and is flagged by ``subject_unclassified``,
which the ``unclassified_subject_templates`` check counts, so a new vendor template
shows up instead of silently counting as a ride.
"""

lazy import polars as pl

type Kind = str
"""One of KINDS."""

# Report order: rider-spend kinds first, then the rest.
KINDS: tuple[Kind, ...] = (
    "ride",
    "tip_update",
    "fare_correction",
    "cancel_noshow",
    "payment_failed",
    "driver_earnings",
    "pass_or_credit",
    "micromobility",
    "food_order",
    "freight",
    "rental",
)

RIDER_SPEND_KINDS: frozenset[Kind] = frozenset(
    {"ride", "tip_update", "fare_correction", "cancel_noshow"}
)

# Kinds that are not a rideshare trip, so the category "Travel" is wrong for them
# (check V2). A pass, a bike or a cancellation is still travel.
MISMATCHED_CATEGORY_KINDS: frozenset[Kind] = frozenset(
    {"food_order", "freight", "rental", "driver_earnings"}
)

# The vendor's code for a driver-earnings receipt; no other kind carries it.
ORDER_PICKUP_EARNINGS = 4

UNKNOWN_TEMPLATE_LIMIT = 120
"""Characters of a normalised subject template kept in a report."""

# Mail-gateway prefixes and bracketed tags that precede the vendor's own subject:
# "(EXTERNAL SENDER)", "{Disarmed}", "[Business]", "[Personal - csr]", "[Card X] -".
_PREFIX = r"^(?:\s*(?:\(EXTERNAL SENDER\)|\{Disarmed\}|\[[^\]]*\](?:\s*-)?))+\s*"
# The profile tag: Business or Personal, optionally with a suffix after " - ". It
# must be the first tag, so "[New Business Card]" is not a profile.
_PROFILE = (
    r"(?i)^(?:\s*(?:\(EXTERNAL SENDER\)|\{Disarmed\}))*\s*"
    r"\[(business|personal)(?:\s*-[^\]]*)?\]"
)

_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_WEEKDAYS = (
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo"
)
_DAYPARTS = r"morning|afternoon|evening|night|ma[nñ]ana|tarde|noche|madrugada"


def clean_subject() -> pl.Expr:
    """The email subject without gateway prefixes and leading ``[...]`` tags."""
    return pl.col("email_subject").str.replace(_PREFIX, "")


def subject_profile() -> pl.Expr:
    """``business``, ``personal`` or null, from a leading profile tag."""
    return (
        pl.col("email_subject")
        .str.extract(_PROFILE, 1)
        .str.to_lowercase()
        .alias("subject_profile")
    )


def subject_template() -> pl.Expr:
    """The cleaned subject with variable parts replaced by placeholders.

    Masks (runs of ``*``), digits, month names, weekdays and day parts become
    ``<mask>``, ``<n>``, ``<month>``, ``<weekday>`` and ``<daypart>``, so subjects
    of one template collapse to one string. A null subject is ``<no subject>``.
    """
    return (
        clean_subject()
        .str.replace_all(r"\*+", "<mask>")
        .str.replace_all(r"\d+", "<n>")
        .str.replace_all(rf"(?i)\b(?:{_MONTHS})\b", "<month>")
        .str.replace_all(rf"(?i)\b(?:{_WEEKDAYS})\b", "<weekday>")
        .str.replace_all(rf"(?i)\b(?:{_DAYPARTS})\b", "<daypart>")
        .str.slice(0, UNKNOWN_TEMPLATE_LIMIT)
        .fill_null("<no subject>")
        .alias("subject_template")
    )


def _rules() -> tuple[tuple[Kind, pl.Expr], ...]:
    """(kind, matches) in precedence order, most specific first.

    Each expression may be null for a row (a null subject); a null counts as no
    match. The order matters where one receipt satisfies several rules:

    1. freight: a ``UF-`` order number is a freight shipment whatever else it says.
    2. driver_earnings: ``order_pickup`` 4, the description ``Earnings`` or a
       "getting paid" subject. Before food and rides, so a payout is never a trip.
    3. food_order: a restaurant in ``product_reseller`` (null on every other kind),
       or an "order receipt" subject.
    4. rental: a "rental" subject.
    5. tip_update: "updated your ... trip receipt", "Tip Increase" or the
       description ``Increase Tip``. Before cancel and ride: a tip update may
       mention the ride, and it is the final version of that ride.
    6. fare_correction: "corrected the fare".
    7. cancel_noshow: cancel, no-show or missed ride in the subject, or a
       "cancel" description (``Ride cancelled``).
    8. payment_failed: "charge failed" or "payment failed".
    9. pass_or_credit: a pass or subscription (``product_subtitle`` is filled, or
       the subject names a pass, subscription, credits, gift card or savings plan
       purchase).
    10. micromobility: a bike, e-bike, scooter or JUMP subject. After cancel and
        failed payment, so "cancelled bike trip" stays a cancellation.
    11. ride: a known ride template (a trip subject, or a distance in the
        description). Nothing special; it exists so that ``subject_unclassified``
        can tell a known ride from an unknown template that defaulted to ride.
    """
    s = clean_subject()
    d = pl.col("product_description").fill_null("")
    return (
        ("freight", pl.col("order_number").str.starts_with("UF-")),
        (
            "driver_earnings",
            (pl.col("order_pickup") == ORDER_PICKUP_EARNINGS)
            | d.eq("Earnings")
            | s.str.contains(r"(?i)got paid|getting paid"),
        ),
        (
            "food_order",
            pl.col("product_reseller").is_not_null()
            | s.str.contains(r"(?i)order receipt"),
        ),
        ("rental", s.str.contains(r"(?i)\brental\b")),
        (
            "tip_update",
            s.str.contains(r"(?i)\bupdated\b.*\btrip receipt")
            | s.str.contains(r"(?i)tip increase|increase receipt")
            | d.eq("Increase Tip"),
        ),
        ("fare_correction", s.str.contains(r"(?i)corrected the fare")),
        (
            "cancel_noshow",
            s.str.contains(r"(?i)cancel|no-show|missed ride")
            | d.str.contains(r"(?i)cancel"),
        ),
        (
            "payment_failed",
            s.str.contains(r"(?i)charge failed|payment failed|error de pago"),
        ),
        (
            "pass_or_credit",
            pl.col("product_subtitle").is_not_null()
            | s.str.contains(
                r"(?i)\bpass(?:es)?\b|all-access|uber cash|gift card|savings plan"
                r"|subscription|lyft pink|purchase confirmed|credits purchase"
                r"|ride package|thanks for buying"
            ),
        ),
        ("micromobility", s.str.contains(r"(?i)\b(?:jump|bike|scooter)s?\b")),
        (
            "ride",
            s.str.contains(
                r"(?i)\btrip with\b|^your ride with\b|\btook an uber trip\b"
                r"|^tu \S+ del?\b|^sua \S+ de .* com a uber$"
                r"|^copy of your .* receipt$"
                r"|\bminutes riding with\b"
            )
            | d.str.contains(r"(?i)\d\s*(?:mi|miles|km)\b|\bfare \("),
        ),
    )


def _first_match() -> pl.Expr:
    """The kind of the first matching rule, or null where no rule matches.

    Each ``when`` is null where its rule doesn't match, so coalescing them in rule
    order is the ordered when/then/when chain: the first match wins. The three
    columns below are built on this one expression, so polars evaluates it once.
    """
    return pl.coalesce(
        pl.when(matches).then(pl.lit(kind)) for kind, matches in _rules()
    )


def _kind() -> pl.Expr:
    """The kind as text: the first match, else ``ride``."""
    return _first_match().fill_null("ride")


def receipt_kind() -> pl.Expr:
    """The receipt's kind: one of KINDS, as an Enum column ``receipt_kind``.

    The first matching rule of ``_rules`` wins; a row no rule matches is a ride.
    """
    return _kind().cast(pl.Enum(KINDS)).alias("receipt_kind")


def subject_unclassified() -> pl.Expr:
    """True where no kind rule matched, so ``receipt_kind`` defaulted to ``ride``."""
    return _first_match().is_null().alias("subject_unclassified")


def is_rider_spend() -> pl.Expr:
    """True for the kinds that are rider spend (RIDER_SPEND_KINDS), else False."""
    return _kind().is_in(sorted(RIDER_SPEND_KINDS)).alias("is_rider_spend")


DERIVED_NAMES: tuple[str, ...] = (
    "receipt_kind",
    "subject_profile",
    "subject_unclassified",
    "is_rider_spend",
)
"""The names of the columns derived_columns() adds."""


def input_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The frame without the derived columns (a no-op if it has none).

    Structural checks describe the delivered file, so they look at its columns only.
    """
    return lf.drop(DERIVED_NAMES, strict=False)


def derived_columns() -> list[pl.Expr]:
    """The columns added to the typed frame, for one ``with_columns`` call."""
    return [receipt_kind(), subject_profile(), subject_unclassified(), is_rider_spend()]


_MATCHED = "_matched_kind"


def with_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Add the derived columns to `df`: the same result as ``with_columns`` of
    ``derived_columns()``, in about a third of the time.

    ``derived_columns()`` builds each column from the rule chain on its own, so the
    subject regexes run once per column. Here the chain is evaluated once into a
    temporary column and the kind, the unclassified flag and the rider-spend flag
    are read off it.
    """
    matched = pl.col(_MATCHED)
    return (
        df.with_columns(_first_match().alias(_MATCHED), subject_profile())
        .with_columns(
            matched.fill_null("ride").cast(pl.Enum(KINDS)).alias("receipt_kind"),
            matched.is_null().alias("subject_unclassified"),
            matched.fill_null("ride")
            .is_in(sorted(RIDER_SPEND_KINDS))
            .alias("is_rider_spend"),
        )
        .drop(_MATCHED)
        .select(*df.columns, *DERIVED_NAMES)
    )
