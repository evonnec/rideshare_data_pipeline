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
currency column) narrows this to USD spend separately: see ``money.usd_rides``.

Currency risk
-------------
The file has no currency column, and receipts from non-US merchant senders state
amounts in local currency. ``currency_reason`` says why a row is suspect, the first
of: ``non_us_sender`` (``sender_country`` is neither us nor null; see
``sender.py``), ``km_unit`` (the description states kilometres, which a US sender's
receipt does not) and ``foreign_language`` (a Spanish or Portuguese subject, from a
sender that is not a US one; see ``_FOREIGN_SUBJECT``). ``currency_suspect`` is true
where there is a reason. Rows are flagged, never dropped, and no exchange rate is
applied.

A Spanish or Portuguese subject from a US sender is **not** a currency reason: in
the full sample all 2,552 such rows come from US senders, state miles where they
state a distance, and cost a median $2.38/mile, in line with USD. The language
stays findable as the info flag ``foreign_subject`` (true for any such subject,
suspect or not), which gates nothing.

Precedence
----------
The rules form one ordered chain of ``pl.when`` expressions, most specific
first: the first rule that matches names the kind (see ``_rules``). A row that
matches no rule defaults to ``ride`` and is flagged by ``subject_unclassified``,
which the ``unclassified_subject_templates`` check counts, so a new vendor template
shows up instead of silently counting as a ride.

Trip metrics
------------
``product_description`` carries the product, distance and duration of a trip in a
few templates (see ``_raw``). ``distance_mi``,
``distance_unit``, ``duration_s`` and ``product_family`` read them; every one is
null on a line that is not a trip (``Increase Tip``, ``Earnings``, a null), and a
trip that states 0 miles or 0 seconds is 0, never null. ``product_family`` has a
short list of values (``FAMILIES``) so that products can be compared across vendors.
"""

from collections.abc import Mapping

lazy import polars as pl

from rideshare_data_pipeline.dedupe import FLAG_NAMES
from rideshare_data_pipeline.sender import NON_US_EXEMPT, US, sender_country

type Kind = str
"""One of KINDS."""

type Family = str
"""One of FAMILIES."""

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

# Product families, from the product label of a trip line. The keywords below are
# tried in order on the description and the first family that matches wins, so "Express
# Pool" is express_pool (not shared) and "Black VIP" is vip (not black_lux).
# ``lyft_standard`` is the plain Lyft ride (label "Lyft"); ``other`` is a trip line
# whose product is none of these (or has no label, like a Lyft "Base fare" line).
FAMILIES: tuple[Family, ...] = (
    "uberx",
    "lyft_standard",
    "shared",
    "express_pool",
    "xl",
    "black_lux",
    "comfort",
    "vip",
    "other",
)

# Case-insensitive substrings per family, in matching order (Aho-Corasick, much
# cheaper than a regex per family on 1.9M rows).
_FAMILY_KEYWORDS: tuple[tuple[Family, tuple[str, ...]], ...] = (
    ("express_pool", ("express",)),
    ("shared", ("pool", "share", "line")),
    ("vip", ("vip",)),
    ("black_lux", ("black", "lux", "suv", "premier")),
    ("xl", ("xl",)),
    ("comfort", ("select", "plus", "comfort")),
    ("uberx", ("uberx",)),
    ("lyft_standard", ("lyft",)),
)

DISTANCE_UNITS: tuple[str, ...] = ("mi", "km")

MI_PER_KM = 0.621371
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


# Spanish and Portuguese receipt subjects, on the cleaned subject: "Tu viaje del
# lunes por la mañana ...", "Tu <product> del ...", "Sua viagem de segunda-feira
# ...", "Error de pago: Tu ...". Any of these phrases marks the row.
_FOREIGN_SUBJECT = (
    r"(?i)^(?:error de pago:?\s*)?tu\b|^sua\b"
    r"|\bpor la (?:ma[nñ]ana|tarde|noche|madrugada)\b|\bviaje\b|\bviagem\b"
)

CURRENCY_REASONS: tuple[str, ...] = ("non_us_sender", "km_unit", "foreign_language")
"""Why a row is currency-suspect, in precedence order (the first that holds)."""

# The description is read with a few simple regexes and keyword scans rather than
# one regex per template: a regex that captures several groups is several times
# slower on 1.9M rows than the same work in single-group passes.
#
# A distance and its unit: ``1.63mi``, ``9.93 mi``, ``2.00 miles``, ``7.5 km``,
# ``3 kilometers``, with an optional thousands separator. The unit must end at a
# word boundary, so ``32 min`` is not a distance.
# (?-u) makes \d, \s and \b ASCII-only, which is markedly faster than Unicode.
_UNITS = r"(?:miles|mi|kilometers|kilometres|(?u:kilómetros)|km)\b"
_DISTANCE = rf"(?-u)(\d[\d,]*(?:\.\d+)?|\.\d+)\s*{_UNITS}"
_KM = r"(?-u)\d\s*(?:km|kilomet|(?u:kilómetros))"

# A duration is some of: hours (``1h``, ``1 h``), minutes (``12m``, ``32 min``,
# ``min(s)``), seconds (``23s``), or the clock of the older Uber template
# (``00:15:14 Trip time``, two digits a part); a part a template omits is 0.
_HOURS = r"(?-u)(\d+)\s*h\b"
_MINUTES = r"(?-u)(\d+)\s*m(?:ins?)?\b"
_SECONDS = r"(?-u)(\d+)s\b"
_CLOCK = r"(?-u)(\d\d:\d\d:\d\d)\s+Trip time"

type Raw = Mapping[str, pl.Expr]
"""The raw scans of a description: strings and flags, before any arithmetic."""


def _raw() -> dict[str, pl.Expr]:
    """The scans every trip column is built from, each a single pass over the text.

    ``dist``, ``h``, ``m``, ``s`` and ``clock`` are the matched text (null if
    absent), ``km`` says the unit is kilometres, and ``hit<i>`` whether the
    description names family ``i`` of ``_FAMILY_KEYWORDS``.
    """
    d = pl.col("product_description")
    raw = {
        "dist": d.str.extract(_DISTANCE, 1),
        "km": d.str.contains(_KM),
        "h": d.str.extract(_HOURS, 1),
        "m": d.str.extract(_MINUTES, 1),
        "s": d.str.extract(_SECONDS, 1),
        "clock": d.str.extract(_CLOCK, 1),
    }
    for i, (_, words) in enumerate(_FAMILY_KEYWORDS):
        raw[f"hit{i}"] = d.str.contains_any(list(words), ascii_case_insensitive=True)
    return raw


def _stored(prefix: str) -> dict[str, pl.Expr]:
    """``_raw()`` read back from columns named ``<prefix><key>`` (see with_derived)."""
    return {key: pl.col(f"{prefix}{key}") for key in _raw()}


def _miles(raw: Raw) -> pl.Expr:
    """Distance in miles; null if no distance, 0.0 if it states zero. Km convert."""
    miles = raw["dist"].str.replace_all(",", "", literal=True).cast(pl.Float64)
    return pl.when(raw["km"]).then(miles * MI_PER_KM).otherwise(miles)


def _unit(raw: Raw) -> pl.Expr:
    """``km`` or ``mi`` where a distance was found, else null."""
    return (
        pl.when(raw["dist"].is_not_null())
        .then(pl.when(raw["km"]).then(pl.lit("km")).otherwise(pl.lit("mi")))
        .cast(pl.Enum(DISTANCE_UNITS))
    )


def _seconds(raw: Raw) -> pl.Expr:
    """Duration in seconds; null if the line has no duration, 0 if it states zero.

    Absent parts of a matched template count as 0 (``45s`` is 45, ``| 1 h`` is 3600).
    """
    clock = raw["clock"]

    def part(text: pl.Expr, offset: int) -> pl.Expr:
        # The clock is always ``HH:MM:SS`` (see _CLOCK), so a part is a slice.
        both = pl.coalesce(text, clock.str.slice(offset, 2))
        return both.cast(pl.Int64, strict=False)

    h, m, s = part(raw["h"], 0), part(raw["m"], 3), part(raw["s"], 6)
    return pl.when(pl.coalesce(h, m, s).is_not_null()).then(
        h.fill_null(0) * 3600 + m.fill_null(0) * 60 + s.fill_null(0)
    )


def _family(raw: Raw) -> pl.Expr:
    """The product family of a trip line (one that states a distance); else null.

    The family keywords are looked up in the whole description (the label is the
    only place they occur on a trip line). The first family of ``_FAMILY_KEYWORDS``
    that was hit wins, and a trip line that hit none, like a Lyft "Base fare" line,
    is ``other``.
    """
    chain = pl.coalesce(
        pl.when(raw[f"hit{i}"]).then(pl.lit(family))
        for i, (family, _) in enumerate(_FAMILY_KEYWORDS)
    )
    return (
        pl.when(raw["dist"].is_not_null())
        .then(chain.fill_null("other"))
        .cast(pl.Enum(FAMILIES))
    )


def distance_mi() -> pl.Expr:
    """The trip distance in miles (Float64), km converted; null if not a trip line."""
    return _miles(_raw()).alias("distance_mi")


def distance_unit() -> pl.Expr:
    """The unit the description states, ``mi`` or ``km`` (Enum); null if none."""
    return _unit(_raw()).alias("distance_unit")


def duration_s() -> pl.Expr:
    """The trip duration in seconds (Int64); null if not a trip line."""
    return _seconds(_raw()).alias("duration_s")


def product_family() -> pl.Expr:
    """The product family (Enum FAMILIES); null if the description isn't a trip."""
    return _family(_raw()).alias("product_family")


def foreign_subject() -> pl.Expr:
    """True where the cleaned subject is Spanish or Portuguese (_FOREIGN_SUBJECT)."""
    return clean_subject().str.contains(_FOREIGN_SUBJECT).fill_null(False)


def _reason(country: pl.Expr, km: pl.Expr, foreign: pl.Expr) -> pl.Expr:
    """The first currency reason that holds (see CURRENCY_REASONS), else null.

    ``foreign_language`` needs a sender that isn't a US one (here: a generic sender
    with no country, since a country other than us is already ``non_us_sender``); a
    km description is ``km_unit`` before it gets there.
    """
    return (
        pl.when(country.is_not_null() & ~country.is_in(NON_US_EXEMPT))
        .then(pl.lit("non_us_sender"))
        .when(km.fill_null(False))
        .then(pl.lit("km_unit"))
        .when(foreign & ~country.eq(US).fill_null(False))
        .then(pl.lit("foreign_language"))
        .cast(pl.Enum(CURRENCY_REASONS))
    )


def currency_reason() -> pl.Expr:
    """Why the row is currency-suspect (Enum CURRENCY_REASONS), null if not."""
    return _reason(sender_country(), distance_unit() == "km", foreign_subject()).alias(
        "currency_reason"
    )


def currency_suspect() -> pl.Expr:
    """True where the amount may not be in USD: ``currency_reason`` is set."""
    return currency_reason().is_not_null().alias("currency_suspect")


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
    "sender_country",
    "currency_reason",
    "currency_suspect",
    "foreign_subject",
    "distance_mi",
    "distance_unit",
    "duration_s",
    "product_family",
)
"""The names of the columns derived_columns() adds."""


def input_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The frame without the derived and flag columns (a no-op if it has none).

    Structural checks describe the delivered file, so they look at its columns only.
    """
    return lf.drop((*DERIVED_NAMES, *FLAG_NAMES), strict=False)


def derived_columns() -> list[pl.Expr]:
    """The columns added to the typed frame, for one ``with_columns`` call."""
    return [
        receipt_kind(),
        subject_profile(),
        subject_unclassified(),
        is_rider_spend(),
        sender_country(),
        currency_reason(),
        currency_suspect(),
        foreign_subject().alias("foreign_subject"),
        distance_mi(),
        distance_unit(),
        duration_s(),
        product_family(),
    ]


_MATCHED = "_matched_kind"
_FOREIGN = "_foreign_subject"
_RAW = "_raw_"


def with_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Add the derived columns to `df`: the same result as ``with_columns`` of
    ``derived_columns()``, in a fraction of the time.

    ``derived_columns()`` builds each column from the rule chain on its own, so the
    subject regexes run once per column. Here the chain is evaluated once into a
    temporary column and the kind, the unclassified flag and the rider-spend flag
    are read off it; likewise every description scan (``_raw``) runs once, as its
    own temporary column so that polars runs them in parallel, and the trip
    columns are built from those.
    """
    matched = pl.col(_MATCHED)
    raw = _stored(_RAW)
    temp = [_MATCHED, _FOREIGN, *(f"{_RAW}{key}" for key in raw)]
    return (
        df.with_columns(
            _first_match().alias(_MATCHED),
            subject_profile(),
            sender_country(),
            foreign_subject().alias(_FOREIGN),
            *(expr.alias(f"{_RAW}{key}") for key, expr in _raw().items()),
        )
        .with_columns(
            matched.fill_null("ride").cast(pl.Enum(KINDS)).alias("receipt_kind"),
            matched.is_null().alias("subject_unclassified"),
            matched.fill_null("ride")
            .is_in(sorted(RIDER_SPEND_KINDS))
            .alias("is_rider_spend"),
            _reason(
                pl.col("sender_country"), _unit(raw) == "km", pl.col(_FOREIGN)
            ).alias("currency_reason"),
            _miles(raw).alias("distance_mi"),
            _unit(raw).alias("distance_unit"),
            _seconds(raw).alias("duration_s"),
            _family(raw).alias("product_family"),
        )
        .with_columns(
            pl.col("currency_reason").is_not_null().alias("currency_suspect"),
            pl.col(_FOREIGN).alias("foreign_subject"),
        )
        .drop(temp)
        .select(*df.columns, *DERIVED_NAMES)
    )
