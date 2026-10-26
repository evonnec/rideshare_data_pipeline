"""Receipt kinds, subject profiles and the rider-spend flag (issue #12)."""

import subprocess
import sys
from dataclasses import dataclass, field

import polars as pl
import pytest

from rideshare_data_pipeline.checks.kinds import (
    TOP_TEMPLATES,
    kind_matrix,
    receipt_kinds_check,
    unclassified_templates_check,
)
from rideshare_data_pipeline.parse import (
    DERIVED_NAMES,
    KINDS,
    RIDER_SPEND_KINDS,
    clean_subject,
    derived_columns,
    input_columns,
    subject_template,
    with_derived,
)


@dataclass(frozen=True, slots=True)
class Case:
    """One receipt: the columns the rules read, and what they should decide."""

    subject: str | None
    kind: str
    profile: str | None = None
    unclassified: bool = False
    description: str | None = "UberX 3.10 mi | 9 min"
    reseller: str | None = None
    subtitle: str | None = None
    order_number: str | None = "0b1d2c3e-0000-4000-8000-000000000001"
    pickup: int = 1
    merchant: str = "Uber"
    extra: dict[str, object] = field(default_factory=dict)


def frame(cases: list[Case]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "merchant_name": [c.merchant for c in cases],
            "email_subject": [c.subject for c in cases],
            "product_description": [c.description for c in cases],
            "product_reseller": [c.reseller for c in cases],
            "product_subtitle": [c.subtitle for c in cases],
            "order_number": [c.order_number for c in cases],
            "order_pickup": pl.Series([c.pickup for c in cases], dtype=pl.Int8),
            "product_category": ["Travel"] * len(cases),
        },
        schema_overrides={
            "email_subject": pl.String,
            "product_description": pl.String,
            "product_reseller": pl.String,
            "product_subtitle": pl.String,
            "order_number": pl.String,
        },
    ).with_columns(derived_columns())


# --- every subject template of the synthetic fixture ---------------------------------

FIXTURE_TEMPLATES = [
    Case(
        "Your ride with Alex on March 3",
        "ride",
        description="Lyft fare (2.00mi, 5m 0s)",
    ),
    Case("Your Monday evening trip with Sam", "ride"),
    Case(
        "Your ride with ***** on May 2", "ride", description="Lyft fare (2.20mi, 6m 5s)"
    ),
    Case("Your Friday evening trip with ******", "ride"),
    Case("Your updated trip receipt", "tip_update"),
    Case(
        "Your monthly pass receipt",
        "pass_or_credit",
        description="Pass membership",
        subtitle="Monthly ride pass",
        pickup=0,
        order_number=None,
    ),
    Case(
        "You're getting paid: your earnings summary",
        "driver_earnings",
        description="Earnings",
        subtitle="Weekly payout",
        pickup=4,
        merchant="Lyft",
    ),
    Case(
        "Your order with Café Example",
        "food_order",
        description="Food order",
        reseller="Café Example",
    ),
    Case("Your bike ride receipt", "micromobility", description="Bike ride | 11 min"),
    Case(
        "Your shipment receipt",
        "freight",
        description="Freight shipment",
        order_number="UF-123456789012",
    ),
    # Spanish: the vendor's template, and a variant with no distance in the description
    # (the subject alone says it is a trip).
    Case(
        "Tu viaje del sábado por la mañana",
        "ride",
        description="UberX 6.80 km | 21 min",
    ),
    Case("Tu viaje del sábado por la mañana", "ride", description=None),
]

# --- other vendor templates, with the rules that name them ---------------------------

VENDOR_TEMPLATES = [
    # rides
    Case("Your Tuesday afternoon trip with Uber", "ride"),
    Case("Tu viaje del lunes por la noche con Uber", "ride", description=None),
    Case("Sua viagem de segunda-feira à noite com a Uber", "ride", description=None),
    Case("***** took an Uber trip on Jul 4", "ride", description=None),
    Case("Copy of your Jun 3 Uber receipt", "ride", description=None),
    Case("Alex's last 12 minutes riding with Uber", "ride", description=None),
    # tip updates (Uber and Lyft), including the "Increase Tip" description line
    Case(
        "Thanks for the tip! Alex updated your Monday evening trip receipt",
        "tip_update",
    ),
    Case(
        "Lyft Tip Increase Receipt",
        "tip_update",
        description="Tip Increase",
        merchant="Lyft",
    ),
    Case("Your receipt", "tip_update", description="Increase Tip"),
    # fare corrections, cancellations, failed payments
    Case("We corrected the fare for your ride on Oct 3", "fare_correction", pickup=3),
    Case(
        "Receipt for canceled trip on Friday night", "cancel_noshow", description=None
    ),
    Case("Cancellation Receipt", "cancel_noshow", description=None, merchant="Lyft"),
    Case("No-Show Receipt", "cancel_noshow", description=None, merchant="Lyft"),
    Case("Receipt for your missed ride", "cancel_noshow", description=None),
    Case("Your receipt", "cancel_noshow", description="Ride cancelled"),
    Case("Lyft Ride Receipt - Charge Failed", "payment_failed", merchant="Lyft"),
    Case("Payment Failed - Your Monday evening trip with Uber", "payment_failed"),
    Case("Error de pago: Tu viaje del lunes por la noche con Uber", "payment_failed"),
    # passes and credits
    Case("Your Ride Pass is ready to roll", "pass_or_credit", description=None),
    Case("Your pass purchase (Jan 1 - Feb 1)", "pass_or_credit", description=None),
    Case("You've got Uber Cash at 10% off", "pass_or_credit", description=None),
    Case(
        "Thank you for ordering an Uber gift card", "pass_or_credit", description=None
    ),
    Case(
        "Lyft Smart Savings Plan purchase confirmed",
        "pass_or_credit",
        description="Smart Savings Plan",
    ),
    # bikes and scooters
    Case("Your Monday evening e-bike ride with JUMP", "micromobility"),
    Case(
        "Uber: Your Friday scooter trip receipt",
        "micromobility",
        description="Duration 00:09:12",
    ),
    # other services
    Case("Your rental reservation is confirmed", "rental", description=None),
    Case("Your Uber Eats order receipt", "food_order", description=None),
]

# --- precedence: a receipt that satisfies several rules gets the most specific -------

PRECEDENCE = [
    # a UF- order number is freight, whatever the subject says
    Case(
        "Your Monday evening trip with Uber", "freight", order_number="UF-987654321098"
    ),
    # order_pickup 4 is a driver payout, whatever the subject says
    Case(
        "Your Monday evening trip with Uber",
        "driver_earnings",
        pickup=4,
        merchant="Lyft",
    ),
    Case("Your pass receipt", "driver_earnings", description="Earnings", subtitle="x"),
    # a restaurant makes it a food order even with pass-like fields
    Case("Your pass receipt", "food_order", reseller="Place", subtitle="Weekly"),
    # a tip update that also mentions the ride stays a tip update
    Case(
        "Thanks! Alex updated your ride trip receipt",
        "tip_update",
        description="UberX 2.00 mi | 5 min",
    ),
    # ... and so does one that mentions a cancellation or a bike
    Case("Alex updated your canceled bike trip receipt", "tip_update"),
    # a cancelled bike trip is a cancellation; a failed pass payment a failed payment
    Case("Receipt for your cancelled bike trip", "cancel_noshow"),
    Case("Payment failed: your pass purchase", "payment_failed"),
    # a ride whose description mentions "cancel" is a cancellation
    Case(
        "Your Monday evening trip with Uber",
        "cancel_noshow",
        description="Ride cancelled",
    ),
    # a pass subtitle makes a pass of an otherwise ordinary subject
    Case(
        "Your Monday evening trip with Uber",
        "pass_or_credit",
        subtitle="Lyft Lyft rides",
    ),
]

# --- unknown templates fall back to ride and are flagged -----------------------------

UNKNOWN = [
    Case("Your receipt", "ride", unclassified=True, description=None),
    Case("A brand new vendor template", "ride", unclassified=True, description=None),
    Case(None, "ride", unclassified=True, description=None),
    Case("A brand new vendor template", "ride", description="UberX 2.00 mi | 5 min"),
]

# --- gateway prefixes and profile tags -----------------------------------------------

PROFILES = [
    Case("(EXTERNAL SENDER) Your Monday evening trip with Uber", "ride"),
    Case("{Disarmed} Your Monday evening trip with Uber", "ride"),
    Case("[Business] Your Monday evening trip with Uber", "ride", profile="business"),
    Case("[Personal] Your Monday evening trip with Uber", "ride", profile="personal"),
    Case(
        "(EXTERNAL SENDER) [Business] Your Monday evening trip with Uber",
        "ride",
        profile="business",
    ),
    Case(
        "{Disarmed} (EXTERNAL SENDER) [Personal] Your trip with Uber",
        "ride",
        profile="personal",
    ),
    Case(
        "[Personal - csr] Your Monday evening trip with Uber",
        "ride",
        profile="personal",
    ),
    Case(
        "[Business] Receipt for canceled trip on Friday night",
        "cancel_noshow",
        profile="business",
    ),
    Case(
        "[Personal] We corrected the fare for your ride on Sep 3",
        "fare_correction",
        profile="personal",
    ),
    Case(
        "(EXTERNAL SENDER) We corrected the fare for your ride on Sep 3",
        "fare_correction",
    ),
    # another bracket tag is stripped for matching but is not a profile
    Case("[New Business Card] Your Tuesday afternoon trip with Uber", "ride"),
    Case("[*****] Your Monday evening trip with Uber", "ride"),
    Case("[Card X] - Your Monday evening trip with Uber", "ride"),
    # a tag that is not at the start is no profile
    Case("Your Monday trip with Uber [Business]", "ride"),
]

ALL_CASES = FIXTURE_TEMPLATES + VENDOR_TEMPLATES + PRECEDENCE + UNKNOWN + PROFILES


IDS = [f"{i:02d}-{c.kind}" for i, c in enumerate(ALL_CASES)]


@pytest.mark.parametrize("case", ALL_CASES, ids=IDS)
def test_kind_profile_and_unclassified(case: Case) -> None:
    row = frame([case]).row(0, named=True)
    assert row["receipt_kind"] == case.kind
    assert row["subject_profile"] == case.profile
    assert row["subject_unclassified"] is case.unclassified
    assert row["is_rider_spend"] is (case.kind in RIDER_SPEND_KINDS)


def test_every_kind_is_covered_by_the_table() -> None:
    assert {c.kind for c in ALL_CASES} == set(KINDS)


def test_known_ride_templates_are_not_unclassified() -> None:
    rides = [c for c in FIXTURE_TEMPLATES + VENDOR_TEMPLATES if c.kind == "ride"]
    assert rides
    assert not frame(rides)["subject_unclassified"].any()


# --- the rider-spend rule ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "spend"),
    [
        ("ride", True),
        ("tip_update", True),
        ("fare_correction", True),
        ("cancel_noshow", True),
        ("payment_failed", False),
        ("driver_earnings", False),
        ("pass_or_credit", False),
        ("micromobility", False),
        ("food_order", False),
        ("freight", False),
        ("rental", False),
    ],
)
def test_rider_spend_by_kind(kind: str, spend: bool) -> None:
    assert (kind in RIDER_SPEND_KINDS) is spend


def test_kinds_are_an_enum_column_in_canonical_order() -> None:
    df = frame([Case("Cancellation Receipt", "cancel_noshow", description=None)])
    assert df.schema["receipt_kind"] == pl.Enum(KINDS)
    assert df.schema["is_rider_spend"] == pl.Boolean
    assert df.columns[-len(DERIVED_NAMES) :] == list(DERIVED_NAMES)
    assert input_columns(df.lazy()).collect_schema().names()[-1] == "product_category"


def test_with_derived_equals_with_columns_of_derived_columns() -> None:
    raw = frame(ALL_CASES).select(pl.exclude(*DERIVED_NAMES))
    assert with_derived(raw).equals(raw.with_columns(derived_columns()))
    assert with_derived(raw).columns == [*raw.columns, *DERIVED_NAMES]


# --- subject cleaning and templates --------------------------------------------------


@pytest.mark.parametrize(
    ("subject", "cleaned"),
    [
        ("Your trip", "Your trip"),
        ("(EXTERNAL SENDER) Your trip", "Your trip"),
        ("{Disarmed} Your trip", "Your trip"),
        ("(EXTERNAL SENDER) [Business] Your trip", "Your trip"),
        ("[Personal - csr] Your trip", "Your trip"),
        ("[Card X] - Your trip", "Your trip"),
        ("Your trip [Business]", "Your trip [Business]"),
        ("Fwd: (EXTERNAL SENDER) Your trip", "Fwd: (EXTERNAL SENDER) Your trip"),
    ],
)
def test_clean_subject(subject: str, cleaned: str) -> None:
    out = pl.DataFrame({"email_subject": [subject]}).select(clean_subject()).item()
    assert out == cleaned


@pytest.mark.parametrize(
    ("subject", "template"),
    [
        (
            "[Business] Your Monday evening trip with ****",
            "Your <weekday> <daypart> trip with <mask>",
        ),
        (
            "We corrected the fare for your ride on Oct 31",
            "We corrected the fare for your ride on <month> <n>",
        ),
        (
            "Tu viaje del sábado por la mañana",
            "Tu viaje del <weekday> por la <daypart>",
        ),
        (
            "Tu viaje del lunes por la noche con Uber",
            "Tu viaje del <weekday> por la <daypart> con Uber",
        ),
        (
            "Your pass purchase (Jan 1 - Feb 1)",
            "Your pass purchase (<month> <n> - <month> <n>)",
        ),
        ("Get 20% off", "Get <n>% off"),
        (None, "<no subject>"),
    ],
)
def test_subject_template(subject: str | None, template: str) -> None:
    df = pl.DataFrame({"email_subject": [subject]}, schema={"email_subject": pl.String})
    assert df.select(subject_template()).item() == template


def test_subject_template_is_truncated() -> None:
    df = pl.DataFrame({"email_subject": ["x" * 500]})
    assert len(df.select(subject_template()).item()) == 120


# --- the checks ----------------------------------------------------------------------


def test_kind_matrix_and_check_details() -> None:
    cases = [
        Case("Your Monday evening trip with Uber", "ride"),
        Case("Your Monday evening trip with Uber", "ride"),
        Case("Your Monday evening trip with Uber", "ride", merchant="Lyft"),
        Case(
            "Cancellation Receipt", "cancel_noshow", description=None, merchant="Lyft"
        ),
        Case("You just got paid", "driver_earnings", pickup=4, merchant="Lyft"),
    ]
    lf = frame(cases).lazy()
    matrix = kind_matrix(lf)
    assert list(matrix) == ["Lyft", "Uber"]
    assert all(list(row) == list(KINDS) for row in matrix.values())
    assert matrix["Uber"]["ride"] == 2
    assert matrix["Lyft"]["ride"] == 1
    assert matrix["Lyft"]["driver_earnings"] == 1
    assert matrix["Uber"]["driver_earnings"] == 0

    k1 = receipt_kinds_check(lf)
    assert (k1.check_id, k1.passed, k1.threshold) == ("K1", True, None)
    assert (k1.n_failed, k1.n_evaluated) == (1, 5)  # only the payout is non-spend
    assert k1.details["counts"] == {
        **dict.fromkeys(KINDS, 0),
        "ride": 3,
        "cancel_noshow": 1,
        "driver_earnings": 1,
    }


def test_kind_check_on_an_empty_frame() -> None:
    k1 = receipt_kinds_check(frame([]).lazy())
    assert (k1.n_failed, k1.n_evaluated, k1.metric) == (0, 0, None)
    assert k1.details["by_merchant"] == {}


def test_unknown_templates_are_counted_and_listed() -> None:
    cases = [
        Case("Your Monday evening trip with Uber", "ride"),
        Case(
            "Hello 12 from ***** on May 3", "ride", unclassified=True, description=None
        ),
        Case(
            "Hello 99 from ***** on May 4", "ride", unclassified=True, description=None
        ),
        Case("Another thing", "ride", unclassified=True, description=None),
        Case(None, "ride", unclassified=True, description=None),
    ]
    res = unclassified_templates_check(frame(cases).lazy())
    assert (res.name, res.passed, res.n_failed, res.n_evaluated) == (
        "unclassified_subject_templates",
        True,
        4,
        5,
    )
    assert res.details["distinct"] == 3
    assert res.details["templates"] == [
        {"template": "Hello <n> from <mask> on <month> <n>", "count": 2},
        {"template": "<no subject>", "count": 1},
        {"template": "Another thing", "count": 1},
    ]


def test_at_most_twenty_templates_are_listed() -> None:
    cases = [
        Case(f"Unknown template {chr(97 + i)}{chr(97 + i)}", "ride", description=None)
        for i in range(25)
    ]
    res = unclassified_templates_check(frame(cases).lazy())
    assert res.n_failed == 25
    assert res.details["distinct"] == 25
    templates = res.details["templates"]
    assert isinstance(templates, list)
    assert len(templates) == TOP_TEMPLATES == 20


def test_parse_module_does_not_load_polars() -> None:
    code = (
        "import sys\n"
        "import rideshare_data_pipeline.parse, rideshare_data_pipeline.checks.kinds\n"
        "sys.exit('polars' in sys.modules)\n"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
