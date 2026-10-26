"""Sender country, the currency gate, ``usd_rides``, CUR1, V1, V3 and the spend table.

Every address and amount here is made up; the only addresses that appear are the
merchant's public receipt senders and an invented person-like shape.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest

from rideshare_data_pipeline.checks import RowRule, default_rules, run_checks
from rideshare_data_pipeline.checks.currency import currency_suspect_check
from rideshare_data_pipeline.dedupe import supersede_flags
from rideshare_data_pipeline.money import (
    STEPS,
    spend_waterfall,
    usd_rides,
    waterfall_exprs,
)
from rideshare_data_pipeline.parse import (
    CURRENCY_REASONS,
    currency_reason,
    currency_suspect,
    foreign_subject,
    with_derived,
)
from rideshare_data_pipeline.sender import known_sender, sender_country

type Spec = dict[str, Any]

TRIP = "UberX 3.10 mi | 9 min"


def senders(*addresses: str | None) -> pl.DataFrame:
    return pl.DataFrame({"from_domain": pl.Series(list(addresses), dtype=pl.String)})


# --- sender_country -------------------------------------------------------------

COUNTRIES: list[tuple[str | None, str | None]] = [
    # country senders
    ("uber.colombia@uber.com", "co"),
    ("uber.chile@uber.com", "cl"),
    ("uber.costa.rica@uber.com", "cr"),
    ("uber.republica.dominicana@uber.com", "do"),
    ("uber.brasil@uber.com", "br"),
    ("uber.uk@uber.com", "gb"),
    ("UBER.Mexico@Uber.com", "mx"),
    # the US, and Puerto Rico (it uses the dollar)
    ("uber.us@uber.com", "us"),
    ("uber.pr@uber.com", "us"),
    # an unknown token of the country pattern is "unknown", not null
    ("uber.atlantis@uber.com", "unknown"),
    # city senders: a US city, a foreign one, and one no table knows (us)
    ("receipts.nyc.suburbs@uber.com", "us"),
    ("receipts.eugene.or@uber.com", "us"),
    ("receipts.puerto.vallarta@uber.com", "mx"),
    ("receipts.sydney@uber.com", "au"),
    ("receipts.some.new.town@uber.com", "us"),
    # generic merchant senders say nothing about the country
    ("noreply@uber.com", None),
    ("receipts@uber.com", None),
    ("uber@uber.com", None),
    ("email@et.uber.com", None),
    ("no-reply@lyftmail.com", None),
    ("noreply@lyft.com", None),
    ("noreply@rentals.lyftmail.com", None),
    # a person-like address matches no pattern
    ("first.last@uber.com", None),
    ("somebody@example.org", None),
    (None, None),
]


@pytest.mark.parametrize(("address", "country"), COUNTRIES)
def test_sender_country(address: str | None, country: str | None) -> None:
    got = senders(address).select(sender_country()).item()
    assert got == country


# V3's known_sender: a country or city pattern, or a generic merchant sender.
KNOWN: list[tuple[str | None, bool | None]] = [
    ("uber.colombia@uber.com", True),
    ("uber.atlantis@uber.com", True),
    ("receipts.macon@uber.com", True),
    ("noreply@uber.com", True),
    ("no-reply@lyft.com", True),
    ("noreply@rentals.lyftmail.com", True),
    ("first.last@uber.com", False),  # person-like: matches nothing
    ("first@uber.com", False),
    ("noreply@example.com", False),
    (None, None),
]


@pytest.mark.parametrize(("address", "known"), KNOWN)
def test_known_sender(address: str | None, known: bool | None) -> None:
    got = senders(address).select(known_sender()).item()
    # A null address is not a verdict either way (the rule's applies_to skips it).
    assert got == known or (known is None and got is False)


# --- currency_reason / currency_suspect --------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    sender: str | None
    subject: str = "Your Monday evening trip with Sam"
    description: str = TRIP


def derived(*rows: Row) -> pl.DataFrame:
    return with_derived(
        pl.DataFrame(
            {
                "merchant_name": ["Uber"] * len(rows),
                "from_domain": pl.Series([r.sender for r in rows], dtype=pl.String),
                "email_subject": pl.Series([r.subject for r in rows], dtype=pl.String),
                "product_description": pl.Series(
                    [r.description for r in rows], dtype=pl.String
                ),
                "product_reseller": pl.Series([None] * len(rows), dtype=pl.String),
                "product_subtitle": pl.Series([None] * len(rows), dtype=pl.String),
                "order_number": pl.Series([None] * len(rows), dtype=pl.String),
                "order_pickup": pl.Series([1] * len(rows), dtype=pl.Int8),
                "product_category": ["Travel"] * len(rows),
            }
        )
    )


REASONS: list[tuple[str, Row, str | None]] = [
    ("us sender", Row("uber.us@uber.com"), None),
    ("generic sender", Row("noreply@uber.com"), None),
    ("us city", Row("receipts.macon@uber.com"), None),
    ("foreign country", Row("uber.chile@uber.com"), "non_us_sender"),
    ("foreign city", Row("receipts.cusco@uber.com"), "non_us_sender"),
    ("unknown token", Row("uber.atlantis@uber.com"), "non_us_sender"),
    # the km rule fires on a generic sender, and on a US one
    (
        "km, generic",
        Row("noreply@uber.com", description="UberX 6.80 km | 11 min"),
        "km_unit",
    ),
    (
        "km, us",
        Row("uber.us@uber.com", description="UberX 6.80 km | 11 min"),
        "km_unit",
    ),
    (
        "miles, generic",
        Row("noreply@uber.com", description="UberX 6.80 mi | 11 min"),
        None,
    ),
    (
        "spanish subject",
        Row("uber.us@uber.com", subject="Tu viaje del lunes por la tarde"),
        "foreign_language",
    ),
    (
        "spanish product",
        Row("noreply@uber.com", subject="Tu UberX del martes por la noche"),
        "foreign_language",
    ),
    (
        "spanish payment",
        Row("noreply@uber.com", subject="Error de pago: Tu viaje del lunes"),
        "foreign_language",
    ),
    (
        "portuguese subject",
        Row("noreply@uber.com", subject="Sua viagem de segunda-feira"),
        "foreign_language",
    ),
    (
        "gateway tag, english",
        Row("noreply@uber.com", subject="[Business] Your trip with Sam"),
        None,
    ),
    # the first reason wins
    (
        "country beats km",
        Row("uber.chile@uber.com", description="UberX 6.80 km | 11 min"),
        "non_us_sender",
    ),
    (
        "country beats language",
        Row("uber.chile@uber.com", subject="Tu viaje del lunes"),
        "non_us_sender",
    ),
    (
        "km beats language",
        Row(
            "noreply@uber.com",
            subject="Tu viaje del lunes",
            description="UberX 6.80 km | 11 min",
        ),
        "km_unit",
    ),
    # no sender, no description: nothing to suspect
    ("null sender", Row(None), None),
    ("no description", Row("noreply@uber.com", description="Increase Tip"), None),
]


@pytest.mark.parametrize(
    ("row", "reason"), [(r, e) for _, r, e in REASONS], ids=[n for n, _, _ in REASONS]
)
def test_currency_reason(row: Row, reason: str | None) -> None:
    df = derived(row)
    assert df["currency_reason"].item() == reason
    assert df["currency_suspect"].item() is (reason is not None)
    assert df.schema["currency_reason"] == pl.Enum(CURRENCY_REASONS)


def test_standalone_factories_equal_with_derived() -> None:
    df = derived(*(r for _, r, _ in REASONS))
    alone = df.select(
        "currency_reason",
        "currency_suspect",
        "sender_country",
    )
    again = df.select(sender_country(), currency_reason(), currency_suspect()).select(
        alone.columns
    )
    assert again.equals(alone)
    flags = df.select(foreign_subject()).to_series().to_list()
    assert sum(flags) == 6  # the language rule is independent of the reason order


# --- usd_rides ------------------------------------------------------------------


def money_frame() -> pl.DataFrame:
    """One row per way to be in or out of the USD ride population."""
    base: Spec = {
        "merchant_name": "Uber",
        "order_number": None,
        "email_time": None,
        "update_time": None,
        "checksum": "x",
    }
    cases: list[tuple[str, Spec]] = [
        ("usd", {}),
        ("usd tip update", {"kind": "tip_update", "amount": 3.0}),
        ("suspect", {"suspect": True}),
        ("superseded", {"superseded": True}),
        ("not spend", {"spend": False, "kind": "freight"}),
        ("null amount", {"amount": None}),
        ("suspect and superseded", {"suspect": True, "superseded": True}),
    ]
    rows = [
        {
            "case": name,
            "kind": "ride",
            "amount": 10.0,
            "spend": True,
            "suspect": False,
            "superseded": False,
            **base,
            **spec,
        }
        for name, spec in cases
    ]
    return pl.DataFrame(
        {
            "case": [r["case"] for r in rows],
            "order_total_amount": pl.Series(
                [r["amount"] for r in rows], dtype=pl.Float64
            ),
            "is_rider_spend": [r["spend"] for r in rows],
            "currency_suspect": [r["suspect"] for r in rows],
            "is_superseded": [r["superseded"] for r in rows],
        }
    )


def test_usd_rides_excludes_suspect_superseded_non_spend_and_null_amounts() -> None:
    df = money_frame()
    assert usd_rides(df)["case"].to_list() == ["usd", "usd tip update"]
    lazy = usd_rides(df.lazy()).collect()
    assert lazy.equals(usd_rides(df))


def test_waterfall_partitions_the_naive_total() -> None:
    df = money_frame()
    w = spend_waterfall(df.select(waterfall_exprs()).row(0, named=True))
    assert list(w)[: len(STEPS)] == list(STEPS)
    # Six rows have an amount; the null one is in no step.
    assert w["naive_total"] == {"rows": 6, "amount": 53.0}
    assert w["currency_suspect"] == {"rows": 2, "amount": 20.0}
    assert w["non_rider_spend"] == {"rows": 1, "amount": 10.0}
    assert w["superseded"] == {"rows": 1, "amount": 10.0}
    assert w["clean_usd_rider_spend"] == {"rows": 2, "amount": 13.0}
    assert w["suspect_rider_spend"] == {"rows": 1, "amount": 10.0}
    taken = sum(w[k]["amount"] for k in STEPS[1:4])
    assert w["naive_total"]["amount"] - taken == w["clean_usd_rider_spend"]["amount"]
    assert w["clean_usd_rider_spend"]["rows"] == usd_rides(df).height


# --- V1 and V3 ------------------------------------------------------------------


def rule(name: str) -> RowRule:
    return next(r for r in default_rules() if r.name == name)


def sender_frame(*pairs: tuple[str, str | None]) -> pl.LazyFrame:
    df = pl.DataFrame(
        {
            "merchant_name": [m for m, _ in pairs],
            "from_domain": pl.Series([a for _, a in pairs], dtype=pl.String),
        }
    )
    return df.with_columns(sender_country()).lazy()


def failing(name: str, lf: pl.LazyFrame) -> tuple[int, int]:
    (res,) = run_checks_rule(name, lf)
    return res.n_failed, res.n_evaluated


def run_checks_rule(name: str, lf: pl.LazyFrame) -> list[Any]:
    return run_checks(lf, rules=[rule(name)], dataset_checks=[])


V1_CASES: list[tuple[str, tuple[str, str | None], bool]] = [
    ("uber from uber", ("Uber", "noreply@uber.com"), False),
    ("uber from a country sender", ("Uber", "uber.chile@uber.com"), False),
    ("uber from a subdomain", ("Uber", "email@et.uber.com"), False),
    ("lyft from lyftmail", ("Lyft", "no-reply@lyftmail.com"), False),
    ("lyft from lyft.com", ("Lyft", "noreply@lyft.com"), False),
    ("lyft from a lyftmail subdomain", ("Lyft", "noreply@rentals.lyftmail.com"), False),
    ("uber from lyft", ("Uber", "no-reply@lyftmail.com"), True),
    ("lyft from uber", ("Lyft", "noreply@uber.com"), True),
    ("uber from an unrelated domain", ("Uber", "noreply@example.com"), True),
    ("lookalike domain", ("Uber", "noreply@notuber.com"), True),
]


@pytest.mark.parametrize(
    ("pair", "fails"), [(p, f) for _, p, f in V1_CASES], ids=[n for n, _, _ in V1_CASES]
)
def test_v1_merchant_sender_consistent(
    pair: tuple[str, str | None], fails: bool
) -> None:
    assert failing("merchant_sender_consistent", sender_frame(pair)) == (int(fails), 1)


def test_v1_skips_unknown_merchants_and_null_senders() -> None:
    lf = sender_frame(("Other", "noreply@example.com"), ("Uber", None))
    assert failing("merchant_sender_consistent", lf) == (0, 0)
    r = rule("merchant_sender_consistent")
    assert (r.check_id, r.severity.value) == ("V1", "error")


def test_v3_unknown_sender_and_its_sample_never_shows_the_address() -> None:
    lf = sender_frame(
        ("Uber", "noreply@uber.com"),
        ("Uber", "uber.colombia@uber.com"),
        ("Uber", "first.last@uber.com"),
        ("Uber", None),
    )
    (res,) = run_checks_rule("unknown_sender", lf)
    assert (res.check_id, res.severity.value) == ("V3", "warn")
    assert (res.n_failed, res.n_evaluated) == (1, 3)
    assert res.sample is not None
    assert "from_domain" not in res.sample.columns
    assert "last" not in str(res.sample.to_dicts())
    for r in default_rules():
        assert "from_domain" not in r.sample_cols


# --- CUR1 -----------------------------------------------------------------------


def test_cur1_details_count_reasons_countries_and_amounts() -> None:
    rows = [
        Row("uber.us@uber.com"),
        Row("uber.chile@uber.com"),
        Row("uber.chile@uber.com"),
        Row("uber.colombia@uber.com", subject="Tu viaje del lunes"),
        Row("noreply@uber.com", description="UberX 6.80 km | 11 min"),
        Row("noreply@uber.com", subject="Tu viaje del lunes por la tarde"),
    ]
    df = derived(*rows).with_columns(
        order_total_amount=pl.Series([10.0, 1000.0, 2000.0, 3000.0, 5.0, 6.0]),
        email_time=pl.lit(None, dtype=pl.Datetime("ms")),
        update_time=pl.lit(None, dtype=pl.Datetime("ms")),
        checksum=pl.Series([str(i) for i in range(6)]),
    )
    res = currency_suspect_check(supersede_flags(df).lazy())
    assert (res.check_id, res.severity.value, res.passed) == ("CUR1", "warn", False)
    assert (res.n_failed, res.n_evaluated) == (5, 6)
    assert res.details["by_reason"] == {
        "non_us_sender": {"rows": 3, "amount": 6000.0},
        "foreign_language": {"rows": 1, "amount": 6.0},
        "km_unit": {"rows": 1, "amount": 5.0},
    }
    assert res.details["by_country"] == {
        "cl": {"rows": 2, "amount": 3000.0},
        "(none)": {"rows": 2, "amount": 11.0},
        "co": {"rows": 1, "amount": 3000.0},
    }
    assert res.details["suspect_sum"] == {
        "rows": 5,
        "amount": 6011.0,
        "note": "local currency, not summable as USD",
    }
    assert res.details["suspect_rider_spend"] == {"rows": 5, "amount": 6011.0}
    spend = res.details["spend"]
    assert isinstance(spend, dict)
    assert spend["clean_usd_rider_spend"] == {"rows": 1, "amount": 10.0}


def test_cur1_passes_when_nothing_is_suspect() -> None:
    df = derived(Row("uber.us@uber.com")).with_columns(
        order_total_amount=pl.lit(4.0),
        email_time=pl.lit(None, dtype=pl.Datetime("ms")),
        update_time=pl.lit(None, dtype=pl.Datetime("ms")),
        checksum=pl.lit("a"),
    )
    res = currency_suspect_check(df.lazy())
    assert res.passed
    assert res.details["by_reason"] == {}
    assert res.details["suspect_sum"]["rows"] == 0  # type: ignore[index]


# --- imports --------------------------------------------------------------------


def test_new_modules_do_not_import_polars() -> None:
    code = (
        "import sys\n"
        "import rideshare_data_pipeline.sender\n"
        "import rideshare_data_pipeline.money\n"
        "import rideshare_data_pipeline.checks.currency\n"
        "sys.exit('polars' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=env, capture_output=True
    )
