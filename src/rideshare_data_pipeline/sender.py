"""The sender of a receipt: its country, and whether it is a known receipt sender.

``from_domain`` is misnamed: it holds the **full sender address**. The merchant
sends from a few public, automated addresses, and Uber encodes the country or city of
the receipt in the local part. That is the only currency evidence the file has (there
is no currency column): an ``uber.colombia@`` receipt states its amount in pesos.

Patterns (matched on the lower-cased address)
---------------------------------------------
``uber.<token>@...``
    A country sender. ``<token>`` is looked up in ``COUNTRY_TOKENS`` (``colombia`` is
    ``co``, ``costa.rica`` is ``cr``). ``uber.us@`` is ``us``. A token the table
    doesn't know is ``unknown``: the receipt is from a country sender we can't place,
    so ``currency_suspect`` treats it as non-USD. ``uber.pr@`` (Puerto Rico) is ``us``,
    which uses the dollar. Some other tokens (Panama, Ecuador, El Salvador) also
    use the dollar, but their receipts are kept apart anyway: a conservative gate
    only costs a smaller USD population.
``receipts.<city>@...``
    A city sender. A city in ``FOREIGN_CITIES`` has its country; any other city is
    ``us``: the vendor's city senders are overwhelmingly US cities, and a US city
    can't be enumerated. A foreign city missing from the table would pass as USD, so
    extend the table when the data shows one.
generic senders (``noreply@``, ``no-reply@``, ``receipts@``, ``uber@``, ``email@``
on the merchants' domains and subdomains)
    null: the address says nothing about the country.

Privacy
-------
Only the merchant's public receipt senders are encoded here. Some rows come from
addresses of individual people (freight): they match none of these shapes, so
``sender_country`` is null for them and ``known_sender`` is false (check V3). No
such address appears in code, tests or reports.
"""

lazy import polars as pl

UNKNOWN = "unknown"
"""The country of an ``uber.<token>@`` sender whose token is not in the table."""

US = "us"
NON_US_EXEMPT: tuple[str, ...] = (US,)
"""Country values that are not suspect (null is exempt too)."""

# Token of ``uber.<token>@`` (dots stand for spaces) to an ISO 3166 alpha-2 code.
COUNTRY_TOKENS: dict[str, str] = {
    "us": "us",
    "pr": "us",
    "canada": "ca",
    "mexico": "mx",
    "colombia": "co",
    "chile": "cl",
    "brazil": "br",
    "brasil": "br",
    "argentina": "ar",
    "peru": "pe",
    "ecuador": "ec",
    "uruguay": "uy",
    "bolivia": "bo",
    "panama": "pa",
    "guatemala": "gt",
    "costa.rica": "cr",
    "el.salvador": "sv",
    "republica.dominicana": "do",
    "tt": "tt",
    "uk": "gb",
    "belarus": "by",
    "turkiye": "tr",
    "lebanon": "lb",
    "jordan": "jo",
    "egypt": "eg",
    "israel": "il",
    "uae": "ae",
    "south.africa": "za",
    "tanzania": "tz",
    "kenya": "ke",
    "ghana": "gh",
    "bangladesh": "bd",
    "cambodia": "kh",
    "singapore": "sg",
    "hongkong": "hk",
    "taiwan": "tw",
    "australia": "au",
    "new.zealand": "nz",
}

# Cities of ``receipts.<city>@`` that are not in the US; every other city is us.
FOREIGN_CITIES: dict[str, str] = {
    "puerto.vallarta": "mx",
    "nuevo.vallarta": "mx",
    "punta.del.este": "uy",
    "punta.arenas": "cl",
    "cusco": "pe",
    "gatineau": "ca",
    "mecca": "sa",
    "madinah": "sa",
    "sydney": "au",
}

# One extraction for both patterns; the key is ``uber.<token>`` or ``receipts.<city>``.
_SENDER_KEY = r"^((?:uber|receipts)\.[a-z.]+)@"
_CITY_PREFIX = "receipts."
# Generic merchant senders: a fixed local part on a merchant domain or subdomain.
_GENERIC_SENDER = (
    r"^(?:no-?reply|receipts|uber|email)@(?:[a-z0-9-]+\.)*(?:uber|lyft|lyftmail)\.com$"
)

UBER_DOMAINS: tuple[str, ...] = ("uber.com",)
LYFT_DOMAINS: tuple[str, ...] = ("lyft.com", "lyftmail.com")


def _address() -> pl.Expr:
    return pl.col("from_domain").str.to_lowercase()


def _key_table() -> dict[str, str]:
    """Sender keys (``uber.colombia``, ``receipts.sydney``) to countries."""
    return {
        **{f"uber.{token}": country for token, country in COUNTRY_TOKENS.items()},
        **{
            f"{_CITY_PREFIX}{city}": country for city, country in FOREIGN_CITIES.items()
        },
    }


def sender_country() -> pl.Expr:
    """The country of the sender: a code, ``unknown`` or null (String).

    ``uber.<token>@`` gives the table's country (``unknown`` for a token not in it),
    ``receipts.<city>@`` the city's country (``us`` unless in FOREIGN_CITIES), and
    every other sender, generic or unrecognised, null.
    """
    key = _address().str.extract(_SENDER_KEY, 1)
    # A key the table lacks: a city is us, a country token is unknown.
    default = (
        pl.when(key.str.starts_with(_CITY_PREFIX))
        .then(pl.lit(US))
        .otherwise(pl.lit(UNKNOWN))
    )
    country = key.replace_strict(_key_table(), default=default)
    return pl.when(key.is_not_null()).then(country).alias("sender_country")


def known_sender(country: pl.Expr | None = None) -> pl.Expr:
    """True where the sender matches a known receipt-sender pattern.

    That is a country or city sender (``sender_country`` not null) or a generic
    merchant sender. Null where ``from_domain`` is null.
    """
    country = sender_country() if country is None else country
    return (country.is_not_null() | _address().str.contains(_GENERIC_SENDER)).alias(
        "known_sender"
    )


def _on_domain(domains: tuple[str, ...]) -> pl.Expr:
    """True where the address is on one of `domains` or a subdomain of one."""
    address = _address()
    return pl.any_horizontal(
        address.str.ends_with(f"@{d}") | address.str.ends_with(f".{d}") for d in domains
    )


def is_uber_sender() -> pl.Expr:
    """True where the sender is on an Uber domain (uber.com or a subdomain)."""
    return _on_domain(UBER_DOMAINS)


def is_lyft_sender() -> pl.Expr:
    """True where the sender is on a Lyft domain (lyft.com, lyftmail.com, or below)."""
    return _on_domain(LYFT_DOMAINS)
