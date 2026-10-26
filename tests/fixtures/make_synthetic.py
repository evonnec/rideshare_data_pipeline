"""A seeded generator of ~300 synthetic receipts that reproduce known data issues.

Every value is made up; real data is never committed. The output is a CSV in the
exact 33-column layout (UTF-8, CRLF) and, for the same seed, byte-identical bytes.

Layout: 248 clean baseline rides (8 panelists x 31 spaced days, both merchants,
fares tightly tied to distance and duration), then one small injection block per
issue below. A panelist has at most one baseline ride a day, so the baseline fires
no warning or error check. Every injection uses its own panelist(s) and its own
dates, and is built to trigger only its intended check; the few overlaps are
listed in the MANIFEST comments.

INJECTIONS maps each block's name to the row positions it produced (0-based, in
file order). That label is not a CSV column.

    block                          rows  what it reproduces
    duplicate_order_key               2  same (merchant, order_number), other email_time
    cross_panelist_duplicate          2  same order number under two user_ids
    ride_tip_update                   2  a ride and its tip-updated receipt
    near_duplicate                    2  two rides identical but for order_number
    monthly_pass                      3  pass charges identical but for email_time
    driver_earnings                   1  Lyft payout row (order_pickup 4)
    food_orders                       2  food orders with a restaurant name
    bike                              1  micromobility row, negative total
    freight                           1  UF- order number, person-like sender
    non_us_sender                     1  Spanish subject, local-currency amount
    km_generic_sender                 1  km description from a US sender
    uber_date_only                    1  Uber order_time at 00:00:00.000
    lyft_order_5h_before_email        1  Lyft order_time exactly 5 h before email
    apac_next_day                     1  Asia-Pacific order date = email date + 1
    ancient_order                     1  order dated 2010, years before the window
    discount_gt_subtotal              1  order_discount > order_subtotal
    tax_gt_total                      1  order_tax > order_total_amount
    zero_fare_ride                    1  a ride with total 0
    negative_total_ride               1  a car ride with a negative total (A1)
    zero_duration_trip                1  a 2-mile Lyft ride lasting 0 s
    implausible_speed                 1  60 miles in 20 minutes
    fare_outlier_10x                  1  fare at 10x the typical $/mile and $/min
    overlapping_rides                 2  two Lyft rides of one panelist that overlap
    many_rides_one_day                12 12 rides, one panelist, one day
    masked_subjects                   2  subjects with an asterisk mask

Columns that are all-null, constant, or first-equals-last but not constant come
out of the layout itself (see ALL_NULL and CONSTANT in the tests): the file starts
and ends on an Uber row.

Run it as a script to write a CSV for a manual ``rideshare-qc`` run:

    python tests/fixtures/make_synthetic.py OUT.csv
"""

import csv
import hashlib
import random
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from rideshare_data_pipeline.io import typed
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS, raw_schema

type Row = dict[str, str]

SEED = 20190301

# The expected `n_failed` of each catalog check on the full
# synthetic frame. Keys with a dot are sub-counts (per kind, in groups or in
# panelist-days). Counting conventions where the catalog leaves one open: a
# "duplicate" rule fails every row of a repeated key; U4 counts excess rows;
# U6 flags the earlier trip of an overlapping pair; "ride rows" are kinds ride
# and tip_update. Checks whose value is 0 are negative controls: they must keep
# passing however the injections change.
MANIFEST: dict[str, int] = {
    # --- implemented now (asserted in tests/test_synthetic.py) -----------------
    "S1": 0,  # the header is the 33 expected columns, in order
    "S2": 0,  # no value fails its cast
    "S3": 0,  # valid UTF-8
    # Negative control: every ride has a total, an order number and an order time,
    # so the null rates are 0 unless an injection removes one.
    "C2": 0,
    "C3": 0,  # row_count_band: the default band (at least 1 row) holds
    "U2b": 0,  # after the dedupe no key repeats among the kept rows
    # A1 applies to rider-spend receipts only: the bike's -4.50 (a non-spend kind) is
    # not counted. The injected negative car ride is, and it is the only A1 failure.
    # Later checks must not count it twice: A3 (zero fare) means exactly 0, and the
    # fare-outlier checks (O4) must skip non-positive totals, which A1 owns.
    "A1": 1,  # negative_total_ride: total -5.75
    "U0": 0,  # no exact duplicate row (checksums differ, so even pairs differ)
    "U1": 0,  # checksum is md5 of the row position
    "S4": 5,  # order_points, order_shipping, SKU, item_id, file_id
    "S5": 4,  # order_total_qty, item_quantity, digital_transaction, product_category
    # --- reserved for later PRs -------------------------------------------------
    "U2": 6,  # rows sharing a non-null key: duplicate_order_key 2 + cross 2 + tip 2
    "U2.groups": 3,  # one key each: duplicate_order_key, cross, ride_tip_update
    # One row of each group is superseded: the later email wins both rides-only
    # groups (duplicate_order_key, cross), and the tip update wins its pair, so the
    # superseded rows are the first duplicate_order_key row, the first cross row and
    # the original ride. The 3 monthly pass rows have no order number: never superseded.
    "U2.superseded": 3,  # = U2.groups: every group has exactly 2 rows
    "U3": 2,  # cross_panelist_duplicate (2 user_ids on one key)
    "U3.groups": 1,
    # near_duplicate pair: 1 excess row. The duplicate_order_key pair is identical
    # on user, merchant, date, total and description too, but shares its order
    # number, so a U4 rule must require distinct order numbers (else 2 excess).
    "U4": 1,
    "U4.rows": 2,
    "K1.driver_earnings": 1,  # driver_earnings
    "K1.pass": 3,  # monthly_pass
    "K1.micromobility": 1,  # bike
    "K1.food_order": 2,  # food_orders
    "K1.freight": 1,  # freight
    "K1.tip_update": 1,  # ride_tip_update (the updated receipt)
    "K1.ride": 284,  # every other row: baseline, injected rides, masked subjects
    "K1.non_spend": 8,  # earnings 1 + pass 3 + bike 1 + food 2 + freight 1
    "V2": 4,  # category 'Travel' on earnings 1 + food 2 + freight 1
    "V3": 1,  # freight sender is person-like
    "S6": 1,  # freight order number 'UF-...' is not a UUID
    # non_us_sender, apac_next_day (a sender country that is not us) and
    # km_generic_sender. non_us_sender has a Spanish subject too, still 1 row. Its
    # 5,718 total and the km row's distance would be $/mile outliers if CUR1 did
    # not gate them out of the USD stats (O4), so O4 must apply the gate.
    "CUR1": 3,
    "CUR1.non_us": 2,
    "CUR1.km_generic": 1,
    # The population of money.usd_rides: rider-spend rows (293 - 8 non-spend = 285)
    # that are not superseded (3) and not currency-suspect (3). The three groups
    # are disjoint, and every row has a total.
    "USD.rides": 279,
    "T2": 0,  # negative control: apac_next_day is within tolerance, Lyft 5 h row too
    "T3": 1,  # ancient_order is > 7 days late (every other lag is <= 2 days)
    "T3.30d": 1,  # ... and > 30 days late
    "T4": 1,  # ancient_order is before the window
    "A3": 1,  # zero_fare_ride
    "A4": 1,  # discount_gt_subtotal
    "A5": 1,  # tax_gt_total
    "O1": 1,  # implausible_speed: 60 mi in 1,200 s
    "O2": 1,  # zero_duration_trip (miles are never 0)
    "O3": 0,  # negative control: nothing exceeds 100 miles or 3 hours
    "O4": 1,  # fare_outlier_10x (implausible_speed has a $/min outlier only)
    "U5": 12,  # many_rides_one_day: rows in the over-threshold panelist-day
    "U5.panelist_days": 1,
    "U6": 1,  # overlapping_rides: the earlier trip
    # Every ride and tip-update row has a parseable "<product> <d> mi|km | <m> min" or
    # "Lyft fare (<d>mi, <m>m <s>s)" description (the one tip update copies its
    # ride's), so no row is unparsed.
    "PC1": 0,
    "PR1": 2,  # masked_subjects
    # More negative controls: the baseline and injections must not trip these.
    "C1": 0,  # required columns are always present (asserted in test_synthetic)
    "V1": 0,  # merchant and sender domain agree in every row
    "A2": 0,  # item_price == order_subtotal in every row
    "A6": 0,  # no unexplained residual above max(20, subtotal)
    "T1": 0,  # insert <= update <= source_dttimestamp <= dttimestamp
    "T5": 0,  # email_time <= update_time in every row (tip update included)
}

# The sum of order_total_amount over the USD.rides rows: the naive total less what the
# three groups take off (derived from the rows, then pinned: 9,762.45 - 5,734.35 -
# 228.82 - 29.36).
USD_AMOUNT = 3769.92

_FIRST_NAMES = ("Alex", "Sam", "Jordan", "Taylor", "Casey", "Morgan")
_US_UBER_SENDERS = (
    "noreply@uber.com",
    "receipts.example@uber.com",
    "uber.us@uber.com",
)
_LYFT_SENDER = "no-reply@lyftmail.com"
_DAYPARTS = ("morning", "afternoon", "evening", "night")

_WINDOW_START = date(2019, 3, 1)
_N_PANELISTS = 8
_N_GROUPS = 31  # baseline rows: 31 spaced days x 8 panelists = 248


def _ts(dt: datetime) -> str:
    return f"{dt:%Y-%m-%d %H:%M:%S}.{dt.microsecond // 1000:03d}"


def _money(x: float) -> str:
    return f"{x:.2f}"


def _user_id(i: int) -> str:
    return hashlib.sha256(f"synthetic-user-{i}".encode()).hexdigest()


def _lyft_fare(miles: float, secs: int) -> float:
    return round(0.40 + 1.60 * miles + 0.20 * secs / 60, 2)


def _uber_fare(miles: float, mins: int) -> float:
    return round(0.40 + 1.55 * miles + 0.22 * mins, 2)


def _typical_secs(miles: float) -> int:
    """The duration a trip of this length usually takes (22.5 mph)."""
    return int(160 * miles)


def _secs_for(miles: float, rng: random.Random) -> int:
    """A baseline duration: typical, give or take 10 s."""
    return _typical_secs(miles) + rng.randint(-10, 10)


class _Generator:
    """Builds rows from one seeded RNG and a running counter; no global state."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.counter = 0

    def lyft_order_number(self) -> str:
        self.counter += 1
        digest = hashlib.sha256(f"lyft-order-{self.counter}".encode()).hexdigest()
        return str(10**18 + int(digest[:15], 16))  # always 19 digits

    def uber_order_number(self) -> str:
        self.counter += 1
        digest = hashlib.md5(f"uber-order-{self.counter}".encode()).digest()
        return str(uuid.UUID(bytes=digest))

    def stamps(self, email: datetime, insert: datetime | None = None) -> Row:
        """The vendor timestamps of a receipt that arrived at `email` (UTC).

        insert_time normally follows the email; a tip update keeps the original
        (earlier) insert_time. update_time never precedes the email.
        """
        rng = self.rng
        insert = insert or email + timedelta(minutes=rng.randint(2, 40))
        update = max(insert, email) + timedelta(minutes=rng.choice((0, 0, 3, 17)))
        source = update + timedelta(minutes=rng.randint(30, 180))
        dtstamp = source + timedelta(minutes=rng.randint(1, 59))
        batch_day = datetime.combine(dtstamp.date(), datetime.min.time())
        return {
            "email_time": _ts(email),
            "insert_time": _ts(insert),
            "update_time": _ts(update),
            "delivery_date": _ts(batch_day),
            "start_source_folder_date": f"{dtstamp:%Y-%m-%d}",
            "end_source_folder_date": f"{dtstamp:%Y-%m-%d}",
            "source_dttimestamp": _ts(source),
            "dttimestamp": _ts(dtstamp),
        }

    def row(self, merchant: str, user: int, email: datetime, **fields: str) -> Row:
        """A row of defaults for a charged one-item receipt, then `fields` on top."""
        base: Row = dict.fromkeys(EXPECTED_COLUMNS, "")
        base.update(
            merchant_name=merchant,
            user_id=_user_id(user),
            order_total_qty="1.0",
            item_quantity="1.0",
            digital_transaction="0",
            product_category="Travel",
            order_pickup="1",
            from_domain=_LYFT_SENDER if merchant == "Lyft" else "noreply@uber.com",
        )
        base.update(self.stamps(email))
        base.update(fields)
        return base

    def _name(self) -> str:
        return self.rng.choice(_FIRST_NAMES)

    def lyft_trip(
        self,
        user: int,
        order: datetime,
        miles: float,
        secs: int,
        *,
        email: datetime | None = None,
        **fields: str,
    ) -> Row:
        """A Lyft ride; the email arrives about 5 h (a US offset) after the trip."""
        rng = self.rng
        fare = _money(_lyft_fare(miles, secs))
        email = email or order + timedelta(
            hours=5, minutes=rng.randint(3, 20), seconds=rng.randint(0, 59)
        )
        defaults = {
            "order_number": self.lyft_order_number(),
            "order_time": _ts(order),
            "order_total_amount": fare,
            "order_subtotal": fare,
            "item_price": fare,
            "product_description": (
                f"Lyft fare ({miles:.2f}mi, {secs // 60}m {secs % 60}s)"
            ),
            "email_subject": f"Your ride with {self._name()} on {order:%B} {order.day}",
        }
        return self.row("Lyft", user, email, **{**defaults, **fields})

    def uber_trip(
        self,
        user: int,
        day: date,
        miles: float,
        secs: int,
        *,
        email: datetime | None = None,
        **fields: str,
    ) -> Row:
        """An Uber ride: date-only order_time, email later (UTC, so maybe next day)."""
        rng = self.rng
        mins = max(1, round(secs / 60))
        fare = _money(_uber_fare(miles, mins))
        midnight = datetime.combine(day, datetime.min.time())
        email = email or midnight + timedelta(
            hours=rng.randint(14, 30),
            minutes=rng.randint(0, 59),
            seconds=rng.randint(0, 59),
        )
        subject = f"Your {day:%A} {rng.choice(_DAYPARTS)} trip with {self._name()}"
        defaults = {
            "order_number": self.uber_order_number(),
            "order_time": _ts(midnight),
            "order_total_amount": fare,
            "order_subtotal": fare,
            "item_price": fare,
            "product_description": f"UberX {miles:.2f} mi | {mins} min",
            "email_subject": subject,
            "from_domain": rng.choice(_US_UBER_SENDERS),
        }
        return self.row("Uber", user, email, **{**defaults, **fields})

    def typical_trip(self) -> tuple[float, int]:
        miles = round(self.rng.uniform(1.5, 11.0), 2)
        return miles, _secs_for(miles, self.rng)


def _baseline(g: _Generator) -> list[Row]:
    """Clean rides: each panelist rides at most once on any day."""
    rows: list[Row] = []
    for group in range(_N_GROUPS):
        day = _WINDOW_START + timedelta(days=group * 92 // _N_GROUPS)
        for user in range(_N_PANELISTS):
            miles, secs = g.typical_trip()
            if (user + group) % 2:
                order = datetime.combine(day, datetime.min.time()) + timedelta(
                    hours=g.rng.randint(7, 21), minutes=g.rng.randint(0, 59)
                )
                rows.append(g.lyft_trip(user, order, miles, secs))
            else:
                rows.append(g.uber_trip(user, day, miles, secs))
            # Fares scatter +-12% around the formula, so that robust fare statistics
            # have a realistic spread and no baseline ride looks like an outlier.
            fare = _money(
                float(rows[-1]["order_total_amount"]) * (1 + g.rng.uniform(-0.12, 0.12))
            )
            rows[-1].update(
                order_total_amount=fare, order_subtotal=fare, item_price=fare
            )
            # Now and then a small tax or promo, well inside the fare, so that
            # order_tax and order_discount aren't constant columns.
            k = len(rows) - 1
            if k % 40 == 7:
                rows[k]["order_tax"] = "0.85"
            if k % 37 == 5:
                rows[k]["order_discount"] = "1.00"
    return rows


# --- injections: each returns the rows of one block --------------------------------
# Panelists 8.. are reserved one block at a time (see _inject), so no injection
# shares a user or a day with the baseline or with another block.


def _duplicate_order_key(g: _Generator, user: int) -> list[Row]:
    first = g.uber_trip(user, date(2019, 3, 12), 4.2, _typical_secs(4.2))
    email = datetime.fromisoformat(first["email_time"]) + timedelta(hours=3)
    return [first, {**first, **g.stamps(email)}]


def _cross_panelist_duplicate(g: _Generator, user: int) -> list[Row]:
    first = g.uber_trip(user, date(2019, 3, 14), 3.4, _typical_secs(3.4))
    email = datetime.fromisoformat(first["email_time"]) + timedelta(hours=1)
    return [first, {**first, "user_id": _user_id(user + 1), **g.stamps(email)}]


def _ride_tip_update(g: _Generator, user: int) -> list[Row]:
    ride = g.uber_trip(user, date(2019, 3, 20), 5.6, _typical_secs(5.6))
    email = datetime.fromisoformat(ride["email_time"]) + timedelta(days=1)
    insert = datetime.fromisoformat(ride["insert_time"])  # kept: precedes the email
    update = {
        **ride,
        "order_total_amount": _money(float(ride["order_total_amount"]) + 3.50),
        "email_subject": "Your updated trip receipt",
        **g.stamps(email, insert),
    }
    return [ride, update]


def _near_duplicate(g: _Generator, user: int) -> list[Row]:
    day = date(2019, 3, 25)
    first = g.uber_trip(user, day, 2.9, _typical_secs(2.9))
    second = g.uber_trip(
        user,
        day,
        2.9,
        _typical_secs(2.9),
        order_time=first["order_time"],
        email_subject=first["email_subject"],
        from_domain=first["from_domain"],
    )
    return [first, second]


def _monthly_pass(g: _Generator, user: int) -> list[Row]:
    rows = []
    for month in (3, 4, 5):
        email = datetime(2019, month, 5, 14, 2, 11)
        rows.append(
            g.row(
                "Uber",
                user,
                email,
                order_total_amount="9.99",
                order_subtotal="9.99",
                item_price="9.99",
                order_pickup="0",
                product_description="Pass membership",
                product_subtitle="Monthly ride pass",
                email_subject="Your monthly pass receipt",
            )
        )
    return rows


def _driver_earnings(g: _Generator, user: int) -> list[Row]:
    return [
        g.row(
            "Lyft",
            user,
            datetime(2019, 4, 8, 15, 30, 5),
            order_total_amount="42.10",
            order_subtotal="42.10",
            item_price="42.10",
            order_pickup="4",
            product_description="Earnings",
            product_subtitle="Weekly payout",
            email_subject="You're getting paid: your earnings summary",
        )
    ]


def _food_orders(g: _Generator, user: int) -> list[Row]:
    rows = []
    for day, place, total in (
        (date(2019, 4, 2), "Café Example", "23.40"),
        (date(2019, 4, 9), "Example Noodle House", "17.85"),
    ):
        midnight = datetime.combine(day, datetime.min.time())
        rows.append(
            g.row(
                "Uber",
                user,
                midnight + timedelta(hours=26, minutes=12),
                order_number=g.uber_order_number(),
                order_time=_ts(midnight),
                order_total_amount=total,
                order_subtotal=total,
                item_price=total,
                product_description="Food order",
                product_reseller=place,
                email_subject=f"Your order with {place}",
            )
        )
    return rows


def _bike(g: _Generator, user: int) -> list[Row]:
    midnight = datetime(2019, 4, 4)
    return [
        g.row(
            "Uber",
            user,
            midnight + timedelta(hours=19, minutes=40),
            order_number=g.uber_order_number(),
            order_time=_ts(midnight),
            order_total_amount="-4.50",
            order_subtotal="-4.50",
            item_price="-4.50",
            product_description="Bike ride | 11 min",
            email_subject="Your bike ride receipt",
        )
    ]


def _freight(g: _Generator, user: int) -> list[Row]:
    midnight = datetime(2019, 4, 11)
    return [
        g.row(
            "Uber",
            user,
            midnight + timedelta(hours=21, minutes=5),
            order_number=f"UF-{g.rng.randint(10**11, 10**12 - 1)}",
            order_time=_ts(midnight),
            order_total_amount="120.00",
            order_subtotal="120.00",
            item_price="120.00",
            product_description="Freight shipment",
            from_domain="person.name@uber.com",
            email_subject="Your shipment receipt",
        )
    ]


def _non_us_sender(g: _Generator, user: int) -> list[Row]:
    return [
        g.uber_trip(
            user,
            date(2019, 4, 13),
            6.8,
            1260,
            order_total_amount="5718.00",
            order_subtotal="5718.00",
            item_price="5718.00",
            product_description="UberX 6.80 km | 21 min",
            from_domain="uber.colombia@uber.com",
            email_subject="Tu viaje del sábado por la mañana",
        )
    ]


def _km_generic_sender(g: _Generator, user: int) -> list[Row]:
    # 6.8 km is 4.23 mi: the fare follows the miles, so only the unit is odd.
    fare = _money(_uber_fare(4.23, 11))
    return [
        g.uber_trip(
            user,
            date(2019, 4, 15),
            4.23,
            _typical_secs(4.23),
            order_total_amount=fare,
            order_subtotal=fare,
            item_price=fare,
            product_description="UberX 6.80 km | 11 min",
            from_domain="noreply@uber.com",
        )
    ]


def _uber_date_only(g: _Generator, user: int) -> list[Row]:
    day = date(2019, 4, 16)
    email = datetime.combine(day, datetime.min.time()) + timedelta(hours=23, minutes=30)
    return [g.uber_trip(user, day, 3.3, _typical_secs(3.3), email=email)]


def _lyft_order_5h_before_email(g: _Generator, user: int) -> list[Row]:
    order = datetime(2019, 4, 17, 13, 20)
    return [
        g.lyft_trip(
            user, order, 4.1, _typical_secs(4.1), email=order + timedelta(hours=5)
        )
    ]


def _apac_next_day(g: _Generator, user: int) -> list[Row]:
    # Local (UTC+10) trip on the 12th, UTC email still on the 11th.
    return [
        g.uber_trip(
            user,
            date(2019, 4, 12),
            3.1,
            _typical_secs(3.1),
            email=datetime(2019, 4, 11, 14, 20, 9),
            from_domain="receipts.sydney@uber.com",
        )
    ]


def _ancient_order(g: _Generator, user: int) -> list[Row]:
    return [
        g.lyft_trip(
            user,
            datetime(2010, 4, 27, 9, 30),
            3.8,
            _typical_secs(3.8),
            email=datetime(2019, 4, 18, 14, 34, 7),
        )
    ]


def _discount_gt_subtotal(g: _Generator, user: int) -> list[Row]:
    # The total stays at the undiscounted fare so that only A4 fires.
    return [
        g.uber_trip(
            user, date(2019, 4, 19), 4.0, _typical_secs(4.0), order_discount="11.00"
        )
    ]


def _tax_gt_total(g: _Generator, user: int) -> list[Row]:
    return [
        g.uber_trip(user, date(2019, 4, 21), 1.0, _typical_secs(1.0), order_tax="9.25")
    ]


def _zero_fare_ride(g: _Generator, user: int) -> list[Row]:
    return [
        g.uber_trip(
            user,
            date(2019, 4, 22),
            2.4,
            _typical_secs(2.4),
            order_total_amount="0.00",
            order_subtotal="0.00",
            item_price="0.00",
            order_pickup="0",
        )
    ]


def _negative_total_ride(g: _Generator, user: int) -> list[Row]:
    # An ordinary ride whose total is negative (a refunded fare): the one A1 failure.
    return [
        g.uber_trip(
            user,
            date(2019, 4, 28),
            3.6,
            _typical_secs(3.6),
            order_total_amount="-5.75",
            order_subtotal="-5.75",
            item_price="-5.75",
        )
    ]


def _zero_duration_trip(g: _Generator, user: int) -> list[Row]:
    return [g.lyft_trip(user, datetime(2019, 4, 23, 11, 5), 2.0, 0)]


def _implausible_speed(g: _Generator, user: int) -> list[Row]:
    # The fare is the typical $/mile for 60 miles (160 min), so the row is a speed
    # outlier only, not a $/mile outlier.
    fare = _money(_uber_fare(60.0, 160))
    return [
        g.uber_trip(
            user,
            date(2019, 4, 24),
            60.0,
            1200,
            order_total_amount=fare,
            order_subtotal=fare,
            item_price=fare,
        )
    ]


def _fare_outlier_10x(g: _Generator, user: int) -> list[Row]:
    fare = _money(10 * _uber_fare(5.0, 13))
    return [
        g.uber_trip(
            user,
            date(2019, 4, 25),
            5.0,
            _typical_secs(5.0),
            order_total_amount=fare,
            order_subtotal=fare,
            item_price=fare,
        )
    ]


def _overlapping_rides(g: _Generator, user: int) -> list[Row]:
    # The first trip runs 22 min 40 s from 10:00; the second starts at 10:15.
    return [
        g.lyft_trip(user, datetime(2019, 4, 26, 10, 0), 8.5, _typical_secs(8.5)),
        g.lyft_trip(user, datetime(2019, 4, 26, 10, 15), 3.0, _typical_secs(3.0)),
    ]


def _many_rides_one_day(g: _Generator, user: int) -> list[Row]:
    rows = []
    for k in range(12):
        miles = round(1.5 + 0.4 * k, 2)  # distinct distances, so distinct totals
        rows.append(
            g.uber_trip(user, date(2019, 4, 10), miles, _secs_for(miles, g.rng))
        )
    return rows


def _masked_subjects(g: _Generator, user: int) -> list[Row]:
    # Last, and Uber, so the file ends on the merchant it starts with.
    return [
        g.lyft_trip(
            user,
            datetime(2019, 5, 2, 18, 45),
            2.2,
            _typical_secs(2.2),
            email_subject="Your ride with ***** on May 2",
        ),
        g.uber_trip(
            user + 1,
            date(2019, 5, 3),
            2.6,
            _typical_secs(2.6),
            email_subject="Your Friday evening trip with ******",
        ),
    ]


# (name, builder, panelists the block uses); panelists are numbered from 8, in order.
_BLOCKS: tuple[tuple[str, Callable[[_Generator, int], list[Row]], int], ...] = (
    ("duplicate_order_key", _duplicate_order_key, 1),
    ("cross_panelist_duplicate", _cross_panelist_duplicate, 2),
    ("ride_tip_update", _ride_tip_update, 1),
    ("near_duplicate", _near_duplicate, 1),
    ("monthly_pass", _monthly_pass, 1),
    ("driver_earnings", _driver_earnings, 1),
    ("food_orders", _food_orders, 1),
    ("bike", _bike, 1),
    ("freight", _freight, 1),
    ("non_us_sender", _non_us_sender, 1),
    ("km_generic_sender", _km_generic_sender, 1),
    ("uber_date_only", _uber_date_only, 1),
    ("lyft_order_5h_before_email", _lyft_order_5h_before_email, 1),
    ("apac_next_day", _apac_next_day, 1),
    ("ancient_order", _ancient_order, 1),
    ("discount_gt_subtotal", _discount_gt_subtotal, 1),
    ("tax_gt_total", _tax_gt_total, 1),
    ("zero_fare_ride", _zero_fare_ride, 1),
    ("negative_total_ride", _negative_total_ride, 1),
    ("zero_duration_trip", _zero_duration_trip, 1),
    ("implausible_speed", _implausible_speed, 1),
    ("fare_outlier_10x", _fare_outlier_10x, 1),
    ("overlapping_rides", _overlapping_rides, 1),
    ("many_rides_one_day", _many_rides_one_day, 1),
    ("masked_subjects", _masked_subjects, 2),
)


def build_rows(seed: int = SEED) -> tuple[list[Row], dict[str, tuple[int, ...]]]:
    """Return the raw text rows and, per injection block, the positions it produced."""
    g = _Generator(seed)
    rows = _baseline(g)
    injections: dict[str, tuple[int, ...]] = {}
    user = _N_PANELISTS
    for name, build, n_users in _BLOCKS:
        block = build(g, user)
        injections[name] = tuple(range(len(rows), len(rows) + len(block)))
        rows.extend(block)
        user += n_users
    for position, row in enumerate(rows):
        row["checksum"] = hashlib.md5(
            f"synthetic-row-{seed}-{position}".encode()
        ).hexdigest()
    return rows, injections


_DEFAULT_ROWS, INJECTIONS = build_rows()


def write_synthetic_csv(path: Path, seed: int = SEED) -> Path:
    """Write the synthetic CSV (UTF-8, CRLF) to `path`; same seed, same bytes."""
    rows = build_rows(seed)[0] if seed != SEED else _DEFAULT_ROWS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerows([row[col] for col in EXPECTED_COLUMNS] for row in rows)
    return path


def build_synthetic(seed: int = SEED) -> pl.DataFrame:
    """Return the synthetic receipts as the typed frame the checks run on."""
    rows = build_rows(seed)[0] if seed != SEED else _DEFAULT_ROWS
    raw = pl.DataFrame(
        [{col: row[col] or None for col in EXPECTED_COLUMNS} for row in rows],
        schema=raw_schema(),
    )
    return typed(raw.lazy()).collect()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python tests/fixtures/make_synthetic.py OUT.csv", file=sys.stderr)
        return 2
    path = write_synthetic_csv(Path(args[0]))
    print(f"wrote {path} ({len(_DEFAULT_ROWS)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
