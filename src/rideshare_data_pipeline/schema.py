"""The input CSV's columns, the dtype each is cast to, and the timestamp formats."""

# PEP 810: polars is bound now but only imported when a schema function first runs, so
# importing this module (e.g. for EXPECTED_COLUMNS) stays cheap.
lazy import polars as pl

# Every timestamp in the CSV looks like "2018-09-25 09:00:00.000"; the date-only
# columns look like "2020-02-18". These are chrono formats (polars), not strptime.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%.3f"
DATE_FORMAT = "%Y-%m-%d"

# The 33 header names, in file order. Plain strings, so they need no polars.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "merchant_name",
    "user_id",
    "order_number",
    "order_time",
    "email_time",
    "insert_time",
    "update_time",
    "order_total_amount",
    "order_points",
    "order_shipping",
    "order_tax",
    "order_subtotal",
    "order_total_qty",
    "product_description",
    "product_subtitle",
    "item_quantity",
    "item_price",
    "digital_transaction",
    "checksum",
    "product_reseller",
    "product_category",
    "order_discount",
    "SKU",
    "item_id",
    "order_pickup",
    "from_domain",
    "email_subject",
    "delivery_date",
    "start_source_folder_date",
    "end_source_folder_date",
    "file_id",
    "source_dttimestamp",
    "dttimestamp",
)


def raw_schema() -> pl.Schema:
    """Return the schema the CSV is read with: every column as String, in file order.

    Reading everything as text means no dtype is inferred and no value is rejected
    by the reader; typed() then casts, and cast_failures() counts what didn't
    convert. A function rather than a constant so that importing this module
    doesn't import polars; each call returns an independent Schema.
    """
    return pl.Schema(dict.fromkeys(EXPECTED_COLUMNS, pl.String()))


def casts() -> dict[str, pl.DataType]:
    """Return the dtype of each column that is cast away from String, in file order.

    The columns left out (names, hashes, IDs, free text) stay String. Chosen by
    profiling the full sample (~866 MB, 1.88M rows): every value converts, and
    empty fields are null.
    """
    ts = pl.Datetime("ms")  # the timestamps carry milliseconds
    money = pl.Float64()
    return {
        "order_time": ts,
        "email_time": ts,
        "insert_time": ts,
        "update_time": ts,
        "order_total_amount": money,
        "order_points": money,
        "order_shipping": money,
        "order_tax": money,
        "order_subtotal": money,
        # Written as "1.0", which an integer cast would reject.
        "order_total_qty": pl.Float64(),
        "item_quantity": pl.Float64(),
        "item_price": money,
        "digital_transaction": pl.Int8(),  # a 0/1 flag (always 0 in the sample)
        "order_discount": money,
        "order_pickup": pl.Int8(),  # a small code: 0, 1, 3 or 4
        "delivery_date": ts,
        "start_source_folder_date": pl.Date(),
        "end_source_folder_date": pl.Date(),
        "source_dttimestamp": ts,
        "dttimestamp": ts,
        # Not cast, deliberately: user_id, order_number (19-digit integers mixed
        # with UUIDs), checksum, SKU, item_id and file_id (all empty in the
        # sample) are identifiers, and an identifier is text.
    }


# Python's datetime and date cover the years 1 to 9999; polars parses year 0 and
# beyond 9999, which can't be converted to Python (min(), isoformat, row()).
_MIN_YEAR, _MAX_YEAR = 1, 9999


def cast_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    """Return an expression casting the String column to dtype; failures become null.

    Timestamps and dates use the explicit formats above (the CSV reader would only
    infer them). A value that doesn't convert is null, never an error, so that
    cast_failures() can count it; compare with the raw column to tell a failure
    from an empty field. Two kinds of value that do parse are failures too, so no
    later step has to cope with them: a Float64 that is not finite (polars' float
    parser accepts "nan", "inf" and "1e400", which overflows to inf), and a
    timestamp or date outside the years 1 to 9999 (year 0 can't be a Python date).
    """
    col = pl.col(column)
    if isinstance(dtype, pl.Datetime | pl.Date):
        if isinstance(dtype, pl.Datetime):
            parsed = col.str.to_datetime(
                TIMESTAMP_FORMAT, time_unit=dtype.time_unit, strict=False
            )
        else:
            parsed = col.str.to_date(DATE_FORMAT, strict=False)
        return pl.when(parsed.dt.year().is_between(_MIN_YEAR, _MAX_YEAR)).then(parsed)
    cast = col.cast(dtype, strict=False)
    if isinstance(dtype, pl.Float64):
        return pl.when(cast.is_finite()).then(cast)
    return cast


def csv_schema() -> pl.Schema:
    """Return the dtype of every column after typed(), in file order."""
    return pl.Schema({**raw_schema(), **casts()})
