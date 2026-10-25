"""The input CSV's columns, the dtype each is parsed to, and the timestamp formats."""

# PEP 810: polars is bound now but only imported when csv_schema() first runs, so
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


def csv_schema() -> pl.Schema:
    """Return the dtype each column is parsed to, in file order.

    polars matches the schema to the header by name and raises SchemaError if a
    column is missing or extra. This is a function rather than a module constant
    so that importing the module doesn't import polars; each call returns a new,
    independent Schema.
    """
    # Chosen by profiling the full sample (~866 MB, 1.88M rows). Every value
    # parses as its dtype; empty fields are null.
    ts = pl.Datetime("ms")  # the timestamps carry milliseconds
    return pl.Schema(
        {
            "merchant_name": pl.String(),
            "user_id": pl.String(),  # 64-hex-digit hash
            # mostly 19-digit integers (Lyft) or UUIDs (Uber), plus a few other
            # formats, so not numeric
            "order_number": pl.String(),
            "order_time": ts,
            "email_time": ts,
            "insert_time": ts,
            "update_time": ts,
            "order_total_amount": pl.Float64(),
            "order_points": pl.Float64(),
            "order_shipping": pl.Float64(),
            "order_tax": pl.Float64(),
            "order_subtotal": pl.Float64(),
            "order_total_qty": pl.Float64(),  # written as "1.0"
            "product_description": pl.String(),
            "product_subtitle": pl.String(),
            "item_quantity": pl.Float64(),  # written as "1.0"
            "item_price": pl.Float64(),
            "digital_transaction": pl.Int64(),  # always 0 in the sample
            "checksum": pl.String(),  # 32-hex-digit hash
            "product_reseller": pl.String(),
            "product_category": pl.String(),
            "order_discount": pl.Float64(),
            "SKU": pl.String(),
            "item_id": pl.String(),
            "order_pickup": pl.Int64(),  # a code: 0, 1, 3 or 4
            "from_domain": pl.String(),
            "email_subject": pl.String(),
            "delivery_date": ts,
            "start_source_folder_date": pl.Date(),
            "end_source_folder_date": pl.Date(),
            "file_id": pl.String(),
            "source_dttimestamp": ts,
            "dttimestamp": ts,
        }
    )
