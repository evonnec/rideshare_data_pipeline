"""The column list, the typed schema and the timestamp formats."""

import datetime as dt
import importlib.metadata

import polars as pl

import rideshare_data_pipeline
from rideshare_data_pipeline.schema import (
    DATE_FORMAT,
    EXPECTED_COLUMNS,
    TIMESTAMP_FORMAT,
    csv_schema,
)


def test_expected_columns_are_the_33_unique_header_names() -> None:
    assert len(EXPECTED_COLUMNS) == 33
    assert len(set(EXPECTED_COLUMNS)) == 33


def test_schema_covers_expected_columns_in_file_order() -> None:
    assert tuple(csv_schema().names()) == EXPECTED_COLUMNS


def test_schema_dtypes_by_kind() -> None:
    schema = csv_schema()

    def columns_of(dtype: pl.DataType) -> set[str]:
        return {col for col, d in schema.items() if d == dtype}

    assert columns_of(pl.Datetime("ms")) == {
        "order_time",
        "email_time",
        "insert_time",
        "update_time",
        "delivery_date",
        "source_dttimestamp",
        "dttimestamp",
    }
    assert columns_of(pl.Date()) == {
        "start_source_folder_date",
        "end_source_folder_date",
    }
    assert columns_of(pl.Int64()) == {"digital_transaction", "order_pickup"}
    # IDs and hashes stay text: order numbers mix 19-digit integers and UUIDs.
    ids = {"user_id", "order_number", "checksum", "SKU", "item_id", "file_id"}
    assert ids <= columns_of(pl.String())
    assert "order_total_amount" in columns_of(pl.Float64())


def test_csv_schema_returns_an_independent_copy() -> None:
    schema = csv_schema()
    schema["extra"] = pl.String()
    assert "extra" not in csv_schema()


def test_timestamp_format_parses_millisecond_timestamps() -> None:
    parsed = pl.Series(["2018-09-25 09:00:00.123"]).str.to_datetime(
        TIMESTAMP_FORMAT, time_unit="ms"
    )
    assert parsed.item() == dt.datetime(2018, 9, 25, 9, 0, 0, 123_000)


def test_date_format_parses_dates() -> None:
    parsed = pl.Series(["2020-02-18"]).str.to_date(DATE_FORMAT)
    assert parsed.item() == dt.date(2020, 2, 18)


def test_version_comes_from_package_metadata() -> None:
    expected = importlib.metadata.version("rideshare-data-pipeline")
    assert rideshare_data_pipeline.__version__ == expected
