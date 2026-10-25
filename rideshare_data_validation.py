#!/usr/bin/env python

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Never

# PEP 810 lazy imports: each module below is bound now but only imported on first
# use. polars and pytest are the heavy ones (pytest is only needed by
# Testdata.test_data_types_per_col); tarfile and zipfile are each used by a single
# extraction helper. Annotations are deferred (PEP 649), so the pl.DataFrame /
# pl.Series hints in the signatures below never force the polars import either.
lazy import tarfile
lazy from zipfile import ZipFile

lazy import polars as pl
lazy import pytest

# The sample contains records of transactions with Uber and Lyft (as tagged by the
# vendor) made by a subset of the panel's active users.  An active user is defined as
# a user who has made at least a couple of transaction every quarter for the last
# several years - not necessarily Uber/Lyft transactions, but any transaction.

# Using this data, please brainstorm and develop some quality control and
# outlier detection tests.

# Although under typical circumstances you would have the chance to discuss the data
# with the vendor, we want to see how you would explore the data and infer the meaning
# of the fields based on their contents.  We are leaving it up to you to examine the
# data, determine which QC and outlier tests are most practical, and then execute
# against those ideas in code.

# Please use Python and be prepared to share/discuss your thought process,
# assumptions, and code.  Let us know if you have any questions or difficulty
# accessing the file.

# We look forward to reviewing your results.

# PEP 695 aliases for the shapes that recur across the signatures below. Their values
# are evaluated lazily, so `Column` doesn't import polars when this module is
# imported (an eager `Column: TypeAlias = pl.Series` would reify the lazy import).
type StrPath = str | os.PathLike[str]
type Labels = list[str]  # column names
type Column = pl.Series  # a single column, e.g. df["col"]

# Every timestamp in the CSV looks like "2018-09-25 09:00:00.000"; the date-only
# columns look like "2020-02-18".
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%.3f"
DATE_FORMAT = "%Y-%m-%d"


# a class of file retrieval
class Retrieve:
    @staticmethod
    def extract_tar_to_local(tar_file: StrPath, path: StrPath) -> None:
        """
        extract tar file to local
        """
        with tarfile.open(tar_file, "r:bz2") as tar_temp:
            tar_temp.extractall(path, filter="data")

    @staticmethod
    def extract_zip_to_local(zip_file: StrPath, path: StrPath) -> None:
        """
        extract zip file to local
        """
        with ZipFile(zip_file, "r") as zip_temp:
            zip_temp.extractall(path)

    @staticmethod
    def get_fields(csv_file: StrPath) -> list[str]:
        """
        obtain fields from csv file as a list
        """
        with Path(csv_file).open(encoding="utf-8") as data_temp:
            field_names = data_temp.readline()
            field_names = field_names.strip()

        fields = field_names.split(",")
        return fields

    @staticmethod
    def csv_schema() -> pl.Schema:
        """
        the CSV's 33 columns, in file order, with the dtype each is parsed to.
        polars matches them to the header by name and raises SchemaError if a
        column is missing or extra. (A function rather than a module constant,
        so that importing this module doesn't import polars.)
        """
        # Chosen by profiling the full sample (~866 MB, 1.88M rows). Every value
        # parses as its dtype; empty fields are null.
        ts = pl.Datetime("ms")  # the timestamps carry milliseconds
        return pl.Schema(
            {
                "merchant_name": pl.String(),
                "user_id": pl.String(),  # 64-hex-digit hash
                # mostly 19-digit integers (Lyft) or UUIDs (Uber), plus a few
                # other formats, so not numeric
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

    @staticmethod
    def _parse_text(col: str, dtype: pl.DataType, *, strict: bool) -> pl.Expr:
        """
        expression converting the text column `col` to `dtype`, using an
        explicit format for timestamps and dates; with strict=False a value
        that doesn't convert becomes null
        """
        if isinstance(dtype, pl.Datetime):
            return pl.col(col).str.to_datetime(
                TIMESTAMP_FORMAT, time_unit=dtype.time_unit, strict=strict
            )
        if isinstance(dtype, pl.Date):
            return pl.col(col).str.to_date(DATE_FORMAT, strict=strict)
        return pl.col(col).cast(dtype, strict=strict)

    @staticmethod
    def readin_df_from_csv(in_csv_file: StrPath) -> pl.LazyFrame:
        """
        take in a csv and make a polars LazyFrame from it (nothing is read
        until it is collected), with the dtypes from csv_schema(); a value that
        doesn't parse makes the collect raise rather than becoming null
        """
        schema = Retrieve.csv_schema()
        # The CSV reader only infers datetime patterns, so timestamps and dates
        # are read as text and then parsed with an explicit format.
        as_text = {
            col: pl.String() if dtype.is_temporal() else dtype
            for col, dtype in schema.items()
        }
        return pl.scan_csv(Path(in_csv_file), schema=as_text).with_columns(
            Retrieve._parse_text(col, dtype, strict=True)
            for col, dtype in schema.items()
            if dtype.is_temporal()
        )

    @staticmethod
    def count_parse_failures(csv_file: StrPath) -> dict[str, int]:
        """
        for each non-text column in csv_schema(), count the values that are
        present but don't parse as its dtype; columns with none are left out
        """
        schema = Retrieve.csv_schema()
        as_text = pl.scan_csv(Path(csv_file), schema=dict.fromkeys(schema, pl.String()))
        counts = as_text.select(
            (
                pl.col(col).is_not_null()
                & Retrieve._parse_text(col, dtype, strict=False).is_null()
            )
            .sum()
            .alias(col)
            for col, dtype in schema.items()
            if dtype != pl.String
        ).collect(engine="streaming")
        return {col: n for col, n in counts.row(0, named=True).items() if n}

    @staticmethod
    def import_csv_clean(clean_csv_file: StrPath) -> pl.DataFrame:
        """
        take in a csv and make a polars dataframe from it, in one pass of the
        streaming engine; raises ValueError with per-column counts if any value
        doesn't parse as its column's dtype
        """
        try:
            return Retrieve.readin_df_from_csv(clean_csv_file).collect(
                engine="streaming"
            )
        except (
            pl.exceptions.ComputeError,
            pl.exceptions.InvalidOperationError,
        ) as err:
            # The error stops at the first bad batch, so it can't say how many
            # values are bad; a second pass, taken only on failure, counts them.
            failures = Retrieve.count_parse_failures(clean_csv_file)
            if not failures:
                raise
            msg = f"values that don't parse as their column's dtype: {failures}"
            raise ValueError(msg) from err

    def __init__(self, name: str, file: StrPath, path: StrPath) -> None:
        self.name = name
        self.file = file
        self.path = path


def _constant_cols(df: pl.DataFrame) -> Labels:
    """
    columns where all the values are the same, found in one aggregation pass
    """
    # Constant means one distinct value and no nulls. So an all-null column is
    # not constant (it belongs to determine_null_cols only), and neither is
    # [x, null, x]: main() drops constant columns, and a column that holds one
    # value or nothing is a presence flag whose null pattern is the information.
    # (Counting only the non-null values would drop it.)
    is_constant = df.select((pl.all().null_count() == 0) & (pl.all().n_unique() == 1))
    return [s.name for s in is_constant.iter_columns() if s.item()]


class IdConstraint:
    """
    Identify the constraints, return issues found to be passed to Wrangle class
    """

    @staticmethod
    def determine_null_cols(df: pl.DataFrame) -> Labels:
        """
        determine which columns where all the values are null
        """
        # null_count() aggregates every column in a single pass.
        null_counts = df.null_count().row(0, named=True)
        return [col for col, n in null_counts.items() if n == df.height]

    @staticmethod
    def determine_sameval_cols(df: pl.DataFrame) -> Labels:
        """
        determine which columns where all the values are the same
        """
        sameval_cols = _constant_cols(df)
        print(sameval_cols)
        return sameval_cols

    @staticmethod
    def has_consistent_dtype(col: Column) -> Never:
        """
        not implemented
        """
        raise NotImplementedError("has_consistent_dtype is not implemented")

    @staticmethod
    def col_is_complete(col: Column) -> Never:
        """
        not implemented
        """
        raise NotImplementedError("col_is_complete is not implemented")


class Wrangle:
    """
    Change the dataset using inputs from class IdConstraint
    """

    @staticmethod
    def drop_cols(df: pl.DataFrame, list_of_cols: Labels) -> pl.DataFrame:
        """
        drop columns from df in which all the values are the same or null;
        returns a new frame (callers must use the return value)
        """
        return df.drop(list_of_cols)

    @staticmethod
    def drop_rows(df: pl.DataFrame, rows: Labels) -> Never:
        """
        not implemented i.e. if entire row isempty, drop the row
        """
        # df.drop(rows=rows, axis=0, inplace=True)
        raise NotImplementedError("drop_rows is not implemented")

    @staticmethod
    def assign_dtype_to_col(list_of_cols: Labels) -> Never:
        """
        for columns with dates, text, totals -- assign a data type
        """
        raise NotImplementedError("assign_dtype_to_col is not implemented")

    @staticmethod
    def max_min_is_inrange(col: Column) -> Never:
        """
        range that would be acceptable for columns with numeric type
        """
        raise NotImplementedError("max_min_is_inrange is not implemented")


class Testdata:
    """
    Test validity of dataset to be used
    """

    @staticmethod
    def test_check_for_duplicates(df: pl.DataFrame) -> None:
        """
        check for sets of data that appear to be duplicates
        """
        # Counts the copies beyond the first of each repeated row (pandas'
        # `duplicated().sum()`); nulls compare equal. Identical rows hash alike,
        # so only rows whose 64-bit row hash repeats can be duplicates; comparing
        # just those exactly avoids building a full-width key for every row.
        candidates = df.filter(df.hash_rows().is_duplicated())
        n_duplicates = candidates.height - candidates.n_unique()
        if n_duplicates > 0:
            raise ValueError(f"found {n_duplicates} duplicate rows")

    @staticmethod
    def test_col_null_values(df: pl.DataFrame) -> None:
        """
        test for all null values in a df
        """
        assert IdConstraint.determine_null_cols(df) == []

    @staticmethod
    def test_col_values_same(df: pl.DataFrame) -> None:
        """
        test values in col are not all the same value
        """
        assert IdConstraint.determine_sameval_cols(df) == []

    @staticmethod
    def test_vals_of_all_df_cols_same(df: pl.DataFrame) -> None:
        """
        determine elements per col are not all same
        if this is run before Wrangle, this should assert False,
        i.e. columns 'product_category', 'digital_transaction', 'item_quantity',
        and 'order_total_qty'
        these cols return the same value for each record per col. drop cols to be
        done.
        """
        sameval_cols = _constant_cols(df)
        assert not sameval_cols, f"every value is the same in {sameval_cols}"

    @staticmethod
    def test_vals_of_one_col_same(s: Column) -> None:
        """
        s should be a df['col']
        test individual columns
        """
        assert not _constant_cols(s.to_frame()), (
            f"every value in {s.name!r} is the same"
        )

    @staticmethod
    def test_val_in_col1_is_less_than_col2(col1: Column, col2: Column) -> Never:
        """
        not implemented i.e. taxes less than order total
        """
        raise NotImplementedError(
            "test_val_in_col1_is_less_than_col2 is not implemented"
        )

    @staticmethod
    def test_correlation_between_cols(col1: Column, col2: Column) -> None:
        """
        not implemented i.e. subtotal correlated to order total
        """
        # df.corr()
        assert True

    @staticmethod
    def test_data_types_per_col() -> None:
        """
        not implemented i.e. test that data types of records in col are same
        """
        with pytest.raises(ValueError):
            main()  # some function to test data type consistency in cols

    @staticmethod
    def test_missing_rows() -> None:
        """
        not implemented - polars marks a missing value as null (None in Python)
        """
        assert True

    @staticmethod
    def test_performance() -> None:
        start = datetime.now()
        start_counter = time.perf_counter()
        # run main function here
        elapsed = time.perf_counter() - start_counter
        end = datetime.now()

        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        start_str = start.strftime("%H:%M:%S.%f")
        end_str = end.strftime("%H:%M:%S.%f")

        print(
            f"minutes: {minutes} , seconds: {seconds} , "
            f"start time: {start_str} , end time: {end_str}"
        )


class Output:
    """
    to-do: combine formats into one function, pass arguments to type
    """

    @staticmethod
    def write_df_to_csv(df: pl.DataFrame, out_csv_file: StrPath) -> None:
        # Timestamps go out in the format they came in (polars' default is ISO
        # 8601 with a "T"). polars has no index, so there is no index column.
        df.write_csv(Path(out_csv_file), datetime_format=TIMESTAMP_FORMAT)

    @staticmethod
    def write_df_to_feather(df: pl.DataFrame, out_feather_file: StrPath) -> None:
        # Feather v2 is the Arrow IPC file format; LZ4 is pandas' default codec.
        df.write_ipc(Path(out_feather_file), compression="lz4")

    @staticmethod
    def write_df_to_parquet(df: pl.DataFrame, out_parquet_file: StrPath) -> None:
        df.write_parquet(Path(out_parquet_file), compression="zstd")


def main() -> None:
    tar_file = Path("Data_Rideshare.tar.bz2")
    # zip_file = Path("Data_Rideshare.zip")
    path = Path("extract")
    csv_file = "Data_Rideshare.csv"
    parquet_file = Path("Data_Rideshare.parquet")

    new_folder = datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
    extract_dir = path / new_folder
    Retrieve.extract_tar_to_local(tar_file, extract_dir)
    # Retrieve.extract_zip_to_local(zip_file, extract_dir)

    # lf_csv = Retrieve.readin_df_from_csv(extract_dir / csv_file)
    df = Retrieve.import_csv_clean(extract_dir / csv_file)

    # identify problem columns, wrangle dataset, test
    cols_to_drop = IdConstraint.determine_null_cols(df)
    df = Wrangle.drop_cols(df, cols_to_drop)
    Testdata.test_col_null_values(df)
    # identify problem columns, wrangle dataset, test
    cols_to_drop2 = IdConstraint.determine_sameval_cols(df)
    df = Wrangle.drop_cols(df, cols_to_drop2)
    Testdata.test_vals_of_all_df_cols_same(df)

    # some more tests
    Testdata.test_col_values_same(df)
    Testdata.test_check_for_duplicates(df)
    # pytest.main() #ideally run pytest on all tests instead of one by one

    Output.write_df_to_parquet(df, parquet_file)


if __name__ == "__main__":
    main()
