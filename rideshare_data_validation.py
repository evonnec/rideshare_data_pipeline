#!/usr/bin/env python

import os
import time
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any, Never

# PEP 810 lazy imports: each module below is bound now but only imported on first
# use. pandas, numpy and pytest are the heavy ones (pytest is only needed by
# Testdata.test_data_types_per_col); tarfile and zipfile are each used by a single
# extraction helper. Annotations are deferred (PEP 649), so the pd.DataFrame /
# pd.Series hints in the signatures below never force the pandas import either.
lazy import tarfile
lazy from zipfile import ZipFile

lazy import numpy as np
lazy import pandas as pd
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
# are evaluated lazily, which `Column` relies on: pd.Series is generic only in
# pandas-stubs (the runtime class is not subscriptable), so an eager
# `Column: TypeAlias = pd.Series[Any]` would raise TypeError at import time.
type StrPath = str | os.PathLike[str]
type Labels = list[Hashable]  # row or column labels
type Column = pd.Series[Any]  # a single column, e.g. df["col"]


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
    def readin_df_from_csv(in_csv_file: StrPath) -> pd.DataFrame:
        """
        take in a csv and make a pandas dataframe from it
        """
        data_df = pd.read_csv(in_csv_file, low_memory=False)
        return data_df

    @staticmethod
    def import_csv_clean(clean_csv_file: StrPath) -> pd.DataFrame:
        """
        take in a csv and make a pandas dataframe from it
        """
        clean_df = pd.read_csv(clean_csv_file, low_memory=False)
        return clean_df

    def __init__(self, name: str, file: StrPath, path: StrPath) -> None:
        self.name = name
        self.file = file
        self.path = path


class IdConstraint:
    """
    Identify the constraints, return issues found to be passed to Wrangle class
    """

    @staticmethod
    def determine_null_cols(df: pd.DataFrame) -> Labels:
        """
        determine which columns where all the values are null
        """
        # The original compared the count against the shape tuple `(n,)`, which
        # only worked because NumPy broadcasts `np.int64 == tuple` to a 1-element
        # bool array; comparing against shape[0] gives the same truth value.
        list_of_null_cols = [
            col for col in df if df[col].isnull().sum() == df[col].shape[0]
        ]
        return list_of_null_cols

    @staticmethod
    def determine_sameval_cols(df: pd.DataFrame) -> Labels:
        """
        determine which columns where all the values are the same
        """
        sameval_cols = []
        for col in df:
            i = df[col].to_numpy()
            # NumPy 2 made ndarray.all() return a bool for object arrays; the
            # original NumPy 1 behaviour (Python `and` semantics) is
            # np.logical_and.reduce, which is identical for non-object arrays.
            if i[0] == np.logical_and.reduce(i[1:]):
                sameval_cols.append(col)
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
    def drop_cols(df: pd.DataFrame, list_of_cols: Labels) -> pd.DataFrame:
        """
        drop columns from df in which all the values are the same or null;
        returns a new frame (callers must use the return value)
        """
        # pandas 3.1 deprecates inplace=True here (PDEP-8) and rejects `axis`
        # alongside `columns`; pandas 1.5 ignored `axis` when `columns` was given.
        return df.drop(columns=list_of_cols)

    @staticmethod
    def drop_rows(df: pd.DataFrame, rows: Labels) -> Never:
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
    def test_check_for_duplicates(df: pd.DataFrame) -> None:
        """
        check for sets of data that appear to be duplicates
        """
        if len(df) - len(df.drop_duplicates()) > 1e-5:
            raise ValueError
        assert len(df) - len(df.drop_duplicates()) < 1e-5

    @staticmethod
    def test_col_null_values(df: pd.DataFrame) -> None:
        """
        test for all null values in a df
        """
        assert IdConstraint.determine_null_cols(df) == []

    @staticmethod
    def test_col_values_same(df: pd.DataFrame) -> None:
        """
        test values in col are not all the same value
        """
        assert IdConstraint.determine_sameval_cols(df) == []

    @staticmethod
    def test_vals_of_all_df_cols_same(df: pd.DataFrame) -> None:
        """
        determine elements per col are not all same
        if this is run before Wrangle, this should assert False,
        i.e. columns 'product_category', 'digital_transaction', 'item_quantity',
        and 'order_total_qty'
        these cols return the same value for each record per col. drop cols to be
        done.
        """
        a = df.to_numpy()  # df.values (pandas<0.24)
        # np.logical_and.reduce keeps the NumPy 1 ndarray.all() semantics for
        # object arrays (see IdConstraint.determine_sameval_cols).
        arr = a[0] == np.logical_and.reduce(a[1:], axis=0)
        for ele in arr:
            assert not ele
        assert True

    @staticmethod
    def test_vals_of_one_col_same(s: Column) -> None:
        """
        s should be a df['col'] -- could use pandas series is_unique
        test individual columns
        """
        a = s.to_numpy()  # s.values (pandas<0.24)
        assert a[0] != np.logical_and.reduce(a[1:])

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
        not implemented - For NumPy or Pandas, use np.NaN; For python, use None
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
    def write_df_to_csv(df: pd.DataFrame, out_csv_file: StrPath) -> None:
        df.to_csv(out_csv_file)

    @staticmethod
    def write_df_to_feather(df: pd.DataFrame, out_feather_file: StrPath) -> None:
        df.to_feather(out_feather_file)  # TODO: change to parquet


def main() -> None:
    tar_file = Path("Data_Rideshare.tar.bz2")
    # zip_file = Path("Data_Rideshare.zip")
    path = Path("extract")
    csv_file = "Data_Rideshare.csv"
    feather_file = Path("Data_Rideshare.feather")  # TODO: change to parquet

    new_folder = datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
    extract_dir = path / new_folder
    Retrieve.extract_tar_to_local(tar_file, extract_dir)
    # Retrieve.extract_zip_to_local(zip_file, extract_dir)

    # df_csv = Retrieve.readin_df_from_csv(extract_dir / csv_file)
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

    Output.write_df_to_feather(df, feather_file)  # TODO: change to parquet


if __name__ == "__main__":
    main()
