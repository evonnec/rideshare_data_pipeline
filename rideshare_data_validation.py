#!/usr/bin/env python

import sys
import os
import csv
import pytest
import tarfile
from zipfile import ZipFile
from datetime import datetime

import modin.pandas as pdm
import pandas as pd
import numpy as np

#import seaborn as sns
#import matplotlib.pyplot as plt 
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

# The sample contains records of transactions with Uber and Lyft (as tagged by the vendor) 
# made by a subset of the panel’s active users.  An active user is defined as a user who has 
# made at least a couple of transaction every quarter for the last several years - not 
# necessarily Uber/Lyft transactions, but any transaction.  

# Using this data, please brainstorm and develop some quality control and 
# outlier detection tests. 

# Although under typical circumstances you would have the chance to discuss the data 
# with the vendor, we want to see how you would explore the data and infer the meaning 
# of the fields based on their contents.  We are leaving it up to you to examine the data, 
# determine which QC and outlier tests are most practical, and then execute against those 
# ideas in code.

# Please use Python and be prepared to share/discuss your thought process, assumptions, 
# and code.  Let us know if you have any questions or difficulty accessing the file.

# We look forward to reviewing your results. 

#lines 40-130 implmentation unused
class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        self.__dict__.update(opts)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

def typed(expected_type, cls=None):
    if cls is None:
        return lambda cls: typed(expected_type, cls)

    super_set = cls.__set__
    def __set__(self, instance, value):
        if not isinstance(value, expected_type):
            raise TypeError('expected ' + str(expected_type))
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

def unsigned(cls):
    super_set = cls.__set__
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Expected >= 0')
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

def maxsized(cls):
    super_init = cls.__init__
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        self.size = opts['size']
        super_init(self, name, **opts)
    cls.__init__ = __init__

    super_set = cls.__set__
    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

@typed(int)
class Integer(Descriptor):
    pass

@unsigned
class UnsignedInteger(Integer):
    pass

@typed(float)
class Float(Descriptor):
    pass

@unsigned
class UnsignedFloat(Float):
    pass

@typed(str)
class String(Descriptor):
    pass

@maxsized
class SizedString(String):
    pass

# Class decorator to apply constraints
def check_attributes(**kwargs):
    def decorate(cls):
        for key, value in kwargs.items():
            if isinstance(value, Descriptor):
                value.name = key
                setattr(cls, key, value)
            else:
                setattr(cls, key, value(key))
        return cls
    return decorate

# A metaclass that applies checking
class Checkedmeta(type):
    def __new__(cls, clsname, bases, methods):
        # Attach attribute names to the descriptors
        for key, value in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(cls, clsname, bases, methods)

# a class of file retrieval
class Retrieve:
    def extract_tar_to_local(tar_file, path):
        """ 
        extract tar file to local 
        """
        with tarfile.open(tar_file, "r:bz2") as tar_temp:
            tar_temp.extractall(path)

    def extract_zip_to_local(zip_file, path):
        """ 
        extract zip file to local 
        """
        with ZipFile(zip_file, 'r') as zip_temp:
            zip_temp.extractall(path)
    
    def get_fields(csv_file):
        """
        obtain fields from csv file as a list
        """
        with open(csv_file, mode='r') as data_temp:
            field_names = data_temp.readline()
            field_names = field_names.strip()

        fields = field_names.split(",")
        return fields
 
    def readin_df_from_csv(in_csv_file):
        """
        take in a csv and make a modin pandas dataframe from it
        """
        data_df = pdm.read_csv(in_csv_file, low_memory=False)
        return data_df

    def import_csv_clean(clean_csv_file):
        """
        take in a csv and make a pandas dataframe from it
        """
        clean_df = pd.read_csv(clean_csv_file, low_memory=False)
        return clean_df

    def __init__(self, name, file, path):
        self.name = name
        self.file = file
        self.path = path

class IdConstraint:
    """
    Identify the constraints, return issues found to be passed to Wrangle class
    """
    def determine_null_cols(df):
        """
        determine which columns where all the values are null
        """
        list_of_null_cols = [col for col in df if df[col].isnull().sum() == df[col].shape]
        return list_of_null_cols

    def determine_sameval_cols(df):
        """
        determine which columns where all the values are the same
        """
        sameval_cols = []
        for col in df:
            i = df[col].to_numpy()
            if i[0] == i[1:].all():
                sameval_cols.append(col)
        print(sameval_cols)
        return sameval_cols

    def has_consistent_dtype(col):
        """
        not implemented
        """
        return _

    def col_is_complete(col):
        """
        not implemented
        """
        return _

class Wrangle:
    """
    Change the dataset using inputs from class IdConstraint
    """
    def drop_cols(df, list_of_cols):
        """
        drop columns from df in which all the values are the same or null
        """  
        df.drop(columns=list_of_cols, axis=1, inplace=True)
        return df

    def drop_rows(df, rows):
        """
        not implemented i.e. if entire row isempty, drop the row
        """
        # df.drop(rows=rows, axis=0, inplace=True)
        return _

    def assign_dtype_to_col(list_of_cols):
        """
        for columns with dates, text, totals -- assign a data type
        """
        return _

    def max_min_is_inrange(col):
        """
        range that would be acceptable for columns with numeric type
        """
        return _


class Testdata:
    """
    Test validity of dataset to be used
    """
    def test_check_for_duplicates(df): 
        """
        check for sets of data that appear to be duplicates
        """
        if len(df) - len(df.drop_duplicates()) > 1e-5:
            raise ValueError 
        assert len(df) - len(df.drop_duplicates()) < 1e-5

    def test_col_null_values(df):
        """
        test for all null values in a df
        """
        assert IdConstraint.determine_null_cols(df) == []

    def test_col_values_same(df):
        """
        test values in col are not all the same value
        """
        assert IdConstraint.determine_sameval_cols(df) == []

    def test_vals_of_all_df_cols_same(df):
        """
        determine elements per col are not all same
        if this is run before Wrangle, this should assert False, 
        i.e. columns 'product_category', 'digital_transaction', 'item_quantity', and 'order_total_qty'
        these cols return the same value for each record per col. drop cols to be done.
        """
        a = df.to_numpy() # df.values (pandas<0.24)
        arr = a[0] == a[1:].all(0)
        for ele in arr:
            if ele == False:
                continue
            elif ele == True:
                assert False
        assert True

    def test_vals_of_one_col_same(s):
        """
        s should be a df['col'] -- could use pandas series is_unique
        test individual columns
        """
        a = s.to_numpy() # s.values (pandas<0.24)
        assert a[0] != a[1:].all()

    def test_val_in_col1_is_less_than_col2(col1, col2):
        """
        not implemented i.e. taxes less than order total
        """
        return _
        
    def test_correlation_between_cols(col1, col2):
        """
        not implemented i.e. subtotal correlated to order total
        """
        # df.corr()
        assert True 

    def test_data_types_per_col():
        """
        not implemented i.e. test that data types of records in col are same
        """
        with pytest.raises(ValueError): 
            main() #some function to test data type consistency in cols

    def test_missing_rows():
        """ 
        not implemented - For NumPy or Pandas, use np.NaN; For python, use None
        """
        assert True

    def test_performance():
        start = datetime.now()
        # run main function here
        end = datetime.now()
        diff = end - start

        minutes = (diff.total_seconds() % 3600) // 60
        seconds = diff.total_seconds() % 60

        start_str = start.strftime("%H:%M:%S.%f")
        end_str = end.strftime("%H:%M:%S.%f")
        
        print("minutes: {} , seconds: {} , start time: {} , end time: {}".format(minutes, seconds, start_str, end_str))

class Output:
    """
    to-do: combine formats into one function, pass arguments to type
    """
    def write_df_to_csv(df, out_csv_file):
        df.to_csv(out_csv_file)

    def write_df_to_feather(df, out_feather_file):
        df.to_feather(out_feather_file) #TODO: change to parquet

def main():
    tar_file = "Data_Rideshare.tar.bz2"
    #zip_file = "Data_Rideshare.zip"
    path = "extract/"
    csv_file = "Data_Rideshare.csv"
    feather_file = "Data_Rideshare.feather" #TODO: change to parquet

    new_folder = datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
    Retrieve.extract_tar_to_local(tar_file, path + new_folder + "/")
    #Retrieve.extract_zip_to_local(zip_file, path + new_folder + "/")

    #df_csv = Retrieve.readin_df_from_csv(path + new_folder + "/" + csv_file)
    df = Retrieve.import_csv_clean(path + new_folder + "/" + csv_file)

    #identify problem columns, wrangle dataset, test
    cols_to_drop = IdConstraint.determine_null_cols(df)
    Wrangle.drop_cols(df, cols_to_drop)
    Testdata.test_col_null_values(df)
    #identify problem columns, wrangle dataset, test
    cols_to_drop2 = IdConstraint.determine_sameval_cols(df)
    Wrangle.drop_cols(df, cols_to_drop2)
    Testdata.test_vals_of_all_df_cols_same(df)
    
    #some more tests
    Testdata.test_col_values_same(df)
    Testdata.test_check_for_duplicates(df)
    #pytest.main() #ideally run pytest on all tests instead of one by one

    Output.write_df_to_feather(df, feather_file) #TODO: change to parquet
    

if __name__ == "__main__":
    main()