#!/bin/bash

find . -name '*.csv' | xargs isutf8

pytest rideshare_data_validation.py::Testdata::test_check_for_duplicates