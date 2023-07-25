#!/bin/bash

find . -name '*.csv' | xargs isutf8

pytest eminence.py::Testdata::test_check_for_duplicates