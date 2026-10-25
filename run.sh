#!/usr/bin/env bash
set -euo pipefail

# Check that data CSVs are valid UTF-8 (requires `isutf8` from moreutils).
# Virtual environments are skipped: installed packages ship their own CSVs.
find . \( -path ./.venv -o -path ./rideshare \) -prune -o -name '*.csv' -print0 \
    | xargs -0 -r isutf8

uv run pytest rideshare_data_validation.py::Testdata::test_check_for_duplicates
