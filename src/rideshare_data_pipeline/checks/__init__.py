"""QC checks: pure functions of a polars LazyFrame, grouped by category.

A check runs a query and returns what it found. It never prints, writes or
changes the data; acting on a finding (dropping a column, say) is up to the caller.
"""
