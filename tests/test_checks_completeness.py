"""Structural and completeness checks (#16): S1, S3, C1, S6, C2, row-count band."""

import dataclasses
from typing import Any

import polars as pl
import pytest

from rideshare_data_pipeline.checks import run_checks
from rideshare_data_pipeline.checks.model import Severity
from rideshare_data_pipeline.checks.structural import (
    REQUIRED_COLUMNS,
    RIDE_NULL_LIMITS,
    all_null_columns_check,
    column_profile_checks,
    constant_columns_check,
    header_valid_check,
    ride_completeness_check,
    row_count_band_check,
    rules,
    utf8_valid_check,
)
from rideshare_data_pipeline.config import ConfigError, QCConfig, parse_config
from rideshare_data_pipeline.io import FileFacts
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

USER = "a" * 64
SUM = "b" * 32
UUID = "0b1d2c3e-0000-4000-8000-000000000001"
LYFT_ORDER = "1" * 19


def facts(**changes: Any) -> FileFacts:
    base = FileFacts(
        size_bytes=10,
        sha256="0" * 64,
        has_bom=False,
        line_ending="lf",
        header=EXPECTED_COLUMNS,
    )
    return dataclasses.replace(base, **changes)


def row(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "merchant_name": "Uber",
        "user_id": USER,
        "order_number": UUID,
        "email_time": "t",
        "checksum": SUM,
        "from_domain": "uber.com",
        "email_subject": "s",
    }
    return base | changes


def frame(*rows: dict[str, Any]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {k: [r[k] for r in rows] for k in rows[0]},
        schema=dict.fromkeys(rows[0], pl.String()),
    )


def rule_result(name: str, *rows: dict[str, Any]) -> Any:
    chosen = [r for r in rules() if r.name == name]
    (res,) = run_checks(frame(*rows), rules=chosen, dataset_checks=[])
    return res


# --- S1 and S3 ----------------------------------------------------------------------


def test_header_valid_passes_with_the_expected_header() -> None:
    res = header_valid_check(facts())
    assert (res.check_id, res.severity, res.passed) == ("S1", Severity.ERROR, True)
    assert res.details == {"columns": 33, "in_order": True}
    assert (res.metric, res.threshold) == (0.0, 0.0)


@pytest.mark.parametrize(
    "header",
    [EXPECTED_COLUMNS[:-1], tuple(reversed(EXPECTED_COLUMNS))],
    ids=["missing-column", "out-of-order"],
)
def test_header_valid_fails_on_a_different_header(header: tuple[str, ...]) -> None:
    res = header_valid_check(facts(header=header))
    assert (res.passed, res.n_failed, res.metric) == (False, 1, 1.0)
    assert res.details["in_order"] is False


@pytest.mark.parametrize(
    ("has_bom", "line_ending"), [(False, "lf"), (True, "crlf"), (False, "mixed")]
)
def test_utf8_valid_records_bom_and_line_ending(
    has_bom: bool, line_ending: str
) -> None:
    res = utf8_valid_check(facts(has_bom=has_bom, line_ending=line_ending))
    assert (res.check_id, res.severity, res.passed) == ("S3", Severity.ERROR, True)
    assert res.details == {"has_bom": has_bom, "line_ending": line_ending}


# --- C1 ---------------------------------------------------------------------------


def test_required_fields_passes_when_nothing_is_null() -> None:
    res = rule_result("required_fields", row(), row())
    assert (res.check_id, res.severity) == ("C1", Severity.ERROR)
    assert (res.passed, res.n_failed, res.n_evaluated) == (True, 0, 2)
    assert res.details["by_field"] == dict.fromkeys(REQUIRED_COLUMNS, 0)


def test_required_fields_counts_rows_and_per_column_nulls() -> None:
    res = rule_result(
        "required_fields",
        row(),
        row(user_id=None),
        row(user_id=None, checksum=None),  # one row, two columns
        row(email_subject=None),
    )
    assert (res.passed, res.n_failed, res.n_evaluated) == (False, 3, 4)
    assert res.details["by_field"] == {
        "merchant_name": 0,
        "user_id": 2,
        "email_time": 0,
        "checksum": 1,
        "from_domain": 0,
        "email_subject": 1,
    }
    assert res.sample is not None
    assert res.sample.height == 3


# --- S6 ---------------------------------------------------------------------------


def test_id_formats_passes_on_well_formed_ids() -> None:
    res = rule_result(
        "id_formats",
        row(),
        row(merchant_name="Lyft", order_number=LYFT_ORDER),
        row(merchant_name="Lyft", order_number="2" * 18),
    )
    assert (res.check_id, res.severity) == ("S6", Severity.WARN)
    assert (res.passed, res.n_failed) == (True, 0)
    assert res.details["by_field"] == {
        "user_id": 0,
        "checksum": 0,
        "lyft_order_number": 0,
        "uber_order_number": 0,
    }


def test_id_formats_ignores_nulls() -> None:
    # A null id is C1's (user_id, checksum) or C2's (order_number), not S6's.
    res = rule_result("id_formats", row(user_id=None, checksum=None, order_number=None))
    assert (res.passed, res.n_failed) == (True, 0)


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"user_id": "A" * 64}, "user_id"),  # uppercase hex
        ({"user_id": "a" * 63}, "user_id"),
        ({"checksum": "b" * 33}, "checksum"),
        ({"checksum": "g" * 32}, "checksum"),
        ({"order_number": "UF-123456789012"}, "uber_order_number"),  # freight
        ({"order_number": UUID.upper()}, "uber_order_number"),
        ({"merchant_name": "Lyft", "order_number": "1" * 17}, "lyft_order_number"),
        ({"merchant_name": "Lyft", "order_number": "1" * 20}, "lyft_order_number"),
        ({"merchant_name": "Lyft", "order_number": UUID}, "lyft_order_number"),
    ],
)
def test_id_formats_flags_each_malformed_id(
    changes: dict[str, str], field: str
) -> None:
    res = rule_result("id_formats", row(), row(**changes))
    assert (res.passed, res.n_failed, res.n_evaluated) == (False, 1, 2)
    assert {k: v for k, v in res.details["by_field"].items() if v} == {field: 1}


def test_id_formats_counts_a_row_once_however_many_fields_fail() -> None:
    res = rule_result("id_formats", row(user_id="x", checksum="y"))
    assert res.n_failed == 1
    assert (
        res.details["by_field"]["user_id"],
        res.details["by_field"]["checksum"],
    ) == (
        1,
        1,
    )


# --- C2 ---------------------------------------------------------------------------


def ride(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "receipt_kind": "ride",
        "is_superseded": False,
        "order_total_amount": 12.5,
        "order_number": UUID,
        "order_time": "t",
    }
    return base | changes


def rides(n: int, **changes: Any) -> list[dict[str, Any]]:
    return [ride(**changes) for _ in range(n)]


def ride_frame(data: list[dict[str, Any]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {k: [r[k] for r in data] for k in data[0]},
        schema={
            "receipt_kind": pl.String(),
            "is_superseded": pl.Boolean(),
            "order_total_amount": pl.Float64(),
            "order_number": pl.String(),
            "order_time": pl.String(),
        },
    )


def test_ride_completeness_passes_on_complete_rides() -> None:
    res = ride_completeness_check(ride_frame(rides(10)))
    assert (res.check_id, res.severity, res.passed) == ("C2", Severity.WARN, True)
    assert (res.n_failed, res.n_evaluated, res.metric) == (0, 10, 0.0)
    assert res.details["limits"] == RIDE_NULL_LIMITS


@pytest.mark.parametrize(
    ("changes", "worst"),
    [
        ({"order_total_amount": None}, "order_total_amount"),
        ({"order_number": None}, "order_number"),
        ({"order_time": None}, "order_time"),
    ],
)
def test_ride_completeness_fails_when_a_field_is_over_its_limit(
    changes: dict[str, Any], worst: str
) -> None:
    # 1 null in 100 rides is 1%: over every limit (0.01%, 0.5%, 0.5%).
    res = ride_completeness_check(ride_frame([ride(**changes), *rides(99)]))
    assert (res.passed, res.n_failed, res.n_evaluated) == (False, 1, 100)
    assert res.details["worst"] == worst
    assert (res.metric, res.threshold) == (0.01, RIDE_NULL_LIMITS[worst])


def test_ride_completeness_allows_nulls_within_the_limit() -> None:
    # 1 null order number in 1,000 rides is 0.1%, within 0.5%; it still counts in
    # n_failed.
    res = ride_completeness_check(ride_frame([ride(order_number=None), *rides(999)]))
    assert (res.passed, res.n_failed, res.metric) == (True, 1, 0.001)
    assert res.details["worst"] == "order_number"


def test_ride_completeness_counts_a_ride_missing_two_fields_once() -> None:
    res = ride_completeness_check(
        ride_frame([ride(order_number=None, order_time=None), *rides(9)])
    )
    assert (res.n_failed, res.n_evaluated) == (1, 10)
    rates = res.details["rates"]
    assert isinstance(rates, dict)
    assert rates["order_number"] == rates["order_time"] == 0.1


def test_ride_completeness_looks_only_at_rides_that_are_not_superseded() -> None:
    data = [
        ride(receipt_kind="bike", order_number=None),
        ride(receipt_kind="driver_earnings", order_total_amount=None),
        ride(is_superseded=True, order_time=None),
        *rides(2),
    ]
    res = ride_completeness_check(ride_frame(data))
    assert (res.passed, res.n_failed, res.n_evaluated) == (True, 0, 2)


def test_ride_completeness_with_no_rides_passes_without_a_metric() -> None:
    res = ride_completeness_check(ride_frame([ride(receipt_kind="bike")]))
    assert (res.passed, res.n_evaluated, res.metric) == (True, 0, None)


# --- row-count band -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "low", "high", "passed", "threshold"),
    [
        (5, 1, None, True, 1.0),
        (5, 5, 5, True, 5.0),
        (5, None, None, True, None),
        (5, None, 10, True, 10.0),
        (0, 1, None, False, 1.0),  # a header-only file under the default band
        (3, 5, 10, False, 5.0),
        (11, 5, 10, False, 10.0),
        (0, 0, None, True, 0.0),  # min_rows = 0 turns the lower bound off
    ],
)
def test_row_count_band(
    n: int, low: int | None, high: int | None, passed: bool, threshold: float | None
) -> None:
    res = row_count_band_check(n, low, high)
    assert (res.check_id, res.severity) == ("C3", Severity.WARN)
    assert (res.passed, res.metric, res.threshold) == (passed, float(n), threshold)
    assert res.n_failed == (0 if passed else 1)
    assert res.details == {"rows": n, "min_rows": low, "max_rows": high}


# --- S4 and S5 share one aggregation -----------------------------------------------


def test_column_profile_pair_matches_the_standalone_checks() -> None:
    lf = pl.LazyFrame(
        {"empty": [None, None], "same": [1, 1], "varies": [1, 2]},
        schema_overrides={"empty": pl.String()},
    )
    all_null, constant = column_profile_checks()
    assert all_null(lf) == all_null_columns_check(lf)
    assert constant(lf) == constant_columns_check(lf)
    assert all_null(lf).details == {"columns": ["empty"]}
    assert constant(lf).details == {"columns": ["same"]}


def test_column_profile_pair_scans_the_frame_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rideshare_data_pipeline.checks import structural

    calls: list[int] = []
    real = structural.degenerate_columns

    def counting(lf: pl.LazyFrame) -> structural.DegenerateColumns:
        calls.append(1)
        return real(lf)

    monkeypatch.setattr(structural, "degenerate_columns", counting)
    lf = pl.LazyFrame({"a": [1, 1], "b": [1, 2]})
    all_null, constant = column_profile_checks()
    all_null(lf)
    constant(lf)
    assert len(calls) == 1
    constant(pl.LazyFrame({"a": [1, 1]}))  # another frame is profiled afresh
    assert len(calls) == 2


# --- config ------------------------------------------------------------------------


def test_row_bounds_default_to_at_least_one_row() -> None:
    config = QCConfig()
    assert (config.min_rows, config.max_rows) == (1, None)
    assert parse_config({}) == config


def test_row_bounds_are_read_from_the_top_level() -> None:
    config = parse_config({"min_rows": 0, "max_rows": 10})
    assert (config.min_rows, config.max_rows) == (0, 10)
    assert parse_config({"min_rows": 5, "max_rows": 5}).max_rows == 5


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"min_rows": -1}, "min_rows: expected a non-negative integer"),
        ({"min_rows": 1.5}, "min_rows: expected a non-negative integer"),
        ({"min_rows": True}, "min_rows: expected a non-negative integer"),
        ({"max_rows": "10"}, "max_rows: expected a non-negative integer"),
        ({"max_rows": -5}, "max_rows: expected a non-negative integer"),
        ({"min_rows": 10, "max_rows": 5}, "min_rows .10. is greater than max_rows .5."),
    ],
)
def test_bad_row_bounds_are_config_errors(
    data: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(data)
