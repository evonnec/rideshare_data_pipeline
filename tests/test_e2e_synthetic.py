"""The whole pipeline on the synthetic fixture: one CLI run, every check's count.

The run writes into a temporary directory and must exit 1 (the planted negative
ride, A1, is an error-severity failure). Every check in report.json is compared with
the fixture manifest (fixtures/make_synthetic.py): the mapping below names the
manifest key of each check, and a check that has no manifest count has to be listed in
NO_MANIFEST_COUNT, so a newly added check fails this test until someone decides what
the fixture should hold for it. The same run is also the time budget: well under
10 s on a CI runner.
"""

import json
import time
from pathlib import Path
from typing import Any

import jsonschema
import polars as pl
import pytest
from fixtures.make_synthetic import MANIFEST, write_synthetic_csv

from rideshare_data_pipeline.cli import main
from rideshare_data_pipeline.config import KNOWN_CHECKS
from rideshare_data_pipeline.report import schema_text

BUDGET_SECONDS = 10.0

# report name -> the MANIFEST key holding its expected n_failed.
CHECK_TO_MANIFEST: dict[str, str] = {
    "header_valid": "S1",
    "utf8_valid": "S3",
    "cast_failures": "S2",
    "row_count_band": "C3",
    "required_fields": "C1",
    "id_formats": "S6",
    "non_negative_total": "A1",
    "item_price_equals_subtotal": "A2",
    "zero_fare_rides": "A3",
    "discount_within_subtotal": "A4",
    "tax_within_total": "A5",
    "unexplained_residual": "A6",
    "checksum_unique": "U1",
    "category_mismatch": "V2",
    "merchant_sender_consistent": "V1",
    "unknown_sender": "V3",
    "ingestion_order": "T1",
    "order_before_receipt": "T2",
    "late_receipts": "T3",
    "order_before_window": "T4",
    "email_after_update": "T5",
    "implausible_speed": "O1",
    "zero_trip_metrics": "O2",
    "extreme_trip": "O3",
    "all_null_columns": "S4",
    "constant_columns": "S5",
    "duplicate_rows": "U0",
    "ride_completeness": "C2",
    "identity_share": "A7",
    "receipt_kinds": "K1.non_spend",
    "duplicate_receipts": "U2",
    "dedupe_key_unique": "U2b",
    "cross_panelist_duplicates": "U3",
    "near_duplicate_rides": "U4",
    "description_parse_coverage": "PC1",
    "currency_suspect_rows": "CUR1",
    "date_only_order_time": "T6",
    "fare_outlier_conditional": "O4",
    "fare_outlier_univariate": "O5",
}

# Checks the manifest has no count for. K2 finds nothing by construction: every
# fixture subject matches a rule (asserted as 0 in tests/test_synthetic.py).
NO_MANIFEST_COUNT: frozenset[str] = frozenset({"unclassified_subject_templates"})


@pytest.fixture(scope="module")
def full_synthetic_csv_module(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_csv(tmp_path_factory.mktemp("fixture") / "synthetic.csv")


@pytest.fixture(scope="module")
def run(
    tmp_path_factory: pytest.TempPathFactory, full_synthetic_csv_module: Path
) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("e2e")
    started = time.perf_counter()
    status = main(["run", "--input", str(full_synthetic_csv_module), "--out", str(out)])
    elapsed = time.perf_counter() - started
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return {"status": status, "elapsed": elapsed, "out": out, "report": report}


def test_exits_1_because_of_the_planted_error(run: dict[str, Any]) -> None:
    assert run["status"] == 1
    assert run["report"]["exit_code"] == 1
    failed_errors = [
        c["name"]
        for c in run["report"]["checks"]
        if c["severity"] == "error" and not c["passed"]
    ]
    assert failed_errors == ["non_negative_total"]


def test_finishes_within_the_budget(run: dict[str, Any]) -> None:
    assert run["elapsed"] < BUDGET_SECONDS


def test_every_enabled_check_has_a_manifest_entry_or_is_listed_without_one() -> None:
    mapped, unmapped = set(CHECK_TO_MANIFEST), NO_MANIFEST_COUNT
    assert not mapped & unmapped
    assert mapped | unmapped == set(KNOWN_CHECKS)
    assert set(CHECK_TO_MANIFEST.values()) <= set(MANIFEST)


def test_every_check_in_the_report_matches_the_manifest(run: dict[str, Any]) -> None:
    checks = {c["name"]: c for c in run["report"]["checks"]}
    # A check missing from the run (disabled, renamed) or one the test doesn't know
    # about both fail here.
    assert set(checks) == set(CHECK_TO_MANIFEST) | NO_MANIFEST_COUNT
    wrong = {
        name: (checks[name]["n_failed"], MANIFEST[key], key)
        for name, key in CHECK_TO_MANIFEST.items()
        if checks[name]["n_failed"] != MANIFEST[key]
    }
    assert wrong == {}, "name: (n_failed, expected, manifest key)"


def test_report_json_is_schema_valid(run: dict[str, Any]) -> None:
    jsonschema.validate(
        run["report"],
        json.loads(schema_text()),
        cls=jsonschema.Draft202012Validator,
    )


def test_curated_parquet_has_every_fixture_row(
    run: dict[str, Any], full_synthetic_frame: pl.DataFrame
) -> None:
    curated = run["out"] / "synthetic.curated.parquet"
    assert pl.scan_parquet(curated).select(pl.len()).collect().item() == (
        full_synthetic_frame.height
    )
    assert run["report"]["run"]["rows"] == full_synthetic_frame.height
