"""The curated Parquet: schema contract, per-row QC flags, report entries (#22).

Every row is kept (owner decision: no row dropping and no --no-dedupe; dedupe is the
``is_superseded`` flag). The expectations below are built row by row from the
synthetic fixture's INJECTIONS and the MANIFEST conventions.
"""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import polars as pl
import pytest
from fixtures.make_synthetic import (
    INJECTIONS,
    MANIFEST,
    USD_AMOUNT,
    write_synthetic_csv,
)
from synthetic import synthetic_rows, write_csv_text

import rideshare_data_pipeline.checks.outliers as outliers_module
from rideshare_data_pipeline.cli import main
from rideshare_data_pipeline.config import KNOWN_CHECKS
from rideshare_data_pipeline.curated import (
    FLAGS,
    PREFIX,
    curated_schema,
    flagged_names,
)
from rideshare_data_pipeline.dedupe import FLAG_NAMES
from rideshare_data_pipeline.parse import DERIVED_NAMES
from rideshare_data_pipeline.report import schema_text
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

N_ROWS = 294


@dataclass(frozen=True, slots=True)
class Run:
    out: Path
    curated: pl.DataFrame
    report: dict[str, object]


def run_main(csv: Path, out: Path, *extra: str) -> Run:
    main(["run", "--input", str(csv), "--out", str(out), *extra])
    document = json.loads((out / "report.json").read_text(encoding="utf-8"))
    schema = json.loads(schema_text())
    jsonschema.validate(document, schema, cls=jsonschema.Draft202012Validator)
    curated = pl.read_parquet(next(out.glob("*.curated.parquet")))
    return Run(out, curated, document)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("curated")
    csv = write_synthetic_csv(root / "synthetic.csv")
    return run_main(csv, root / "out")


def n_failed(run: Run) -> dict[str, int]:
    checks = run.report["checks"]
    assert isinstance(checks, list)
    return {c["name"]: c["n_failed"] for c in checks}


# --- the schema contract -------------------------------------------------------


def test_schema_equals_the_contract_and_the_columns_are_in_order(
    synthetic_run: Run,
) -> None:
    written = synthetic_run.curated
    contract = curated_schema(KNOWN_CHECKS)
    assert written.schema == contract
    assert written.columns == list(contract)
    columns = written.columns
    flags = [c for c in columns if c.startswith(PREFIX) and c != FLAGS]
    row_flags = [c for c in flags if c not in {"qc_error", "qc_warn"}]
    # The groups, in order.
    assert columns == [
        *EXPECTED_COLUMNS,
        *DERIVED_NAMES,
        *FLAG_NAMES,
        "is_usd_rider_spend",
        *row_flags,
        "qc_error",
        "qc_warn",
        "qc_flags",
    ]
    assert row_flags == [f"qc_{n}" for n in KNOWN_CHECKS if n in flagged_names()]
    assert not [c for c in columns if c.startswith("_")]  # no helper column survives


def test_the_contract_types() -> None:
    schema = curated_schema(KNOWN_CHECKS)
    assert schema["qc_flags"] == pl.List(pl.String)
    assert isinstance(schema["receipt_kind"], pl.Enum)
    for name in ("is_superseded", "is_usd_rider_spend", "qc_error", "qc_warn"):
        assert schema[name] == pl.Boolean
    assert all(schema[f"qc_{n}"] == pl.Boolean for n in flagged_names())


def test_the_flagged_checks_are_rules_plus_the_row_identifiable_dataset_checks() -> (
    None
):
    assert set(flagged_names()) >= {
        "duplicate_rows",
        "duplicate_receipts",
        "cross_panelist_duplicates",
        "near_duplicate_rides",
        "currency_suspect_rows",
        "description_parse_coverage",
        "fare_outlier_conditional",
    }
    dataset_level = {
        "header_valid",
        "utf8_valid",
        "cast_failures",
        "row_count_band",
        "all_null_columns",
        "constant_columns",
        "receipt_kinds",
        "unclassified_subject_templates",
        "identity_share",
        "date_only_order_time",
        "fare_outlier_univariate",
        "ride_completeness",
        "dedupe_key_unique",
    }
    assert not dataset_level & set(flagged_names())


# --- qc_flags per injected row -----------------------------------------------------

type Names = tuple[str, ...]

DUP = ("duplicate_receipts",)
CROSS = ("duplicate_receipts", "cross_panelist_duplicates")
NONE: Names = ()

# Row by row, per INJECTIONS block. The MANIFEST conventions: a duplicate rule fails
# every row of a repeated key (U2, U3); U4 flags the excess rows, so the second ride
# of the near-duplicate pair (the first in file order is not flagged); CUR1 flags
# non_us_sender, km_generic_sender and apac_next_day but not us_spanish_subject (F6);
# V2 (category_mismatch) fires on earnings, food and freight; the freight row also
# has a non-UUID order number (S6) and a person-like sender (V3); the ancient order is
# late (T3) and before the window (T4); O4 flags the 10x fare only. The blocks with
# no flags are listed too. U5, U6 and PR1 are catalog rows with no check yet.
EXPECTED: dict[str, tuple[Names, ...]] = {
    "duplicate_order_key": (DUP, DUP),
    "cross_panelist_duplicate": (CROSS, CROSS),
    "ride_tip_update": (DUP, DUP),
    "near_duplicate": (NONE, ("near_duplicate_rides",)),
    "monthly_pass": (NONE, NONE, NONE),
    "driver_earnings": (("category_mismatch",),),
    "food_orders": (("category_mismatch",), ("category_mismatch",)),
    "bike": (NONE,),
    "freight": (("id_formats", "category_mismatch", "unknown_sender"),),
    "non_us_sender": (("currency_suspect_rows",),),
    "km_generic_sender": (("currency_suspect_rows",),),
    "uber_date_only": (NONE,),
    "lyft_order_5h_before_email": (NONE,),
    "apac_next_day": (("currency_suspect_rows",),),
    "ancient_order": (("late_receipts", "order_before_window"),),
    "discount_gt_subtotal": (("discount_within_subtotal",),),
    "tax_gt_total": (("tax_within_total",),),
    "zero_fare_ride": (("zero_fare_rides",),),
    "negative_total_ride": (("non_negative_total",),),
    "zero_duration_trip": (("zero_trip_metrics",),),
    "implausible_speed": (("implausible_speed",),),
    "fare_outlier_10x": (("fare_outlier_conditional",),),
    "overlapping_rides": (NONE, NONE),
    "many_rides_one_day": (NONE,) * 12,
    "masked_subjects": (NONE, NONE),
    "us_spanish_subject": (NONE,),
}


def test_the_table_covers_every_injected_block() -> None:
    assert set(EXPECTED) == set(INJECTIONS)
    for block, rows in EXPECTED.items():
        assert len(rows) == len(INJECTIONS[block]), block


@pytest.mark.parametrize("block", list(EXPECTED))
def test_qc_flags_of_an_injected_block(synthetic_run: Run, block: str) -> None:
    flags = synthetic_run.curated[FLAGS].to_list()
    found = [tuple(flags[i]) for i in INJECTIONS[block]]
    # In report order, so a list reads the same on every run.
    expected = [tuple(sorted(n, key=KNOWN_CHECKS.index)) for n in EXPECTED[block]]
    assert found == expected


def test_the_clean_baseline_has_no_flags(synthetic_run: Run) -> None:
    injected = {i for rows in INJECTIONS.values() for i in rows}
    baseline = [i for i in range(N_ROWS) if i not in injected]
    assert len(baseline) == 248
    assert all(synthetic_run.curated[FLAGS][i].to_list() == [] for i in baseline)
    assert synthetic_run.curated[baseline]["qc_error"].sum() == 0
    assert synthetic_run.curated[baseline]["qc_warn"].sum() == 0


def test_qc_flags_is_never_null_and_agrees_with_the_columns(
    synthetic_run: Run,
) -> None:
    curated = synthetic_run.curated
    assert curated[FLAGS].null_count() == 0
    columns = [f"{PREFIX}{n}" for n in flagged_names()]
    rebuilt = curated.select(
        pl.concat_list(
            [pl.when(pl.col(c)).then(pl.lit(c.removeprefix(PREFIX))) for c in columns]
        )
        .list.drop_nulls()
        .alias("rebuilt")
    )["rebuilt"]
    assert rebuilt.to_list() == curated[FLAGS].to_list()


# --- the flag-sum invariant ---------------------------------------------------------


@pytest.mark.parametrize("name", flagged_names())
def test_each_flag_column_sums_to_the_checks_n_failed(
    synthetic_run: Run, name: str
) -> None:
    assert (
        synthetic_run.curated[f"{PREFIX}{name}"].sum() == n_failed(synthetic_run)[name]
    )


def test_dataset_level_checks_have_no_column(synthetic_run: Run) -> None:
    flag_columns = {
        c for c in synthetic_run.curated.columns if c.startswith(PREFIX)
    } - {"qc_error", "qc_warn", "qc_flags"}
    assert flag_columns == {f"{PREFIX}{n}" for n in flagged_names()}
    for name in ("header_valid", "row_count_band", "identity_share", "receipt_kinds"):
        assert f"{PREFIX}{name}" not in synthetic_run.curated.columns


def test_manifest_counts_hold_in_the_flag_columns(synthetic_run: Run) -> None:
    curated = synthetic_run.curated
    assert curated["is_superseded"].sum() == MANIFEST["U2.superseded"]
    assert curated["qc_duplicate_receipts"].sum() == MANIFEST["U2"]
    assert curated["qc_cross_panelist_duplicates"].sum() == MANIFEST["U3"]
    assert curated["qc_near_duplicate_rides"].sum() == MANIFEST["U4"]
    assert curated["qc_currency_suspect_rows"].sum() == MANIFEST["CUR1"]
    assert curated["qc_fare_outlier_conditional"].sum() == MANIFEST["O4"]
    assert curated["qc_description_parse_coverage"].sum() == MANIFEST["PC1"]
    assert curated["qc_duplicate_rows"].sum() == MANIFEST["U0"]


def test_clean_usd_rider_spend_is_a_filter_on_the_flags(synthetic_run: Run) -> None:
    curated = synthetic_run.curated
    kept = curated.filter(pl.col("is_usd_rider_spend") & ~pl.col("is_superseded"))
    assert kept.height == MANIFEST["USD.rides"]
    assert round(kept["order_total_amount"].sum(), 2) == USD_AMOUNT


def test_qc_error_and_qc_warn(synthetic_run: Run) -> None:
    curated = synthetic_run.curated
    # The only error-severity row failure is A1, the negative car ride.
    assert curated.filter("qc_error").height == 1
    assert curated["qc_error"][INJECTIONS["negative_total_ride"][0]]
    # qc_warn: a row failing any warn-severity flag; the error-only row is not one.
    warned = {i for i, failed in enumerate(curated["qc_warn"]) if failed}
    assert INJECTIONS["negative_total_ride"][0] not in warned
    assert INJECTIONS["fare_outlier_10x"][0] in warned
    assert INJECTIONS["monthly_pass"][0] not in warned


# --- every row is kept; the stream; no helper columns ------------------------------


def test_every_row_is_kept(synthetic_run: Run) -> None:
    run = synthetic_run.report["run"]
    assert isinstance(run, dict)
    assert synthetic_run.curated.height == run["rows"] == N_ROWS
    # The input columns are unchanged: the same values, in the same order.
    raw = pl.read_csv(synthetic_run.out.parent / "synthetic.csv", infer_schema=False)
    assert synthetic_run.curated["checksum"].to_list() == raw["checksum"].to_list()


def test_exact_duplicate_rows_are_flagged_after_the_first_and_kept(
    tmp_path: Path,
) -> None:
    rows = synthetic_rows(3)
    csv = write_csv_text(tmp_path / "dups.csv", [*rows, rows[0], rows[0]])
    run = run_main(csv, tmp_path / "out")
    assert run.curated.height == 5
    assert run.curated["qc_duplicate_rows"].to_list() == [False] * 3 + [True] * 2
    assert run.curated["qc_duplicate_rows"].sum() == n_failed(run)["duplicate_rows"]
    # U1 counts every row of a repeated checksum, so all three copies fail it.
    assert (
        run.curated["qc_checksum_unique"].to_list() == [True, False, False] + [True] * 2
    )


def test_the_outlier_scores_are_computed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real = outliers_module.score

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(outliers_module, "score", counting)
    csv = write_synthetic_csv(tmp_path / "synthetic.csv")
    run = run_main(csv, tmp_path / "out")
    assert run.curated["qc_fare_outlier_conditional"].sum() == 1
    assert len(calls) == 1


# --- config, drop-columns -----------------------------------------------------------


def write_config(path: Path, text: str) -> Path:
    tomllib.loads(text)
    path.write_text(text, encoding="utf-8")
    return path


def test_disabled_checks_have_no_column(tmp_path: Path) -> None:
    config = write_config(
        tmp_path / "qc.toml",
        "[checks.near_duplicate_rides]\nenabled = false\n"
        "[checks.zero_fare_rides]\nenabled = false\n"
        "[checks.fare_outlier_conditional]\nenabled = false\n"
        "[checks.duplicate_rows]\nenabled = false\n",
    )
    csv = write_synthetic_csv(tmp_path / "synthetic.csv")
    run = run_main(csv, tmp_path / "out", "--config", str(config))
    off = {
        "near_duplicate_rides",
        "zero_fare_rides",
        "fare_outlier_conditional",
        "duplicate_rows",
    }
    enabled = [n for n in KNOWN_CHECKS if n not in off]
    assert run.curated.schema == curated_schema(enabled)
    assert not any(f"{PREFIX}{n}" in run.curated.columns for n in off)
    flags = run.curated[FLAGS].explode().drop_nulls().unique().to_list()
    assert not off & set(flags)
    assert INJECTIONS["fare_outlier_10x"][0] not in {
        i for i, failed in enumerate(run.curated["qc_warn"]) if failed
    }


@pytest.mark.parametrize(
    ("severity", "error", "warn"),
    [("error", True, False), ("warn", False, True), ("info", False, False)],
)
def test_a_severity_override_feeds_qc_error_and_qc_warn(
    tmp_path: Path, severity: str, error: bool, warn: bool
) -> None:
    # Two rows are flagged by checks under test: the negative car ride (A1, error by
    # default) and, as a control, the zero-fare ride (A3, warn). The override moves A1.
    config = write_config(
        tmp_path / "qc.toml", f'[checks.non_negative_total]\nseverity = "{severity}"\n'
    )
    csv = write_synthetic_csv(tmp_path / "synthetic.csv")
    run = run_main(csv, tmp_path / "out", "--config", str(config))
    row = INJECTIONS["negative_total_ride"][0]
    curated = run.curated
    assert curated["qc_error"][row] is error
    assert curated["qc_warn"][row] is warn
    assert curated["qc_error"].sum() == int(error)
    zero_fare = INJECTIONS["zero_fare_ride"][0]
    assert curated["qc_warn"][zero_fare]
    # The flag and the name are still there whatever the severity.
    assert curated["qc_non_negative_total"][row]
    assert "non_negative_total" in curated[FLAGS][row].to_list()


def test_drop_columns_leaves_out_inputs_only_and_flags_stay_correct(
    tmp_path: Path,
) -> None:
    dropped = ["order_total_amount", "product_description", "checksum", "from_domain"]
    csv = write_synthetic_csv(tmp_path / "synthetic.csv")
    run = run_main(csv, tmp_path / "out", "--drop-columns", ",".join(dropped))
    assert run.curated.schema == curated_schema(KNOWN_CHECKS, dropped)
    assert not set(dropped) & set(run.curated.columns)
    for name in (*DERIVED_NAMES, *FLAG_NAMES, "qc_flags", "qc_error", "qc_warn"):
        assert name in run.curated.columns
    # The checks ran on the whole frame, so the flags don't depend on the drop.
    row = INJECTIONS["negative_total_ride"][0]
    assert run.curated["qc_non_negative_total"][row]
    assert run.curated["qc_checksum_unique"].sum() == 0
    columns = run.report["outputs"]
    assert isinstance(columns, dict)
    curated = columns["curated"]
    assert isinstance(curated, dict)
    assert curated["columns"] == len(run.curated.columns)


# --- the report ----------------------------------------------------------------------


def test_the_report_records_the_curated_file(synthetic_run: Run) -> None:
    outputs = synthetic_run.report["outputs"]
    assert isinstance(outputs, dict)
    assert outputs["curated"] == {
        "name": "synthetic.curated.parquet",
        "rows": N_ROWS,
        "columns": len(curated_schema(KNOWN_CHECKS)),
    }
    assert outputs["parquet"] == "synthetic.curated.parquet"
    markdown = (synthetic_run.out / "report.md").read_text(encoding="utf-8")
    section = markdown.split("## Curated dataset")[1].split("## Samples")[0]
    assert "synthetic.curated.parquet" in section
    assert f"{N_ROWS} rows" in section
    assert f"{len(curated_schema(KNOWN_CHECKS))} columns" in section
