"""The synthetic fixture: determinism, shape, and the manifest of expected counts.

Checks that exist run against the frame and must match MANIFEST; the manifest's other
entries are reserved for the PRs that add those checks (each asserts its own key).
"""

from pathlib import Path

import polars as pl
import pytest
from fixtures.make_synthetic import (
    INJECTIONS,
    MANIFEST,
    build_rows,
    build_synthetic,
    main,
    write_synthetic_csv,
)
from synthetic import ALL_NULL, CONSTANT, read_typed_frame

from rideshare_data_pipeline.checks import run_checks
from rideshare_data_pipeline.checks.structural import cast_failures_check
from rideshare_data_pipeline.cli import main as qc_main
from rideshare_data_pipeline.io import cast_failures, read_raw
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

N_BASELINE = 248
N_ROWS = N_BASELINE + sum(len(p) for p in INJECTIONS.values())


# --- the generator ---------------------------------------------------------------


def test_same_seed_gives_identical_bytes(tmp_path: Path) -> None:
    a = write_synthetic_csv(tmp_path / "a.csv").read_bytes()
    b = write_synthetic_csv(tmp_path / "b.csv").read_bytes()
    assert a == b
    assert write_synthetic_csv(tmp_path / "c.csv", seed=1).read_bytes() != a


def test_file_shape(full_synthetic_csv: Path) -> None:
    raw = full_synthetic_csv.read_bytes()
    assert raw.count(b"\r\n") == N_ROWS + 1
    assert raw.count(b"\n") == raw.count(b"\r\n")  # no bare LF
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert len(raw) < 200_000
    assert 280 <= N_ROWS <= 320
    assert any(b > 127 for b in raw)
    assert raw.decode("utf-8").splitlines()[0] == ",".join(EXPECTED_COLUMNS)


def test_csv_matches_the_typed_frame(
    full_synthetic_csv: Path, full_synthetic_frame: pl.DataFrame
) -> None:
    assert read_typed_frame(full_synthetic_csv).equals(full_synthetic_frame)
    assert full_synthetic_frame.height == N_ROWS
    assert full_synthetic_frame.columns == list(EXPECTED_COLUMNS)


def test_identifiers_are_synthetic_and_well_formed(
    full_synthetic_frame: pl.DataFrame,
) -> None:
    df = full_synthetic_frame
    assert df["user_id"].str.contains(r"^[0-9a-f]{64}$").all()
    assert df["checksum"].str.contains(r"^[0-9a-f]{32}$").all()
    lyft = df.filter(
        (pl.col("merchant_name") == "Lyft") & pl.col("order_number").is_not_null()
    )
    assert lyft["order_number"].str.contains(r"^\d{19}$").all()
    uber = df.filter(
        (pl.col("merchant_name") == "Uber")
        & ~pl.col("order_number").str.starts_with("UF-")
    )
    uuid = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    assert uber["order_number"].drop_nulls().str.contains(uuid).all()


def test_injection_registry() -> None:
    positions = [p for block in INJECTIONS.values() for p in block]
    assert positions == list(range(N_BASELINE, N_ROWS))  # contiguous, disjoint
    assert all(INJECTIONS.values())
    rows, again = build_rows(seed=7)  # positions don't depend on the seed
    assert again == INJECTIONS
    assert len(rows) == N_ROWS


def test_script_writes_a_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out.csv"
    assert main([str(out)]) == 0
    assert out.read_bytes() == write_synthetic_csv(tmp_path / "x.csv").read_bytes()
    assert f"{N_ROWS} rows" in capsys.readouterr().out
    assert main([]) == 2


# --- the layout the checks rely on --------------------------------------------------


def test_layout_degenerate_columns(full_synthetic_frame: pl.DataFrame) -> None:
    df = full_synthetic_frame
    for col in ALL_NULL:
        assert df[col].null_count() == df.height
    for col in CONSTANT:
        assert df[col].drop_nulls().n_unique() == 1
    # first equals last, yet not constant (the legacy constant check got this wrong)
    merchant = df["merchant_name"]
    assert merchant[0] == merchant[-1]
    assert merchant.n_unique() == 2


def test_baseline_is_clean(full_synthetic_frame: pl.DataFrame) -> None:
    base = full_synthetic_frame.head(N_BASELINE)
    assert base.filter(pl.col("item_price") != pl.col("order_subtotal")).is_empty()
    assert base.filter(pl.col("order_total_amount") <= 0).is_empty()
    assert base["order_number"].is_unique().all()
    daily = base.group_by("user_id", pl.col("order_time").dt.date()).len()
    assert daily["len"].max() == 1
    assert base.filter(pl.col("insert_time") < pl.col("email_time")).is_empty()
    assert base.filter(
        pl.col("email_time").dt.date() < pl.col("order_time").dt.date()
    ).is_empty()
    assert set(base["merchant_name"]) == {"Uber", "Lyft"}


def test_injected_rows_are_where_the_registry_says(
    full_synthetic_frame: pl.DataFrame,
) -> None:
    df = full_synthetic_frame
    (bike,) = INJECTIONS["bike"]
    assert df["order_total_amount"][bike] == -4.5
    (freight,) = INJECTIONS["freight"]
    assert df["order_number"][freight].startswith("UF-")
    (old,) = INJECTIONS["ancient_order"]
    assert df["order_time"][old].year == 2010
    passes = df[list(INJECTIONS["monthly_pass"])]
    assert passes["order_number"].null_count() == 3
    varying = [c for c in EXPECTED_COLUMNS if passes[c].n_unique() > 1]
    assert "email_time" in varying
    assert not {"order_total_amount", "product_description", "email_subject"} & set(
        varying
    )
    dup = df[list(INJECTIONS["duplicate_order_key"])]
    assert dup["order_number"].n_unique() == 1
    assert dup["email_time"].n_unique() == 2
    cross = df[list(INJECTIONS["cross_panelist_duplicate"])]
    assert cross["order_number"].n_unique() == 1
    assert cross["user_id"].n_unique() == 2
    ride, tip = df[list(INJECTIONS["ride_tip_update"])].rows(named=True)
    assert ride["order_number"] == tip["order_number"]
    assert tip["order_total_amount"] > ride["order_total_amount"]
    assert tip["insert_time"] < tip["email_time"]
    masked = df[list(INJECTIONS["masked_subjects"])]
    assert masked["email_subject"].str.contains(r"\*{5}").all()
    (apac,) = INJECTIONS["apac_next_day"]
    assert (df["order_time"][apac].date() - df["email_time"][apac].date()).days == 1


def test_manifest_matches_the_injections() -> None:
    n = {name: len(rows) for name, rows in INJECTIONS.items()}
    assert (
        MANIFEST["U2"]
        == n["duplicate_order_key"]
        + n["cross_panelist_duplicate"]
        + n["ride_tip_update"]
    )
    assert MANIFEST["U3"] == n["cross_panelist_duplicate"]
    assert MANIFEST["U4.rows"] == n["near_duplicate"]
    assert MANIFEST["K1.pass"] == n["monthly_pass"]
    assert MANIFEST["K1.food_order"] == n["food_orders"]
    kinds = ("driver_earnings", "pass", "micromobility", "food_order", "freight")
    assert MANIFEST["K1.non_spend"] == sum(MANIFEST[f"K1.{k}"] for k in kinds)
    assert MANIFEST["V2"] == (
        MANIFEST["K1.driver_earnings"]
        + MANIFEST["K1.food_order"]
        + MANIFEST["K1.freight"]
    )
    assert MANIFEST["U5"] == n["many_rides_one_day"]
    assert MANIFEST["PR1"] == n["masked_subjects"]
    assert MANIFEST["CUR1"] == MANIFEST["CUR1.non_us"] + MANIFEST["CUR1.km_generic"]
    assert MANIFEST["S4"] == len(ALL_NULL)
    assert MANIFEST["S5"] == len(CONSTANT)


# --- the implemented checks against the manifest -------------------------------------


def test_implemented_checks_match_the_manifest(
    full_synthetic_csv: Path, full_synthetic_frame: pl.DataFrame
) -> None:
    results = run_checks(full_synthetic_frame.lazy())
    results.append(cast_failures_check(cast_failures(read_raw(full_synthetic_csv))))
    found = {r.check_id: r.n_failed for r in results}
    assert found == {k: MANIFEST[k] for k in ("S2", "A1", "U1", "S4", "S5", "U0")}


def test_degenerate_columns_are_the_documented_ones(
    full_synthetic_frame: pl.DataFrame,
) -> None:
    by_id = {r.check_id: r for r in run_checks(full_synthetic_frame.lazy())}
    assert by_id["S4"].details["columns"] == list(ALL_NULL)
    assert by_id["S5"].details["columns"] == list(CONSTANT)


# --- end to end ----------------------------------------


def test_cli_end_to_end_on_the_full_fixture(
    full_synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    # Only exact duplicate rows change the exit status; the bike's A1 failure doesn't.
    assert qc_main(["--input", str(full_synthetic_csv), "--out", str(out)]) == 0

    summary, checks = capsys.readouterr().out.split("\n\n")
    fields = dict(line.split(":", 1) for line in summary.splitlines())
    assert fields["encoding"].strip() == "UTF-8, no BOM, CRLF line endings"
    assert fields["rows"].strip() == str(N_ROWS)
    assert fields["columns"].strip() == "33"
    assert fields["all-null columns"].strip() == f"5 ({', '.join(ALL_NULL)})"
    assert fields["constant columns"].strip() == f"4 ({', '.join(CONSTANT)})"
    assert fields["duplicate rows"].strip() == "0"

    lines = {line.split()[2]: line.split() for line in checks.splitlines()}
    assert list(lines) == [
        "cast_failures",
        "non_negative_total",
        "checksum_unique",
        "all_null_columns",
        "constant_columns",
        "duplicate_rows",
    ]
    assert lines["non_negative_total"][1] == "FAIL"
    assert lines["non_negative_total"][3] == f"{MANIFEST['A1']}/{N_ROWS}"
    for name in ("cast_failures", "checksum_unique", "duplicate_rows"):
        assert lines[name][1] == "PASS"
        assert lines[name][3].startswith("0/")
    assert pl.read_parquet(out / "synthetic.parquet").height == N_ROWS


def test_build_synthetic_is_repeatable() -> None:
    assert build_synthetic().equals(build_synthetic())
