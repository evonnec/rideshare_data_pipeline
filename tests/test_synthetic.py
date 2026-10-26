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
    USD_AMOUNT,
    build_rows,
    build_synthetic,
    main,
    write_synthetic_csv,
)
from synthetic import ALL_NULL, CONSTANT, read_typed_frame

from rideshare_data_pipeline.checks import run_checks
from rideshare_data_pipeline.checks.outliers import score
from rideshare_data_pipeline.checks.structural import (
    cast_failures_check,
    header_valid_check,
    row_count_band_check,
    utf8_valid_check,
)
from rideshare_data_pipeline.cli import main as qc_main
from rideshare_data_pipeline.io import cast_failures, inspect_csv, read_raw
from rideshare_data_pipeline.money import usd_rides
from rideshare_data_pipeline.parse import KINDS
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
    (negative_ride,) = INJECTIONS["negative_total_ride"]
    assert df["order_total_amount"][negative_ride] < 0
    assert df["product_description"][negative_ride].startswith("UberX")
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
    assert MANIFEST["U2.superseded"] == MANIFEST["U2.groups"]
    assert MANIFEST["U3"] == n["cross_panelist_duplicate"]
    assert MANIFEST["U4.rows"] == n["near_duplicate"]
    assert MANIFEST["K1.pass"] == n["monthly_pass"]
    assert MANIFEST["K1.food_order"] == n["food_orders"]
    kinds = ("driver_earnings", "pass", "micromobility", "food_order", "freight")
    assert MANIFEST["K1.non_spend"] == sum(MANIFEST[f"K1.{k}"] for k in kinds)
    assert (
        MANIFEST["K1.ride"]
        == N_ROWS - MANIFEST["K1.non_spend"] - MANIFEST["K1.tip_update"]
    )
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
    full_synthetic_csv: Path, full_synthetic_derived: pl.DataFrame
) -> None:
    results = run_checks(full_synthetic_derived.lazy())
    results.append(cast_failures_check(cast_failures(read_raw(full_synthetic_csv))))
    facts = inspect_csv(full_synthetic_csv)
    results += [
        header_valid_check(facts),
        utf8_valid_check(facts),
        row_count_band_check(full_synthetic_derived.height, 1, None),
    ]
    found = {r.check_id: r.n_failed for r in results}
    expected = {
        k: MANIFEST[k]
        for k in (
            "S1",
            "S2",
            "S3",
            "C1",
            "C2",
            "C3",
            "S6",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "U1",
            "S4",
            "S5",
            "U0",
            "V2",
            "U2",
            "U2b",
            "U3",
            "U4",
            "CUR1",
            "V1",
            "V3",
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6",
            "O1",
            "O2",
            "O3",
            "O4",
            "O5",
        )
    }
    # K1's n_failed is the non-spend rows; K2 (unclassified templates) finds none.
    assert found == {
        **expected,
        "K1": MANIFEST["K1.non_spend"],
        "K2": 0,
        "PC1": MANIFEST["PC1"],
    }


def test_currency_and_sender_checks_match_the_manifest(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    df = full_synthetic_derived
    by_id = {r.check_id: r for r in run_checks(df.lazy())}
    cur1, v1, v3 = by_id["CUR1"], by_id["V1"], by_id["V3"]
    assert cur1.n_failed == MANIFEST["CUR1"]
    assert not cur1.passed
    by_reason = cur1.details["by_reason"]
    assert isinstance(by_reason, dict)
    assert {k: v["rows"] for k, v in by_reason.items()} == {
        "non_us_sender": MANIFEST["CUR1.non_us"],
        "km_unit": MANIFEST["CUR1.km_generic"],
    }
    # F6: the US sender's Spanish subject is tagged (foreign_subject), not suspect,
    # and it is a clean USD ride. The non-US row's Spanish subject is tagged too.
    tag = cur1.details["foreign_subject"]
    assert isinstance(tag, dict)
    assert tag["rows"] == MANIFEST["CUR1.foreign_us"]
    (spanish,) = INJECTIONS["us_spanish_subject"]
    assert df["foreign_subject"][spanish]
    assert not df["currency_suspect"][spanish]
    assert df.filter(pl.col("foreign_subject")).height == (
        MANIFEST["CUR1.foreign_us"] + 1
    )
    assert (v1.n_failed, v1.passed) == (MANIFEST["V1"], True)
    assert (v3.n_failed, v3.passed) == (MANIFEST["V3"], False)
    # V3 finds the freight sender, and its sample never carries the address.
    assert v3.sample is not None
    (freight,) = INJECTIONS["freight"]
    assert v3.sample["checksum"].to_list() == [df["checksum"][freight]]
    assert "from_domain" not in v3.sample.columns


def test_usd_rides_excludes_the_gated_rows(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    df = full_synthetic_derived
    usd = usd_rides(df)
    assert usd.height == MANIFEST["USD.rides"]
    assert usd["order_total_amount"].sum() == pytest.approx(USD_AMOUNT)
    kept = set(usd["checksum"])
    for name in ("non_us_sender", "km_generic_sender", "apac_next_day"):
        (i,) = INJECTIONS[name]
        assert df["checksum"][i] not in kept
        assert df["currency_suspect"][i]
    # None of the suspect rows is a fare outlier of the USD baseline: the 5,718
    # local-currency total is gone from the statistics.
    assert usd.select(pl.col("order_total_amount").max() < 1000).item()
    assert df.filter(pl.col("currency_suspect")).height == MANIFEST["CUR1"]
    (spanish,) = INJECTIONS["us_spanish_subject"]
    assert df["checksum"][spanish] in kept


def test_spend_waterfall_of_the_fixture(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    df = full_synthetic_derived
    spend = next(r for r in run_checks(df.lazy()) if r.check_id == "CUR1").details[
        "spend"
    ]
    assert isinstance(spend, dict)
    assert spend["naive_total"]["rows"] == N_ROWS
    assert spend["currency_suspect"]["rows"] == MANIFEST["CUR1"]
    assert spend["non_rider_spend"]["rows"] == MANIFEST["K1.non_spend"]
    assert spend["superseded"]["rows"] == MANIFEST["U2.superseded"]
    assert spend["micromobility"]["rows"] == MANIFEST["K1.micromobility"]
    assert spend["clean_usd_rider_spend"] == {
        "rows": MANIFEST["USD.rides"],
        "amount": USD_AMOUNT,
    }


def test_duplicate_checks_match_the_manifest(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    df = full_synthetic_derived
    by_id = {r.check_id: r for r in run_checks(df.lazy())}
    u2, u2b, u3, u4 = (by_id[k] for k in ("U2", "U2b", "U3", "U4"))
    assert (u2.n_failed, u2.details["groups"]) == (
        MANIFEST["U2"],
        MANIFEST["U2.groups"],
    )
    assert u2.details["excess_rows"] == MANIFEST["U2.superseded"]
    assert (u3.n_failed, u3.details["groups"]) == (
        MANIFEST["U3"],
        MANIFEST["U3.groups"],
    )
    assert u4.n_failed == MANIFEST["U4"]
    assert u4.details["rows"] == MANIFEST["U4.rows"]
    assert (u2b.passed, u2b.n_failed) == (True, 0)
    assert [r.passed for r in (u2, u3, u4)] == [False] * 3
    # is_cross_panelist_copy: exactly the rows U3 counts (the two panelists' copies)
    copies = {i for i, v in enumerate(df["is_cross_panelist_copy"]) if v}
    assert copies == set(INJECTIONS["cross_panelist_duplicate"])
    assert len(copies) == u3.n_failed == MANIFEST["U3"]
    assert u2.details["kind_pairs"] == [
        {"pair": "ride+ride", "groups": 2, "excess_spend": pytest.approx(16.98)},
        {"pair": "ride+tip_update", "groups": 1, "excess_spend": pytest.approx(12.38)},
    ]
    assert u2.details["excess_spend"] == pytest.approx(29.36)


def test_superseded_rows_are_the_manifest_ones(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    df = full_synthetic_derived
    flagged = {i for i, v in enumerate(df["is_superseded"]) if v}
    assert len(flagged) == MANIFEST["U2.superseded"]
    # The earlier email loses, except the ride, which its tip update supersedes.
    assert flagged == {
        INJECTIONS["duplicate_order_key"][0],
        INJECTIONS["cross_panelist_duplicate"][0],
        INJECTIONS["ride_tip_update"][0],
    }
    assert not flagged & set(INJECTIONS["monthly_pass"])
    assert not df["is_superseded"][INJECTIONS["ride_tip_update"][1]]


# manifest key -> receipt kind
KIND_KEYS = {
    "K1.ride": "ride",
    "K1.tip_update": "tip_update",
    "K1.driver_earnings": "driver_earnings",
    "K1.pass": "pass_or_credit",
    "K1.micromobility": "micromobility",
    "K1.food_order": "food_order",
    "K1.freight": "freight",
}


def test_kind_counts_equal_the_manifest(full_synthetic_derived: pl.DataFrame) -> None:
    counts = dict(
        full_synthetic_derived["receipt_kind"].cast(pl.String).value_counts().rows()
    )
    assert counts == {kind: MANIFEST[key] for key, kind in KIND_KEYS.items()}
    # The kinds the fixture doesn't inject are absent, not misclassified.
    for kind in ("fare_correction", "cancel_noshow", "payment_failed", "rental"):
        assert kind not in counts
    spend = full_synthetic_derived["is_rider_spend"].sum()
    assert N_ROWS - spend == MANIFEST["K1.non_spend"]


def test_kind_check_details_match_the_manifest(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    by_id = {r.check_id: r for r in run_checks(full_synthetic_derived.lazy())}
    k1 = by_id["K1"]
    assert k1.details["counts"] == {
        **dict.fromkeys(KINDS, 0),
        **{kind: MANIFEST[key] for key, kind in KIND_KEYS.items()},
    }
    by_merchant = k1.details["by_merchant"]
    assert isinstance(by_merchant, dict)
    assert sorted(by_merchant) == ["Lyft", "Uber"]
    assert by_merchant["Lyft"]["driver_earnings"] == 1
    assert by_merchant["Uber"]["food_order"] == 2
    assert (k1.n_failed, k1.n_evaluated) == (MANIFEST["K1.non_spend"], N_ROWS)
    # No injected subject is an unknown template.
    assert by_id["K2"].n_failed == 0
    assert by_id["K2"].details["templates"] == []


def test_v2_flags_exactly_the_manifest_rows(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    v2 = next(
        r for r in run_checks(full_synthetic_derived.lazy()) if r.check_id == "V2"
    )
    assert not v2.passed
    assert v2.n_failed == MANIFEST["V2"]
    assert v2.sample is not None
    assert sorted(v2.sample["receipt_kind"].cast(pl.String).to_list()) == [
        "driver_earnings",
        "food_order",
        "food_order",
        "freight",
    ]


def test_a1_fails_only_the_negative_car_ride(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    a1 = next(
        r for r in run_checks(full_synthetic_derived.lazy()) if r.check_id == "A1"
    )
    assert a1.n_failed == MANIFEST["A1"] == 1
    assert a1.sample is not None
    assert a1.sample["order_total_amount"].to_list() == [-5.75]
    # The bike's negative total is not counted: it isn't rider spend.
    (bike,) = INJECTIONS["bike"]
    assert full_synthetic_derived["order_total_amount"][bike] < 0
    assert not full_synthetic_derived["is_rider_spend"][bike]
    assert a1.n_evaluated == full_synthetic_derived["is_rider_spend"].sum()


def test_arithmetic_checks_hit_their_injections_and_nothing_else(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    by_id = {r.check_id: r for r in run_checks(full_synthetic_derived.lazy())}
    d = full_synthetic_derived
    for check_id, key in (
        ("A2", "A2"),
        ("A3", "A3"),
        ("A4", "A4"),
        ("A5", "A5"),
        ("A6", "A6"),
    ):
        assert by_id[check_id].n_failed == MANIFEST[key], check_id
    assert [by_id[k].passed for k in ("A2", "A3", "A4", "A5", "A6")] == [
        True,
        False,
        False,
        False,
        True,
    ]
    # Each warning fires on its own injection row.
    for check_id, block in (
        ("A3", "zero_fare_ride"),
        ("A4", "discount_gt_subtotal"),
        ("A5", "tax_gt_total"),
    ):
        sample = by_id[check_id].sample
        assert sample is not None
        (pos,) = INJECTIONS[block]
        assert sample["checksum"].to_list() == [d["checksum"][pos]], check_id
    assert by_id["A3"].details["by_field"] == {"Uber": 1, "Lyft": 0, "other": 0}
    # A3 means exactly 0 (the negative ride is A1's), and A6 judges USD.rides.
    assert by_id["A6"].n_evaluated == MANIFEST["USD.rides"]

    a7 = by_id["A7"]
    assert a7.passed
    assert a7.n_failed == MANIFEST["A7"] == a7.details["holds"]
    assert a7.details["population"] == MANIFEST["USD.rides"]
    assert a7.details["holds"] == MANIFEST["A7"]
    assert a7.details["above"] == MANIFEST["A7.above"]
    assert a7.details["below"] == MANIFEST["A7.below"]
    assert a7.details["not_computable"] == 0
    assert a7.metric == MANIFEST["A7"] / MANIFEST["USD.rides"]


def test_temporal_checks_hit_their_injections_and_nothing_else(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    by_id = {r.check_id: r for r in run_checks(full_synthetic_derived.lazy())}
    d = full_synthetic_derived
    for check_id in ("T1", "T2", "T3", "T4", "T5"):
        assert by_id[check_id].n_failed == MANIFEST[check_id], check_id
    assert [by_id[k].passed for k in ("T1", "T2", "T3", "T4", "T5")] == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert by_id["T3"].details["by_field"] == {"over_30_days": MANIFEST["T3.30d"]}
    # T3 and T4 fire on the ancient order, and only on it.
    (ancient,) = INJECTIONS["ancient_order"]
    for check_id in ("T3", "T4"):
        sample = by_id[check_id].sample
        assert sample is not None
        assert sample["checksum"].to_list() == [d["checksum"][ancient]], check_id
    # The negative controls: the APAC row (order date = email date + 1) is within
    # the tolerance but is after its email as an instant; the Lyft 5 h row is not.
    (apac,) = INJECTIONS["apac_next_day"]
    assert d["order_time"][apac] > d["email_time"][apac]
    assert by_id["T2"].details["by_field"] == {"order_after_email_no_tolerance": 1}
    # The tip update keeps its ride's insert_time, which precedes its email_time.
    _, update = INJECTIONS["ride_tip_update"]
    assert d["insert_time"][update] < d["email_time"][update]


def test_date_only_check_matches_the_manifest(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    t6 = next(
        r for r in run_checks(full_synthetic_derived.lazy()) if r.check_id == "T6"
    )
    assert t6.passed
    assert t6.n_failed == MANIFEST["T6"]
    merchants = t6.details["by_merchant"]
    assert isinstance(merchants, dict)
    assert {m: v["date_only"] for m, v in merchants.items()} == {
        m: MANIFEST[f"T6.{m}"] for m in merchants if f"T6.{m}" in MANIFEST
    } | {m: 0 for m in merchants if f"T6.{m}" not in MANIFEST}
    # Uber order_time is date-only throughout; Lyft carries a time of day.
    assert merchants["Uber"]["share"] == 1.0
    assert "T6.Lyft" not in MANIFEST
    assert merchants["Lyft"]["date_only"] == 0


def test_fare_outliers_hit_the_injection_and_nothing_else(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    by_id = {r.check_id: r for r in run_checks(full_synthetic_derived.lazy())}
    d = full_synthetic_derived
    o4, o5 = by_id["O4"], by_id["O5"]
    assert (o4.n_failed, o4.passed) == (MANIFEST["O4"], False)
    assert o4.n_evaluated == 277  # USD rides with >= 0.5 mi, >= 60 s and a family
    (outlier,) = INJECTIONS["fare_outlier_10x"]
    assert o4.sample is not None
    assert o4.sample["checksum"].to_list() == [d["checksum"][outlier]]
    # Not flagged: the 60-mile trip at a normal $/mile (a $/minute outlier only), the
    # local-currency Colombia row (gated out by the currency gate) and the km row.
    (fast,) = INJECTIONS["implausible_speed"]
    (colombia,) = INJECTIONS["non_us_sender"]
    assert d["checksum"][fast] not in o4.sample["checksum"].to_list()
    assert d["checksum"][colombia] not in o4.sample["checksum"].to_list()
    assert d["order_total_amount"][colombia] == 5718.0
    assert d["currency_suspect"][colombia]
    # O5 is context: it flags the injected fare and the (legitimate) 60-mile trip.
    assert (o5.n_failed, o5.passed) == (MANIFEST["O5"], True)
    merchants = o5.details["by_merchant"]
    assert isinstance(merchants, dict)
    assert merchants["Uber"]["flagged"] == MANIFEST["O5"]
    assert merchants["Lyft"]["flagged"] == 0
    scores = score(d.lazy()).frame
    o5_rows = scores.filter(pl.col("rz_total").abs() > 3.5)["_row"].to_list()
    assert sorted(o5_rows) == sorted([outlier, fast])
    # every baseline segment is big enough to be scored
    assert o4.details["scored"] == o4.n_evaluated
    assert o4.details["by_direction"] == {"high": 1, "low": 0, "mixed": 0}
    assert o4.details["k"] == 4.0


def test_degenerate_columns_are_the_documented_ones(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    by_id = {r.check_id: r for r in run_checks(full_synthetic_derived.lazy())}
    # The derived columns are not part of the delivered file's profile.
    assert by_id["S4"].details["columns"] == list(ALL_NULL)
    assert by_id["S5"].details["columns"] == list(CONSTANT)


# --- end to end ----------------------------------------


def test_cli_end_to_end_on_the_full_fixture(
    full_synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    # The negative car ride fails A1, an error check, so the run exits 1.
    assert qc_main(["run", "--input", str(full_synthetic_csv), "--out", str(out)]) == 1

    summary, checks, _outcome = capsys.readouterr().out.split("\n\n")
    fields = dict(line.split(":", 1) for line in summary.splitlines())
    assert fields["encoding"].strip() == "UTF-8, no BOM, CRLF line endings"
    assert fields["rows"].strip() == str(N_ROWS)
    assert fields["columns"].strip() == "33"
    assert fields["all-null columns"].strip() == f"5 ({', '.join(ALL_NULL)})"
    assert fields["constant columns"].strip() == f"4 ({', '.join(CONSTANT)})"
    assert fields["duplicate rows"].strip() == "0"

    lines = {line.split()[2]: line.split() for line in checks.splitlines()}
    assert list(lines) == [
        "header_valid",
        "utf8_valid",
        "cast_failures",
        "row_count_band",
        "required_fields",
        "id_formats",
        "non_negative_total",
        "item_price_equals_subtotal",
        "zero_fare_rides",
        "discount_within_subtotal",
        "tax_within_total",
        "unexplained_residual",
        "checksum_unique",
        "category_mismatch",
        "merchant_sender_consistent",
        "unknown_sender",
        "ingestion_order",
        "order_before_receipt",
        "late_receipts",
        "order_before_window",
        "email_after_update",
        "implausible_speed",
        "zero_trip_metrics",
        "extreme_trip",
        "all_null_columns",
        "constant_columns",
        "duplicate_rows",
        "ride_completeness",
        "identity_share",
        "receipt_kinds",
        "unclassified_subject_templates",
        "duplicate_receipts",
        "dedupe_key_unique",
        "cross_panelist_duplicates",
        "near_duplicate_rides",
        "description_parse_coverage",
        "currency_suspect_rows",
        "date_only_order_time",
        "fare_outlier_conditional",
        "fare_outlier_univariate",
    ]
    assert lines["non_negative_total"][1] == "FAIL"
    spend = N_ROWS - MANIFEST["K1.non_spend"]
    assert lines["non_negative_total"][3] == f"{MANIFEST['A1']}/{spend}"
    assert lines["category_mismatch"][1] == "FAIL"
    assert lines["category_mismatch"][3] == f"{MANIFEST['V2']}/{N_ROWS}"
    assert lines["receipt_kinds"][3] == f"{MANIFEST['K1.non_spend']}/{N_ROWS}"
    assert lines["unclassified_subject_templates"][3] == f"0/{N_ROWS}"
    for name in ("duplicate_receipts", "cross_panelist_duplicates"):
        assert lines[name][1] == "FAIL"
    assert lines["near_duplicate_rides"][1] == "FAIL"
    assert lines["dedupe_key_unique"][1] == "PASS"
    for name in ("cast_failures", "checksum_unique", "duplicate_rows"):
        assert lines[name][1] == "PASS"
        assert lines[name][3].startswith("0/")
    assert pl.read_parquet(out / "synthetic.curated.parquet").height == N_ROWS
    report = (out / "report.md").read_text(encoding="utf-8")
    section = report.split("## Description parsing")[1].split("## Column profile")[0]
    assert "| Lyft |" in section
    assert "| 100.00% |" in section
    assert "Distance unit: mi " in section
    assert "| uberx |" in section
    assert "| lyft_standard |" in section


def test_build_synthetic_is_repeatable() -> None:
    assert build_synthetic().equals(build_synthetic())


def test_parsed_trip_columns_of_the_injections(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    df = full_synthetic_derived
    (zero,) = INJECTIONS["zero_duration_trip"]
    # A parsed zero is 0, not null: O2 counts it.
    assert df["duration_s"][zero] == 0
    assert df["distance_mi"][zero] == 2.0
    (fast,) = INJECTIONS["implausible_speed"]
    assert df["distance_mi"][fast] == 60.0
    assert df["duration_s"][fast] == 1200
    (km,) = INJECTIONS["km_generic_sender"]
    assert df["distance_unit"][km] == "km"
    assert df["distance_mi"][km] == pytest.approx(6.8 * 0.621371)
    assert df["duration_s"][km] == 11 * 60
    assert df["product_family"][km] == "uberx"


def test_product_families_of_the_baseline(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    base = full_synthetic_derived.head(N_BASELINE)
    families = {
        m: set(base.filter(pl.col("merchant_name") == m)["product_family"])
        for m in ("Uber", "Lyft")
    }
    assert families == {"Uber": {"uberx"}, "Lyft": {"lyft_standard"}}
    assert base["distance_mi"].null_count() == 0
    assert base["duration_s"].null_count() == 0
    assert set(base["distance_unit"]) == {"mi"}


def test_pc1_counts_no_unparsed_ride_row(
    full_synthetic_derived: pl.DataFrame,
) -> None:
    by_id = {r.check_id: r for r in run_checks(full_synthetic_derived.lazy())}
    pc1 = by_id["PC1"]
    assert pc1.n_failed == MANIFEST["PC1"]
    assert pc1.passed
    merchants = pc1.details["merchants"]
    assert isinstance(merchants, dict)
    assert set(merchants) == {"Uber", "Lyft"}
