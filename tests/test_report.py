"""The QC report: JSON and Markdown, samples, exit codes, thresholds."""

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import polars as pl
import pytest
from fixtures.make_synthetic import MANIFEST, USD_AMOUNT
from synthetic import make_tar, write_csv_text

from rideshare_data_pipeline import __version__
from rideshare_data_pipeline.checks.model import CheckResult, Severity
from rideshare_data_pipeline.cli import main
from rideshare_data_pipeline.config import QCConfig
from rideshare_data_pipeline.parse import KINDS, RIDER_SPEND_KINDS
from rideshare_data_pipeline.report import (
    RunInfo,
    exit_code,
    gates,
    render_markdown,
    sample_name,
    schema_text,
    summarize,
)

IMPLEMENTED = (
    "S1",
    "S2",
    "S3",
    "C1",
    "C2",
    "C3",
    "S6",
    "A1",
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
)
DUPLICATE_CHECKS = (
    "id_formats",
    "currency_suspect_rows",
    "unknown_sender",
    "duplicate_receipts",
    "cross_panelist_duplicates",
    "near_duplicate_rides",
)
LOOSE_V2 = "[checks.category_mismatch]\nmax_failed_rate = 0.02\n" + "".join(
    f"[checks.{name}]\nmax_failed_rate = 0.05\n" for name in DUPLICATE_CHECKS
)


def run_cli(csv: Path, out: Path, *extra: str, config: str | None = None) -> int:
    argv = ["run", "--input", str(csv), "--out", str(out), *extra]
    if config is not None:
        path = out.parent / "qc.toml"
        path.write_text(config, encoding="utf-8")
        argv += ["--config", str(path)]
    return main(argv)


def load(out: Path) -> dict[str, object]:
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def checks_by_name(report: dict[str, object]) -> dict[str, dict[str, object]]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return {c["name"]: c for c in checks}


def make_result(**changes: object) -> CheckResult:
    base = CheckResult(
        name="c",
        check_id="C1",
        category="test",
        severity=Severity.ERROR,
        passed=False,
        metric=0.5,
        threshold=0.0,
        n_failed=1,
        n_evaluated=2,
        predicate="a | b",
        sample=pl.DataFrame({"k": ["x|y", "line\nbreak"], "n": [1, 2]}),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def make_run(**changes: object) -> RunInfo:
    base = RunInfo(
        tool_version="1",
        input_name="in.csv",
        sha256="ab" * 32,
        size_bytes=10,
        has_bom=False,
        line_ending="lf",
        rows=1234,
        panelists=7,
        window_start="2019-01-01T00:00:00",
        window_end="2019-02-01T00:00:00",
        started="s",
        finished="f",
        elapsed_seconds=0.1,
        config=QCConfig(),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


# --- the synthetic fixture, end to end -----------------------------------------


def test_run_writes_reports_and_samples(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert run_cli(full_synthetic_csv, out) == 1  # A1 is an error and fails
    assert (out / "report.json").is_file()
    assert (out / "report.md").is_file()
    assert (out / "synthetic.parquet").is_file()
    assert sorted(p.name for p in (out / "samples").iterdir()) == [
        "category_mismatch.parquet",
        "cross_panelist_duplicates.parquet",
        "duplicate_receipts.parquet",
        "id_formats.parquet",
        "near_duplicate_rides.parquet",
        "non_negative_total.parquet",
        "unknown_sender.parquet",
    ]
    sample = pl.read_parquet(out / "samples" / "non_negative_total.parquet")
    assert sample.height == MANIFEST["A1"]
    assert "subject" not in " ".join(sample.columns)


def test_rerun_removes_samples_of_checks_that_now_pass(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert run_cli(full_synthetic_csv, out) == 1
    assert (out / "samples" / "non_negative_total.parquet").is_file()
    passing = "[checks.non_negative_total]\nmax_failed_rate = 0.01\n" + LOOSE_V2
    assert run_cli(full_synthetic_csv, out, config=passing) == 0
    assert list((out / "samples").glob("*.parquet")) == []


def test_report_json_validates_against_the_schema(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    schema = json.loads(schema_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(load(out), schema, cls=jsonschema.Draft202012Validator)


def test_report_json_content_matches_the_manifest(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    report = load(out)
    checks = report["checks"]
    assert isinstance(checks, list)
    assert {c["check_id"]: c["n_failed"] for c in checks} == {
        **{k: MANIFEST[k] for k in IMPLEMENTED},
        "K1": MANIFEST["K1.non_spend"],
        "K2": 0,
        "PC1": MANIFEST["PC1"],
    }
    assert report["schema_version"] == 1
    assert report["exit_code"] == 1
    assert report["summary"] == {
        "errors_failed": 1,
        "warnings_failed": 7,  # V2, U2, U3, U4, CUR1, V3, S6
        "info": 4,
        "passed": 15,
        "total": 23,
    }
    run = report["run"]
    assert isinstance(run, dict)
    import hashlib

    assert (
        run["input_sha256"]
        == hashlib.sha256(full_synthetic_csv.read_bytes()).hexdigest()
    )
    assert run["input_name"] == "synthetic.csv"
    assert str(tmp_path) not in json.dumps(report)  # no local path leaks
    assert run["tool_version"] == __version__
    assert run["rows"] == 294
    assert (
        run["panelists"]
        == pl.read_parquet(out / "synthetic.parquet")["user_id"].n_unique()
    )
    assert run["window"]["start"] < run["window"]["end"]
    a1 = checks_by_name(report)["non_negative_total"]
    assert a1["sample_file"] == "samples/non_negative_total.parquet"
    assert a1["threshold"] == 0.0
    assert checks_by_name(report)["duplicate_rows"]["sample_file"] is None


def test_report_markdown_layout(full_synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    lines = (out / "report.md").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# QC report: synthetic.csv (")
    assert "| check | severity | status | failed | rate | threshold | rule |" in lines
    table = [
        ln for ln in lines if ln.startswith("| ") and ("FAIL" in ln or "PASS" in ln)
    ]
    assert "non_negative_total" in table[0]  # failures first
    for heading in (
        "## Summary",
        "## Composition by receipt kind",
        "## Duplicate receipts",
        "## Spend",
        "## Column profile",
        "## Samples",
        "### non_negative_total",
    ):
        assert heading in lines


# --- composition by receipt kind -----------------------------------------------


def test_composition_in_json_and_markdown(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    report = load(out)
    schema = json.loads(schema_text())
    jsonschema.validate(report, schema, cls=jsonschema.Draft202012Validator)
    assert report["schema_version"] == 1

    table = report["composition"]
    assert isinstance(table, dict)
    assert table["merchants"] == ["Lyft", "Uber"]
    kinds = {k["kind"]: k for k in table["kinds"]}
    assert list(kinds) == list(KINDS)  # canonical order, zero rows included
    assert table["rows"] == sum(k["total"] for k in kinds.values())
    expected = {
        "ride": MANIFEST["K1.ride"],
        "tip_update": MANIFEST["K1.tip_update"],
        "driver_earnings": MANIFEST["K1.driver_earnings"],
        "pass_or_credit": MANIFEST["K1.pass"],
        "micromobility": MANIFEST["K1.micromobility"],
        "food_order": MANIFEST["K1.food_order"],
        "freight": MANIFEST["K1.freight"],
    }
    assert {k: v["total"] for k, v in kinds.items() if v["total"]} == expected
    assert {k for k, v in kinds.items() if v["rider_spend"]} == RIDER_SPEND_KINDS
    for kind in kinds.values():
        assert sum(kind["by_merchant"].values()) == kind["total"]
    assert kinds["driver_earnings"]["by_merchant"] == {"Lyft": 1, "Uber": 0}

    lines = (out / "report.md").read_text(encoding="utf-8").splitlines()
    start = lines.index("## Composition by receipt kind")
    section = lines[start : lines.index("## Duplicate receipts")]
    assert "| kind | rider spend | Lyft | Uber | total |" in section
    assert "| driver_earnings | no | 1 | 0 | 1 |" in section
    assert "| food_order | no | 0 | 2 | 2 |" in section
    assert (
        "| ride | yes | "
        + "".join(
            f"{n:,} | "
            for n in (
                kinds["ride"]["by_merchant"]["Lyft"],
                kinds["ride"]["by_merchant"]["Uber"],
            )
        )
        + f"{MANIFEST['K1.ride']:,} |"
        in section
    )
    assert "| **total** |  | " in "\n".join(section)
    assert section[-2].endswith(f"| {table['rows']:,} |")


def test_composition_is_omitted_when_the_check_is_disabled(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out, config="[checks.receipt_kinds]\nenabled = false\n")
    report = load(out)
    assert "composition" not in report
    jsonschema.validate(report, json.loads(schema_text()))
    assert "_Not available: the receipt_kinds check is disabled._" in (
        out / "report.md"
    ).read_text(encoding="utf-8")


def test_schema_rejects_unknown_composition_keys(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    report = load(out)
    composition = report["composition"]
    assert isinstance(composition, dict)
    composition["extra"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(report, json.loads(schema_text()))


# --- spend --------------------------------------------------------------------


def test_spend_in_json_and_markdown(full_synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    report = load(out)
    jsonschema.validate(report, json.loads(schema_text()))
    spend = report["spend"]
    assert isinstance(spend, dict)
    assert spend["naive_total"]["rows"] == report["run"]["rows"]  # type: ignore[index]
    assert spend["currency_suspect"]["rows"] == MANIFEST["CUR1"]
    assert spend["clean_usd_rider_spend"] == {
        "rows": MANIFEST["USD.rides"],
        "amount": USD_AMOUNT,
    }
    assert spend["suspect_rider_spend"]["rows"] == MANIFEST["CUR1"]
    assert "local currency" in spend["note"]
    # the waterfall adds up
    taken = sum(
        spend[k]["amount"]
        for k in ("currency_suspect", "non_rider_spend", "superseded")
    )
    assert spend["naive_total"]["amount"] - taken == pytest.approx(
        spend["clean_usd_rider_spend"]["amount"]
    )

    text = (out / "report.md").read_text(encoding="utf-8")
    section = text.split("## Spend")[1].split("## Description parsing")[0]
    assert "| minus currency-suspect rows (local currency) | 3 | -5,734.35 |" in section
    assert "| **= clean USD rider spend** | 280 | 3,777.57 |" in section
    assert "| non_us_sender | 2 |" in section
    assert "| co | 1 |" in section
    assert "not summable as USD" in section


def test_spend_is_omitted_when_cur1_is_disabled(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(
        full_synthetic_csv,
        out,
        config="[checks.currency_suspect_rows]\nenabled = false\n",
    )
    report = load(out)
    assert "spend" not in report
    jsonschema.validate(report, json.loads(schema_text()))
    assert "_Not available: the currency_suspect_rows check is disabled._" in (
        out / "report.md"
    ).read_text(encoding="utf-8")


def test_schema_rejects_a_malformed_spend(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    report = load(out)
    spend = report["spend"]
    assert isinstance(spend, dict)
    del spend["naive_total"]["amount"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(report, json.loads(schema_text()))


# --- thresholds, severities, enabled -------------------------------------------


def test_threshold_override_makes_the_check_pass_and_shows_in_the_report(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    config = "[checks.non_negative_total]\nmax_failed_rate = 0.01\n" + LOOSE_V2
    assert run_cli(full_synthetic_csv, out, config=config) == 0
    report = load(out)
    a1 = checks_by_name(report)["non_negative_total"]
    assert (a1["passed"], a1["threshold"], a1["n_failed"]) == (True, 0.01, 1)
    run = report["run"]
    assert isinstance(run, dict)
    assert run["config"]["checks"]["non_negative_total"]["threshold"] == 0.01
    assert not (out / "samples").exists()
    assert "| non_negative_total | error | PASS | 1 | 0.3497% | 0.01 |" in (
        out / "report.md"
    ).read_text(encoding="utf-8")


def test_warn_severity_and_fail_on(full_synthetic_csv: Path, tmp_path: Path) -> None:
    config = '[checks.non_negative_total]\nseverity = "warn"\n'
    assert run_cli(full_synthetic_csv, tmp_path / "a", config=config) == 0
    report = load(tmp_path / "a")
    assert report["summary"]["warnings_failed"] == 8  # type: ignore[index]  # A1, V2, U2, U3, U4, CUR1, V3, S6
    assert report["exit_code"] == 0
    assert (
        run_cli(full_synthetic_csv, tmp_path / "b", "--fail-on", "warn", config=config)
        == 1
    )
    assert load(tmp_path / "b")["exit_code"] == 1
    assert (
        run_cli(full_synthetic_csv, tmp_path / "c", "--fail-on", "error", config=config)
        == 0
    )


def test_cli_fail_on_overrides_the_config(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    config = 'fail_on = "warn"\n[checks.non_negative_total]\nseverity = "warn"\n'
    assert run_cli(full_synthetic_csv, tmp_path / "a", config=config) == 1
    assert (
        run_cli(full_synthetic_csv, tmp_path / "b", "--fail-on", "error", config=config)
        == 0
    )
    effective = load(tmp_path / "b")["run"]["config"]["fail_on"]  # type: ignore[index]
    assert effective == "error"


def test_info_checks_never_gate(full_synthetic_csv: Path, tmp_path: Path) -> None:
    config = (
        '[checks.non_negative_total]\nseverity = "info"\n'
        '[checks.category_mismatch]\nseverity = "info"\n'
        + "".join(f'[checks.{n}]\nseverity = "info"\n' for n in DUPLICATE_CHECKS)
    )
    assert (
        run_cli(full_synthetic_csv, tmp_path / "a", "--fail-on", "warn", config=config)
        == 0
    )


def test_disabled_check_is_removed(full_synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config = (
        "[checks.non_negative_total]\nenabled = false\n"
        "[checks.category_mismatch]\nenabled = false\n"
        "[checks.duplicate_rows]\nenabled = false\n"
        + "".join(f"[checks.{n}]\nenabled = false\n" for n in DUPLICATE_CHECKS)
    )
    assert run_cli(full_synthetic_csv, out, config=config) == 0
    names = set(checks_by_name(load(out)))
    assert "non_negative_total" not in names
    assert "category_mismatch" not in names
    assert "duplicate_rows" not in names
    assert "checksum_unique" in names
    assert not (out / "samples").exists()


def test_sample_limit_from_config(full_synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config = "sample_limit = 0\n[checks.non_negative_total]\nmax_failed_rate = 0.0\n"
    assert run_cli(full_synthetic_csv, out, config=config) == 1
    # No rows were sampled, so there's no sample file to write.
    assert not (out / "samples").exists()
    assert checks_by_name(load(out))["non_negative_total"]["sample_file"] is None


# --- exit status 2: nothing written --------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        "[checks.not_a_check]\nenabled = false\n",
        "[checks.non_negative_total]\nmax_rate = 0.1\n",
        "typo = 1\n",
    ],
    ids=["unknown-check", "unknown-key", "unknown-top-key"],
)
def test_bad_config_exits_2_and_writes_nothing(
    config: str,
    full_synthetic_csv: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    assert run_cli(full_synthetic_csv, out, config=config) == 2
    assert capsys.readouterr().err.startswith("rideshare-qc: error: qc.toml: ")
    assert not out.exists()


def test_missing_config_file_exits_2(full_synthetic_csv: Path, tmp_path: Path) -> None:
    argv = ["run", "--input", str(full_synthetic_csv), "--out", str(tmp_path / "o")]
    assert main([*argv, "--config", str(tmp_path / "nope.toml")]) == 2


def test_bad_input_exits_2_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert run_cli(tmp_path / "missing.csv", out) == 2
    assert not out.exists()


def test_invalid_fail_on_value_exits_2(
    full_synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        run_cli(full_synthetic_csv, tmp_path / "o", "--fail-on", "info")
    assert info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_clean_input_exits_0(synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert run_cli(synthetic_csv, out) == 0
    assert load(out)["exit_code"] == 0
    assert not (out / "samples").exists()


def test_header_only_input_reports_a_null_window(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert run_cli(write_csv_text(tmp_path / "h.csv", []), out) == 0
    report = load(out)
    schema = json.loads(schema_text())
    jsonschema.validate(report, schema)
    run = report["run"]
    assert isinstance(run, dict)
    assert run["window"] == {"start": None, "end": None}
    assert (run["rows"], run["panelists"]) == (0, 0)
    assert "window n/a" in (out / "report.md").read_text(encoding="utf-8")


# --- pure functions ------------------------------------------------------------------


def test_summarize_counts() -> None:
    results = [
        make_result(),  # error, failed
        make_result(severity=Severity.WARN),  # warn, failed
        make_result(severity=Severity.WARN, passed=True),
        make_result(severity=Severity.INFO, passed=True),
        make_result(severity=Severity.INFO, passed=False),
    ]
    assert summarize(results) == {
        "errors_failed": 1,
        "warnings_failed": 1,
        "info": 2,
        "passed": 2,
        "total": 5,
    }
    assert summarize([]) == dict.fromkeys(
        ("errors_failed", "warnings_failed", "info", "passed", "total"), 0
    )


@pytest.mark.parametrize(
    ("severity", "passed", "fail_on", "expected"),
    [
        (Severity.ERROR, False, Severity.ERROR, True),
        (Severity.ERROR, False, Severity.WARN, True),
        (Severity.ERROR, True, Severity.WARN, False),
        (Severity.WARN, False, Severity.ERROR, False),
        (Severity.WARN, False, Severity.WARN, True),
        (Severity.INFO, False, Severity.WARN, False),
        (Severity.INFO, False, Severity.ERROR, False),
    ],
)
def test_gates(
    severity: Severity, passed: bool, fail_on: Severity, expected: bool
) -> None:
    result = make_result(severity=severity, passed=passed)
    assert gates(result, fail_on) is expected
    assert exit_code([result], fail_on) == int(expected)


def test_markdown_escapes_pipes_and_flattens_newlines() -> None:
    text = render_markdown([make_result()], make_run(), 1)
    assert "| a \\| b |" in text  # the rule cell
    assert "| x\\|y | 1 |" in text  # a sample cell
    assert "| line break | 2 |" in text
    assert "# QC report: in.csv (abababababab)" in text
    assert (
        "rows 1,234 · panelists 7 · window 2019-01-01T00:00:00..2019-02-01T00:00:00"
        in text
    )
    assert "[samples/c.parquet](samples/c.parquet)" in text


def test_markdown_limits_sample_rows() -> None:
    sample = pl.DataFrame({"n": list(range(10))})
    run = make_run(config=QCConfig(sample_limit=3))
    text = render_markdown([make_result(sample=sample, n_failed=10)], run, 1)
    assert "3 of 10 failing rows" in text
    assert "| 2 |" in text
    assert "| 3 |" not in text


def test_markdown_sorts_by_severity_then_failures_first() -> None:
    results = [
        make_result(name="info_pass", severity=Severity.INFO, passed=True, sample=None),
        make_result(name="warn_pass", severity=Severity.WARN, passed=True, sample=None),
        make_result(name="warn_fail", severity=Severity.WARN, sample=None),
        make_result(name="err_pass", passed=True, sample=None),
        make_result(name="err_fail", sample=None),
    ]
    rows = [
        ln.split("|")[1].strip()
        for ln in render_markdown(results, make_run(), 1).splitlines()
        if ln.startswith("| ") and ("PASS" in ln or "FAIL" in ln)
    ]
    assert rows == ["err_fail", "err_pass", "warn_fail", "warn_pass", "info_pass"]


def test_sample_name_only_for_failed_results_with_rows() -> None:
    assert sample_name(make_result()) == "samples/c.parquet"
    assert sample_name(make_result(passed=True)) is None
    assert sample_name(make_result(sample=None)) is None
    assert sample_name(make_result(sample=pl.DataFrame({"k": []}))) is None


def test_duplicate_receipts_in_json_and_markdown(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    u2 = checks_by_name(load(out))["duplicate_receipts"]
    details = u2["details"]
    assert isinstance(details, dict)
    assert details["groups"] == MANIFEST["U2.groups"]
    assert details["excess_rows"] == MANIFEST["U2.superseded"]
    assert details["excess_spend"] == pytest.approx(29.36)
    assert [p["pair"] for p in details["kind_pairs"]] == [
        "ride+ride",
        "ride+tip_update",
    ]

    lines = (out / "report.md").read_text(encoding="utf-8").splitlines()
    start = lines.index("## Duplicate receipts")
    section = lines[start : lines.index("## Column profile")]
    assert any("29.36" in ln for ln in section)
    assert "| ride+tip_update | 1 | 12.38 |" in section


def test_archive_report_with_a_member_validates_against_the_schema(
    synthetic_csv: Path, tmp_path: Path
) -> None:
    archive = make_tar(tmp_path / "a.tar.gz", {"d/x.csv": synthetic_csv.read_bytes()})
    out = tmp_path / "out"
    run_cli(archive, out)
    report = load(out)
    run = report["run"]
    assert isinstance(run, dict)
    assert run["member"]["name"] == "d/x.csv"
    jsonschema.validate(report, json.loads(schema_text()))
