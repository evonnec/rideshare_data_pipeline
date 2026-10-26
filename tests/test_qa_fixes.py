"""Fixes for the QA findings (#45): non-finite numbers, exit codes, blank lines,
outputs, config validation and the report wording.

Every value is made up. F-numbers refer to the findings of the QA report.
"""

import json
import random
import re
import tomllib
from pathlib import Path

import jsonschema
import polars as pl
import pytest
from synthetic import synthetic_rows, write_csv_text

import rideshare_data_pipeline.cli as cli_module
import rideshare_data_pipeline.io as io_module
from rideshare_data_pipeline.checks.model import Severity
from rideshare_data_pipeline.cli import (
    EXIT_GATED,
    EXIT_INPUT_ERROR,
    EXIT_INTERNAL_ERROR,
    EXIT_OK,
    main,
)
from rideshare_data_pipeline.config import (
    KNOWN_CHECKS,
    ConfigError,
    load_config,
    parse_config,
)
from rideshare_data_pipeline.curated import curated_schema
from rideshare_data_pipeline.io import (
    CastError,
    cast_failures,
    input_stem,
    inspect_csv,
    read_raw,
    read_typed,
    typed,
    write_parquet,
)
from rideshare_data_pipeline.report import schema_text
from rideshare_data_pipeline.schema import casts

N_CURATED = len(curated_schema(KNOWN_CHECKS))


def run_cli(csv: Path, out: Path, *extra: str) -> int:
    return main(["run", "--input", str(csv), "--out", str(out), *extra])


def report(out: Path) -> dict[str, object]:
    document = json.loads((out / "report.json").read_text(encoding="utf-8"))
    schema = json.loads(schema_text())
    jsonschema.validate(document, schema, cls=jsonschema.Draft202012Validator)
    assert isinstance(document, dict)
    return document


# --- F1: non-finite numbers are cast failures ------------------------------------

NON_FINITE = ["nan", "NaN", "inf", "-inf", "Infinity", "-Infinity", "1e400", "-1e400"]
MONEY_COLUMNS = [
    c for c, d in casts().items() if isinstance(d, pl.Float64) and c != "order_points"
]


@pytest.mark.parametrize("value", NON_FINITE)
def test_non_finite_money_is_a_cast_failure(tmp_path: Path, value: str) -> None:
    path = write_csv_text(
        tmp_path / "in.csv", synthetic_rows(3, order_total_amount={1: value})
    )
    raw = read_raw(path)
    assert cast_failures(raw)["order_total_amount"] == 1
    # typed() nulls it, so no non-finite value can reach a check
    assert typed(raw).collect()["order_total_amount"].is_finite().all()
    assert typed(raw).collect()["order_total_amount"].null_count() == 1
    with pytest.raises(CastError) as info:
        read_typed(path)
    assert info.value.counts == {"order_total_amount": 1}


@pytest.mark.parametrize("column", MONEY_COLUMNS)
def test_every_float_column_rejects_non_finite_values(
    tmp_path: Path, column: str
) -> None:
    path = write_csv_text(
        tmp_path / "in.csv", synthetic_rows(2, **{column: {0: "inf"}})
    )
    assert cast_failures(read_raw(path))[column] == 1


@pytest.mark.parametrize("value", NON_FINITE)
def test_non_finite_money_exits_2_and_writes_nothing(
    tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(
        tmp_path / "in.csv", synthetic_rows(3, order_total_amount={2: value})
    )
    out = tmp_path / "out"
    assert run_cli(path, out) == EXIT_INPUT_ERROR
    assert "order_total_amount: 1" in capsys.readouterr().err
    assert not out.exists()


def test_finite_extremes_still_parse(tmp_path: Path) -> None:
    path = write_csv_text(
        tmp_path / "in.csv",
        synthetic_rows(3, order_total_amount={0: "1e3", 1: "-0.5", 2: "1e300"}),
    )
    assert cast_failures(read_raw(path))["order_total_amount"] == 0


# --- F2: input-caused exceptions exit 2, internal errors exit 3 ------------------


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("email_time", "0000-01-01 00:00:00.000"),
        ("order_time", "0000-06-01 00:00:00.000"),
        ("email_time", "-0001-01-01 00:00:00.000"),
        ("start_source_folder_date", "0000-01-01"),
        ("delivery_date", "0000-01-01 00:00:00.000"),
    ],
)
def test_timestamp_outside_python_years_is_a_cast_failure(
    tmp_path: Path,
    column: str,
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_csv_text(
        tmp_path / "in.csv", synthetic_rows(3, **{column: {1: value}})
    )
    assert cast_failures(read_raw(path))[column] == 1
    out = tmp_path / "out"
    assert run_cli(path, out) == EXIT_INPUT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("rideshare-qc: error: ")
    assert "Traceback" not in err
    assert not out.exists()


def test_the_first_and_last_valid_years_parse(tmp_path: Path) -> None:
    path = write_csv_text(
        tmp_path / "in.csv",
        synthetic_rows(
            2,
            email_time={0: "0001-01-01 00:00:00.000", 1: "9999-12-31 23:59:59.999"},
        ),
    )
    assert cast_failures(read_raw(path))["email_time"] == 0
    assert run_cli(path, tmp_path / "out") in (EXIT_OK, EXIT_GATED)


def test_an_unexpected_exception_exits_3_with_one_line(
    synthetic_csv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        msg = "something broke\nsecond line stays out"
        raise RuntimeError(msg)

    monkeypatch.setattr(cli_module, "supersede_flags", boom)
    assert run_cli(synthetic_csv, tmp_path / "out") == EXIT_INTERNAL_ERROR == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "rideshare-qc: internal error: RuntimeError: something broke (please report)\n"
    )


def test_keyboard_interrupt_is_not_caught(
    synthetic_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_cli(synthetic_csv, tmp_path / "out")


def test_exit_codes_are_documented_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    out = capsys.readouterr().out
    for line in (
        "0  every check",
        "1  an error-severity",
        "2  input error",
        "3  internal",
    ):
        assert line in out
    assert (EXIT_OK, EXIT_GATED, EXIT_INPUT_ERROR, EXIT_INTERNAL_ERROR) == (0, 1, 2, 3)


def test_readme_documents_the_exit_codes() -> None:
    text = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    for code in "0123":
        assert f"| {code} |" in text
    assert "**Internal error**" in text


# --- F3: blank lines -------------------------------------------------------------


def test_a_trailing_blank_line_is_dropped_and_counted(tmp_path: Path) -> None:
    path = write_csv_text(tmp_path / "in.csv", synthetic_rows(5))
    clean_rows = read_typed(path)[0].height
    path.write_bytes(path.read_bytes() + b"\r\n")
    df, facts = read_typed(path)
    assert (df.height, facts.blank_lines) == (clean_rows, 1)
    assert df["checksum"].null_count() == 0


@pytest.mark.parametrize(
    ("suffix", "blanks"),
    [(b"\r\n", 1), (b"\r\n\r\n", 2), (b"\r\n\r\n\r\n", 3), (b"\n\n", 2), (b"", 0)],
)
def test_blank_line_counts_by_ending(
    tmp_path: Path, suffix: bytes, blanks: int
) -> None:
    path = write_csv_text(tmp_path / "in.csv", synthetic_rows(3))
    path.write_bytes(path.read_bytes() + suffix)
    assert inspect_csv(path).blank_lines == blanks
    assert read_typed(path)[0].height == 3


def test_blank_lines_in_the_middle_and_after_the_header(tmp_path: Path) -> None:
    lines = write_csv_text(tmp_path / "in.csv", synthetic_rows(4)).read_bytes()
    parts = lines.split(b"\r\n")
    path = tmp_path / "gaps.csv"
    path.write_bytes(b"\r\n".join([parts[0], b"", parts[1], b"", b"", *parts[2:]]))
    df, facts = read_typed(path)
    assert (df.height, facts.blank_lines) == (4, 3)


def test_two_blank_lines_no_longer_fail_duplicate_rows(
    tmp_path: Path,
) -> None:
    path = write_csv_text(tmp_path / "in.csv", synthetic_rows(4))
    path.write_bytes(path.read_bytes() + b"\r\n\r\n")
    out = tmp_path / "out"
    assert run_cli(path, out) == EXIT_OK
    document = report(out)
    by_name = {c["name"]: c for c in document["checks"]}  # type: ignore[attr-defined]
    assert by_name["duplicate_rows"]["passed"] is True
    assert by_name["utf8_valid"]["details"]["blank_lines"] == 2
    run = document["run"]
    assert isinstance(run, dict)
    assert (run["rows"], run["blank_lines"]) == (4, 2)
    assert pl.read_parquet(out / "in.curated.parquet").height == 4


def test_the_summary_mentions_blank_lines_only_when_there_are_some(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv_text(tmp_path / "in.csv", synthetic_rows(3))
    run_cli(path, tmp_path / "o1")
    assert "blank lines" not in capsys.readouterr().out
    path.write_bytes(path.read_bytes() + b"\r\n")
    run_cli(path, tmp_path / "o2")
    assert re.search(r"blank lines:\s+1 \(dropped\)", capsys.readouterr().out)


BLANK_SAMPLES = [
    b"a,b\r\n\r\n1,2\r\n\r\n\r\n3,4\r\n\r\n",
    b"a\n\n\n\nb\n\nc\n",
    b"a\r\n\r\n\r\n\r\n\r\nb\r\n",
    b"\n\n\nx\r\n\r\n",
    b"x\r\r\n\r\n\n\r\n",
]


@pytest.mark.parametrize("data", BLANK_SAMPLES)
@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 7, 1024])
def test_blank_lines_are_counted_exactly_across_any_chunk_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: bytes, chunk: int
) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes(data)
    monkeypatch.setattr(io_module, "_CHUNK_BYTES", 1024)
    monkeypatch.setattr(io_module, "_CHUNK_BYTES", chunk)
    # An empty line is a line break right after another one, "\r" ignored.
    text = data.replace(b"\r", b"")
    expected = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1] == 10)
    assert io_module._scan_bytes(path)[4] == expected


def test_blank_line_counts_match_a_reference_on_random_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(45)
    path = tmp_path / "fuzz.csv"
    for _ in range(300):
        data = bytes(rng.choice(b"\r\nab") for _ in range(rng.randint(0, 60)))
        path.write_bytes(data)
        text = data.replace(b"\r", b"")
        expected = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1] == 10)
        monkeypatch.setattr(io_module, "_CHUNK_BYTES", rng.randint(1, 9))
        assert io_module._scan_bytes(path)[4] == expected, data


# --- F9: atomic outputs, and which files belong to a run -------------------------


def test_rerun_into_the_same_out_lists_only_this_runs_files(tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = write_csv_text(tmp_path / "first.csv", synthetic_rows(3))
    second = write_csv_text(tmp_path / "second.csv", synthetic_rows(4))
    assert run_cli(first, out) == EXIT_OK
    assert report(out)["outputs"] == {
        "parquet": "first.curated.parquet",
        "curated": {"name": "first.curated.parquet", "rows": 3, "columns": N_CURATED},
        "samples": [],
    }
    assert run_cli(second, out) == EXIT_OK
    # the old Parquet is not deleted (the directory may hold other runs' files),
    # but report.json says which file is this run's
    assert (out / "first.curated.parquet").exists()
    assert (out / "second.curated.parquet").exists()
    assert report(out)["outputs"] == {
        "parquet": "second.curated.parquet",
        "curated": {"name": "second.curated.parquet", "rows": 4, "columns": N_CURATED},
        "samples": [],
    }


def test_outputs_lists_the_sample_files(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert run_cli(full_synthetic_csv, out) == EXIT_GATED
    outputs = report(out)["outputs"]
    assert isinstance(outputs, dict)
    assert outputs["parquet"] == "synthetic.curated.parquet"
    samples = outputs["samples"]
    assert isinstance(samples, list)
    assert samples
    assert sorted(samples) == sorted(
        f"samples/{p.name}" for p in (out / "samples").iterdir()
    )


def test_a_failed_write_leaves_the_old_file_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.parquet"
    df = pl.DataFrame({"a": [1, 2]})
    write_parquet(df, target)
    before = target.read_bytes()

    def partial(self: pl.DataFrame, file: Path, **_kwargs: object) -> None:
        Path(file).write_bytes(b"half")
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", partial)
    with pytest.raises(OSError, match="disk full"):
        write_parquet(pl.DataFrame({"a": [3]}), target)
    assert target.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["data.parquet"]


def test_a_first_failed_write_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def partial(self: pl.DataFrame, file: Path, **_kwargs: object) -> None:
        Path(file).write_bytes(b"half")
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", partial)
    with pytest.raises(OSError, match="disk full"):
        write_parquet(pl.DataFrame({"a": [3]}), tmp_path / "new.parquet")
    assert list(tmp_path.iterdir()) == []


def test_report_files_are_written_atomically(
    synthetic_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    assert run_cli(synthetic_csv, out) == EXIT_OK
    old = (out / "report.md").read_text(encoding="utf-8")

    def broken(path: object, text: str) -> None:
        raise OSError(text[:0] or "no space")

    # if the report can't be written, the previous file stays whole
    monkeypatch.setattr("rideshare_data_pipeline.report.write_text_atomic", broken)
    assert run_cli(synthetic_csv, out) == EXIT_INPUT_ERROR
    assert (out / "report.md").read_text(encoding="utf-8") == old
    assert not [p for p in out.iterdir() if ".tmp-" in p.name]


# --- F15: a file named .csv ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "stem"),
    [
        (".csv", "input"),
        (".CSV", "input"),
        ("..csv", "input"),
        (".tar.bz2", "input"),
        (".hidden.csv", "hidden"),
        ("dir/.csv", "input"),
        ("Data.csv", "Data"),
    ],
)
def test_input_stem_is_never_empty_or_hidden(name: str, stem: str) -> None:
    assert input_stem(name) == stem


def test_a_file_named_dot_csv_writes_input_parquet(tmp_path: Path) -> None:
    path = write_csv_text(tmp_path / ".csv", synthetic_rows(3))
    out = tmp_path / "out"
    assert run_cli(path, out) == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == [
        "input.curated.parquet",
        "report.json",
        "report.md",
    ]
    assert report(out)["outputs"] == {
        "parquet": "input.curated.parquet",
        "curated": {"name": "input.curated.parquet", "rows": 3, "columns": N_CURATED},
        "samples": [],
    }


# --- F11, F12: config validation -------------------------------------------------


def test_cast_failures_cannot_be_disabled() -> None:
    with pytest.raises(ConfigError, match=r"cast_failures can't be disabled"):
        parse_config({"checks": {"cast_failures": {"enabled": False}}})


def test_cast_failures_may_keep_other_overrides() -> None:
    config = parse_config({"checks": {"cast_failures": {"severity": "warn"}}})
    assert config.checks["cast_failures"].severity is Severity.WARN


def test_a_config_with_every_check_disabled_is_an_error(tmp_path: Path) -> None:
    from rideshare_data_pipeline.config import KNOWN_CHECKS

    text = "".join(f"[checks.{n}]\nenabled = false\n" for n in KNOWN_CHECKS)
    path = tmp_path / "off.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="every check is disabled"):
        load_config(path)


def test_all_checks_disabled_exits_2(
    synthetic_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rideshare_data_pipeline.config import KNOWN_CHECKS

    config = tmp_path / "off.toml"
    config.write_text(
        "".join(f"[checks.{n}]\nenabled = false\n" for n in KNOWN_CHECKS),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert run_cli(synthetic_csv, out, "--config", str(config)) == EXIT_INPUT_ERROR
    assert "every check is disabled" in capsys.readouterr().err
    assert not out.exists()


def test_every_other_check_may_be_disabled(synthetic_csv: Path, tmp_path: Path) -> None:
    from rideshare_data_pipeline.config import KNOWN_CHECKS

    config = tmp_path / "one.toml"
    config.write_text(
        "".join(
            f"[checks.{n}]\nenabled = false\n"
            for n in KNOWN_CHECKS
            if n != "cast_failures"
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert run_cli(synthetic_csv, out, "--config", str(config)) == EXIT_OK
    names = [c["name"] for c in report(out)["checks"]]  # type: ignore[attr-defined]
    assert names == ["cast_failures"]


# --- micromobility line and the spend report -------------------------------------


def test_the_micromobility_line_is_part_of_the_non_rider_step(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert run_cli(full_synthetic_csv, out) == EXIT_GATED
    spend = report(out)["spend"]
    assert isinstance(spend, dict)
    micro = spend["micromobility"]
    assert micro == {"rows": 1, "amount": -4.5}  # the one synthetic bike
    steps = ("currency_suspect", "non_rider_spend", "superseded")
    naive, clean = spend["naive_total"], spend["clean_usd_rider_spend"]
    assert naive["rows"] - sum(spend[s]["rows"] for s in steps) == clean["rows"]
    assert (
        round(naive["amount"] - sum(spend[s]["amount"] for s in steps), 2)
        == (clean["amount"])
    )
    assert spend["non_rider_spend"]["rows"] >= micro["rows"]
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "of which micromobility (excluded; not rider spend) | 1 | -4.50" in markdown
    # the sub-line sits between the non-rider and the superseded steps
    order = [
        markdown.index(text)
        for text in (
            "minus non-rider-spend kinds",
            "of which micromobility",
            "minus superseded duplicates",
        )
    ]
    assert order == sorted(order)


# --- F13: report wording ---------------------------------------------------------


def test_the_near_duplicate_caption_says_what_the_counts_are(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    markdown = (out / "report.md").read_text(encoding="utf-8")
    section = markdown.split("### near_duplicate_rides", 1)[1].split("###", 1)[0]
    assert "sample rows of the 2 rows in 1 near-duplicate groups" in section
    assert "1 failing rows is the excess" in section
    assert " of 1 failing rows" not in section


def test_the_spend_section_explains_the_superseded_difference(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    markdown = (out / "report.md").read_text(encoding="utf-8")
    spend = markdown.split("## Spend", 1)[1].split("## Description parsing", 1)[0]
    assert 'Why "superseded" is ' in spend
    assert "rows here but 3 under Duplicate receipts" in spend


def test_the_parsing_section_labels_its_metric(
    full_synthetic_csv: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(full_synthetic_csv, out)
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "worst merchant unparsed rate" in markdown
    assert "`Increase Tip` / `Tip Increase`" in markdown


# --- F10: the sdist --------------------------------------------------------------


def test_the_sdist_is_configured_to_ship_tests_tools_example_and_lock() -> None:
    root = Path(__file__).parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    include = project["tool"]["uv"]["build-backend"]["source-include"]
    assert {"tests/**", "tools/**", "qc.example.toml", "uv.lock"} <= set(include)
    assert not (root / "pyproject.toml.orig").exists()
