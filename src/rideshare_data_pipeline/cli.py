"""The ``rideshare-qc`` command: ``run`` checks an input and writes the reports."""

import argparse
import dataclasses
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# None of these import polars at import time (they use PEP 810 lazy imports), so
# `rideshare-qc --help` and argument errors return without loading it.
from rideshare_data_pipeline import __version__
from rideshare_data_pipeline.cache import CacheStatus, load_typed
from rideshare_data_pipeline.checks import (
    default_dataset_checks,
    default_rules,
    run_checks,
)
from rideshare_data_pipeline.checks.model import CheckResult, Severity
from rideshare_data_pipeline.checks.structural import (
    cast_failures_check,
    header_valid_check,
    row_count_band_check,
    utf8_valid_check,
)
from rideshare_data_pipeline.config import (
    QCConfig,
    configure_results,
    configure_rules,
    load_config,
)
from rideshare_data_pipeline.curated import (
    curated_plan,
    flag_table,
    row_flags,
    write_curated,
)
from rideshare_data_pipeline.dedupe import supersede_flags
from rideshare_data_pipeline.io import (
    FileFacts,
    InputError,
    input_stem,
)
from rideshare_data_pipeline.parse import with_derived
from rideshare_data_pipeline.report import (
    MemberInfo,
    RunInfo,
    Written,
    exit_code,
    metric_text,
    summarize,
    write_report,
)
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS, casts

EXIT_OK = 0
EXIT_GATED = 1
EXIT_INPUT_ERROR = 2
EXIT_INTERNAL_ERROR = 3

CACHE_ENV = "RIDESHARE_QC_CACHE"

# The parser keeps this text's line breaks (RawDescriptionHelpFormatter).
_DESCRIPTION = """\
Quality-control checks for the rideshare e-receipt CSV.
"""

_RUN_DESCRIPTION = """\
Read the rideshare e-receipt CSV (or an archive holding it): check its UTF-8
encoding and header, cast the columns to their types, run the QC checks and write
into DIR: report.json, report.md, samples/<check>.parquet for each failed check,
and <input stem>.curated.parquet: every row, the typed input columns (unless
--drop-columns names some to leave out), the derived columns, the dedupe flags and
one qc_<check> boolean per row-flagged check (True: the row fails it), with
qc_error, qc_warn and qc_flags (the failing checks' names). No row is dropped:
filter on the flags. Blank lines are dropped and counted in the report.
Reports hold hashed user ids and masked subjects only, but
keep DIR outside the repository anyway.

With --cache-dir (or the environment variable RIDESHARE_QC_CACHE) the validated,
typed data is also kept as Parquet there, keyed by the input's SHA-256, and a later
run on the same file skips extraction and parsing. That cache is a typed copy of the
real data: keep it outside the repository too. Without either, nothing is cached.
"""

_RUN_EPILOG = f"""\
exit status:
  {EXIT_OK}  every check that gates passed
  {EXIT_GATED}  an error-severity check failed (with --fail-on warn, a failed warn
      check too); info checks never gate. The reports and the Parquet are still written.
  {EXIT_INPUT_ERROR}  input error (bad encoding or header, a damaged archive, values
      that don't cast to their column's type, including non-finite numbers and
      years outside 1 to 9999, a bad config file), invalid arguments, or
      unwritable output. Nothing is written when the input can't be read.
  {EXIT_INTERNAL_ERROR}  internal error: a bug in this tool, reported as one line on
      stderr (please report it). Not a verdict on the data.

Files are written atomically (to a temporary name, then renamed), so a file is
never partial. DIR may hold files from other runs: report.json's "outputs" lists
the files of this run.
"""


@dataclass(frozen=True, slots=True)
class Summary:
    """What a run found and wrote."""

    rows: int
    columns: int
    all_null: list[str] | None
    """None when the check was disabled in the config (likewise the next two)."""

    constant: list[str] | None
    duplicate_rows: int | None
    results: list[CheckResult]
    facts: FileFacts
    output: Path
    output_columns: int
    written: Written
    config: QCConfig
    exit_code: int
    cache: CacheStatus = "off"


def _column_list(text: str) -> list[str]:
    """Parse ``--drop-columns a,b``; every name must be a schema column."""
    columns = [c.strip() for c in text.split(",") if c.strip()]
    unknown = [c for c in columns if c not in EXPECTED_COLUMNS]
    if unknown:
        msg = f"not in the schema: {', '.join(unknown)}"
        raise argparse.ArgumentTypeError(msg)
    return columns


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``rideshare-qc`` (one subcommand: run)."""
    parser = argparse.ArgumentParser(
        prog="rideshare-qc",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run_parser = commands.add_parser(
        "run",
        help="check an input file and write the reports",
        description=_RUN_DESCRIPTION,
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "the CSV, or a .tar.bz2/.tar.gz/.tar.xz/.tar or .zip archive holding "
            "exactly one CSV (extracted to a temporary directory that is removed "
            "afterwards)"
        ),
    )
    run_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="directory for the reports and the Parquet; created if missing",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        metavar="FILE",
        help="TOML file with thresholds and severities (see qc.example.toml)",
    )
    run_parser.add_argument(
        "--fail-on",
        choices=[Severity.ERROR.value, Severity.WARN.value],
        help=(
            "the lowest severity whose failure exits 1 (default: the config's "
            "fail_on, else error)"
        ),
    )
    run_parser.add_argument(
        "--drop-columns",
        type=_column_list,
        default=[],
        metavar="A,B",
        help="comma-separated columns to leave out of the Parquet (default: none)",
    )
    run_parser.add_argument(
        "--cache-dir",
        type=Path,
        metavar="DIR",
        help=(
            "keep a typed Parquet copy of the input in DIR (also read from "
            f"${CACHE_ENV}); later runs on the same file skip extraction and "
            "parsing. Opt-in: "
            "without it nothing is cached. Holds real data, so keep it outside "
            "the repository"
        ),
    )
    run_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="rebuild the cache entry for this input even if a valid one exists",
    )
    return parser


def _iso(value: object) -> str | None:
    """ISO 8601 text for a datetime, None for the null of an empty column."""
    return value.isoformat() if isinstance(value, datetime) else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(
    input_path: Path,
    out_dir: Path,
    config: QCConfig | None = None,
    drop_columns: Sequence[str] = (),
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
) -> Summary:
    """Read the input, run the checks, write the Parquet and the report.

    Raises InputError if the input can't be read, and OSError if the output
    can't be written. Nothing is written until the input has been read. With a
    `cache_dir` the typed frame comes from, and is saved to, the Parquet cache (see
    cache.py); a cache that can't be used is a warning, never an error.
    """
    config = config or QCConfig()
    started, clock = _utc_now(), time.perf_counter()
    loaded = load_typed(input_path, cache_dir, refresh=refresh_cache)
    df, facts = loaded.df, loaded.facts
    # read_typed raises HeaderError, EncodingError or CastError on a bad header,
    # bad bytes or any cast failure, so reaching here means there were none;
    # recording that keeps the S1, S3 and S2 lines in the report.
    # The derived columns (receipt kind, profile, rider-spend flag) are computed
    # once, then the dedupe flag; the checks read them and the curated Parquet
    # keeps them.
    lf = supersede_flags(with_derived(df)).lazy()
    rules = configure_rules(default_rules(config.check_params()), config)
    # One instance of the dataset checks: the outlier pair caches its scores in it,
    # and the row flags below reuse them.
    dataset_checks = default_dataset_checks(config.check_params(), config.sample_limit)
    results = configure_results(
        [
            header_valid_check(facts),
            utf8_valid_check(facts),
            cast_failures_check(dict.fromkeys(casts(), 0)),
            row_count_band_check(df.height, config.min_rows, config.max_rows),
            *run_checks(
                lf,
                rules=rules,
                dataset_checks=dataset_checks,
                sample_limit=config.sample_limit,
            ),
        ],
        config,
    )
    by_name = {r.name: r for r in results}

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{input_stem(input_path)}.curated.parquet"
    flags = row_flags(lf, rules, dataset_checks, results)
    plan = curated_plan(lf, flag_table(lf, flags), drop_columns)
    output_columns = plan.collect_schema().len()
    write_curated(plan, output)

    code = exit_code(results, config.fail_on)
    window = df["email_time"]
    member = None
    if loaded.member is not None:
        member = MemberInfo(
            loaded.member,
            facts.sha256,
            facts.size_bytes,
            facts.has_bom,
            facts.line_ending,
        )
    info = RunInfo(
        tool_version=__version__,
        input_name=Path(input_path).name,
        sha256=loaded.input_sha256,
        size_bytes=loaded.input_size_bytes,
        has_bom=facts.has_bom,
        line_ending=facts.line_ending,
        rows=df.height,
        panelists=df["user_id"].drop_nulls().n_unique(),
        window_start=_iso(window.min()),
        window_end=_iso(window.max()),
        started=started,
        finished=_utc_now(),
        elapsed_seconds=time.perf_counter() - clock,
        config=config,
        member=member,
        cache=loaded.cache,
        blank_lines=facts.blank_lines,
        parquet_name=output.name,
        curated_columns=output_columns,
    )
    written = write_report(out_dir, results, info, code)
    return Summary(
        rows=df.height,
        columns=df.width,
        all_null=_columns(by_name.get("all_null_columns")),
        constant=_columns(by_name.get("constant_columns")),
        duplicate_rows=(
            by_name["duplicate_rows"].n_failed if "duplicate_rows" in by_name else None
        ),
        results=results,
        facts=facts,
        output=output,
        output_columns=output_columns,
        written=written,
        config=config,
        exit_code=code,
        cache=loaded.cache,
    )


def _columns(result: CheckResult | None) -> list[str] | None:
    """The column names an info check recorded in its details."""
    if result is None:
        return None
    columns = result.details["columns"]
    assert isinstance(columns, list)
    return [str(c) for c in columns]


def _names(columns: list[str] | None) -> str:
    if columns is None:
        return "disabled"
    return f"{len(columns)} ({', '.join(columns)})" if columns else "0"


def format_summary(summary: Summary) -> str:
    """Render the summary as aligned ``label: value`` lines."""
    facts = summary.facts
    duplicates = summary.duplicate_rows
    lines = {
        "encoding": (
            f"UTF-8, {'BOM' if facts.has_bom else 'no BOM'}, "
            f"{facts.line_ending.upper()} line endings"
        ),
        "rows": f"{summary.rows:,}",
        "columns": f"{summary.columns}",
        "all-null columns": _names(summary.all_null),
        "constant columns": _names(summary.constant),
        "duplicate rows": "disabled" if duplicates is None else f"{duplicates:,}",
        "wrote": f"{summary.output} ({summary.output_columns} columns)",
    }
    if facts.blank_lines:
        lines["blank lines"] = f"{facts.blank_lines:,} (dropped)"
    if summary.cache != "off":
        lines["cache"] = summary.cache
    width = max(map(len, lines)) + 2
    return "\n".join(f"{f'{label}:':<{width}}{value}" for label, value in lines.items())


def format_check(result: CheckResult) -> str:
    """Render one result as an aligned line: severity, status, name, counts, rate."""
    rate = metric_text(result)
    return (
        f"{result.severity.value:<5}  {'PASS' if result.passed else 'FAIL'}  "
        f"{result.name:<30}  {result.n_failed:>9,}/{result.n_evaluated:<9,}  {rate:>9}"
    )


def format_checks(results: Sequence[CheckResult]) -> str:
    """Render every result, one line each."""
    return "\n".join(format_check(r) for r in results)


def format_outcome(summary: Summary) -> str:
    """The counts, the exit status and the report paths, as the last block."""
    counts = summarize(summary.results)
    written = summary.written
    lines = [
        (
            f"errors failed: {counts['errors_failed']}  "
            f"warnings failed: {counts['warnings_failed']}  "
            f"passed: {counts['passed']}/{counts['total']}  "
            f"fail-on: {summary.config.fail_on.value}  exit: {summary.exit_code}"
        ),
        f"wrote: {written.json_path}",
        f"wrote: {written.markdown_path}",
    ]
    if written.samples:
        folder = written.samples[0].parent
        lines.append(f"wrote: {len(written.samples)} sample file(s) in {folder}")
    return "\n".join(lines)


def _first_line(err: Exception) -> str:
    """The first line of the message; polars errors append the whole query plan."""
    return str(err).partition("\n")[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cache_dir = args.cache_dir or (
        Path(os.environ[CACHE_ENV]) if os.environ.get(CACHE_ENV) else None
    )
    if args.refresh_cache and cache_dir is None:
        parser.error(f"--refresh-cache needs --cache-dir or ${CACHE_ENV}")
    try:
        config = QCConfig() if args.config is None else load_config(args.config)
        if args.fail_on is not None:
            config = dataclasses.replace(config, fail_on=Severity(args.fail_on))
        summary = run(
            args.input,
            args.out,
            config,
            args.drop_columns,
            cache_dir,
            args.refresh_cache,
        )
    except (InputError, OSError) as err:
        print(f"rideshare-qc: error: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except Exception as err:
        # Anything else is a bug here, not a fault in the input: say so with its
        # own status (3), distinct from a failed check (1). KeyboardInterrupt and
        # SystemExit are not Exceptions and pass through.
        print(
            f"rideshare-qc: internal error: {type(err).__name__}: "
            f"{_first_line(err)} (please report)",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR
    print(format_summary(summary))
    print()
    print(format_checks(summary.results))
    print()
    print(format_outcome(summary))
    return EXIT_GATED if summary.exit_code else EXIT_OK
