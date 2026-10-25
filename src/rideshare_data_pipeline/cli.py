"""The ``rideshare-qc`` command: read the input, print a QC summary, write Parquet."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# None of these import polars at import time (they use PEP 810 lazy imports), so
# `rideshare-qc --help` and argument errors return without loading it.
from rideshare_data_pipeline import __version__
from rideshare_data_pipeline.checks import run_checks
from rideshare_data_pipeline.checks.model import CheckResult
from rideshare_data_pipeline.checks.structural import cast_failures_check
from rideshare_data_pipeline.io import (
    FileFacts,
    InputError,
    input_csv,
    input_stem,
    read_typed,
    write_parquet,
)
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS, casts

EXIT_OK = 0
EXIT_DUPLICATES = 1
EXIT_INPUT_ERROR = 2

# The parser keeps this text's line breaks (RawDescriptionHelpFormatter).
_DESCRIPTION = """\
Read the rideshare e-receipt CSV (or an archive holding it): check its UTF-8
encoding and header, cast the columns to their types, print a structural QC
summary, and write the data to DIR/<input stem>.parquet. Every column and row is
kept unless --drop-columns names columns to leave out.
"""

_EPILOG = f"""\
exit status:
  {EXIT_OK}  success
  {EXIT_DUPLICATES}  exact duplicate rows were found (the Parquet is still written)
  {EXIT_INPUT_ERROR}  input error (bad encoding or header, values that don't cast to
      their column's type), invalid arguments, or unwritable output
"""


@dataclass(frozen=True, slots=True)
class Summary:
    """What a run found and wrote."""

    rows: int
    columns: int
    all_null: list[str]
    constant: list[str]
    duplicate_rows: int
    results: list[CheckResult]
    facts: FileFacts
    output: Path
    output_columns: int


def _column_list(text: str) -> list[str]:
    """Parse ``--drop-columns a,b``; every name must be a schema column."""
    columns = [c.strip() for c in text.split(",") if c.strip()]
    unknown = [c for c in columns if c not in EXPECTED_COLUMNS]
    if unknown:
        msg = f"not in the schema: {', '.join(unknown)}"
        raise argparse.ArgumentTypeError(msg)
    return columns


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``rideshare-qc``."""
    parser = argparse.ArgumentParser(
        prog="rideshare-qc",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
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
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="directory for the Parquet output; created if missing",
    )
    parser.add_argument(
        "--drop-columns",
        type=_column_list,
        default=[],
        metavar="A,B",
        help="comma-separated columns to leave out of the Parquet (default: none)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def run(input_path: Path, out_dir: Path, drop_columns: Sequence[str] = ()) -> Summary:
    """Read the input, run the structural checks and write the Parquet output.

    Raises InputError if the input can't be read, and OSError if the output
    can't be written. Nothing is written until the input has been read.
    """
    with input_csv(input_path) as csv_path:
        df, facts = read_typed(csv_path)
    # read_typed raises CastError on any cast failure, so reaching here means there
    # were none; recording that keeps the S2 line in the report.
    results = [
        cast_failures_check(dict.fromkeys(casts(), 0)),
        *run_checks(df.lazy()),
    ]
    by_name = {r.name: r for r in results}

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{input_stem(input_path)}.parquet"
    kept = df.drop(drop_columns)
    write_parquet(kept, output)
    return Summary(
        rows=df.height,
        columns=df.width,
        all_null=_columns(by_name["all_null_columns"]),
        constant=_columns(by_name["constant_columns"]),
        duplicate_rows=by_name["duplicate_rows"].n_failed,
        results=results,
        facts=facts,
        output=output,
        output_columns=kept.width,
    )


def _columns(result: CheckResult) -> list[str]:
    """The column names an info check recorded in its details."""
    columns = result.details["columns"]
    assert isinstance(columns, list)
    return [str(c) for c in columns]


def _names(columns: list[str]) -> str:
    return f"{len(columns)} ({', '.join(columns)})" if columns else "0"


def format_summary(summary: Summary) -> str:
    """Render the summary as aligned ``label: value`` lines."""
    facts = summary.facts
    lines = {
        "encoding": (
            f"UTF-8, {'BOM' if facts.has_bom else 'no BOM'}, "
            f"{facts.line_ending.upper()} line endings"
        ),
        "rows": f"{summary.rows:,}",
        "columns": f"{summary.columns}",
        "all-null columns": _names(summary.all_null),
        "constant columns": _names(summary.constant),
        "duplicate rows": f"{summary.duplicate_rows:,}",
        "wrote": f"{summary.output} ({summary.output_columns} columns)",
    }
    width = max(map(len, lines)) + 2
    return "\n".join(f"{f'{label}:':<{width}}{value}" for label, value in lines.items())


def format_check(result: CheckResult) -> str:
    """Render one result as an aligned line: severity, status, name, counts, rate."""
    rate = "n/a" if result.metric is None else f"{result.metric:.4%}"
    return (
        f"{result.severity.value:<5}  {'PASS' if result.passed else 'FAIL'}  "
        f"{result.name:<20}  {result.n_failed:>9,}/{result.n_evaluated:<9,}  {rate:>9}"
    )


def format_checks(results: Sequence[CheckResult]) -> str:
    """Render every result, one line each."""
    return "\n".join(format_check(r) for r in results)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status."""
    args = build_parser().parse_args(argv)
    try:
        summary = run(args.input, args.out, args.drop_columns)
    except (InputError, OSError) as err:
        print(f"rideshare-qc: error: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    print(format_summary(summary))
    print()
    print(format_checks(summary.results))
    return EXIT_DUPLICATES if summary.duplicate_rows else EXIT_OK
