"""The ``rideshare-qc`` command: read the input, print a QC summary, write Parquet."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# None of these import polars at import time (they use PEP 810 lazy imports), so
# `rideshare-qc --help` and argument errors return without loading it.
from rideshare_data_pipeline import __version__
from rideshare_data_pipeline.checks.structural import (
    degenerate_columns,
    duplicate_row_count,
)
from rideshare_data_pipeline.io import (
    InputError,
    input_csv,
    input_stem,
    read_typed_csv,
    write_parquet,
)
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

EXIT_OK = 0
EXIT_DUPLICATES = 1
EXIT_INPUT_ERROR = 2

# The parser keeps this text's line breaks (RawDescriptionHelpFormatter).
_DESCRIPTION = """\
Read the rideshare e-receipt CSV (or an archive holding it) with its typed
schema, print a structural QC summary, and write the data to
DIR/<input stem>.parquet. Every column and row is kept unless --drop-columns
names columns to leave out.
"""

_EPILOG = f"""\
exit status:
  {EXIT_OK}  success
  {EXIT_DUPLICATES}  exact duplicate rows were found (the Parquet is still written)
  {EXIT_INPUT_ERROR}  input or parse error, invalid arguments, or unwritable output
"""


@dataclass(frozen=True, slots=True)
class Summary:
    """What a run found and wrote."""

    rows: int
    columns: int
    all_null: list[str]
    constant: list[str]
    duplicate_rows: int
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
        df = read_typed_csv(csv_path)
    lf = df.lazy()
    found = degenerate_columns(lf)
    duplicates = duplicate_row_count(lf)

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{input_stem(input_path)}.parquet"
    kept = df.drop(drop_columns)
    write_parquet(kept, output)
    return Summary(
        rows=df.height,
        columns=df.width,
        all_null=found.all_null,
        constant=found.constant,
        duplicate_rows=duplicates,
        output=output,
        output_columns=kept.width,
    )


def _names(columns: list[str]) -> str:
    return f"{len(columns)} ({', '.join(columns)})" if columns else "0"


def format_summary(summary: Summary) -> str:
    """Render the summary as aligned ``label: value`` lines."""
    lines = {
        "rows": f"{summary.rows:,}",
        "columns": f"{summary.columns}",
        "all-null columns": _names(summary.all_null),
        "constant columns": _names(summary.constant),
        "duplicate rows": f"{summary.duplicate_rows:,}",
        "wrote": f"{summary.output} ({summary.output_columns} columns)",
    }
    width = max(map(len, lines)) + 2
    return "\n".join(f"{f'{label}:':<{width}}{value}" for label, value in lines.items())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status."""
    args = build_parser().parse_args(argv)
    try:
        summary = run(args.input, args.out, args.drop_columns)
    except (InputError, OSError) as err:
        print(f"rideshare-qc: error: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    print(format_summary(summary))
    return EXIT_DUPLICATES if summary.duplicate_rows else EXIT_OK
