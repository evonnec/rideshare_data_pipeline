"""Input and output: archive extraction, the typed CSV scan and the writers."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

# PEP 810 lazy imports: each module below is bound now but only imported on first
# use, so importing this module (e.g. from the CLI before it has parsed its
# arguments) doesn't load polars. Annotations are deferred (PEP 649), so the
# pl.DataFrame / pl.LazyFrame hints below never force the import either.
lazy import tarfile
lazy import tempfile
lazy from zipfile import BadZipFile, ZipFile

lazy import polars as pl

from rideshare_data_pipeline.schema import DATE_FORMAT, TIMESTAMP_FORMAT, csv_schema

type StrPath = str | os.PathLike[str]
type InputKind = Literal["csv", "tar", "zip"]

# The input file names recognised, matched against the end of the lowercased name.
_SUFFIX_KINDS: tuple[tuple[str, InputKind], ...] = (
    (".csv", "csv"),
    (".zip", "zip"),
    (".tar.bz2", "tar"),
    (".tbz2", "tar"),
    (".tar.gz", "tar"),
    (".tgz", "tar"),
    (".tar.xz", "tar"),
    (".txz", "tar"),
    (".tar", "tar"),
)


class InputError(ValueError):
    """The input can't be read as the expected CSV.

    Raised for a missing or unsupported input file, an archive that can't be
    extracted or doesn't hold exactly one CSV, an empty file, a header that
    doesn't match the schema, and values that don't parse as their column's dtype.
    """


def _split_input_name(path: StrPath) -> tuple[str, InputKind]:
    """Split an input file name into its stem and kind: "a.tar.bz2" -> ("a", "tar")."""
    name = Path(path).name
    for suffix, kind in _SUFFIX_KINDS:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or name, kind
    supported = ", ".join(suffix for suffix, _ in _SUFFIX_KINDS)
    msg = f"unsupported input {name!r}: expected one of {supported}"
    raise InputError(msg)


def input_stem(path: StrPath) -> str:
    """Return the input's file name without its CSV or archive suffix.

    "Data.csv" and "Data.tar.bz2" both give "Data". Raises InputError for a name
    that isn't a supported input.
    """
    return _split_input_name(path)[0]


def extract_tar(archive: StrPath, dest: StrPath) -> None:
    """Extract a tar archive (uncompressed, bz2, gzip or xz) into dest.

    The "data" filter rejects members that would land outside dest (``../x``),
    links that point outside it, and device files; it strips leading slashes and
    unsafe permission bits. A rejected member raises tarfile.FilterError.
    """
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(dest, filter="data")


def extract_zip(archive: StrPath, dest: StrPath) -> None:
    """Extract a zip archive into dest.

    zipfile sanitises member names itself: leading slashes and ``..`` components
    are removed, so every member lands inside dest.
    """
    with ZipFile(archive) as zip_file:
        zip_file.extractall(dest)


def extract_archive(archive: StrPath, dest: StrPath) -> Path:
    """Extract a tar or zip archive into dest and return the one CSV it holds.

    macOS metadata files (``._name.csv`` AppleDouble sidecars, which archives made
    on a Mac often carry) don't count as CSVs. Raises InputError if the archive
    can't be extracted, or if it holds no CSV or more than one.
    """
    archive, dest = Path(archive), Path(dest)
    _, kind = _split_input_name(archive)
    if kind == "csv":
        msg = f"{archive.name} is a CSV, not an archive"
        raise InputError(msg)
    try:
        if kind == "tar":
            extract_tar(archive, dest)
        else:
            extract_zip(archive, dest)
    except (tarfile.TarError, BadZipFile) as err:
        msg = f"can't extract {archive.name}: {err}"
        raise InputError(msg) from err

    csvs = sorted(
        p
        for p in dest.rglob("*")
        if p.is_file() and p.suffix.lower() == ".csv" and not p.name.startswith("._")
    )
    if len(csvs) != 1:
        found = ", ".join(str(p.relative_to(dest)) for p in csvs) or "none"
        msg = f"expected exactly one CSV in {archive.name}, found {len(csvs)}: {found}"
        raise InputError(msg)
    return csvs[0]


@contextmanager
def input_csv(path: StrPath) -> Iterator[Path]:
    """Yield the CSV to read for `path`, a CSV file or an archive holding one.

    A CSV is yielded as is. An archive is extracted into a new temporary
    directory, which is removed, with everything in it, when the block exits;
    nothing is written next to the input or to the working directory. Raises
    InputError if `path` isn't a file or isn't a supported input.
    """
    path = Path(path)
    _, kind = _split_input_name(path)
    if not path.is_file():
        msg = f"input not found: {path}"
        raise InputError(msg)
    if kind == "csv":
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="rideshare-qc-") as tmp:
        yield extract_archive(path, tmp)


def _parse_text(col: str, dtype: pl.DataType, *, strict: bool) -> pl.Expr:
    """Return an expression converting the text column `col` to `dtype`.

    Timestamps and dates use the explicit formats from the schema module. With
    strict=False a value that doesn't convert becomes null.
    """
    if isinstance(dtype, pl.Datetime):
        return pl.col(col).str.to_datetime(
            TIMESTAMP_FORMAT, time_unit=dtype.time_unit, strict=strict
        )
    if isinstance(dtype, pl.Date):
        return pl.col(col).str.to_date(DATE_FORMAT, strict=strict)
    return pl.col(col).cast(dtype, strict=strict)


def scan_typed_csv(path: StrPath) -> pl.LazyFrame:
    """Lazily scan the CSV with the dtypes from schema.csv_schema().

    Nothing is read until the frame is collected. A value that doesn't parse
    makes the collect raise rather than becoming null, and so does a header that
    is missing a column or has an extra one (columns are matched by name).
    """
    schema = csv_schema()
    # The CSV reader only infers datetime patterns, so timestamps and dates are
    # read as text and then parsed with an explicit format.
    as_text = {
        col: pl.String() if dtype.is_temporal() else dtype
        for col, dtype in schema.items()
    }
    return pl.scan_csv(Path(path), schema=as_text).with_columns(
        _parse_text(col, dtype, strict=True)
        for col, dtype in schema.items()
        if dtype.is_temporal()
    )


def count_parse_failures(path: StrPath) -> dict[str, int]:
    """Count, per non-text column, the values present that don't parse as its dtype.

    Columns with no failures are left out, so a clean file gives {}.
    """
    schema = csv_schema()
    as_text = pl.scan_csv(Path(path), schema=dict.fromkeys(schema, pl.String()))
    counts = as_text.select(
        (pl.col(col).is_not_null() & _parse_text(col, dtype, strict=False).is_null())
        .sum()
        .alias(col)
        for col, dtype in schema.items()
        if dtype != pl.String
    ).collect(engine="streaming")
    return {col: n for col, n in counts.row(0, named=True).items() if n}


def _first_line(err: Exception) -> str:
    """polars errors append the query plan after the first line; keep the message."""
    return str(err).partition("\n")[0]


def read_typed_csv(path: StrPath) -> pl.DataFrame:
    """Read the CSV into a DataFrame with the schema's dtypes, in one streaming pass.

    Raises InputError (a ValueError) if the file is empty, if its header doesn't
    match the schema, or if any value doesn't parse as its column's dtype; in the
    last case the message gives the number of failures per column.
    """
    name = Path(path).name
    try:
        return scan_typed_csv(path).collect(engine="streaming")
    except pl.exceptions.NoDataError as err:
        msg = f"{name} is empty"
        raise InputError(msg) from err
    except pl.exceptions.SchemaError as err:
        msg = f"the header of {name} doesn't match the schema: {_first_line(err)}"
        raise InputError(msg) from err
    except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as err:
        # The error stops at the first bad batch, so it can't say how many values
        # are bad; a second pass, taken only on failure, counts them.
        try:
            failures = count_parse_failures(path)
        except pl.exceptions.PolarsError:
            failures = {}  # malformed in a way the count can't get past either
        if failures:
            msg = f"{name} has values that don't parse as their dtype: {failures}"
        else:
            msg = f"can't parse {name}: {_first_line(err)}"
        raise InputError(msg) from err


def write_csv(df: pl.DataFrame, path: StrPath) -> None:
    """Write df as CSV, with timestamps and dates in the format they came in.

    (polars' default datetime format is ISO 8601 with a "T".)
    """
    df.write_csv(Path(path), datetime_format=TIMESTAMP_FORMAT, date_format=DATE_FORMAT)


def write_parquet(df: pl.DataFrame, path: StrPath) -> None:
    """Write df as zstd-compressed Parquet, keeping every column and dtype."""
    df.write_parquet(Path(path), compression="zstd")
