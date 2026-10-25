"""Input and output: archive extraction, CSV validation, typed reading and writers."""

import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# PEP 810 lazy imports: each module below is bound now but only imported on first
# use, so importing this module (e.g. from the CLI before it has parsed its
# arguments) doesn't load polars. Annotations are deferred (PEP 649), so the
# pl.DataFrame / pl.LazyFrame hints below never force the import either.
lazy import codecs
lazy import csv
lazy import hashlib
lazy import tarfile
lazy import tempfile
lazy from zipfile import BadZipFile, ZipFile

lazy import polars as pl

from rideshare_data_pipeline.schema import (
    DATE_FORMAT,
    EXPECTED_COLUMNS,
    TIMESTAMP_FORMAT,
    cast_expr,
    casts,
)

type StrPath = str | os.PathLike[str]
type InputKind = Literal["csv", "tar", "zip"]
type LineEnding = Literal["crlf", "lf", "mixed"]

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
    doesn't match the schema, bytes that aren't UTF-8, and values that don't convert
    to their column's dtype. The subclasses below say which.
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


class HeaderError(InputError):
    """The header doesn't match the 33 expected columns.

    The lists name every column at fault, so a message never needs a second look
    at the file. `reordered` is True when the names are right but their order is
    not (only judged when nothing is missing, extra or duplicated).
    """

    def __init__(
        self,
        message: str,
        *,
        missing: Sequence[str] = (),
        extra: Sequence[str] = (),
        duplicated: Sequence[str] = (),
        reordered: bool = False,
    ) -> None:
        super().__init__(message)
        self.missing = list(missing)
        self.extra = list(extra)
        self.duplicated = list(duplicated)
        self.reordered = reordered


class EncodingError(InputError):
    """The file isn't valid UTF-8; `offset` is the byte offset of the first bad byte."""

    def __init__(self, message: str, *, offset: int) -> None:
        super().__init__(message)
        self.offset = offset


class CastError(InputError):
    """Values are present that don't convert to their column's dtype.

    `counts` maps each affected column to its number of failures; columns without
    failures are left out.
    """

    def __init__(self, counts: Mapping[str, int]) -> None:
        self.counts = dict(counts)
        detail = "\n".join(f"  {col}: {n:,}" for col, n in self.counts.items())
        super().__init__(f"values that don't convert to their column's type:\n{detail}")


@dataclass(frozen=True, slots=True)
class FileFacts:
    """What the byte-level pass learned about a CSV."""

    size_bytes: int
    sha256: str
    """Hex SHA-256 of the file's bytes, computed in the same pass as the UTF-8 check."""

    has_bom: bool
    """The file starts with a UTF-8 byte order mark (it is stripped on reading)."""

    line_ending: LineEnding
    """The style of the line breaks, counting any inside quoted fields. A file
    with no line break at all is reported as "lf"."""

    header: tuple[str, ...]
    """The column names, as parsed from the first line (after any BOM)."""


# A UTF-8 pass over the 866 MB sample at this chunk size costs about a second;
# larger chunks don't help, smaller ones add per-call overhead.
_CHUNK_BYTES = 4 * 1024 * 1024
_BOM = b"\xef\xbb\xbf"


def _line_ending(*, crlf: int, bare_lf: int) -> LineEnding:
    """Name the style from the counts of "\\r\\n" and lone "\\n" line breaks."""
    if crlf and bare_lf:
        return "mixed"
    return "crlf" if crlf else "lf"


def _scan_bytes(path: Path) -> tuple[int, str, bool, LineEnding]:
    """Validate the file as UTF-8 in one streaming pass and describe its bytes.

    Returns the size, the SHA-256 (hex), whether it starts with a BOM, and the
    line-ending style; the hash uses the same chunks as the validation.
    Raises EncodingError with the absolute offset of the first invalid byte.
    """
    # An incremental decoder keeps a multi-byte sequence that straddles two chunks
    # buffered, so a chunk boundary is never mistaken for an error.
    decoder = codecs.getincrementaldecoder("utf-8")()
    digest = hashlib.sha256()
    position = crlf = lf = 0
    has_bom = False
    previous_tail = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_BYTES)
            final = not chunk
            digest.update(chunk)
            if position == 0 and chunk:
                has_bom = chunk.startswith(_BOM)
            pending = len(decoder.getstate()[0])
            try:
                decoder.decode(chunk, final=final)
            except UnicodeDecodeError as err:
                # err.start is relative to the buffered bytes plus this chunk.
                offset = position - pending + err.start
                what = (
                    "truncated sequence"
                    if err.reason.startswith("unexpected end")
                    else "invalid byte"
                )
                msg = f"{path.name} isn't valid UTF-8: {what} at byte offset {offset:,}"
                raise EncodingError(msg, offset=offset) from err
            if final:
                break
            # Prefix the last byte of the previous chunk so a "\r\n" split across
            # the boundary is counted once; the tail alone can't hold a pair.
            data = previous_tail + chunk
            crlf += data.count(b"\r\n")
            lf += data.count(b"\n")
            previous_tail = chunk[-1:]
            position += len(chunk)
    ending = _line_ending(crlf=crlf, bare_lf=lf - crlf)
    return position, digest.hexdigest(), has_bom, ending


def _read_header(path: Path) -> tuple[str, ...]:
    """Parse the first record as CSV (so a quoted comma stays in its name).

    Raises InputError if the file has no content. utf-8-sig drops any BOM.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        first = next(csv.reader(f), None)
    if first is None:
        msg = f"{path.name} is empty"
        raise InputError(msg)
    return tuple(first)


def check_header(header: Sequence[str]) -> None:
    """Raise HeaderError unless header is exactly EXPECTED_COLUMNS, in order.

    Missing, extra, duplicated and reordered columns each get their own report;
    order matters because polars matches columns by name, and would quietly accept
    a permuted header.
    """
    expected = set(EXPECTED_COLUMNS)
    counts = Counter(header)
    missing = [c for c in EXPECTED_COLUMNS if c not in counts]
    extra = [c for c in dict.fromkeys(header) if c not in expected]
    duplicated = [c for c, n in counts.items() if n > 1]
    problems = [
        f"{label} {', '.join(map(repr, cols))}"
        for label, cols in (
            ("missing", missing),
            ("extra", extra),
            ("duplicated", duplicated),
        )
        if cols
    ]
    reordered = False
    if not problems and tuple(header) != EXPECTED_COLUMNS:
        reordered = True
        first = next(
            i
            for i, (got, want) in enumerate(zip(header, EXPECTED_COLUMNS, strict=True))
            if got != want
        )
        problems.append(
            f"columns out of order (column {first + 1} is {header[first]!r}, "
            f"expected {EXPECTED_COLUMNS[first]!r})"
        )
    if problems:
        msg = f"the header doesn't match the schema: {'; '.join(problems)}"
        raise HeaderError(
            msg,
            missing=missing,
            extra=extra,
            duplicated=duplicated,
            reordered=reordered,
        )


def inspect_csv(path: StrPath) -> FileFacts:
    """Validate the file's encoding and header; return what the bytes looked like.

    Reads the whole file once for UTF-8 validity (about a second per GB), then the
    first line for the header. Raises EncodingError, HeaderError, or InputError
    for an empty file.
    """
    path = Path(path)
    size, sha256, has_bom, line_ending = _scan_bytes(path)
    header = _read_header(path)
    check_header(header)
    return FileFacts(size, sha256, has_bom, line_ending, header)


def read_raw(path: StrPath, *, inspected: FileFacts | None = None) -> pl.LazyFrame:
    """Lazily scan the CSV with every column as String; nothing is inferred.

    The file is validated first (see inspect_csv) unless the caller has already
    done so and passes the result as `inspected`. Nothing is read by polars until
    the frame is collected. Cast it with typed() and count its failures with
    cast_failures().
    """
    path = Path(path)
    if inspected is None:
        inspect_csv(path)
    # polars strips a BOM and accepts CRLF or LF itself. A row with too many
    # fields makes the collect fail; one with too few is padded with nulls, which
    # isn't detected (see the ragged-row check planned in the edge-case suite).
    return pl.scan_csv(path, infer_schema=False)


def typed(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Cast the raw frame's columns to the dtypes in schema.casts(), non-strictly.

    A value that doesn't convert becomes null; cast_failures() on the same raw
    frame says how many did, so nothing is lost silently.
    """
    return lf.with_columns(cast_expr(col, dtype) for col, dtype in casts().items())


def cast_failures(lf: pl.LazyFrame) -> dict[str, int]:
    """Count, per cast column, values that are non-null raw but null after the cast.

    `lf` is the raw frame from read_raw(). One lazy aggregation covers every
    column, so the file is read once. Every column in schema.casts() is a key,
    with 0 for a clean one.
    """
    counts = lf.select(
        (pl.col(col).is_not_null() & cast_expr(col, dtype).is_null()).sum().alias(col)
        for col, dtype in casts().items()
    ).collect(engine="streaming")
    return counts.row(0, named=True)


def _first_line(err: Exception) -> str:
    """polars errors append the query plan after the first line; keep the message."""
    return str(err).partition("\n")[0]


def read_typed(path: StrPath) -> tuple[pl.DataFrame, FileFacts]:
    """Validate the CSV, then read it into a DataFrame with the schema's dtypes.

    The raw frame is scanned twice, both times streaming: once to count cast
    failures (CastError if there are any, before anything is built), once to cast
    and collect. Raises InputError (or a subclass) for anything wrong with the
    file; returns the frame and the byte-level facts.
    """
    name = Path(path).name
    facts = inspect_csv(path)
    raw = read_raw(path, inspected=facts)
    try:
        failures = {col: n for col, n in cast_failures(raw).items() if n}
        if failures:
            raise CastError(failures)
        return typed(raw).collect(engine="streaming"), facts
    except pl.exceptions.PolarsError as err:
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
