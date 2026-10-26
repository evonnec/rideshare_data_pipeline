"""Input and output: archive extraction, CSV validation, typed reading and writers.

Ragged rows (a record without exactly 33 fields) are found in two steps, because
polars pads a short row with nulls silently and has no option to raise: a
vectorized polars pass counts commas outside quotes per line (a fraction of a
second on a ~0.9 GB file, no false negatives), and only if it flags a line does
a quote-aware stdlib ``csv`` pass locate the physical line for RaggedRowError. See
check_ragged_rows.
"""

import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Literal

# PEP 810 lazy imports: each module below is bound now but only imported on first
# use, so importing this module (e.g. from the CLI before it has parsed its
# arguments) doesn't load polars. Annotations are deferred (PEP 649), so the
# pl.DataFrame / pl.LazyFrame hints below never force the import either.
lazy import codecs
lazy import csv
lazy import fnmatch
lazy import hashlib
lazy import lzma
lazy import re
lazy import shutil
lazy import stat
lazy import tarfile
lazy import tempfile
lazy import zlib
lazy from concurrent.futures import Future, ThreadPoolExecutor
lazy from zipfile import BadZipFile, ZipFile, ZipInfo

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


FALLBACK_STEM = "input"
"""The output stem when the input's name has none (a file named just ".csv")."""


def _safe_stem(stem: str) -> str:
    """The stem without leading dots, so the output is never a hidden file."""
    return stem.lstrip(".") or FALLBACK_STEM


def _split_input_name(path: StrPath) -> tuple[str, InputKind]:
    """Split an input file name into its stem and kind: "a.tar.bz2" -> ("a", "tar")."""
    name = Path(path).name
    for suffix, kind in _SUFFIX_KINDS:
        if name.lower().endswith(suffix):
            return _safe_stem(name[: -len(suffix)]), kind
    supported = ", ".join(suffix for suffix, _ in _SUFFIX_KINDS)
    msg = f"unsupported input {name!r}: expected one of {supported}"
    raise InputError(msg)


def input_stem(path: StrPath) -> str:
    """Return the input's file name without its CSV or archive suffix.

    "Data.csv" and "Data.tar.bz2" both give "Data". A name with no stem (".csv") or
    only leading dots falls back to "input", so the output file is never hidden.
    Raises InputError for a name
    that isn't a supported input.
    """
    return _split_input_name(path)[0]


class ArchiveError(InputError):
    """The archive can't give up exactly one safe CSV.

    Raised when it holds no CSV or several (the message lists the member names, at
    most 20), or when the one CSV is a link or has an unsafe name (absolute, or
    with a ``..`` component).
    """


# The one CSV is always written under this name, whatever the member is called, so
# a member's own name never reaches the file system.
EXTRACTED_NAME = "data.csv"
_MAX_LISTED = 20


def _listing(names: Sequence[str]) -> str:
    """Join at most 20 names; "and N more" covers the rest, "none" an empty list."""
    shown = ", ".join(names[:_MAX_LISTED]) or "none"
    more = len(names) - _MAX_LISTED
    return f"{shown}, and {more} more" if more > 0 else shown


def _is_sidecar(name: str) -> bool:
    """A macOS AppleDouble metadata file: ``._x`` anywhere in the path."""
    return PurePosixPath(name.replace("\\", "/")).name.startswith("._")


def _unsafe_name(name: str) -> str | None:
    """Say why a member name is unsafe to trust (absolute, ``..``), else None."""
    norm = name.replace("\\", "/")
    if norm.startswith("/") or PureWindowsPath(norm).drive:
        return "an absolute path"
    if ".." in norm.split("/"):
        return "a path with a '..' component"
    return None


def _wanted(name: str, pattern: str) -> bool:
    """True for a member the pattern selects: a case-insensitive glob on the path."""
    return fnmatch.fnmatchcase(name.lower(), pattern.lower()) and not _is_sidecar(name)


def _copy_to(source: IO[bytes], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source, target.open("wb") as out:
        shutil.copyfileobj(source, out, _CHUNK_BYTES)


def _settle(
    archive: Path,
    pattern: str,
    members: Sequence[str],
    chosen: Sequence[str],
    problem: str | None,
    target: Path,
) -> tuple[Path, str]:
    """Turn what the listing found into (extracted path, member), or an ArchiveError."""
    if len(chosen) != 1:
        target.unlink(missing_ok=True)
        if chosen:
            what = f"found {len(chosen)} matching {pattern!r}: {_listing(chosen)}"
        else:
            what = f"found none matching {pattern!r}; members: {_listing(members)}"
        msg = f"expected exactly one {pattern!r} member in {archive.name}, {what}"
        raise ArchiveError(msg)
    if problem is not None:
        msg = f"unsafe member {chosen[0]!r} in {archive.name}: {problem}"
        raise ArchiveError(msg)
    return target, chosen[0]


def _tar_problem(info: tarfile.TarInfo) -> str | None:
    if info.issym():
        return "a symlink"
    if info.islnk():
        return "a hardlink"
    if not info.isreg():
        return "not a regular file"
    unsafe = _unsafe_name(info.name)
    return None if unsafe is None else f"{unsafe}"


def _extract_tar_member(archive: Path, dest: Path, pattern: str) -> tuple[Path, str]:
    # One sequential pass: listing a compressed tar means decompressing it, and
    # seeking back to a member afterwards would decompress it all over again. So the
    # first match is written as it goes by, and a second one discards it.
    target = dest / EXTRACTED_NAME
    members: list[str] = []
    chosen: list[str] = []
    problem: str | None = None
    with tarfile.open(archive, "r:*") as tar:
        for info in tar:
            if info.isdir():
                continue
            members.append(info.name)
            if not _wanted(info.name, pattern):
                continue
            chosen.append(info.name)
            if len(chosen) == 1:
                problem = _tar_problem(info)
                if problem is None:
                    source = tar.extractfile(info)
                    assert source is not None  # a regular file
                    _copy_to(source, target)
    return _settle(archive, pattern, members, chosen, problem, target)


def _zip_problem(info: ZipInfo) -> str | None:
    if info.flag_bits & 0x1:
        return "encrypted"
    if stat.S_ISLNK(info.external_attr >> 16):
        return "a symlink"
    return _unsafe_name(info.filename)


def _extract_zip_member(archive: Path, dest: Path, pattern: str) -> tuple[Path, str]:
    target = dest / EXTRACTED_NAME
    with ZipFile(archive) as zip_file:
        infos = [i for i in zip_file.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        picked = [i for i in infos if _wanted(i.filename, pattern)]
        problem = _zip_problem(picked[0]) if picked else None
        if problem == "encrypted":
            msg = (
                f"can't extract {archive.name}: encrypted zip members are not supported"
            )
            raise ArchiveError(msg)
        if len(picked) == 1 and problem is None:
            _copy_to(zip_file.open(picked[0]), target)
    return _settle(
        archive, pattern, names, [i.filename for i in picked], problem, target
    )


def extract_member_named(
    archive: StrPath, dest: StrPath, member: str = "*.csv"
) -> tuple[Path, str]:
    """Like extract_member, but also return the member's name inside the archive."""
    archive, dest = Path(archive), Path(dest)
    _, kind = _split_input_name(archive)
    if kind == "csv":
        msg = f"{archive.name} is a CSV, not an archive"
        raise InputError(msg)
    try:
        if kind == "tar":
            return _extract_tar_member(archive, dest, member)
        return _extract_zip_member(archive, dest, member)
    except RuntimeError as err:
        # zipfile reports an encrypted member this way.
        what = (
            "encrypted zip members are not supported"
            if "encrypted" in str(err)
            else f"archive is truncated or corrupt: {err}"
        )
        msg = f"can't extract {archive.name}: {what}"
        raise ArchiveError(msg) from err
    except (
        tarfile.TarError,
        BadZipFile,
        EOFError,
        OSError,
        zlib.error,
        lzma.LZMAError,
        NotImplementedError,
    ) as err:
        msg = f"can't extract {archive.name}: archive is truncated or corrupt: {err}"
        raise ArchiveError(msg) from err


def extract_member(archive: StrPath, dest: StrPath, member: str = "*.csv") -> Path:
    """Extract exactly one member of a tar or zip archive into dest; return its path.

    `member` is a case-insensitive glob on the member's path (default: any CSV).
    Directories and macOS ``._*`` sidecars never count. Zero matches or several
    raise ArchiveError naming the members (at most 20); so does a match that is a
    symlink, a hardlink or another non-regular member, or whose name is absolute
    or has a ``..`` component. Nothing else in the archive is extracted.

    The bytes are streamed to ``dest/data.csv`` (see EXTRACTED_NAME), a fixed name
    chosen here, rather than extracted under the member's own path: so there is
    no path to escape by, and tarfile's extraction filters aren't needed. Every
    failure to open, list or read the archive (truncated, corrupt, an encrypted
    zip member) is an ArchiveError, so a damaged archive is an input error
    (exit 2), never a crash.
    """
    return extract_member_named(archive, dest, member)[0]


def validate_input(path: StrPath) -> InputKind:
    """Return the input's kind; raise InputError unless it is a supported file."""
    path = Path(path)
    _, kind = _split_input_name(path)
    if not path.is_file():
        msg = f"input not found: {path}"
        raise InputError(msg)
    return kind


@contextmanager
def open_input(path: StrPath) -> Iterator[tuple[Path, str | None]]:
    """Yield (the CSV to read, its member name) for `path`, a CSV or an archive.

    A CSV is yielded as is, with no member name. An archive's one CSV (see
    extract_member) is extracted into a new temporary directory, which is removed,
    with everything in it, when the block exits; the member name is its path
    inside the archive. Nothing is written next to the input or to the working
    directory. Raises InputError if `path` isn't a file or isn't a supported input.
    """
    path = Path(path)
    if validate_input(path) == "csv":
        yield path, None
        return
    with tempfile.TemporaryDirectory(prefix="rideshare-qc-") as tmp:
        csv_path, name = extract_member_named(path, tmp)
        yield csv_path, name


@contextmanager
def input_csv(path: StrPath) -> Iterator[Path]:
    """Yield the CSV to read for `path`; see open_input."""
    with open_input(path) as (csv_path, _):
        yield csv_path


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


class RaggedRowError(InputError):
    """A row has too few or too many fields.

    `line` is the 1-based physical line in the file where the bad record starts
    (the header is line 1; a quoted field may hold a line break, so this can
    differ from the record number), `expected` the column count and `found` the
    number of fields the record has.
    """

    def __init__(self, message: str, *, line: int, expected: int, found: int) -> None:
        super().__init__(message)
        self.line = line
        self.expected = expected
        self.found = found


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

    blank_lines: int = 0
    """Lines that are empty once any ``\\r`` is removed. They are dropped on reading
    (by position, see blank_rows), not an error; a row of commas alone is not blank.
    Counted from the bytes, so a blank line inside a quoted field is counted too."""


# A UTF-8 pass over a ~0.9 GB file at this chunk size costs about a second;
# larger chunks don't help, smaller ones add per-call overhead.
_CHUNK_BYTES = 4 * 1024 * 1024
_BOM = b"\xef\xbb\xbf"
# A "\n" that is followed, ignoring "\r"s, by another "\n": one blank line each.
_BLANK_LINE = re.compile(rb"\n(?=\r*\n)")
# A chunk's last "\n" and the "\r"s after it, searched for in the last few bytes.
_TRAILING_BREAK = re.compile(rb"\n\r*\Z")
_TAIL_BYTES = 16
# What follows a held-back "\n" (and its "\r"s) for it to end a blank line.
_LEADING_BREAK = re.compile(rb"\r*\n")
# A regex search beats bytes.count for the two-byte "\r\n" (about 4x on a large chunk).
_CRLF = re.compile(rb"\r\n")


def file_sha256(path: StrPath) -> str:
    """Hex SHA-256 of the file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _line_ending(*, crlf: int, bare_lf: int) -> LineEnding:
    """Name the style from the counts of "\\r\\n" and lone "\\n" line breaks."""
    if crlf and bare_lf:
        return "mixed"
    return "crlf" if crlf else "lf"


def _scan_bytes(
    path: Path, name: str | None = None
) -> tuple[int, str, bool, LineEnding, int]:
    """Validate the file as UTF-8 in one streaming pass and describe its bytes.

    Returns the size, the SHA-256 (hex), whether it starts with a BOM, the
    line-ending style and the number of blank lines (empty once ``\\r`` is removed);
    the hash uses the same chunks as the validation.
    Raises EncodingError with the absolute offset of the first invalid byte.

    The chunks are never copied or joined: a "\\r\\n" or a blank line that
    straddles two chunks is settled from a few carried bytes. The hash runs on a
    worker thread (hashlib releases the GIL on a large buffer) while this thread
    decodes and counts, so the two overlap; one chunk is in flight at a time.
    """
    # An incremental decoder keeps a multi-byte sequence that straddles two chunks
    # buffered, so a chunk boundary is never mistaken for an error.
    decoder = codecs.getincrementaldecoder("utf-8")()
    digest = hashlib.sha256()
    position = crlf = lf = blank = 0
    has_bom = False
    held_back = b""
    previous_last = b""
    hashing: Future[None] | None = None
    with path.open("rb") as f, ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            chunk = f.read(_CHUNK_BYTES)
            final = not chunk
            if hashing is not None:
                hashing.result()  # the worker is done with the previous chunk
            if not final:
                hashing = pool.submit(digest.update, chunk)
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
                shown = name or path.name
                msg = f"{shown} isn't valid UTF-8: {what} at byte offset {offset:,}"
                raise EncodingError(msg, offset=offset) from err
            if final:
                break
            # A "\r\n" split across the boundary (the last byte of the previous
            # chunk is "\r", the first of this one "\n") is counted once.
            lf += chunk.count(b"\n")
            crlf += len(_CRLF.findall(chunk))
            if previous_last == b"\r" and chunk.startswith(b"\n"):
                crlf += 1
            previous_last = chunk[-1:]
            position += len(chunk)
            # A blank line is a "\n" followed by "\r"s and another "\n", so a run of
            # k line breaks holds k - 1. The last "\n" of a chunk (with any "\r"s
            # after it) is held back (it is at most a few bytes) and judged against
            # the start of the next chunk, where its lookahead can see what follows.
            if len(chunk) < _TAIL_BYTES:
                # Too short for the tail window to stay inside it: join (a copy of
                # a few bytes).
                data = held_back + chunk
                blank += len(_BLANK_LINE.findall(data))
                held = _TRAILING_BREAK.search(data, max(0, len(data) - _TAIL_BYTES))
                held_back = data[held.start() :] if held else b""
                continue
            if held_back and _LEADING_BREAK.match(chunk):
                blank += 1
            blank += len(_BLANK_LINE.findall(chunk))
            held = _TRAILING_BREAK.search(chunk, len(chunk) - _TAIL_BYTES)
            held_back = chunk[held.start() :] if held else b""
    ending = _line_ending(crlf=crlf, bare_lf=lf - crlf)
    return position, digest.hexdigest(), has_bom, ending, blank


def _read_header(path: Path, name: str | None = None) -> tuple[str, ...]:
    """Parse the first record as CSV (so a quoted comma stays in its name).

    Raises InputError if the file has no content. utf-8-sig drops any BOM.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        first = next(csv.reader(f), None)
    if first is None:
        msg = f"{name or path.name} is empty"
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


def inspect_csv(path: StrPath, name: str | None = None) -> FileFacts:
    """Validate the file's encoding and header; return what the bytes looked like.

    Reads the whole file once for UTF-8 validity (about a second per GB), then the
    first line for the header. Raises EncodingError, HeaderError, or InputError
    for an empty file.
    """
    path = Path(path)
    size, sha256, has_bom, line_ending, blank_lines = _scan_bytes(path, name)
    header = _read_header(path, name)
    check_header(header)
    return FileFacts(size, sha256, has_bom, line_ending, header, blank_lines)


# The ragged-row check (see check_ragged_rows). polars pads a row that has too few
# fields with nulls and offers no option to raise instead (checked on polars 2.0:
# truncate_ragged_lines, missing_columns and extra_columns don't), so the check
# works on the lines themselves.
# A byte that doesn't occur in the data, so each line stays one field (a file that
# does hold it makes the scan fail, which counts as a hit).
_NO_SEPARATOR = "\x01"
_QUOTE = '"'
_QUOTED_SEGMENT = r'"[^"]*"'


def _irregular_lines() -> pl.Expr:
    """True for a line that isn't a record with exactly 33 fields on its own.

    Quoted segments are removed before counting commas, so a quoted comma isn't a
    separator. A line with an odd number of quotes is a record continuing onto the
    next line (or a stray quote); it is flagged for the slow, exact locator.
    Blank lines are read as null and are never flagged.
    """
    line = pl.col("line")
    quotes = line.str.count_matches(_QUOTE, literal=True)
    unquoted = (
        pl.when(quotes == 0)
        .then(line)
        .otherwise(line.str.replace_all(_QUOTED_SEGMENT, ""))
    )
    commas = unquoted.str.count_matches(",", literal=True)
    wrong = (commas != len(EXPECTED_COLUMNS) - 1) | (quotes % 2 == 1)
    return wrong & line.is_not_null()


def _may_be_ragged(path: Path) -> bool:
    """Cheap, vectorized test: might some line not be a 33-field record?

    Reads each line as one string column and counts commas outside quotes in
    polars' own multi-threaded engine (a fraction of a second on a ~0.9 GB file, warm),
    with no per-line Python. No false negatives; false positives are lines that
    legitimately span several lines, which the locator then clears.
    """
    lines = pl.scan_csv(
        path,
        has_header=False,
        separator=_NO_SEPARATOR,
        quote_char=None,
        infer_schema=False,
        new_columns=["line"],
    )
    try:
        hits = lines.select(_irregular_lines().sum()).collect(engine="streaming")
    except pl.exceptions.PolarsError:
        return True
    return bool(hits.item())


def locate_ragged_row(path: StrPath, name: str | None = None) -> None:
    """Raise RaggedRowError for the first record without exactly 33 fields.

    Exact and quote-aware (stdlib csv: a quoted comma isn't a separator, a quoted
    field may hold line breaks), and slow: pure Python, several seconds per
    million rows. Returns normally when every record is the right width. Blank
    lines are skipped, as on reading.
    """
    path = Path(path)
    expected = len(EXPECTED_COLUMNS)
    line = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for record in reader:
            start = line + 1
            line = reader.line_num
            if record and len(record) != expected:
                found = len(record)
                msg = (
                    f"{name or path.name} line {start}: "
                    f"expected {expected} fields, found {found}"
                )
                raise RaggedRowError(msg, line=start, expected=expected, found=found)


def check_ragged_rows(path: StrPath, name: str | None = None) -> None:
    """Raise RaggedRowError unless every record has exactly 33 fields.

    Detects cheaply and locates slowly: _may_be_ragged is a vectorized polars pass
    that costs well under a second on a ~0.9 GB file; only when it flags a line
    does the exact stdlib locator run to find the physical line number. A valid
    file with line breaks inside quoted fields also takes the slow path.
    """
    if _may_be_ragged(Path(path)):
        locate_ragged_row(path, name)


def _blank_rows_by_lines(path: Path) -> list[int] | None:
    """Data-row indices of the blank lines, from one vectorized pass over the lines.

    Reads each physical line as one string (blank lines are null), as the ragged
    check does. Physical lines equal records only if no quoted field spans lines,
    so a line with an odd number of quotes (a record continuing, or a stray quote)
    makes the mapping ambiguous: None is returned and the caller takes the exact
    path. The header is line 0, so data row ``i`` is line ``i + 1``.
    """
    line = pl.col("line")
    odd = (line.str.count_matches(_QUOTE, literal=True) % 2 == 1).fill_null(False)
    lines = pl.scan_csv(
        path,
        has_header=False,
        separator=_NO_SEPARATOR,
        quote_char=None,
        infer_schema=False,
        new_columns=["line"],
    )
    try:
        hits = (
            lines.with_row_index("n")
            .select("n", blank=line.is_null(), odd=odd)
            .filter(pl.col("blank") | pl.col("odd"))
            .collect(engine="streaming")
        )
    except pl.exceptions.PolarsError:
        return None
    if hits["odd"].any():
        return None
    return [n - 1 for n in hits["n"] if n >= 1]


def _blank_rows_by_records(path: Path) -> list[int]:
    """Data-row indices of the blank lines, exactly, with the stdlib csv reader.

    Quote-aware: a blank line inside a quoted field is part of its record, not a
    row of its own. Pure Python, several seconds per million rows.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [n - 1 for n, record in enumerate(csv.reader(f)) if n and not record]


def blank_rows(path: StrPath, blank_lines: int) -> list[int]:
    """0-based data-row indices (as polars reads them) of the blank lines.

    polars has no option to skip blank lines: it reads each as a row of nulls, and
    a row of 32 commas is indistinguishable by value, so only position tells them
    apart. Only called when the byte scan found blank lines (`blank_lines` > 0), so
    a file without any pays nothing. The vectorized line pass is used when its
    count agrees with the byte scan's; otherwise (a quoted field spans lines, or
    an odd line break) the exact csv-module locator runs, which is slow.
    """
    path = Path(path)
    found = _blank_rows_by_lines(path)
    if found is not None and len(found) == blank_lines:
        return found
    return _blank_rows_by_records(path)


def _without_rows(df: pl.DataFrame, rows: Sequence[int]) -> pl.DataFrame:
    """Drop the rows at the given 0-based positions (any past the end are ignored)."""
    drop = pl.Series("drop", rows, dtype=pl.Int64)
    return df.filter(~pl.int_range(df.height, eager=True).is_in(drop.implode()))


def read_raw(path: StrPath, *, inspected: FileFacts | None = None) -> pl.LazyFrame:
    """Lazily scan the CSV with every column as String; nothing is inferred.

    The file is validated first (see inspect_csv) unless the caller has already
    done so and passes the result as `inspected`. Nothing is read by polars until
    the frame is collected. Cast it with typed() and count its failures with
    cast_failures().
    """
    path = Path(path)
    facts = inspect_csv(path) if inspected is None else inspected
    # polars strips a BOM and accepts CRLF or LF itself. A row with too many
    # fields makes the collect fail; one with too few is padded with nulls.
    # read_typed runs check_ragged_rows first, which catches both with a line number.
    lf = pl.scan_csv(path, infer_schema=False)
    if facts.blank_lines == 0:
        # The byte scan found no blank line: nothing to drop, nothing to pay.
        return lf
    # A blank line is read as a row of nulls, so it is dropped by position (see
    # blank_rows). A row of commas alone is a real, all-empty record: it stays,
    # and C1 flags it.
    drop = pl.Series("drop", blank_rows(path, facts.blank_lines), dtype=pl.Int64)
    return (
        lf.with_row_index("_row")
        .filter(~pl.col("_row").is_in(drop.implode()))
        .drop("_row")
    )


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


_FAILED = "_cast_failed_"


def _typed_with_failures(path: Path) -> pl.DataFrame:
    """Scan the CSV once: the typed columns plus one ``_cast_failed_<col>`` flag each.

    The same lazy query casts (as typed() does) and flags the values that are
    non-null raw but null after the cast (as cast_failures() counts them), so the
    file is parsed once instead of twice. The flags are bit-packed booleans (a few
    MB for a ~2M-row file); the caller sums and drops them.
    """
    lf = pl.scan_csv(path, infer_schema=False)
    return lf.with_columns(
        *(cast_expr(col, dtype) for col, dtype in casts().items()),
        *(
            (pl.col(col).is_not_null() & cast_expr(col, dtype).is_null()).alias(
                f"{_FAILED}{col}"
            )
            for col, dtype in casts().items()
        ),
    ).collect(engine="streaming")


def read_typed(
    path: StrPath, name: str | None = None
) -> tuple[pl.DataFrame, FileFacts]:
    """Validate the CSV, then read it into a DataFrame with the schema's dtypes.

    The CSV is parsed once, streaming, into the typed columns and the per-column
    cast-failure flags together (CastError if any column has a failure, before
    anything is built). That parse and the ragged-row detection run on threads
    (polars releases the GIL) while this thread does the byte scan, and their
    results are judged in a fixed order: encoding and header, then ragged rows,
    then the parse, then casts. Raises InputError (or a subclass) for anything wrong
    with the file; returns the frame and the byte-level facts.
    """
    path = Path(path)
    name = name or path.name
    with ThreadPoolExecutor(max_workers=2) as pool:
        ragged = pool.submit(_may_be_ragged, path)
        parse = pool.submit(_typed_with_failures, path)
        facts = inspect_csv(path, name)
        if ragged.result():
            locate_ragged_row(path, name)
        try:
            frame = parse.result()
        except pl.exceptions.PolarsError as err:
            locate_ragged_row(
                path, name
            )  # a precise error if a row's width is the cause
            msg = f"can't parse {name}: {_first_line(err)}"
            raise InputError(msg) from err
    flags = [f"{_FAILED}{col}" for col in casts()]
    sums = frame.select(pl.col(flags).sum()).row(0)
    failures = {col: n for col, n in zip(casts(), sums, strict=True) if n}
    if failures:
        raise CastError(failures)
    df = frame.drop(flags)
    if facts.blank_lines:
        # Dropped by position, so an all-empty record (a row of commas) is kept for
        # C1; see blank_rows.
        df = _without_rows(df, blank_rows(path, facts.blank_lines))
    return df, facts


def write_csv(df: pl.DataFrame, path: StrPath) -> None:
    """Write df as CSV, with timestamps and dates in the format they came in.

    (polars' default datetime format is ISO 8601 with a "T".)
    """
    df.write_csv(Path(path), datetime_format=TIMESTAMP_FORMAT, date_format=DATE_FORMAT)


@contextmanager
def atomic_path(path: StrPath) -> Iterator[Path]:
    """Yield a temporary path beside `path`, renamed over it if the block succeeds.

    A reader never sees a half-written file, and a failed write leaves any older
    `path` as it was and no temporary file behind.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        yield tmp
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text_atomic(path: StrPath, text: str) -> None:
    """Write UTF-8 text to `path` atomically (see atomic_path)."""
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


def write_parquet(df: pl.DataFrame, path: StrPath) -> None:
    """Write df as zstd-compressed Parquet, keeping every column and dtype.

    Atomic (see atomic_path): an interrupted run never leaves a partial file.
    """
    with atomic_path(path) as tmp:
        df.write_parquet(tmp, compression="zstd")
