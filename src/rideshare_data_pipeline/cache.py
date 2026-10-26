"""An opt-in, content-addressed Parquet cache of the validated, typed frame.

Extracting the 866 MB sample from its bz2 archive takes about 22 s; the same data as
typed Parquet is a sixth of the size and loads in a few seconds. With a cache
directory configured, the first run on an input writes::

    <cache>/<sha256 of the input file as given>/data.parquet   typed, zstd
    <cache>/<sha256 of the input file as given>/meta.json
    <cache>/index.json                       (path, size, mtime) -> sha

and later runs on the same bytes read the Parquet instead of extracting, validating
and casting. The cache holds a typed copy of the real data: keep it outside the
repository. Nothing here runs unless a cache directory is given.

An entry is valid only if its format version and schema fingerprint (a hash of the
column list, the dtypes and the timestamp formats) match this code's; the tool
version is recorded but deliberately not part of the key, so a release that doesn't
change the schema keeps the cache. The cache never fails a run: anything wrong with
it is a warning on stderr and a rebuild.
"""

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal
lazy import hashlib

lazy import polars as pl

from rideshare_data_pipeline import __version__
from rideshare_data_pipeline import io as rio
from rideshare_data_pipeline.dedupe import FLAG_NAMES, supersede_flags
from rideshare_data_pipeline.parse import DERIVED_NAMES, with_derived
from rideshare_data_pipeline.schema import (
    DATE_FORMAT,
    EXPECTED_COLUMNS,
    TIMESTAMP_FORMAT,
    casts,
    csv_schema,
)

FORMAT_VERSION = 3
"""Bump when the layout of an entry or of index.json changes."""

DATA_FILE = "data.parquet"
DERIVED_FILE = "derived.parquet"
META_FILE = "meta.json"
INDEX_FILE = "index.json"

type CacheStatus = Literal["hit", "miss", "off"]
# @dataclass resolves its annotations when the class is built, which would import
# polars; through this (lazily evaluated) alias importing the module stays free.
type Frame = pl.DataFrame


@dataclass(frozen=True, slots=True)
class Loaded:
    """The typed frame and byte-level facts of an input, and how they were obtained."""

    df: Frame
    facts: rio.FileFacts
    """Of the CSV that was read: the input itself, or the archive's member."""

    cache: CacheStatus
    input_size_bytes: int
    """Of the file the user passed (the archive, or the CSV)."""

    input_sha256: str
    member: str | None
    """The CSV's base name inside the archive; None when the input is a CSV."""


@dataclass(frozen=True, slots=True)
class _Source:
    """The input as the index sees it: where it is, what it looked like, its hash."""

    path: Path
    size: int
    mtime_ns: int
    sha256: str


def schema_fingerprint() -> str:
    """Hash what decides the typed frame: columns, dtypes and timestamp formats."""
    spec = {
        "columns": list(EXPECTED_COLUMNS),
        "casts": {col: str(dtype) for col, dtype in casts().items()},
        "timestamp_format": TIMESTAMP_FORMAT,
        "date_format": DATE_FORMAT,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


DERIVATION_MODULES: tuple[str, ...] = ("parse", "sender", "dedupe")
"""The modules (files of this package) whose code decides the derived columns and the
dedupe flags: ``with_derived`` and ``supersede_flags`` live in the first and last,
the sender rules in ``sender``. A test checks that nothing else of this package is
imported by them."""

DERIVED_NAMES_STORED: tuple[str, ...] = (*DERIVED_NAMES, *FLAG_NAMES)


def derivation_fingerprint() -> str:
    """Hash what decides the derived and dedupe-flag columns, for the derived cache.

    The tool version, the polars version (a regex or cast could change under the
    same code) and the exact source text of DERIVATION_MODULES. The source hash is
    the point: editing the derivation logic changes the fingerprint on its own,
    with no constant to remember to bump, so a stale derived column can't be served;
    the versions only add more invalidation (a release or a polars upgrade
    rebuilds the derived columns, which costs a second on the sample).
    """
    digest = hashlib.sha256()
    digest.update(f"tool={__version__}\0polars={pl.__version__}\0".encode())
    package = files("rideshare_data_pipeline")
    for module in DERIVATION_MODULES:
        digest.update(f"{module}\0".encode())
        digest.update(package.joinpath(f"{module}.py").read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _warn(message: str) -> None:
    print(f"rideshare-qc: warning: {message}", file=sys.stderr)


def _write_json(path: Path, document: object) -> None:
    """Write JSON next to its destination, then rename it into place."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_index(cache_dir: Path) -> dict[str, Any]:
    """The index's entries (resolved path -> size, mtime_ns, sha256); {} if unusable."""
    path = cache_dir / INDEX_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        entries = document["entries"]
        if document["format_version"] != FORMAT_VERSION or not isinstance(
            entries, dict
        ):
            return {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, KeyError, TypeError) as err:
        _warn(f"ignoring unreadable cache index {path}: {err}")
        return {}
    return entries


def _indexed_sha(
    entries: dict[str, Any], source_path: Path, size: int, mtime_ns: int
) -> str | None:
    entry = entries.get(str(source_path))
    if (
        isinstance(entry, dict)
        and entry.get("size") == size
        and entry.get("mtime_ns") == mtime_ns
        and isinstance(entry.get("sha256"), str)
    ):
        return str(entry["sha256"])
    return None


def _update_index(cache_dir: Path, source: _Source) -> None:
    entries = _read_index(cache_dir)
    entries[str(source.path)] = {
        "size": source.size,
        "mtime_ns": source.mtime_ns,
        "sha256": source.sha256,
    }
    _write_json(
        cache_dir / INDEX_FILE, {"format_version": FORMAT_VERSION, "entries": entries}
    )


def _try_load(cache_dir: Path, sha256: str) -> Loaded | None:
    """Load the entry for this hash if it exists and is valid; None means rebuild.

    A missing entry or one from another format version or schema is a plain miss;
    one that exists but can't be read back (bad JSON, bad or mismatched Parquet) is
    a miss with a warning.
    """
    entry = cache_dir / sha256
    meta_path = entry / META_FILE
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta["format_version"] != FORMAT_VERSION
            or meta["schema_fingerprint"] != schema_fingerprint()
        ):
            return None
        csv = meta["csv"]
        match csv["line_ending"]:
            case "crlf" | "lf" | "mixed" as ending:
                pass
            case other:
                msg = f"unknown line ending {other!r}"
                raise ValueError(msg)
        facts = rio.FileFacts(
            size_bytes=int(csv["size_bytes"]),
            sha256=str(csv["sha256"]),
            has_bom=bool(csv["bom"]),
            line_ending=ending,
            header=EXPECTED_COLUMNS,
            blank_lines=int(csv["blank_lines"]),
        )
        member = csv["member"]
        if member is not None and not isinstance(member, str):
            msg = "bad member name"
            raise ValueError(msg)
        input_size = int(meta["input"]["size_bytes"])
        input_sha = str(meta["input"]["sha256"])
        df = pl.read_parquet(entry / DATA_FILE)
        if df.schema != csv_schema() or df.height != int(meta["rows"]):
            msg = "the Parquet doesn't match the recorded schema or row count"
            raise ValueError(msg)
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as err:
        _warn(f"rebuilding unreadable cache entry {entry}: {err}")
        return None
    return Loaded(df, facts, "hit", input_size, input_sha, member)


def _install(tmp: Path, final: Path) -> None:
    """Rename the finished entry into place, replacing any older one."""
    old: Path | None = None
    if final.exists():
        old = final.with_name(f"{final.name}.old-{os.getpid()}")
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
    try:
        tmp.rename(final)
    except OSError:
        # A concurrent run installed the same entry between the two renames; its
        # copy holds the same bytes' data, so it stands.
        if not final.is_dir():
            raise
    finally:
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)


def _store(cache_dir: Path, source: _Source, loaded: Loaded) -> None:
    """Write the entry atomically (temporary directory, then rename) and index it.

    write_parquet from the collected frame: it's already in memory, while a
    sink_parquet would scan the CSV a third time.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{source.sha256}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    df, facts = loaded.df, loaded.facts
    meta = {
        "format_version": FORMAT_VERSION,
        "tool_version": __version__,
        "schema_fingerprint": schema_fingerprint(),
        "input": {
            "name": source.path.name,
            "size_bytes": source.size,
            "mtime_ns": source.mtime_ns,
            "sha256": source.sha256,
        },
        "csv": {
            "member": loaded.member,
            "size_bytes": facts.size_bytes,
            "sha256": facts.sha256,
            "bom": facts.has_bom,
            "line_ending": facts.line_ending,
            "blank_lines": facts.blank_lines,
        },
        "rows": df.height,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        rio.write_parquet(df, tmp / DATA_FILE)
        _write_json(tmp / META_FILE, meta)
        _install(tmp, cache_dir / source.sha256)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _update_index(cache_dir, source)


def _read(path: Path, status: CacheStatus, sha256: str | None = None) -> Loaded:
    """Extract (for an archive), validate, read and cast; describe the input.

    An archive is hashed here (one pass over its bytes) unless the caller already
    has the hash; for a CSV the input's size and hash are the CSV's own.
    """
    with rio.open_input(path) as (csv_path, member):
        df, facts = rio.read_typed(csv_path, member)
    if member is None:
        return Loaded(df, facts, status, facts.size_bytes, facts.sha256, None)
    sha256 = sha256 or rio.file_sha256(path)
    return Loaded(df, facts, status, path.stat().st_size, sha256, member)


def load_typed(
    path: rio.StrPath, cache_dir: rio.StrPath | None = None, *, refresh: bool = False
) -> Loaded:
    """Return the validated, typed frame for `path` (a CSV or an archive).

    Without a cache directory this is extract, inspect, read and cast, as ever.
    With one, the input is statted and looked up in index.json by (resolved path,
    size, mtime_ns): a match whose entry is valid is loaded WITHOUT reading the
    input. Otherwise the input's SHA-256 is computed (a touched file with the same
    bytes still hits) and, failing that, it takes the full path and the result is
    cached, but only after validation passed, so a CastError caches nothing. The
    trade-off of the stat shortcut: an in-place edit that keeps the size and the
    mtime would be missed; refresh=True (``--refresh-cache``) rebuilds regardless.

    Raises InputError exactly as without a cache. A cache that can't be read or
    written is a warning on stderr, never an error.
    """
    path = Path(path)
    rio.validate_input(path)
    if cache_dir is None:
        return _read(path, "off")
    cache = Path(cache_dir)
    stat = path.stat()
    resolved = path.resolve()
    tried: str | None = None
    if not refresh:
        tried = _indexed_sha(
            _read_index(cache), resolved, stat.st_size, stat.st_mtime_ns
        )
        if tried is not None and (loaded := _try_load(cache, tried)) is not None:
            return loaded
    sha256 = rio.file_sha256(path)
    source = _Source(resolved, stat.st_size, stat.st_mtime_ns, sha256)
    if not refresh and sha256 != tried and (loaded := _try_load(cache, sha256)):
        try:
            _update_index(cache, source)
        except OSError as err:
            _warn(f"can't update the cache index in {cache}: {err}")
        return loaded
    loaded = _read(path, "miss", sha256)
    try:
        _store(cache, source, loaded)
    except (OSError, pl.exceptions.PolarsError) as err:
        _warn(f"cache not written to {cache}: {err}")
    return loaded


def _derived_schema() -> pl.Schema:
    """The dtypes of the stored derived columns, from deriving an empty typed frame."""
    empty = pl.DataFrame(schema=csv_schema())
    return supersede_flags(with_derived(empty)).select(DERIVED_NAMES_STORED).schema


def _try_load_derived(entry: Path, rows: int) -> Frame | None:
    """The stored derived columns of this entry, or None unless they are current.

    Current means the entry's meta records the present derivation_fingerprint(),
    and the Parquet has this frame's row count and exactly the expected columns and
    dtypes. A file that exists but can't be read back is a miss with a warning.
    """
    try:
        meta = json.loads((entry / META_FILE).read_text(encoding="utf-8"))
        stored = meta.get("derived")
        if (
            not isinstance(stored, dict)
            or stored.get("fingerprint") != derivation_fingerprint()
            or stored.get("rows") != rows
            or not (entry / DERIVED_FILE).is_file()
        ):
            return None
        frame = pl.read_parquet(entry / DERIVED_FILE)
        if frame.height != rows or frame.schema != _derived_schema():
            msg = "the derived Parquet doesn't match the expected schema or row count"
            raise ValueError(msg)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as err:
        _warn(f"recomputing unreadable derived columns in {entry}: {err}")
        return None
    return frame


def _store_derived(entry: Path, frame: Frame) -> None:
    """Add the derived columns to an existing entry: the Parquet, then its meta record.

    The Parquet is renamed into place first and the meta (which carries the
    fingerprint a reader checks) second, each atomically, so a reader that sees
    the new meta always finds the matching Parquet.
    """
    meta_path = entry / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rio.write_parquet(frame.select(DERIVED_NAMES_STORED), entry / DERIVED_FILE)
    meta["derived"] = {
        "fingerprint": derivation_fingerprint(),
        "rows": frame.height,
        "columns": list(DERIVED_NAMES_STORED),
    }
    _write_json(meta_path, meta)


def derive(loaded: Loaded, cache_dir: rio.StrPath | None = None) -> Frame:
    """The typed frame with the derived and dedupe-flag columns added.

    Deriving costs about 1.2 s on the 1.9M-row sample (the subject and description
    regexes, the dedupe sort). With a cache directory the derived columns are kept in
    the entry of this input (``derived.parquet``) and read back on later runs,
    only while the entry's recorded derivation_fingerprint() matches this code's
    (see there); otherwise they are recomputed and stored again. Without a cache,
    or when the entry can't be written, this is just the computation.
    """
    entry = (
        None
        if cache_dir is None or loaded.cache == "off"
        else Path(cache_dir) / loaded.input_sha256
    )
    if entry is not None:
        columns = _try_load_derived(entry, loaded.df.height)
        if columns is not None:
            return loaded.df.hstack(columns.get_columns())
    frame = supersede_flags(with_derived(loaded.df))
    if entry is not None and (entry / META_FILE).is_file():
        try:
            _store_derived(entry, frame)
        except (OSError, ValueError, pl.exceptions.PolarsError) as err:
            _warn(f"derived columns not cached in {entry}: {err}")
    return frame
