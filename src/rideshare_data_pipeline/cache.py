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
from pathlib import Path
from typing import Any, Literal
lazy import hashlib

lazy import polars as pl

from rideshare_data_pipeline import __version__
from rideshare_data_pipeline import io as rio
from rideshare_data_pipeline.schema import (
    DATE_FORMAT,
    EXPECTED_COLUMNS,
    TIMESTAMP_FORMAT,
    casts,
    csv_schema,
)

FORMAT_VERSION = 1
"""Bump when the layout of an entry or of index.json changes."""

DATA_FILE = "data.parquet"
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
    cache: CacheStatus


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
        )
        df = pl.read_parquet(entry / DATA_FILE)
        if df.schema != csv_schema() or df.height != int(meta["rows"]):
            msg = "the Parquet doesn't match the recorded schema or row count"
            raise ValueError(msg)
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as err:
        _warn(f"rebuilding unreadable cache entry {entry}: {err}")
        return None
    return Loaded(df, facts, "hit")


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


def _store(
    cache_dir: Path, source: _Source, df: pl.DataFrame, facts: rio.FileFacts
) -> None:
    """Write the entry atomically (temporary directory, then rename) and index it.

    write_parquet from the collected frame: it's already in memory, while a
    sink_parquet would scan the CSV a third time.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{source.sha256}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
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
            "size_bytes": facts.size_bytes,
            "sha256": facts.sha256,
            "bom": facts.has_bom,
            "line_ending": facts.line_ending,
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


def _read(path: Path, status: CacheStatus) -> Loaded:
    with rio.input_csv(path) as csv_path:
        df, facts = rio.read_typed(csv_path)
    return Loaded(df, facts, status)


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
    loaded = _read(path, "miss")
    try:
        _store(cache, source, loaded.df, loaded.facts)
    except (OSError, pl.exceptions.PolarsError) as err:
        _warn(f"cache not written to {cache}: {err}")
    return loaded
