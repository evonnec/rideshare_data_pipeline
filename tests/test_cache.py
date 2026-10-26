"""The opt-in Parquet cache: hits, misses, validity rules and failure handling."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from synthetic import make_tar, synthetic_rows, write_csv_text

import rideshare_data_pipeline.cache as cache_module
import rideshare_data_pipeline.io as io_module
from rideshare_data_pipeline.cache import FORMAT_VERSION, load_typed
from rideshare_data_pipeline.cli import CACHE_ENV, main
from rideshare_data_pipeline.io import CastError, InputError
from rideshare_data_pipeline.schema import csv_schema

WATCHED = ("extract_member_named", "inspect_csv", "read_typed", "file_sha256")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count the calls that read or extract an input."""
    counter: Counter[str] = Counter()
    for name in WATCHED:
        real = getattr(io_module, name)

        def wrapper(*args: Any, _real: Any = real, _name: str = name, **kw: Any) -> Any:
            counter[_name] += 1
            return _real(*args, **kw)

        monkeypatch.setattr(io_module, name, wrapper)
    return counter


@pytest.fixture(autouse=True)
def no_env_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CACHE_ENV, raising=False)


@pytest.fixture
def archive(tmp_path: Path, synthetic_csv: Path) -> Path:
    return make_tar(tmp_path / "Data.tar.bz2", {"Data.csv": synthetic_csv.read_bytes()})


def run_cli(
    tmp_path: Path, source: Path, name: str, *extra: str
) -> tuple[int, dict[str, Any]]:
    out = tmp_path / name
    code = main(["run", "--input", str(source), "--out", str(out), *extra])
    report = out / "report.json"
    return code, json.loads(report.read_text()) if report.exists() else {}


def comparable(report: dict[str, Any]) -> dict[str, Any]:
    """The report without what legitimately differs between runs."""
    run = {
        k: v
        for k, v in report["run"].items()
        if k not in {"started", "finished", "elapsed_seconds", "cache"}
    }
    return {**report, "run": run}


def entries(cache: Path) -> list[Path]:
    return sorted(p for p in cache.iterdir() if p.is_dir())


# --- miss, then hit ---------------------------------------------------------------


def test_second_run_is_a_hit_that_neither_reads_nor_extracts(
    tmp_path: Path, archive: Path, calls: Counter[str]
) -> None:
    cache = tmp_path / "cache"
    flag = ("--cache-dir", str(cache))
    assert run_cli(tmp_path, archive, "o0")[1]["run"]["cache"] == "off"
    _, off = run_cli(tmp_path, archive, "o0")
    _, miss = run_cli(tmp_path, archive, "o1", *flag)
    assert calls["extract_member_named"] == 3
    assert calls["file_sha256"] == 3  # each uncached or missed run hashes it once
    before = calls.copy()

    code, hit = run_cli(tmp_path, archive, "o2", *flag)

    assert code == 0
    assert calls == before
    assert (off["run"]["cache"], miss["run"]["cache"], hit["run"]["cache"]) == (
        "off",
        "miss",
        "hit",
    )
    assert comparable(hit) == comparable(miss) == comparable(off)
    assert_frame_equal(
        pl.read_parquet(tmp_path / "o2" / "Data.parquet"),
        pl.read_parquet(tmp_path / "o0" / "Data.parquet"),
    )


def test_summary_names_the_cache_status_only_when_one_is_used(
    tmp_path: Path, archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["run", "--input", str(archive), "--out", str(tmp_path / "a")])
    assert "cache:" not in capsys.readouterr().out
    cache = ["--cache-dir", str(tmp_path / "c")]
    for expected in ("miss", "hit"):
        main(["run", "--input", str(archive), "--out", str(tmp_path / "b"), *cache])
        summary = capsys.readouterr().out.split("\n\n", 1)[0]
        assert summary.splitlines()[-1].split() == ["cache:", expected]


def test_layout_and_metadata(tmp_path: Path, archive: Path) -> None:
    cache = tmp_path / "cache"
    _, report = run_cli(tmp_path, archive, "o", "--cache-dir", str(cache))

    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert [p.name for p in entries(cache)] == [sha]
    assert sorted(p.name for p in cache.iterdir()) == [sha, "index.json"]
    assert sorted(p.name for p in (cache / sha).iterdir()) == [
        "data.parquet",
        "meta.json",
    ]
    meta = json.loads((cache / sha / "meta.json").read_text())
    assert meta["format_version"] == FORMAT_VERSION
    assert meta["schema_fingerprint"] == cache_module.schema_fingerprint()
    assert meta["input"] | {"mtime_ns": 0} == {
        "name": "Data.tar.bz2",
        "size_bytes": archive.stat().st_size,
        "mtime_ns": 0,
        "sha256": sha,
    }
    assert meta["input"]["mtime_ns"] == archive.stat().st_mtime_ns
    assert meta["csv"]["sha256"] == report["run"]["member"]["sha256"]
    assert meta["csv"]["size_bytes"] == report["run"]["member"]["size_bytes"]
    assert meta["csv"]["member"] == report["run"]["member"]["name"] == "Data.csv"
    assert report["run"]["input_sha256"] == sha
    assert meta["csv"]["bom"] is False
    assert meta["csv"]["line_ending"] == "crlf"
    assert meta["rows"] == 3
    assert "tool_version" in meta
    assert meta["created"].endswith("+00:00")
    typed = pl.read_parquet(cache / sha / "data.parquet")
    assert typed.schema == csv_schema()
    index = json.loads((cache / "index.json").read_text())
    assert index["entries"][str(archive.resolve())]["sha256"] == sha


def test_a_csv_input_is_cached_too(
    tmp_path: Path, synthetic_csv: Path, calls: Counter[str]
) -> None:
    cache = tmp_path / "cache"
    flag = ("--cache-dir", str(cache))
    _, miss = run_cli(tmp_path, synthetic_csv, "o1", *flag)
    before = calls.copy()
    _, hit = run_cli(tmp_path, synthetic_csv, "o2", *flag)
    assert (miss["run"]["cache"], hit["run"]["cache"]) == ("miss", "hit")
    assert calls == before
    assert comparable(hit) == comparable(miss)


def test_hit_returns_the_same_frame_and_facts(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    cache = tmp_path / "cache"
    first = load_typed(synthetic_csv, cache)
    second = load_typed(synthetic_csv, cache)
    assert (first.cache, second.cache) == ("miss", "hit")
    assert_frame_equal(first.df, second.df)
    assert first.facts == second.facts


# --- refresh and invalidation -------------------------------------------------------


def test_refresh_cache_rebuilds(
    tmp_path: Path, archive: Path, calls: Counter[str]
) -> None:
    flag = ("--cache-dir", str(tmp_path / "c"))
    run_cli(tmp_path, archive, "o1", *flag)
    calls.clear()
    _, report = run_cli(tmp_path, archive, "o2", *flag, "--refresh-cache")
    assert report["run"]["cache"] == "miss"
    assert calls["read_typed"] == 1
    assert calls["extract_member_named"] == 1
    assert run_cli(tmp_path, archive, "o3", *flag)[1]["run"]["cache"] == "hit"


def test_refresh_cache_without_a_cache_dir_is_a_usage_error(
    tmp_path: Path, archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        run_cli(tmp_path, archive, "o", "--refresh-cache")
    assert info.value.code == 2
    assert "--refresh-cache needs --cache-dir" in capsys.readouterr().err


def test_a_schema_fingerprint_mismatch_invalidates(
    tmp_path: Path,
    archive: Path,
    calls: Counter[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flag = ("--cache-dir", str(tmp_path / "c"))
    run_cli(tmp_path, archive, "o1", *flag)
    monkeypatch.setattr(cache_module, "schema_fingerprint", lambda: "changed")
    calls.clear()
    assert run_cli(tmp_path, archive, "o2", *flag)[1]["run"]["cache"] == "miss"
    assert calls["read_typed"] == 1
    assert run_cli(tmp_path, archive, "o3", *flag)[1]["run"]["cache"] == "hit"
    monkeypatch.undo()
    assert run_cli(tmp_path, archive, "o4", *flag)[1]["run"]["cache"] == "miss"


def test_fingerprint_follows_the_columns_dtypes_and_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = cache_module.schema_fingerprint()
    assert base == cache_module.schema_fingerprint()
    monkeypatch.setattr(cache_module, "TIMESTAMP_FORMAT", "%Y")
    assert cache_module.schema_fingerprint() != base


def test_a_format_version_mismatch_invalidates(
    tmp_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flag = ("--cache-dir", str(tmp_path / "c"))
    run_cli(tmp_path, archive, "o1", *flag)
    monkeypatch.setattr(cache_module, "FORMAT_VERSION", FORMAT_VERSION + 1)
    assert run_cli(tmp_path, archive, "o2", *flag)[1]["run"]["cache"] == "miss"


def test_tool_version_is_not_part_of_the_key(
    tmp_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flag = ("--cache-dir", str(tmp_path / "c"))
    run_cli(tmp_path, archive, "o1", *flag)
    monkeypatch.setattr(cache_module, "__version__", "99.0.0")
    assert run_cli(tmp_path, archive, "o2", *flag)[1]["run"]["cache"] == "hit"


def test_a_touched_input_with_the_same_bytes_hits_by_sha(
    tmp_path: Path, archive: Path, calls: Counter[str]
) -> None:
    flag = ("--cache-dir", str(tmp_path / "c"))
    run_cli(tmp_path, archive, "o1", *flag)
    stat = archive.stat()
    os.utime(archive, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    calls.clear()

    _, report = run_cli(tmp_path, archive, "o2", *flag)

    assert report["run"]["cache"] == "hit"
    assert calls["file_sha256"] == 1  # hashed, because the stat no longer matched
    assert calls["read_typed"] == calls["extract_member_named"] == 0
    run_cli(tmp_path, archive, "o3", *flag)
    assert calls["file_sha256"] == 1  # the index now knows the new mtime


def test_a_copy_of_the_same_bytes_hits_by_sha(
    tmp_path: Path, synthetic_csv: Path, calls: Counter[str]
) -> None:
    flag = ("--cache-dir", str(tmp_path / "c"))
    run_cli(tmp_path, synthetic_csv, "o1", *flag)
    copy = tmp_path / "copy.csv"
    copy.write_bytes(synthetic_csv.read_bytes())
    calls.clear()
    assert run_cli(tmp_path, copy, "o2", *flag)[1]["run"]["cache"] == "hit"
    assert calls["read_typed"] == 0


def test_changed_content_is_a_new_entry(tmp_path: Path) -> None:
    cache = tmp_path / "c"
    path = write_csv_text(tmp_path / "in.csv", synthetic_rows(3))
    load_typed(path, cache)
    write_csv_text(path, synthetic_rows(4))
    again = load_typed(path, cache)
    assert (again.cache, again.df.height) == ("miss", 4)
    assert len(entries(cache)) == 2


# --- failures never fail the run ----------------------------------------------------


@pytest.mark.parametrize("damage", ["meta-garbage", "meta-keys", "parquet", "rows"])
def test_a_damaged_entry_warns_and_rebuilds(
    tmp_path: Path,
    archive: Path,
    capsys: pytest.CaptureFixture[str],
    damage: str,
) -> None:
    cache = tmp_path / "c"
    flag = ("--cache-dir", str(cache))
    run_cli(tmp_path, archive, "o1", *flag)
    (entry,) = entries(cache)
    meta_path = entry / "meta.json"
    match damage:
        case "meta-garbage":
            meta_path.write_text("{not json")
        case "meta-keys":
            meta_path.write_text(json.dumps({"format_version": FORMAT_VERSION}))
        case "parquet":
            (entry / "data.parquet").write_bytes(b"PAR1 nope")
        case "rows":
            meta = json.loads(meta_path.read_text())
            meta_path.write_text(json.dumps(meta | {"rows": 99}))
    capsys.readouterr()

    code, report = run_cli(tmp_path, archive, "o2", *flag)

    assert code == 0
    assert report["run"]["cache"] == "miss"
    assert "warning: rebuilding unreadable cache entry" in capsys.readouterr().err
    assert run_cli(tmp_path, archive, "o3", *flag)[1]["run"]["cache"] == "hit"


def test_a_corrupt_index_warns_and_is_rebuilt(
    tmp_path: Path, archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "c"
    flag = ("--cache-dir", str(cache))
    run_cli(tmp_path, archive, "o1", *flag)
    (cache / "index.json").write_text("[1,")
    capsys.readouterr()
    # Not trusted, so the archive is hashed; the entry is found by its sha.
    _, report = run_cli(tmp_path, archive, "o2", *flag)
    assert report["run"]["cache"] == "hit"
    assert "unreadable cache index" in capsys.readouterr().err
    assert json.loads((cache / "index.json").read_text())["entries"]


def test_an_unwritable_cache_warns_and_the_run_succeeds(
    tmp_path: Path, archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    code, report = run_cli(
        tmp_path, archive, "o", "--cache-dir", str(blocker / "cache")
    )
    assert code == 0
    assert report["run"]["cache"] == "miss"
    assert "warning: cache not written" in capsys.readouterr().err


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_read_only_cache_dir_warns_and_the_run_succeeds(
    tmp_path: Path, archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "c"
    cache.mkdir(mode=0o500)
    try:
        code, _ = run_cli(tmp_path, archive, "o", "--cache-dir", str(cache))
    finally:
        cache.chmod(0o700)
    assert code == 0
    assert "warning: cache not written" in capsys.readouterr().err
    assert list(cache.iterdir()) == []


def test_a_cast_error_caches_nothing(tmp_path: Path) -> None:
    rows = synthetic_rows(3)
    rows[1]["order_time"] = "not a time"
    path = write_csv_text(tmp_path / "bad.csv", rows)
    cache = tmp_path / "c"
    with pytest.raises(CastError):
        load_typed(path, cache)
    args = ["run", "--input", str(path), "--out", str(tmp_path / "o")]
    assert main([*args, "--cache-dir", str(cache)]) == 2
    assert not cache.exists()


def test_input_errors_are_unchanged_with_a_cache(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="input not found"):
        load_typed(tmp_path / "nope.csv", tmp_path / "c")
    assert not (tmp_path / "c").exists()


def test_a_concurrent_install_of_the_same_entry_is_harmless(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    cache = tmp_path / "c"
    load_typed(synthetic_csv, cache, refresh=True)
    load_typed(synthetic_csv, cache, refresh=True)  # replaces the entry
    assert [
        p.name for p in cache.iterdir() if ".tmp-" in p.name or ".old-" in p.name
    ] == []
    assert load_typed(synthetic_csv, cache).cache == "hit"


# --- opting in ------------------------------------------------------------------------


def test_the_environment_variable_configures_the_cache(
    tmp_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "env-cache"
    monkeypatch.setenv(CACHE_ENV, str(cache))
    assert run_cli(tmp_path, archive, "o1")[1]["run"]["cache"] == "miss"
    assert run_cli(tmp_path, archive, "o2")[1]["run"]["cache"] == "hit"
    assert len(entries(cache)) == 1


def test_the_flag_wins_over_the_environment_variable(
    tmp_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "env-cache"))
    run_cli(tmp_path, archive, "o1", "--cache-dir", str(tmp_path / "flag-cache"))
    assert (tmp_path / "flag-cache").is_dir()
    assert not (tmp_path / "env-cache").exists()


def test_an_empty_environment_variable_means_no_cache(
    tmp_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, "")
    assert run_cli(tmp_path, archive, "o1")[1]["run"]["cache"] == "off"


def test_without_a_cache_nothing_is_written_outside_out(
    tmp_path: Path, archive: Path, empty_cwd: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    moved = archive.rename(work / archive.name)
    before = sorted(p.name for p in tmp_path.iterdir())
    out = tmp_path / "out"
    assert main(["run", "--input", str(moved), "--out", str(out)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([*before, "out"])
    assert sorted(p.name for p in work.iterdir()) == [archive.name]
    assert list(empty_cwd.iterdir()) == []


def test_a_hit_reproduces_the_archive_provenance(tmp_path: Path, archive: Path) -> None:
    flag = ("--cache-dir", str(tmp_path / "c"))
    _, miss = run_cli(tmp_path, archive, "o1", *flag)
    _, hit = run_cli(tmp_path, archive, "o2", *flag)
    assert hit["run"]["cache"] == "hit"
    assert hit["run"]["member"] == miss["run"]["member"]
    assert (
        hit["run"]["input_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert hit["run"]["input_size_bytes"] == archive.stat().st_size
