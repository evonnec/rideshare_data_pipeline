"""Opt-in smoke and performance run on the real data, which is never committed.

Set RIDESHARE_QC_DATA to a local CSV or archive to run it; otherwise it is
skipped, and CI never sets it. Run it with::

    RIDESHARE_QC_DATA=/path/to/Data.csv uv run pytest -m slow -s

(``-s`` shows the timing table.) The input is run three times in fresh child
processes, so wall time includes the interpreter start and peak memory is the child's
own: without a cache, with an empty cache (a miss that writes it), and again with it
(a hit). Outputs and the cache go to a temporary directory that is removed afterwards.

Only loose properties are asserted, never a real value: completion, a row count
above a million, rates under a bound, and the time budgets. The budgets are for a
laptop that is otherwise idle: 10 s from the CSV and 2 s from the cache. On a busy
or slower machine set ``RIDESHARE_QC_TIME_SCALE`` (default 1.0) to stretch both,
e.g. ``RIDESHARE_QC_TIME_SCALE=1.5`` allows 15 s and 3 s. The data path is read
from the environment only and never written to a file in the repository.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rideshare_data_pipeline.io import input_stem
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

DATA = os.environ.get("RIDESHARE_QC_DATA", "")
TIME_SCALE = float(os.environ.get("RIDESHARE_QC_TIME_SCALE", "1.0"))

NO_CACHE_BUDGET_SECONDS = 10.0
CACHE_HIT_BUDGET_SECONDS = 2.0

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not Path(DATA).is_file(), reason="RIDESHARE_QC_DATA doesn't name a local file"
    ),
]

# Runs the command given as its arguments and prints the child's peak memory (bytes).
_MEASURE = (
    "import json, resource, subprocess, sys\n"
    "code = subprocess.run(sys.argv[1:]).returncode\n"
    "peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss\n"
    "print(json.dumps({'code': code, 'peak': peak}))\n"
)


@dataclass(frozen=True, slots=True)
class Timed:
    label: str
    status: int
    seconds: float
    peak_mb: float
    out: Path
    # Not in the repr: a failed assertion must not print real figures.
    report: dict[str, Any] = field(repr=False)
    stderr: str = field(repr=False)


def _run(label: str, out: Path, *extra: str) -> Timed:
    command = [
        sys.executable,
        "-c",
        _MEASURE,
        sys.executable,
        "-m",
        "rideshare_data_pipeline",
        "run",
        "--input",
        DATA,
        "--out",
        str(out),
        *extra,
    ]
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    seconds = time.perf_counter() - started
    measured = json.loads(done.stdout.strip().splitlines()[-1])
    # ru_maxrss is bytes on macOS and KiB elsewhere.
    divisor = 2**20 if sys.platform == "darwin" else 2**10
    report_path = out / "report.json"
    report = json.loads(report_path.read_text("utf-8")) if report_path.exists() else {}
    return Timed(
        label,
        measured["code"],
        seconds,
        measured["peak"] / divisor,
        out,
        report,
        done.stderr,
    )


def _table(runs: list[Timed]) -> str:
    lines = [f"{'run':<12}{'cache':>7}{'wall s':>9}{'peak MB':>10}  exit"]
    lines += [
        f"{r.label:<12}{r.report.get('run', {}).get('cache', '?'):>7}"
        f"{r.seconds:>9.2f}{r.peak_mb:>10.0f}  {r.status}"
        for r in runs
    ]
    lines.append(
        f"budgets: no cache < {NO_CACHE_BUDGET_SECONDS * TIME_SCALE:g} s, "
        f"cache hit < {CACHE_HIT_BUDGET_SECONDS * TIME_SCALE:g} s "
        f"(RIDESHARE_QC_TIME_SCALE={TIME_SCALE:g})"
    )
    return "\n".join(lines)


@pytest.fixture(scope="module")
def runs() -> Iterator[list[Timed]]:
    with tempfile.TemporaryDirectory(prefix="rideshare-qc-real-") as tmp:
        root = Path(tmp)
        cache = str(root / "cache")
        found = [
            _run("no cache", root / "o0"),
            _run("cache miss", root / "o1", "--cache-dir", cache),
            _run("cache hit", root / "o2", "--cache-dir", cache),
        ]
        print("\n" + _table(found))
        yield found


def checks(run: Timed) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in run.report["checks"]}


def test_completes_with_exit_0(runs: list[Timed]) -> None:
    for run in runs:
        assert run.status == 0, f"{run.label}: {run.stderr[-500:]}"
        assert run.report["exit_code"] == 0


def test_key_counts_are_within_loose_bounds(runs: list[Timed]) -> None:
    run = runs[0]
    rows = run.report["run"]["rows"]
    assert rows > 1_000_000
    u2 = checks(run)["duplicate_receipts"]
    assert u2["details"]["excess_rows"] / rows < 0.01  # duplicate-order rate
    clean = run.report["spend"]["clean_usd_rider_spend"]
    assert clean["rows"] > 0
    assert clean["amount"] > 0
    assert checks(run)["currency_suspect_rows"]["n_failed"] / rows < 0.05


def test_curated_parquet_is_complete(runs: list[Timed]) -> None:
    run = runs[0]
    curated = pl.scan_parquet(run.out / f"{input_stem(DATA)}.curated.parquet")
    names = curated.collect_schema().names()
    assert names[: len(EXPECTED_COLUMNS)] == list(EXPECTED_COLUMNS)
    assert names[-1] == "qc_flags"
    assert curated.select(pl.len()).collect().item() == run.report["run"]["rows"]


def test_cache_runs_agree_with_the_uncached_run(runs: list[Timed]) -> None:
    assert [r.report["run"]["cache"] for r in runs] == ["off", "miss", "hit"]
    expected = checks(runs[0])
    for run in runs[1:]:
        assert checks(run) == expected
        assert run.report["summary"] == runs[0].report["summary"]


def test_time_budgets(runs: list[Timed]) -> None:
    no_cache, _, hit = runs
    assert no_cache.seconds < NO_CACHE_BUDGET_SECONDS * TIME_SCALE, _table(runs)
    assert hit.seconds < CACHE_HIT_BUDGET_SECONDS * TIME_SCALE, _table(runs)
