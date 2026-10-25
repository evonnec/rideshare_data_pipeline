"""The QC report: a JSON document (report.schema.json) and a Markdown rendering.

Everything except ``write_report`` is a pure function of its arguments. The
Markdown is built from strings, with no template or table library.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from rideshare_data_pipeline.checks.model import CheckResult, Frame, Severity
from rideshare_data_pipeline.config import QCConfig

SCHEMA_VERSION = 1
SAMPLES_DIR = "samples"
SCHEMA_FILE = "report.schema.json"

_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}

type Json = dict[str, object]


@dataclass(frozen=True, slots=True)
class RunInfo:
    """What the report says about the run and the input, besides the checks."""

    tool_version: str
    input_name: str
    """Base name only; the local path never enters a report."""

    sha256: str
    size_bytes: int
    has_bom: bool
    line_ending: str
    rows: int
    panelists: int
    window_start: str | None
    """ISO timestamp of the earliest email_time; None for an empty file."""

    window_end: str | None
    started: str
    finished: str
    elapsed_seconds: float
    config: QCConfig


def gates(result: CheckResult, fail_on: Severity) -> bool:
    """True if this result is a failure that should make the run exit 1.

    Errors always gate; warnings only when fail_on is "warn"; info never does.
    """
    if result.passed:
        return False
    return result.severity is Severity.ERROR or (
        result.severity is Severity.WARN and fail_on is Severity.WARN
    )


def exit_code(results: Sequence[CheckResult], fail_on: Severity) -> int:
    """0 if nothing gated, else 1."""
    return int(any(gates(r, fail_on) for r in results))


def summarize(results: Sequence[CheckResult]) -> dict[str, int]:
    """Count the results by outcome. ``info`` counts info-severity checks."""
    return {
        "errors_failed": sum(
            not r.passed and r.severity is Severity.ERROR for r in results
        ),
        "warnings_failed": sum(
            not r.passed and r.severity is Severity.WARN for r in results
        ),
        "info": sum(r.severity is Severity.INFO for r in results),
        "passed": sum(r.passed for r in results),
        "total": len(results),
    }


def sample_name(result: CheckResult) -> str | None:
    """The relative sample path for a failed result with rows, else None."""
    sample = result.sample
    if result.passed or sample is None or sample.height == 0:
        return None
    return f"{SAMPLES_DIR}/{result.name}.parquet"


def _clean(value: object) -> object:
    """Make `value` strict-JSON safe: non-finite floats become null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if value is None or isinstance(value, bool | int | str):
        return value
    return str(value)


def _check_json(result: CheckResult) -> Json:
    return {
        "name": result.name,
        "check_id": result.check_id,
        "category": result.category,
        "severity": result.severity.value,
        "passed": result.passed,
        "metric": _clean(result.metric),
        "threshold": _clean(result.threshold),
        "n_failed": result.n_failed,
        "n_evaluated": result.n_evaluated,
        "predicate": result.predicate,
        "details": _clean(dict(result.details)),
        "sample_file": sample_name(result),
    }


def build_report(results: Sequence[CheckResult], run: RunInfo, code: int) -> Json:
    """Assemble the report.json document (see report.schema.json)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "tool_version": run.tool_version,
            "input_name": run.input_name,
            "input_sha256": run.sha256,
            "input_size_bytes": run.size_bytes,
            "bom": run.has_bom,
            "line_ending": run.line_ending,
            "rows": run.rows,
            "panelists": run.panelists,
            "window": {"start": run.window_start, "end": run.window_end},
            "started": run.started,
            "finished": run.finished,
            "elapsed_seconds": round(run.elapsed_seconds, 3),
            "config": {
                "fail_on": run.config.fail_on.value,
                "sample_limit": run.config.sample_limit,
                "checks": {
                    r.name: {
                        "threshold": _clean(r.threshold),
                        "severity": r.severity.value,
                    }
                    for r in results
                },
            },
        },
        "summary": summarize(results),
        "checks": [_check_json(r) for r in results],
        "exit_code": code,
    }


def _cell(value: object) -> str:
    """One Markdown table cell: pipes escaped, line breaks flattened."""
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _table(
    header: Sequence[str], rows: Sequence[Sequence[object]], right: Sequence[int] = ()
) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|"
        + "|".join("---:" if i in right else "---" for i in range(len(header)))
        + "|",
    ]
    lines.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return lines


def _pct(metric: float | None) -> str:
    return "n/a" if metric is None else f"{metric:.4%}"


def _num(threshold: float | None) -> str:
    return "n/a" if threshold is None else f"{threshold:g}"


def _sorted(results: Sequence[CheckResult]) -> list[CheckResult]:
    """By severity (errors first), failures before passes; ties keep run order."""
    return sorted(results, key=lambda r: (_SEVERITY_ORDER[r.severity], r.passed))


def _columns(results: Sequence[CheckResult], name: str) -> list[str] | None:
    for r in results:
        if r.name == name:
            cols = r.details.get("columns")
            return [str(c) for c in cols] if isinstance(cols, list) else None
    return None


def _sample_table(sample: Frame, limit: int) -> list[str]:
    head = sample.head(limit)
    rows = [[row[c] for c in head.columns] for row in head.iter_rows(named=True)]
    return _table(head.columns, rows)


def render_markdown(results: Sequence[CheckResult], run: RunInfo, code: int) -> str:
    """Render the report as Markdown."""
    s = summarize(results)
    window = f"{run.window_start}..{run.window_end}" if run.window_start else "n/a"
    out = [
        f"# QC report: {run.input_name} ({run.sha256[:12]})",
        "",
        f"rows {run.rows:,} · panelists {run.panelists:,} · window {window}",
        "",
        "## Summary",
        "",
        (
            f"errors failed: {s['errors_failed']} · warnings failed: "
            f"{s['warnings_failed']} · info: {s['info']} · "
            f"passed: {s['passed']}/{s['total']} · exit code {code}"
        ),
        "",
    ]
    out += _table(
        ["check", "severity", "status", "failed", "rate", "threshold", "rule"],
        [
            [
                r.name,
                r.severity.value,
                "PASS" if r.passed else "FAIL",
                f"{r.n_failed:,}",
                _pct(r.metric),
                _num(r.threshold),
                r.predicate,
            ]
            for r in _sorted(results)
        ],
        right=(3, 4, 5),
    )
    out += [
        "",
        "## Composition by receipt kind",
        "",
        "_Added by a later check (receipt kinds)._",
        "",
    ]
    out += ["## Column profile", ""]
    all_null = _columns(results, "all_null_columns")
    constant = _columns(results, "constant_columns")
    if all_null is None and constant is None:
        out += ["_Added by a later check._", ""]
    else:
        out += _table(
            ["profile", "columns"],
            [
                ["all null", ", ".join(all_null or []) or "none"],
                ["constant", ", ".join(constant or []) or "none"],
            ],
        )
        out.append("")
    out += ["## Samples", ""]
    failing = [r for r in _sorted(results) if sample_name(r) and r.sample is not None]
    if not failing:
        out += ["No failed check has a sample.", ""]
    for r in failing:
        assert r.sample is not None
        shown = min(r.sample.height, run.config.sample_limit)
        out += [
            f"### {r.name}",
            "",
            f"{shown} of {r.n_failed:,} failing rows · "
            f"[{sample_name(r)}]({sample_name(r)})",
            "",
            *_sample_table(r.sample, run.config.sample_limit),
            "",
        ]
    return "\n".join(out)


def schema_text() -> str:
    """The JSON Schema (draft 2020-12) that report.json conforms to."""
    return files("rideshare_data_pipeline").joinpath(SCHEMA_FILE).read_text("utf-8")


@dataclass(frozen=True, slots=True)
class Written:
    """The paths write_report created."""

    json_path: Path
    markdown_path: Path
    samples: tuple[Path, ...]


def write_report(
    out_dir: Path, results: Sequence[CheckResult], run: RunInfo, code: int
) -> Written:
    """Write report.json, report.md and samples/<check>.parquet into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # A re-run into the same directory must not leave samples from checks that
    # pass this time: report.json lists only this run's samples.
    for stale in (out_dir / SAMPLES_DIR).glob("*.parquet"):
        stale.unlink()
    samples: list[Path] = []
    for r in results:
        name = sample_name(r)
        if name is not None and r.sample is not None:
            path = out_dir / name
            path.parent.mkdir(exist_ok=True)
            r.sample.write_parquet(path, compression="zstd")
            samples.append(path)
    json_path = out_dir / "report.json"
    json_path.write_text(
        json.dumps(build_report(results, run, code), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path = out_dir / "report.md"
    md_path.write_text(render_markdown(results, run, code) + "\n", encoding="utf-8")
    return Written(json_path, md_path, tuple(samples))
