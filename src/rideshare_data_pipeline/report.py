"""The QC report: a JSON document (report.schema.json) and a Markdown rendering.

Everything except ``write_report`` is a pure function of its arguments. The
Markdown is built from strings, with no template or table library.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

from rideshare_data_pipeline.checks.model import CheckResult, Frame, Severity
from rideshare_data_pipeline.config import QCConfig
from rideshare_data_pipeline.io import atomic_path, write_text_atomic
from rideshare_data_pipeline.money import STEPS
from rideshare_data_pipeline.parse import FAMILIES, KINDS

SCHEMA_VERSION = 1
SAMPLES_DIR = "samples"
SCHEMA_FILE = "report.schema.json"

_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}

type Json = dict[str, object]


@dataclass(frozen=True, slots=True)
class MemberInfo:
    """The CSV extracted from an archive input (the report's ``run.member``)."""

    name: str
    sha256: str
    size_bytes: int
    has_bom: bool
    line_ending: str


@dataclass(frozen=True, slots=True)
class RunInfo:
    """What the report says about the run and the input, besides the checks."""

    tool_version: str
    input_name: str
    """Base name only; the local path never enters a report."""

    sha256: str
    """Of the file the user passed: the CSV, or the archive."""

    size_bytes: int
    has_bom: bool
    """Of the CSV that was read (the archive's member, for an archive)."""

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
    member: MemberInfo | None = None
    """Set when the input is an archive: sha256 and size_bytes above describe the
    archive then, and this describes the CSV inside it."""

    cache: str = "off"
    """"hit", "miss" or "off" (no cache directory configured)."""

    blank_lines: int = 0
    """Empty lines of the CSV that were read; dropped, not an error."""

    parquet_name: str | None = None
    """File name of the curated Parquet written into the output directory, if any.
    It has `rows` rows (none is dropped) and `curated_columns` columns."""

    curated_columns: int = 0


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


def composition(results: Sequence[CheckResult]) -> Json | None:
    """The merchant x kind composition, from the ``receipt_kinds`` result.

    None when that check didn't run (it was disabled). Otherwise::

        {"rows": N, "merchants": [...],
         "kinds": [{"kind", "rider_spend", "total", "by_merchant": {m: n}}]}

    with every kind in the canonical order, including those with no rows.
    """
    for r in results:
        if r.name != "receipt_kinds":
            continue
        by_merchant = r.details.get("by_merchant")
        spend = r.details.get("rider_spend_kinds")
        if not isinstance(by_merchant, Mapping) or not isinstance(spend, list):
            return None
        merchants = [str(m) for m in by_merchant]
        kinds: list[Json] = []
        for kind in KINDS:
            per = {m: int(by_merchant[m].get(kind, 0)) for m in merchants}
            kinds.append(
                {
                    "kind": kind,
                    "rider_spend": kind in spend,
                    "total": sum(per.values()),
                    "by_merchant": per,
                }
            )
        return {"rows": r.n_evaluated, "merchants": merchants, "kinds": kinds}
    return None


def spend(results: Sequence[CheckResult]) -> Json | None:
    """The spend waterfall, from the ``currency_suspect_rows`` result.

    None when that check didn't run (it was disabled). Otherwise one
    ``{"rows", "amount"}`` object per step of ``money.STEPS`` (the naive total of
    ``order_total_amount``, what the currency-suspect rows, the non-rider-spend rows
    and the superseded rows take off it, and the clean USD rider spend), plus
    ``suspect_rider_spend`` (the suspect rows that would be rider spend),
    ``micromobility`` (the bikes and scooters inside ``non_rider_spend``) and a
    ``note`` on the currency of the suspect amounts.
    """
    for r in results:
        if r.name != "currency_suspect_rows":
            continue
        steps = r.details.get("spend")
        extra = r.details.get("suspect_rider_spend")
        suspect = r.details.get("suspect_sum")
        if not (
            isinstance(steps, Mapping)
            and isinstance(extra, Mapping)
            and isinstance(suspect, Mapping)
        ):
            return None
        table: Json = {step: dict(steps[step]) for step in STEPS}
        micro = steps.get("micromobility")
        if isinstance(micro, Mapping):
            table["micromobility"] = dict(micro)
        table["suspect_rider_spend"] = dict(extra)
        table["note"] = str(suspect["note"])
        return table
    return None


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
    report: Json = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "tool_version": run.tool_version,
            "input_name": run.input_name,
            "input_sha256": run.sha256,
            "input_size_bytes": run.size_bytes,
            "bom": run.has_bom,
            "line_ending": run.line_ending,
            "member": None if run.member is None else asdict(run.member),
            "rows": run.rows,
            "panelists": run.panelists,
            "window": {"start": run.window_start, "end": run.window_end},
            "started": run.started,
            "finished": run.finished,
            "elapsed_seconds": round(run.elapsed_seconds, 3),
            "cache": run.cache,
            "blank_lines": run.blank_lines,
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
        "outputs": {
            "parquet": run.parquet_name,
            "curated": (
                None
                if run.parquet_name is None
                else {
                    "name": run.parquet_name,
                    "rows": run.rows,
                    "columns": run.curated_columns,
                }
            ),
            "samples": [n for r in results if (n := sample_name(r)) is not None],
        },
    }
    table = composition(results)
    if table is not None:
        report["composition"] = table
    waterfall = spend(results)
    if waterfall is not None:
        report["spend"] = waterfall
    return report


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


def metric_text(result: CheckResult) -> str:
    """The metric as text: a percentage, or a count for the row-count band."""
    if result.name == "row_count_band" and result.metric is not None:
        return f"{result.metric:,.0f} rows"
    return _pct(result.metric)


def _num(threshold: float | None) -> str:
    if threshold is None:
        return "n/a"
    return f"{threshold:,.0f}" if threshold >= 1000 else f"{threshold:g}"


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


def _composition_table(results: Sequence[CheckResult]) -> list[str]:
    """The "Composition by receipt kind" section body: merchant columns and a total."""
    table = composition(results)
    if table is None:
        return ["_Not available: the receipt_kinds check is disabled._", ""]
    merchants = table["merchants"]
    kinds = table["kinds"]
    assert isinstance(merchants, list)
    assert isinstance(kinds, list)
    rows = [
        [
            k["kind"],
            "yes" if k["rider_spend"] else "no",
            *(f"{k['by_merchant'][m]:,}" for m in merchants),
            f"{k['total']:,}",
        ]
        for k in kinds
    ]
    totals = [sum(k["by_merchant"][m] for k in kinds) for m in merchants]
    rows.append(["**total**", "", *(f"{n:,}" for n in totals), f"{sum(totals):,}"])
    return [
        "Rider spend: ride, tip_update, fare_correction and cancel_noshow; the other "
        "kinds stay in the data but are left out of spend.",
        "",
        *_table(
            ["kind", "rider spend", *merchants, "total"],
            rows,
            right=range(2, len(merchants) + 3),
        ),
        "",
    ]


def _duplicates_section(results: Sequence[CheckResult]) -> list[str]:
    """The "Duplicate receipts" section body, from the ``duplicate_receipts`` result."""
    for r in results:
        if r.name != "duplicate_receipts":
            continue
        d = r.details
        pairs = d.get("kind_pairs")
        if not isinstance(pairs, list):
            return ["_Not available._", ""]
        spend = d.get("excess_spend")
        spend_text = f"{spend:,.2f}" if isinstance(spend, int | float) else "n/a"
        out = [
            f"{d.get('groups', 0):,} order keys repeat, in {r.n_failed:,} rows. The "
            f"dedupe flags {d.get('excess_rows', 0):,} of them as superseded (rows "
            f"are never dropped); their USD rider spend is {spend_text} "
            f"({d.get('currency', 'USD only')}).",
            "",
        ]
        if pairs:
            out += _table(
                ["kinds in the group", "groups", "excess spend"],
                [
                    [p["pair"], f"{p['groups']:,}", f"{p['excess_spend']:,.2f}"]
                    for p in pairs
                ],
                right=(1, 2),
            )
            out.append("")
        return out
    return ["_Not available: the duplicate_receipts check is disabled._", ""]


def _money(cell: object) -> tuple[str, str]:
    """(rows, amount) text of a ``{"rows", "amount"}`` cell."""
    assert isinstance(cell, Mapping)
    return f"{cell['rows']:,}", f"{cell['amount']:,.2f}"


_SPEND_LABELS = {
    "naive_total": "naive total of order_total_amount (all rows, mixed currencies)",
    "currency_suspect": "minus currency-suspect rows (local currency)",
    "non_rider_spend": "minus non-rider-spend kinds",
    "micromobility": "of which micromobility (excluded; not rider spend)",
    "superseded": "minus superseded duplicates",
    "clean_usd_rider_spend": "**= clean USD rider spend**",
}
_TOP_COUNTRIES = 10


def _superseded_note(results: Sequence[CheckResult]) -> list[str]:
    """Why the dedupe's superseded count and the waterfall's superseded line differ."""
    table = spend(results)
    flagged = next(
        (
            r.details.get("excess_rows")
            for r in results
            if r.name == "duplicate_receipts"
        ),
        None,
    )
    if table is None or not isinstance(flagged, int):
        return []
    n, _ = _money(table["superseded"])
    return [
        f'Why "superseded" is {n} rows here but {flagged:,} under Duplicate '
        "receipts: the dedupe flags a row of any kind, in any currency, with or "
        "without an amount. This line counts only the superseded rows that have an "
        "amount and would otherwise be USD rider spend (rider-spend kinds, not "
        "currency-suspect). The rest sit on the other lines above (a currency-suspect "
        "or non-rider superseded row) or have no amount.",
        "",
    ]


def _spend_section(results: Sequence[CheckResult]) -> list[str]:
    """The "Spend" section body: the waterfall and the currency-suspect breakdown."""
    table = spend(results)
    if table is None:
        return ["_Not available: the currency_suspect_rows check is disabled._", ""]
    rows = []
    for step, label in _SPEND_LABELS.items():
        if step not in table:  # an older document without the micromobility line
            continue
        n, amount = _money(table[step])
        # The micromobility line is part of the non-rider step above it, not a
        # further deduction.
        sign = "" if step in ("naive_total", "clean_usd_rider_spend") else "-"
        if step == "micromobility":
            sign = ""
        rows.append([label, n, f"{sign}{amount}"])
    out = [
        "Receipts with an amount; the file has no currency column. Rows are flagged, "
        "never dropped, and no exchange rate is applied.",
        "",
        *_table(["step", "rows", "amount"], rows, right=(1, 2)),
        "",
        *_superseded_note(results),
    ]
    n, amount = _money(table["suspect_rider_spend"])
    out += [
        f"Currency-suspect rows that would otherwise be rider spend: {n} "
        f"({amount}; {table['note']}).",
        "",
    ]
    for r in results:
        if r.name != "currency_suspect_rows":
            continue
        reasons, countries = r.details.get("by_reason"), r.details.get("by_country")
        if isinstance(reasons, Mapping) and reasons:
            out += [
                *_table(
                    ["currency reason", "rows", "amount (local currency)"],
                    [[k, *_money(v)] for k, v in reasons.items()],
                    right=(1, 2),
                ),
                "",
            ]
        if isinstance(countries, Mapping) and countries:
            top = list(countries.items())[:_TOP_COUNTRIES]
            out += [
                f"Top {len(top)} of {len(countries)} sender countries "
                "((none) is a generic sender):",
                "",
                *_table(
                    ["sender country", "rows", "amount (local currency)"],
                    [[k, *_money(v)] for k, v in top],
                    right=(1, 2),
                ),
                "",
            ]
    return out


def _arithmetic_section(results: Sequence[CheckResult]) -> list[str]:
    """The "Arithmetic" section body, from ``identity_share`` (A7)."""
    for r in results:
        if r.name != "identity_share":
            continue
        d = r.details
        keys = ("holds", "above", "below", "not_computable", "population")
        if not all(isinstance(d.get(k), int) for k in keys):
            return ["_Not available._", ""]
        pop = int(d["population"])  # type: ignore[call-overload]

        def share(n: object, pop: int = pop) -> str:
            return f"{n / pop:.2%}" if pop and isinstance(n, int) else "n/a"

        gap = d.get("median_gap_above")
        return [
            "The identity `total = subtotal + tax - discount` over USD rider-spend "
            "ride receipts that are not superseded (the population of the "
            "`unexplained_residual` check). It cannot be an error rule: tips, tolls "
            "and fees are not itemized and many credits are missing from "
            "`order_discount`, so most rows legitimately differ. Above means the "
            "total exceeds the itemized lines, below that it is under them; a null "
            "tax or discount counts as 0, a null total or subtotal is not computable.",
            "",
            *_table(
                ["identity (within 0.01)", "rows", "share"],
                [
                    ["holds", f"{d['holds']:,}", share(d["holds"])],
                    ["total above", f"{d['above']:,}", share(d["above"])],
                    ["total below", f"{d['below']:,}", share(d["below"])],
                    [
                        "not computable",
                        f"{d['not_computable']:,}",
                        share(d["not_computable"]),
                    ],
                ],
                right=(1, 2),
            ),
            "",
            f"{pop:,} rows in the population"
            + (
                f"; median gap where the total is above: ${gap:,.2f}."
                if isinstance(gap, int | float)
                else "."
            ),
            "",
        ]
    return ["_Not available: the identity_share check is disabled._", ""]


def _parsing_section(results: Sequence[CheckResult]) -> list[str]:
    """The "Description parsing" section body, from ``description_parse_coverage``."""
    for r in results:
        if r.name != "description_parse_coverage":
            continue
        d = r.details
        merchants, units, families = (
            d.get("merchants"),
            d.get("units"),
            d.get("families"),
        )
        if not (
            isinstance(merchants, Mapping)
            and isinstance(units, Mapping)
            and isinstance(families, Mapping)
        ):
            return ["_Not available._", ""]
        names = [str(m) for m in merchants]
        out = [
            "Ride and tip-update receipts that are not superseded, without the "
            "`Increase Tip` / `Tip Increase` stub lines (they name no trip); a row is "
            "parsed when both its distance and its duration are read. The check's "
            f"rate is the {d.get('metric_label', 'worst merchant unparsed rate')}: "
            "the highest unparsed share of any one merchant, not the failed rows "
            "over the whole population.",
            "",
            *_table(
                ["merchant", "rows", "parsed", "coverage"],
                [
                    [
                        m,
                        f"{v['rows']:,}",
                        f"{v['parsed']:,}",
                        f"{v['coverage']:.2%}",
                    ]
                    for m, v in merchants.items()
                ],
                right=(1, 2, 3),
            ),
            "",
            "Distance unit: " + ", ".join(f"{u} {n:,}" for u, n in units.items()) + ".",
            "",
        ]
        kinds = [f for f in (*FAMILIES, "(none)") if any(families[m][f] for m in names)]
        out += [
            *_table(
                ["product family", *names],
                [[f, *(f"{families[m][f]:,}" for m in names)] for f in kinds],
                right=range(1, len(names) + 1),
            ),
            "",
        ]
        shapes = d.get("unparsed_shapes")
        if isinstance(shapes, list) and shapes:
            out += [
                "Most common unparsed description shapes (digits shown as #):",
                "",
                *_table(
                    ["shape", "rows"],
                    [[s["shape"], f"{s['count']:,}"] for s in shapes],
                    right=(1,),
                ),
                "",
            ]
        return out
    return ["_Not available: the description_parse_coverage check is disabled._", ""]


def _by_field(results: Sequence[CheckResult], name: str) -> dict[str, int] | None:
    """The ``by_field`` counts of check `name`, or None when it didn't run."""
    for r in results:
        if r.name == name:
            raw = r.details.get("by_field")
            if isinstance(raw, dict):
                return {str(k): v for k, v in raw.items() if isinstance(v, int)}
    return None


def _time_section(results: Sequence[CheckResult], run: RunInfo) -> list[str]:
    """The "Time" section body: window, T2 tolerance, T3 lateness, date-only shares."""
    window = f"{run.window_start} to {run.window_end}" if run.window_start else "n/a"
    out = [
        "`order_time` is read as the trip's local time and `email_time` as UTC, so "
        "the checks compare dates, not instants (see the README, Assumptions). "
        f"Email window: {window}.",
        "",
    ]
    by_name = {r.name: r for r in results}
    t2, t3 = by_name.get("order_before_receipt"), by_name.get("late_receipts")
    t2f = _by_field(results, "order_before_receipt") or {}
    t3f = _by_field(results, "late_receipts") or {}
    rows: list[list[str]] = []
    if t2 is not None:
        rows.append(
            [
                "order date after email date + 1 day (T2, error)",
                f"{t2.n_failed:,}",
            ]
        )
        rows.append(
            [
                "order time after email time, no tolerance",
                f"{t2f.get('order_after_email_no_tolerance', 0):,}",
            ]
        )
    if t3 is not None:
        rows.append(["email more than 7 days after the order (T3)", f"{t3.n_failed:,}"])
        rows.append(
            [
                "email more than 30 days after the order",
                f"{t3f.get('over_30_days', 0):,}",
            ]
        )
    if rows:
        out += _table(["time check", "rows"], rows, right=(1,))
        out.append("")
    if t2 is not None:
        out += [
            "The T2 tolerance is one day: a trip in a zone east of UTC can be on the "
            "next calendar day locally while its UTC receipt is still on the previous "
            "one. The no-tolerance row is the evidence for the local-vs-UTC "
            "assumption.",
            "",
        ]
    t6 = by_name.get("date_only_order_time")
    if t6 is None:
        out += [
            "_Date-only shares not available: date_only_order_time is disabled._",
            "",
        ]
        return out
    merchants = t6.details.get("by_merchant")
    if isinstance(merchants, dict) and merchants:
        out += _table(
            ["merchant", "order_time rows", "date only", "share"],
            [
                [
                    str(name),
                    f"{v['rows']:,}",
                    f"{v['date_only']:,}",
                    "n/a" if v["share"] is None else f"{v['share']:.2%}",
                ]
                for name, v in merchants.items()
            ],
            right=(1, 2, 3),
        )
        out += [
            "",
            "Date only means `order_time` is exactly midnight, read as a time that "
            "was not recorded.",
            "",
        ]
    return out


def _plausibility_section(results: Sequence[CheckResult]) -> list[str]:
    """The "Trip plausibility" section body, from O1-O3 (one row per check)."""
    names = ("implausible_speed", "zero_trip_metrics", "extreme_trip")
    by_name = {r.name: r for r in results}
    found = [by_name[n] for n in names if n in by_name]
    if not found:
        return ["_Not available: the trip plausibility checks are disabled._", ""]
    rows = []
    for r in found:
        split = _by_field(results, r.name) or {}
        rows.append(
            [
                f"{r.check_id} {r.name}",
                r.severity.value,
                r.predicate,
                f"{r.n_failed:,}",
                f"{r.n_evaluated:,}",
                _pct(r.metric),
                ", ".join(f"{k} {v:,}" for k, v in split.items()) or "none",
            ]
        )
    return [
        "Physical checks on the parsed trip metrics, over ride and tip-update "
        "receipts that are not superseded. A null distance or duration is never "
        "flagged (the parse coverage check owns it), and the speed is computed only "
        "where the duration is above 0.",
        "",
        *_table(
            ["check", "severity", "rule", "flagged", "of", "rate", "breakdown"],
            rows,
            right=(3, 4, 5),
        ),
        "",
    ]


def _outliers_section(results: Sequence[CheckResult]) -> list[str]:
    """The "Outliers" section body, from ``fare_outlier_conditional`` (O4) and
    ``fare_outlier_univariate`` (O5)."""
    by_name = {r.name: r for r in results}
    o4, o5 = (
        by_name.get("fare_outlier_conditional"),
        by_name.get("fare_outlier_univariate"),
    )
    if o4 is None and o5 is None:
        return ["_Not available: both outlier checks are disabled._", ""]
    out: list[str] = []
    if o4 is not None:
        d = o4.details
        out += [
            f"**O4, conditional fare outliers ({o4.severity.value}):** k = "
            f"{_g(d.get('k'))}, flagged {o4.n_failed:,} of {o4.n_evaluated:,} "
            f"eligible rides (flag rate {_pct(o4.metric)}); "
            f"{_count(d.get('scored'))} were scored (a segment under "
            f"{_count(d.get('min_group'))} rows, or with no spread, is not).",
            "",
        ]
        grid = d.get("sensitivity")
        if isinstance(grid, list) and grid:
            out += [
                "Flags by threshold k (a review budget, not a false-positive rate):",
                "",
                *_table(
                    ["k", "flagged", "rate"],
                    [[_g(g["k"]), f"{g['flagged']:,}", _pct(g["rate"])] for g in grid],
                    right=(0, 1, 2),
                ),
                "",
            ]
        direction = d.get("by_direction")
        if isinstance(direction, Mapping):
            out += [
                "Direction of the flagged rides: both prices high is overpriced, both "
                "low is underpriced or near-zero, mixed is one of each.",
                "",
                *_table(
                    ["direction", "flagged"],
                    [[name, f"{n:,}"] for name, n in direction.items()],
                    right=(1,),
                ),
                "",
            ]
        segments = d.get("by_segment")
        if isinstance(segments, list):
            top = [g for g in segments if g["flagged"]][:5]
            if top:
                out += [
                    "Top segments by flags:",
                    "",
                    *_table(
                        ["merchant", "product family", "rides", "flagged", "rate"],
                        [
                            [
                                g["merchant"],
                                g["family"],
                                f"{g['rows']:,}",
                                f"{g['flagged']:,}",
                                _pct(g["rate"]),
                            ]
                            for g in top
                        ],
                        right=(2, 3, 4),
                    ),
                    "",
                ]
        families = d.get("by_family")
        if isinstance(families, list) and families:
            out += [
                "Median price by product family (eligible rides, USD):",
                "",
                *_table(
                    ["product family", "rides", "median $/mile", "median $/minute"],
                    [
                        [
                            g["family"],
                            f"{g['rows']:,}",
                            f"{g['median_per_mile']:.2f}",
                            f"{g['median_per_minute']:.2f}",
                        ]
                        for g in families
                    ],
                    right=(1, 2, 3),
                ),
                "",
            ]
    else:
        out += ["_O4 is disabled._", ""]
    if o5 is not None:
        d = o5.details
        out += [
            f"**O5, univariate fare outliers ({o5.severity.value}, "
            f"{d.get('note', 'context only')}):** k = {_g(d.get('k'))}, "
            f"{o5.n_failed:,} of {o5.n_evaluated:,} rides "
            f"({_pct(o5.metric)}), {_count(d.get('high'))} high and "
            f"{_count(d.get('low'))} low. Contrast with O4: a long trip has a high "
            "fare but an ordinary price per mile.",
            "",
        ]
        merchants = d.get("by_merchant")
        if isinstance(merchants, Mapping) and merchants:
            out += [
                *_table(
                    ["merchant", "rides", "flagged", "high", "low", "rate"],
                    [
                        [
                            m,
                            f"{v['rows']:,}",
                            f"{v['flagged']:,}",
                            f"{v['high']:,}",
                            f"{v['low']:,}",
                            _pct(v["rate"]),
                        ]
                        for m, v in merchants.items()
                    ],
                    right=(1, 2, 3, 4, 5),
                ),
                "",
            ]
    out += [
        "Method: a robust z-score, (x - median) / (1.4826 x MAD), of the log price "
        "within merchant x product family. A ride is an O4 outlier only when its "
        "price per mile and its price per minute are both extreme, so long or slow "
        "trips are not flagged for their length alone. Fares are heavy-tailed and "
        "grow with distance, which a raw z-score or fence ignores (it flags 1-3% of "
        "rows, almost all of them long trips). The threshold k sets how many rows "
        "get reviewed; it is a budget, not a false-positive rate. There is no "
        "machine learning: a z-score on a log scale says why a row was flagged, "
        "and an isolation forest flagged different rows without a reason to read.",
        "",
    ]
    return out


def _g(value: object) -> str:
    return f"{value:g}" if isinstance(value, int | float) else "n/a"


def _count(value: object) -> str:
    return f"{value:,}" if isinstance(value, int) else "n/a"


def _sample_caption(result: CheckResult, shown: int) -> str:
    """The caption of a sample: "20 of 112 failing rows", or what the count means."""
    group_rows = result.details.get("rows")
    if result.name == "near_duplicate_rides" and isinstance(group_rows, int):
        groups = result.details.get("groups", 0)
        return (
            f"{shown} sample rows of the {group_rows:,} rows in {groups:,} near-"
            f"duplicate groups; {result.n_failed:,} failing rows is the excess (every "
            "row of a group but one)"
        )
    return f"{shown} of {result.n_failed:,} failing rows"


def _curated_section(run: RunInfo) -> list[str]:
    if run.parquet_name is None:
        return ["No curated Parquet was written.", ""]
    return [
        (
            f"[{run.parquet_name}]({run.parquet_name}): {run.rows:,} rows (none "
            f"dropped; superseded rows carry `is_superseded`) · "
            f"{run.curated_columns} columns."
        ),
        "",
        (
            "Each `qc_<check>` column is True on the rows that fail that check, and "
            "its count is the check's `failed` figure in the table above; "
            "`qc_error` and `qc_warn` combine them by severity, and `qc_flags` "
            "lists the names."
        ),
        "",
    ]


def render_markdown(results: Sequence[CheckResult], run: RunInfo, code: int) -> str:
    """Render the report as Markdown."""
    s = summarize(results)
    window = f"{run.window_start}..{run.window_end}" if run.window_start else "n/a"
    out = [
        f"# QC report: {run.input_name} ({run.sha256[:12]})",
        "",
        *(
            [
                f"member: {run.member.name} ({run.member.sha256[:12]})",
                "",
            ]
            if run.member
            else []
        ),
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
                metric_text(r),
                _num(r.threshold),
                r.predicate,
            ]
            for r in _sorted(results)
        ],
        right=(3, 4, 5),
    )
    out += ["", "## Composition by receipt kind", ""]
    out += _composition_table(results)
    out += ["## Duplicate receipts", ""]
    out += _duplicates_section(results)
    out += ["## Spend", ""]
    out += _spend_section(results)
    out += ["## Arithmetic", ""]
    out += _arithmetic_section(results)
    out += ["## Description parsing", ""]
    out += _parsing_section(results)
    out += ["## Time", ""]
    out += _time_section(results, run)
    out += ["## Outliers", ""]
    out += _outliers_section(results)
    out += ["## Trip plausibility", ""]
    out += _plausibility_section(results)
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
    out += ["## Curated dataset", ""]
    out += _curated_section(run)
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
            f"{_sample_caption(r, shown)} · [{sample_name(r)}]({sample_name(r)})",
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
    """Write report.json, report.md and samples/<check>.parquet into out_dir.

    Every file is written to a temporary name beside it, then renamed into place,
    so a file is either the old one or the new one, never partial. Nothing is
    deleted but the old samples/*.parquet; other files in out_dir (a Parquet from
    another input, say) are left alone, and report.json's ``outputs`` lists the
    files of this run.
    """
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
            with atomic_path(path) as tmp:
                r.sample.write_parquet(tmp, compression="zstd")
            samples.append(path)
    json_path = out_dir / "report.json"
    write_text_atomic(
        json_path,
        json.dumps(build_report(results, run, code), indent=2, allow_nan=False) + "\n",
    )
    md_path = out_dir / "report.md"
    write_text_atomic(md_path, render_markdown(results, run, code) + "\n")
    return Written(json_path, md_path, tuple(samples))
