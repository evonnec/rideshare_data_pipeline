"""The curated dataset: the typed input, the derived columns and per-row QC flags.

Every input row is kept (the owner decision for #22: nothing is dropped, so a
consumer can always get back to the delivered file). The dedupe policy is the
``is_superseded`` flag, not a deletion: consumers filter on flags.

Columns, in order:

1. the typed input columns, in file order (minus any ``--drop-columns``);
2. the derived columns (``parse.DERIVED_NAMES``);
3. the dedupe flags (``dedupe.FLAG_NAMES``);
4. ``is_usd_rider_spend`` (``money.is_usd_rider_spend``);
5. ``qc_<check name>``, one boolean per row-flagged check that is enabled, in
   report order. True means the row FAILS that check. A check that has a flag is
   every row rule (its predicate AND ``applies_to``, a null counting as False: the
   runner's own rule, so the column sums to the check's ``n_failed``) and the
   dataset checks that carry a ``FlaggedCheck`` row flag. Dataset-level findings
   (header, row-count band, column profiles, ...) have no row to point at and no
   column;
6. ``qc_error`` and ``qc_warn``: the row fails at least one error- (warn-) severity
   flag, severities as configured;
7. ``qc_flags``: the names of the checks the row fails, ``list[str]``, empty (never
   null) on a clean row.

Flags are per row; pass or fail is per rate. A ``qc_<check>`` flag means this row
breaks the rule, whatever the check's ``max_failed_rate``. A check passes when its
failure RATE is within the threshold, so with a loosened threshold a check can PASS
while its flag is still True on the failing rows, and ``qc_error``/``qc_warn``
(which follow the flags, not the verdicts) are still True on them.

The flags are evaluated once into a narrow frame (``flag_table``; the temporary
``POSITION`` column exists only there and never reaches the file), attached to the
typed frame in one lazy ``with_columns`` and written with ``sink_parquet``. The typed
frame itself is already in memory for the checks, so nothing of it is copied.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

lazy import polars as pl

from rideshare_data_pipeline.checks import (
    default_dataset_checks,
    default_rules,
    fails,
)
from rideshare_data_pipeline.checks.model import (
    POSITION,
    CheckResult,
    DatasetCheck,
    Expr,
    FlaggedCheck,
    Lazy,
    RowRule,
    Severity,
)
from rideshare_data_pipeline.config import KNOWN_CHECKS
from rideshare_data_pipeline.dedupe import FLAG_NAMES
from rideshare_data_pipeline.io import atomic_path
from rideshare_data_pipeline.money import is_usd_rider_spend
from rideshare_data_pipeline.parse import (
    CURRENCY_REASONS,
    DERIVED_NAMES,
    DISTANCE_UNITS,
    FAMILIES,
    KINDS,
)
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS, csv_schema

PREFIX = "qc_"
USD_RIDER_SPEND = "is_usd_rider_spend"
ERROR = "qc_error"
WARN = "qc_warn"
FLAGS = "qc_flags"


@dataclass(frozen=True, slots=True)
class RowFlag:
    """One check's per-row failure flag: True on the rows that fail it."""

    name: str
    severity: Severity
    expr: Expr

    @property
    def column(self) -> str:
        return f"{PREFIX}{self.name}"


def flagged_names() -> tuple[str, ...]:
    """The names of the checks that have a per-row flag, in report order."""
    names = {r.name for r in default_rules()}
    names |= {c.name for c in default_dataset_checks() if isinstance(c, FlaggedCheck)}
    return tuple(n for n in KNOWN_CHECKS if n in names)


def row_flags(
    lf: Lazy,
    rules: Iterable[RowRule],
    dataset_checks: Iterable[DatasetCheck],
    results: Sequence[CheckResult],
    evaluated: pl.DataFrame | None = None,
) -> list[RowFlag]:
    """The flags of the checks that ran, in report order.

    `evaluated` is ``evaluate_rules(lf, rules)`` for these `rules`, if the caller
    has it: a rule's flag is then its ``failed_i`` column, already computed, instead
    of its predicate evaluated over the data again.

    `rules` and `dataset_checks` are the ones that were run (the same objects, and
    `lf` the same frame the checks ran on: a flagged check may reuse work it cached
    for that frame, the outlier scores for one). Severities come from `results`, so
    a configured override is what ``qc_error`` and ``qc_warn`` see; a check that is
    not in `results` (disabled) has no flag.
    """
    severity = {r.name: r.severity for r in results}
    found: dict[str, Expr] = {}
    for i, rule in enumerate(rules):
        if rule.name in severity:
            found[rule.name] = (
                fails(rule) if evaluated is None else pl.lit(evaluated[f"failed_{i}"])
            )
    for check in dataset_checks:
        if isinstance(check, FlaggedCheck) and check.name in severity:
            found[check.name] = check.row_flag(lf)
    return [RowFlag(n, severity[n], found[n]) for n in KNOWN_CHECKS if n in found]


def flag_table(lf: Lazy, flags: Sequence[RowFlag]) -> pl.DataFrame:
    """Evaluate the flags once: ``qc_<check>``, ``qc_error``, ``qc_warn``, ``qc_flags``.

    One narrow frame with a row per row of `lf`, in its order. The booleans are
    bit-packed (one bit per flag per row), so this is not a second copy
    of the data. Building ``qc_flags`` per row for every row would dominate the run
    (a 40-way list per row), and only a small share of the rows fail anything, so
    the names are built for the failing rows only and joined back by position.
    """
    columns = [f.column for f in flags]

    def any_of(severity: Severity) -> Expr:
        picked = [pl.col(f.column) for f in flags if f.severity is severity]
        return pl.any_horizontal(picked) if picked else pl.lit(False)

    evaluated = (
        lf.with_row_index(POSITION)
        .select(
            POSITION,
            *(f.expr.fill_null(False).alias(f.column) for f in flags),
        )
        .with_columns(
            any_of(Severity.ERROR).alias(ERROR), any_of(Severity.WARN).alias(WARN)
        )
        .collect(engine="streaming")
    )
    # With no flagged check, a null literal makes the (empty) list of names.
    named = [pl.when(pl.col(c)).then(pl.lit(c.removeprefix(PREFIX))) for c in columns]
    failing = (
        evaluated.filter(pl.any_horizontal(columns) if columns else pl.lit(False))
        .select(
            POSITION,
            pl.concat_list(named or [pl.lit(None, pl.String)])
            .list.drop_nulls()
            .alias(FLAGS),
        )
        .lazy()
    )
    return (
        evaluated.lazy()
        .join(failing, on=POSITION, how="left", maintain_order="left")
        .with_columns(pl.col(FLAGS).fill_null(pl.lit([], dtype=pl.List(pl.String))))
        .select(*columns, ERROR, WARN, FLAGS)
        .collect()
    )


def curated_plan(
    lf: Lazy, table: pl.DataFrame, drop_columns: Iterable[str] = ()
) -> Lazy:
    """The curated frame as a lazy plan (see the module docstring for the columns).

    `lf` is the typed frame with the derived and dedupe columns; `table` is its
    ``flag_table``, attached as literal columns.
    """
    dropped = set(drop_columns)
    inputs = [c for c in EXPECTED_COLUMNS if c not in dropped]
    return lf.with_columns(
        is_usd_rider_spend().alias(USD_RIDER_SPEND),
        *(pl.lit(table[c]) for c in table.columns),
    ).select(*inputs, *DERIVED_NAMES, *FLAG_NAMES, USD_RIDER_SPEND, *table.columns)


def curated_schema(
    enabled_checks: Iterable[str], drop_columns: Iterable[str] = ()
) -> pl.Schema:
    """The dtype of every column of the curated Parquet, in file order.

    `enabled_checks` are the names of the checks that run; only those that have a
    row flag (``flagged_names``) get a ``qc_<name>`` column. `drop_columns` are the
    input columns left out.
    """
    dropped = set(drop_columns)
    enabled = set(enabled_checks)
    boolean = pl.Boolean()
    derived: dict[str, pl.DataType] = {
        "receipt_kind": pl.Enum(KINDS),
        "subject_profile": pl.String(),
        "subject_unclassified": boolean,
        "is_rider_spend": boolean,
        "sender_country": pl.String(),
        "currency_reason": pl.Enum(CURRENCY_REASONS),
        "currency_suspect": boolean,
        "foreign_subject": boolean,
        "distance_mi": pl.Float64(),
        "distance_unit": pl.Enum(DISTANCE_UNITS),
        "duration_s": pl.Int64(),
        "product_family": pl.Enum(FAMILIES),
    }
    return pl.Schema(
        {
            **{c: t for c, t in csv_schema().items() if c not in dropped},
            **{n: derived[n] for n in DERIVED_NAMES},
            **dict.fromkeys(FLAG_NAMES, boolean),
            USD_RIDER_SPEND: boolean,
            **{f"{PREFIX}{n}": boolean for n in flagged_names() if n in enabled},
            ERROR: boolean,
            WARN: boolean,
            FLAGS: pl.List(pl.String()),
        }
    )


def write_curated(plan: Lazy, path: Path) -> None:
    """Stream the plan to a zstd Parquet at `path`, atomically (see atomic_path)."""
    with atomic_path(path) as tmp:
        plan.sink_parquet(tmp, compression="zstd")
