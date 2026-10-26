"""QC checks: pure functions of a polars LazyFrame, grouped by category.

A check runs a query and returns what it found. It never prints, writes or
changes the data; acting on a finding (dropping a column, say) is up to the caller.

The registry is the tuple returned by default_rules(): each category module
(arithmetic.py, uniqueness.py, ...) exposes ``rules()`` and is listed there. No
decorators, no plugins. run_checks() takes the rule sets as arguments, so a
caller (a test, or later a config with per-check thresholds) can pass its own.
"""

from collections.abc import Iterable, Mapping, Sequence
lazy from concurrent.futures import ThreadPoolExecutor

lazy import polars as pl

from rideshare_data_pipeline.checks import (
    arithmetic,
    currency,
    kinds,
    outliers,
    parsing,
    plausibility,
    structural,
    temporal,
    uniqueness,
)
from rideshare_data_pipeline.checks.model import (
    SAMPLE_LIMIT,
    CheckResult,
    DatasetCheck,
    FlaggedCheck,
    RowRule,
)

__all__ = [
    "CheckResult",
    "DatasetCheck",
    "FlaggedCheck",
    "RowRule",
    "default_dataset_checks",
    "default_rules",
    "evaluate_rules",
    "fails",
    "run_checks",
    "run_checks_evaluated",
]


def default_rules(
    params: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[RowRule, ...]:
    """Return every registered row rule; `params` feeds the plausibility rules."""
    return (
        *structural.rules(),
        *arithmetic.rules(),
        *uniqueness.rules(),
        *kinds.rules(),
        *currency.rules(),
        *temporal.rules(),
        *plausibility.rules(params),
    )


def default_dataset_checks(
    params: Mapping[str, Mapping[str, float]] | None = None,
    sample_limit: int = SAMPLE_LIMIT,
) -> tuple[DatasetCheck, ...]:
    """Return every registered dataset check.

    `params` (``{check name: {param: value}}``, from ``QCConfig.check_params``) and
    `sample_limit` are threaded into the checks that take them (the outlier checks).
    """
    return (
        *structural.dataset_checks(),
        *arithmetic.dataset_checks(),
        *kinds.dataset_checks(),
        *uniqueness.dataset_checks(),
        *parsing.dataset_checks(),
        *currency.dataset_checks(),
        *temporal.dataset_checks(),
        *outliers.dataset_checks(params, sample_limit),
    )


def fails(rule: RowRule, predicate: pl.Expr | None = None) -> pl.Expr:
    """The rows that count as failed: applicable, and the predicate is True.

    A null predicate means "can't tell", which is not a failure (a missing value
    is a completeness question, not an arithmetic one); fill_null makes that rule
    explicit so it doesn't depend on how `&` treats nulls. `predicate` defaults to
    the rule's own (a breakdown field passes its own). This is also the filter of
    the sample query, so it holds no length-broadcast literal (``pl.repeat``, as
    `_applies` does): that combined with ``head`` made the streaming engine panic
    now and then.
    """
    failed = (rule.predicate if predicate is None else predicate).fill_null(False)
    if rule.applies_to is not None:
        failed = failed & rule.applies_to.fill_null(False)
    return failed


def _applies(rule: RowRule) -> pl.Expr:
    # A bare literal would sum as a single scalar (1), not once per row, so
    # broadcast it to the frame's height; this keeps `evaluated` a row count.
    applies = pl.repeat(True, pl.len())
    if rule.applies_to is not None:
        applies = applies & rule.applies_to
    return applies.fill_null(False)


def _result(
    rule: RowRule,
    n_failed: int,
    n_evaluated: int,
    sample: pl.DataFrame | None,
    by_field: Mapping[str, int] | None = None,
) -> CheckResult:
    """Judge one rule: passed when failed / evaluated <= max_failed_rate.

    A rule that applied to no rows has nothing to judge: it passes, with no
    metric and no sample.
    """
    metric = n_failed / n_evaluated if n_evaluated else None
    return CheckResult(
        name=rule.name,
        check_id=rule.check_id,
        category=rule.category,
        severity=rule.severity,
        passed=metric is None or metric <= rule.max_failed_rate,
        metric=metric,
        threshold=rule.max_failed_rate,
        n_failed=n_failed,
        n_evaluated=n_evaluated,
        predicate=rule.rule,
        sample=sample,
        details={} if by_field is None else {"by_field": dict(by_field)},
    )


def evaluate_rules(lf: pl.LazyFrame, rules: Sequence[RowRule]) -> pl.DataFrame:
    """Evaluate every row rule in one scan into a narrow table of boolean columns.

    Per rule ``i``: ``failed_i`` (the rows that fail it, see `fails`), ``evaluated_i``
    (the rows it applies to) and ``field_i_j`` (its j-th breakdown field). Booleans
    are bit-packed, so the table is a few MB on the 1.9M-row sample, and counting
    a column from it costs nothing; materializing the columns is also about twice as
    fast as aggregating them in the scan. The curated flags reuse the ``failed_i``
    columns (see ``curated.row_flags``), so each predicate runs over the data once.
    """
    return lf.select(
        expr
        for i, rule in enumerate(rules)
        for expr in (
            fails(rule).alias(f"failed_{i}"),
            _applies(rule).alias(f"evaluated_{i}"),
            *(
                fails(rule, expr).alias(f"field_{i}_{j}")
                for j, expr in enumerate(rule.breakdown.values())
            ),
        )
    ).collect(engine="streaming")


def run_rules(
    lf: pl.LazyFrame,
    rules: Sequence[RowRule],
    *,
    sample_limit: int = SAMPLE_LIMIT,
    evaluated: pl.DataFrame | None = None,
) -> list[CheckResult]:
    """Evaluate every row rule: one scan for the counts, one batch for the samples.

    The counts of all rules come from a single scan (evaluate_rules), so the data is
    read once however many rules there are; pass that table as `evaluated` if the
    caller already has it. Samples are collected only for the rules that failed,
    together in one collect_all (a second pass over the data, and only when
    something failed).
    """
    if not rules:
        return []
    table = evaluate_rules(lf, rules) if evaluated is None else evaluated
    counts = table.select(pl.all().sum()).row(0, named=True)
    failed = [counts[f"failed_{i}"] for i in range(len(rules))]
    evaluated_counts = [counts[f"evaluated_{i}"] for i in range(len(rules))]

    by_field = [
        (
            {name: counts[f"field_{i}_{j}"] for j, name in enumerate(r.breakdown)}
            if r.breakdown
            else None
        )
        for i, r in enumerate(rules)
    ]
    judged = [
        _result(r, f, n, None, b)
        for r, f, n, b in zip(rules, failed, evaluated_counts, by_field, strict=True)
    ]
    failing = [i for i, res in enumerate(judged) if not res.passed]
    if not failing:
        return judged

    present = set(lf.collect_schema().names())
    samples = pl.collect_all(
        [
            lf.filter(fails(rules[i]))
            .select(c for c in rules[i].sample_cols if c in present)
            .head(sample_limit)
            for i in failing
        ],
        engine="streaming",
    )
    for i, sample in zip(failing, samples, strict=True):
        judged[i] = _result(
            rules[i], failed[i], evaluated_counts[i], sample, by_field[i]
        )
    return judged


def run_checks_evaluated(
    lf: pl.LazyFrame,
    *,
    rules: Iterable[RowRule] | None = None,
    dataset_checks: Iterable[DatasetCheck] | None = None,
    sample_limit: int = SAMPLE_LIMIT,
) -> tuple[list[CheckResult], pl.DataFrame | None]:
    """Run row rules, then dataset checks; return the results and the rule table.

    The results are in that order (see run_checks). The table is evaluate_rules()
    of the rules, in order, for a caller that wants the per-row flags without
    evaluating the predicates again; None when there are no rules.

    The row-rule scan and the dataset checks are independent queries on the same
    read-only frame, and polars releases the GIL while it runs one, so the scan
    goes on a second thread. The dataset checks stay in order on one thread: some
    share a cache by frame identity (the outlier pair, the column profile), which
    two threads would fill twice.
    """
    row_rules = tuple(default_rules() if rules is None else rules)
    checks = tuple(
        default_dataset_checks() if dataset_checks is None else dataset_checks
    )
    table: pl.DataFrame | None = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        scan = pool.submit(evaluate_rules, lf, row_rules) if row_rules else None
        dataset_results = [check(lf) for check in checks]
        if scan is not None:
            table = scan.result()
    results = (
        run_rules(lf, row_rules, sample_limit=sample_limit, evaluated=table)
        if table is not None
        else []
    )
    results.extend(dataset_results)
    return results, table


def run_checks(
    lf: pl.LazyFrame,
    *,
    rules: Iterable[RowRule] | None = None,
    dataset_checks: Iterable[DatasetCheck] | None = None,
    sample_limit: int = SAMPLE_LIMIT,
) -> list[CheckResult]:
    """Run row rules, then dataset checks, and return their results in that order.

    None means the registry defaults. Row rules share one scan (see run_rules).
    Thresholds come from each rule's max_failed_rate; a later config layer can
    pass rules rebuilt with other rates (``dataclasses.replace``) without
    changing this function.
    """
    return run_checks_evaluated(
        lf, rules=rules, dataset_checks=dataset_checks, sample_limit=sample_limit
    )[0]
