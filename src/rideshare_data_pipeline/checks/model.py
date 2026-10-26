"""The check contract: what a check is and what it returns.

A check is pure: it reads a LazyFrame and returns a CheckResult. It never
prints, writes or mutates anything. Row-level rules are data (RowRule), so the
runner can evaluate all of them in one scan; checks that need a whole-dataset
view (duplicate rows, column shape) are DatasetChecks.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

lazy import polars as pl

# The columns a failing-row sample shows by default: enough to find the receipt
# in the source, nothing free-text. A rule overrides this with `sample_cols`.
KEY_COLS: tuple[str, ...] = (
    "merchant_name",
    "user_id",
    "order_number",
    "email_time",
    "order_total_amount",
    "checksum",
)

SAMPLE_LIMIT = 20

# @dataclass looks its field annotations up when the class is built, which would
# trigger the lazy polars import. Through these (lazily evaluated) aliases the
# dataclasses below can name polars types and importing this module stays free.
type Frame = pl.DataFrame
type Expr = pl.Expr


class Severity(StrEnum):
    """How much a failed check matters. Info checks report and never fail."""

    ERROR = "error"
    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one check, with the number behind the verdict.

    `metric` is the measured value (the failure rate for a row rule) and
    `threshold` the most it may be for the check to pass; either is None when it
    doesn't apply (an info check has no threshold, and a rule that applied to no
    rows has no rate).
    """

    name: str
    check_id: str
    category: str
    severity: Severity
    passed: bool
    metric: float | None
    threshold: float | None
    n_failed: int
    n_evaluated: int
    predicate: str
    """Human-readable rule, e.g. ``order_total_amount >= 0``."""

    sample: Frame | None = None
    """At most SAMPLE_LIMIT failing rows; None when the check passed or has none."""

    details: Mapping[str, object] = field(default_factory=dict)
    """Extra facts, e.g. the column names an info check found."""


@dataclass(frozen=True, slots=True)
class RowRule:
    """A per-row constraint: the runner counts the rows where `predicate` is True.

    Expressions are built by a function (see checks/arithmetic.py), not at import
    time, so importing a checks module never loads polars.
    """

    name: str
    check_id: str
    category: str
    severity: Severity
    rule: str
    """What a good row satisfies, in words, e.g. ``order_total_amount >= 0``."""

    predicate: Expr
    """True means the row FAILS. A null counts as not failed."""

    applies_to: Expr | None = None
    """Rows the rule is about (the denominator); None means every row."""

    max_failed_rate: float = 0.0
    """The rule passes while failed / evaluated is at most this."""

    sample_cols: tuple[str, ...] = KEY_COLS

    breakdown: Mapping[str, Expr] = field(default_factory=dict)
    """Optional per-field failure predicates (True means failed), counted in the
    same scan as the rule and reported as ``details["by_field"]`` (``{field: n}``,
    among the rows the rule applies to). Not part of the verdict."""


type DatasetCheck = Callable[[pl.LazyFrame], CheckResult]
"""A check on the dataset as a whole: LazyFrame in, CheckResult out."""
