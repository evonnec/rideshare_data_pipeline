"""QC configuration: defaults in code, optional TOML overrides.

The TOML file has two top-level keys and one table per check::

    sample_limit = 20  # failing rows kept per check
    fail_on = "error"  # "error" or "warn": what makes the run exit 1

    [checks.non_negative_total]
    max_failed_rate = 0.01  # the check passes while failed / evaluated <= this
    severity = "warn"  # "error", "warn" or "info"
    enabled = true  # false drops the check from the run and the report

Unknown keys or check names are errors (ConfigError, exit status 2), so a typo
can't silently do nothing. Everything here is pure; ``load_config`` is the only
function that touches the file system.
"""

import contextlib
import dataclasses
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rideshare_data_pipeline.checks.model import (
    SAMPLE_LIMIT,
    CheckResult,
    RowRule,
    Severity,
)
from rideshare_data_pipeline.io import InputError, StrPath

# Every check a run can report, in report order: the registered row rules, then
# the dataset checks. A test keeps this in step with the registry.
KNOWN_CHECKS: tuple[str, ...] = (
    "cast_failures",
    "non_negative_total",
    "checksum_unique",
    "category_mismatch",
    "all_null_columns",
    "constant_columns",
    "duplicate_rows",
    "receipt_kinds",
    "unclassified_subject_templates",
    "duplicate_receipts",
    "dedupe_key_unique",
    "cross_panelist_duplicates",
    "near_duplicate_rides",
)

# `fail_on` can only be a gating severity: info checks never gate.
_FAIL_ON = (Severity.ERROR, Severity.WARN)
_TOP_KEYS = frozenset({"sample_limit", "fail_on", "checks"})
_CHECK_KEYS = frozenset({"max_failed_rate", "severity", "enabled"})


class ConfigError(InputError):
    """The config file can't be read, or holds an unknown key, name or bad value."""


@dataclass(frozen=True, slots=True)
class CheckOverride:
    """Per-check overrides; the defaults leave the check as the code defines it."""

    max_failed_rate: float | None = None
    severity: Severity | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class QCConfig:
    """Run-wide settings and per-check overrides, keyed by check name."""

    sample_limit: int = SAMPLE_LIMIT
    fail_on: Severity = Severity.ERROR
    checks: Mapping[str, CheckOverride] = field(default_factory=dict)

    def override(self, name: str) -> CheckOverride:
        """The override for `name`, or the no-op default."""
        return self.checks.get(name, CheckOverride())


def _severity(value: object, where: str, allowed: Iterable[Severity]) -> Severity:
    options = tuple(allowed)
    severity = None
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            severity = Severity(value)
    if severity is None or severity not in options:
        names = ", ".join(repr(s.value) for s in options)
        msg = f"{where}: expected one of {names}, got {value!r}"
        raise ConfigError(msg)
    return severity


def _unknown(keys: Iterable[str], allowed: Iterable[str], where: str) -> None:
    bad = sorted(set(keys) - set(allowed))
    if bad:
        known = ", ".join(sorted(allowed))
        msg = f"{where}: unknown {', '.join(map(repr, bad))} (known: {known})"
        raise ConfigError(msg)


def _override(name: str, table: object) -> CheckOverride:
    where = f"[checks.{name}]"
    if not isinstance(table, dict):
        msg = f"{where}: expected a table"
        raise ConfigError(msg)
    _unknown(table, _CHECK_KEYS, where)
    rate = table.get("max_failed_rate")
    if rate is not None and (
        isinstance(rate, bool)
        or not isinstance(rate, int | float)
        or not 0 <= rate <= 1
    ):
        msg = f"{where} max_failed_rate: expected a number from 0 to 1, got {rate!r}"
        raise ConfigError(msg)
    severity = table.get("severity")
    enabled = table.get("enabled", True)
    if not isinstance(enabled, bool):
        msg = f"{where} enabled: expected true or false, got {enabled!r}"
        raise ConfigError(msg)
    return CheckOverride(
        max_failed_rate=None if rate is None else float(rate),
        severity=None
        if severity is None
        else _severity(severity, f"{where} severity", Severity),
        enabled=enabled,
    )


def parse_config(data: Mapping[str, object]) -> QCConfig:
    """Build a QCConfig from parsed TOML, rejecting unknown keys and bad values."""
    _unknown(data, _TOP_KEYS, "config")
    defaults = QCConfig()
    limit = data.get("sample_limit", defaults.sample_limit)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        msg = f"sample_limit: expected a non-negative integer, got {limit!r}"
        raise ConfigError(msg)
    fail_on = _severity(data.get("fail_on", defaults.fail_on), "fail_on", _FAIL_ON)
    tables = data.get("checks", {})
    if not isinstance(tables, dict):
        msg = "checks: expected a table of [checks.<name>] tables"
        raise ConfigError(msg)
    _unknown(tables, KNOWN_CHECKS, "[checks]")
    return QCConfig(
        sample_limit=limit,
        fail_on=fail_on,
        checks={name: _override(name, table) for name, table in tables.items()},
    )


def load_config(path: StrPath) -> QCConfig:
    """Read a TOML config file. Raises ConfigError if it is missing or invalid."""
    path = Path(path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        msg = f"config not found: {path.name}"
        raise ConfigError(msg) from None
    except (OSError, tomllib.TOMLDecodeError) as err:
        msg = f"can't read config {path.name}: {err}"
        raise ConfigError(msg) from err
    try:
        return parse_config(data)
    except ConfigError as err:
        msg = f"{path.name}: {err}"
        raise ConfigError(msg) from None


def configure_rules(rules: Iterable[RowRule], config: QCConfig) -> list[RowRule]:
    """Apply the overrides to row rules before they run; drop the disabled ones."""
    out: list[RowRule] = []
    for rule in rules:
        o = config.override(rule.name)
        if not o.enabled:
            continue
        rate = rule.max_failed_rate if o.max_failed_rate is None else o.max_failed_rate
        severity = rule.severity if o.severity is None else o.severity
        out.append(dataclasses.replace(rule, max_failed_rate=rate, severity=severity))
    return out


def configure_results(
    results: Iterable[CheckResult], config: QCConfig
) -> list[CheckResult]:
    """Apply the overrides to results that were computed without them.

    Used for the dataset checks, which don't take thresholds as input: a disabled
    check is dropped, its sample is cut to ``sample_limit`` rows, an overridden
    severity is set, and an overridden
    max_failed_rate replaces the threshold and re-judges ``passed`` against the
    check's metric (a check with no metric has nothing to judge and passes).
    """
    out: list[CheckResult] = []
    for result in results:
        o = config.override(result.name)
        if not o.enabled:
            continue
        sample = result.sample
        if sample is not None and sample.height > config.sample_limit:
            result = dataclasses.replace(
                result, sample=sample.head(config.sample_limit)
            )
        if o.severity is not None:
            result = dataclasses.replace(result, severity=o.severity)
        if o.max_failed_rate is not None:
            result = dataclasses.replace(
                result,
                threshold=o.max_failed_rate,
                passed=result.metric is None or result.metric <= o.max_failed_rate,
            )
        out.append(result)
    return out
