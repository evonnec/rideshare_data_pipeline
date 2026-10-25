"""QCConfig: TOML loading, validation and the override functions."""

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from rideshare_data_pipeline.checks import default_rules, run_checks
from rideshare_data_pipeline.checks.model import CheckResult, Severity
from rideshare_data_pipeline.config import (
    KNOWN_CHECKS,
    CheckOverride,
    ConfigError,
    QCConfig,
    configure_results,
    configure_rules,
    load_config,
    parse_config,
)
from rideshare_data_pipeline.io import InputError


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "qc.toml"
    path.write_text(text, encoding="utf-8")
    return path


def result(**changes: object) -> CheckResult:
    base = CheckResult(
        name="x",
        check_id="X1",
        category="test",
        severity=Severity.ERROR,
        passed=False,
        metric=0.02,
        threshold=0.0,
        n_failed=2,
        n_evaluated=100,
        predicate="p",
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def test_defaults() -> None:
    config = QCConfig()
    assert (config.sample_limit, config.fail_on, dict(config.checks)) == (
        20,
        Severity.ERROR,
        {},
    )
    assert config.override("anything") == CheckOverride()


def test_load_config(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        'sample_limit = 5\nfail_on = "warn"\n'
        '[checks.non_negative_total]\nmax_failed_rate = 0.01\nseverity = "warn"\n'
        "[checks.duplicate_rows]\nenabled = false\n",
    )
    config = load_config(path)
    assert config.sample_limit == 5
    assert config.fail_on is Severity.WARN
    assert config.checks["non_negative_total"] == CheckOverride(
        0.01, Severity.WARN, True
    )
    assert config.checks["duplicate_rows"].enabled is False


def test_empty_file_is_the_defaults(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")) == QCConfig()


def test_shipped_example_is_valid_and_lists_every_check() -> None:
    example = Path(__file__).parent.parent / "qc.example.toml"
    config = load_config(example)
    assert tuple(config.checks) == KNOWN_CHECKS


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("sample_limt = 3", "unknown 'sample_limt'"),
        ("[checks.nope]\nenabled = false", "unknown 'nope'"),
        ("[checks.duplicate_rows]\nmax_rate = 1", "unknown 'max_rate'"),
        ("fail_on = 'info'", "fail_on: expected one of 'error', 'warn'"),
        ("[checks.duplicate_rows]\nseverity = 'fatal'", "severity: expected one of"),
        ("[checks.duplicate_rows]\nmax_failed_rate = 2", "number from 0 to 1"),
        ("[checks.duplicate_rows]\nmax_failed_rate = true", "number from 0 to 1"),
        ("[checks.duplicate_rows]\nenabled = 'no'", "enabled: expected true or false"),
        ("sample_limit = -1", "non-negative integer"),
        ("sample_limit = 'x'", "non-negative integer"),
        ("checks = 3", "expected a table"),
        ("[checks]\nduplicate_rows = 3", "expected a table"),
        ("not toml [", "can't read config"),
    ],
)
def test_invalid_config_raises(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text))


def test_config_error_is_an_input_error_and_missing_file(tmp_path: Path) -> None:
    assert issubclass(ConfigError, InputError)
    with pytest.raises(ConfigError, match=r"config not found: nope\.toml"):
        load_config(tmp_path / "nope.toml")


def test_parse_config_accepts_an_integer_rate() -> None:
    config = parse_config({"checks": {"checksum_unique": {"max_failed_rate": 1}}})
    assert config.checks["checksum_unique"].max_failed_rate == 1.0


def test_configure_rules_replaces_and_drops() -> None:
    config = QCConfig(
        checks={
            "non_negative_total": CheckOverride(0.5, Severity.WARN),
            "checksum_unique": CheckOverride(enabled=False),
        }
    )
    rules = {r.name: r for r in configure_rules(default_rules(), config)}
    assert list(rules) == ["non_negative_total"]
    assert rules["non_negative_total"].max_failed_rate == 0.5
    assert rules["non_negative_total"].severity is Severity.WARN
    # An untouched rule is returned as it was.
    untouched = configure_rules(default_rules(), QCConfig())
    assert [(r.name, r.max_failed_rate, r.severity) for r in untouched] == [
        (r.name, r.max_failed_rate, r.severity) for r in default_rules()
    ]


def test_configure_results_severity_threshold_and_enabled() -> None:
    config = QCConfig(
        checks={
            "x": CheckOverride(max_failed_rate=0.05, severity=Severity.WARN),
            "gone": CheckOverride(enabled=False),
        }
    )
    kept = configure_results([result(), result(name="gone"), result(name="y")], config)
    assert [r.name for r in kept] == ["x", "y"]
    x, y = kept
    assert (x.severity, x.threshold, x.passed) == (Severity.WARN, 0.05, True)
    assert y == result(name="y")  # untouched


def test_configure_results_can_make_a_pass_fail_and_handles_no_metric() -> None:
    tight = QCConfig(checks={"x": CheckOverride(max_failed_rate=0.01)})
    assert (
        configure_results([result(passed=True, threshold=0.5)], tight)[0].passed
        is False
    )
    no_metric = result(metric=None, passed=True)
    assert configure_results([no_metric], tight)[0].passed is True


def test_configure_results_severity_only_keeps_the_verdict() -> None:
    config = QCConfig(checks={"x": CheckOverride(severity=Severity.INFO)})
    out = configure_results([result()], config)[0]
    assert (out.severity, out.passed, out.threshold) == (Severity.INFO, False, 0.0)


def test_known_checks_match_the_registry(full_synthetic_frame: pl.DataFrame) -> None:
    from rideshare_data_pipeline.checks.structural import cast_failures_check

    names = [cast_failures_check({}).name]
    names += [r.name for r in run_checks(full_synthetic_frame.lazy())]
    assert tuple(names) == KNOWN_CHECKS
