"""CheckResult, RowRule and the single-pass runner (issue #9)."""

import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from synthetic import read_typed_frame, synthetic_rows, write_csv_text

from rideshare_data_pipeline.checks import (
    default_dataset_checks,
    default_rules,
    run_checks,
)
from rideshare_data_pipeline.checks.model import KEY_COLS, RowRule, Severity
from rideshare_data_pipeline.checks.structural import (
    all_null_columns_check,
    cast_failures_check,
    constant_columns_check,
    duplicate_rows_check,
)
from rideshare_data_pipeline.parse import derived_columns


def lazy(data: dict[str, list[Any]]) -> pl.LazyFrame:
    return pl.LazyFrame(data)


def rule(
    predicate: pl.Expr, *, applies_to: pl.Expr | None = None, rate: float = 0.0
) -> RowRule:
    return RowRule(
        name="r",
        check_id="T1",
        category="test",
        severity=Severity.ERROR,
        rule="x >= 0",
        predicate=predicate,
        applies_to=applies_to,
        max_failed_rate=rate,
        sample_cols=("x",),
    )


def synthetic_frame(tmp_path: Path, n: int) -> pl.LazyFrame:
    path = write_csv_text(tmp_path / "s.csv", synthetic_rows(n))
    return read_typed_frame(path).with_columns(derived_columns()).lazy()


def one(r: RowRule, lf: pl.LazyFrame, **kw: Any) -> Any:
    (res,) = run_checks(lf, rules=[r], dataset_checks=[], **kw)
    return res


def test_rule_passes_and_fails_on_three_rows() -> None:
    ok = one(rule(pl.col("x") < 0), lazy({"x": [1, 2, 3]}))
    assert ok.passed
    assert (ok.n_failed, ok.n_evaluated, ok.metric) == (0, 3, 0.0)
    assert ok.sample is None

    bad = one(rule(pl.col("x") < 0), lazy({"x": [1, -2, 3]}))
    assert not bad.passed
    assert (bad.n_failed, bad.n_evaluated) == (1, 3)
    assert bad.metric == pytest.approx(1 / 3)
    assert bad.sample is not None
    assert bad.sample["x"].to_list() == [-2]


def test_result_carries_severity_threshold_metric_and_rule_text() -> None:
    res = one(rule(pl.col("x") < 0, rate=0.5), lazy({"x": [1, -2, 3]}))
    assert res.severity is Severity.ERROR
    assert res.threshold == 0.5
    assert res.metric is not None
    assert res.passed  # 1/3 <= 0.5
    assert res.predicate == "x >= 0"
    assert (res.name, res.check_id, res.category) == ("r", "T1", "test")


def test_applies_to_restricts_the_denominator() -> None:
    lf = lazy({"x": [-1, -2, 3, 4], "kind": ["a", "a", "b", "b"]})
    res = one(rule(pl.col("x") < 0, applies_to=pl.col("kind") == "b"), lf)
    assert (res.n_failed, res.n_evaluated, res.passed) == (0, 2, True)
    res = one(rule(pl.col("x") < 0, applies_to=pl.col("kind") == "a"), lf)
    assert (res.n_failed, res.n_evaluated, res.passed) == (2, 2, False)
    assert res.metric == 1.0


def test_null_predicate_is_not_a_failure() -> None:
    res = one(rule(pl.col("x") < 0), lazy({"x": [None, -1, 5]}))
    assert (res.n_failed, res.n_evaluated) == (1, 3)


def test_no_applicable_rows_passes_without_metric() -> None:
    res = one(rule(pl.col("x") < 0, applies_to=pl.lit(False)), lazy({"x": [-1]}))
    assert res.passed
    assert (res.metric, res.n_failed, res.n_evaluated) == (None, 0, 0)
    assert res.sample is None
    assert res.threshold == 0.0
    empty = one(rule(pl.col("x") < 0), pl.LazyFrame({"x": []}, schema={"x": pl.Int64}))
    assert empty.passed
    assert empty.metric is None


def test_sample_is_capped_and_limited_to_present_columns() -> None:
    r = dataclasses.replace(rule(pl.col("x") < 0), sample_cols=("x", "not_a_column"))
    lf = lazy({"x": list(range(-50, 0)), "y": list(range(50))})
    res = one(r, lf)
    assert res.sample is not None
    assert res.sample.height == 20
    assert res.sample.columns == ["x"]
    small = one(r, lf, sample_limit=3)
    assert small.sample is not None
    assert small.sample.height == 3


def test_default_sample_columns_are_the_key_columns() -> None:
    assert rule(pl.col("x") < 0).sample_cols == ("x",)
    assert RowRule("n", "T", "c", Severity.WARN, "r", pl.lit(False)).sample_cols == (
        KEY_COLS
    )


class _Counter:
    """Counts LazyFrame.collect and pl.collect_all calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.collects = 0
        self.batches = 0
        collect = pl.LazyFrame.collect
        collect_all = pl.collect_all
        counter = self

        def counting_collect(self: pl.LazyFrame, *a: Any, **k: Any) -> Any:
            counter.collects += 1
            return collect(self, *a, **k)

        def counting_collect_all(*a: Any, **k: Any) -> Any:
            counter.batches += 1
            return collect_all(*a, **k)

        monkeypatch.setattr(pl.LazyFrame, "collect", counting_collect)
        monkeypatch.setattr(pl, "collect_all", counting_collect_all)


def test_all_rules_share_one_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    rules = [rule(pl.col("x") < 0, rate=0.0) for _ in range(5)]
    counter = _Counter(monkeypatch)
    results = run_checks(lazy({"x": [1, 2, 3]}), rules=rules, dataset_checks=[])
    assert len(results) == 5
    assert (counter.collects, counter.batches) == (1, 0)


def test_samples_for_all_failing_rules_add_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = [rule(pl.col("x") < 0), rule(pl.col("x") < 1), rule(pl.col("x") > 99)]
    counter = _Counter(monkeypatch)
    results = run_checks(lazy({"x": [-1, 0, 3]}), rules=rules, dataset_checks=[])
    assert [r.passed for r in results] == [False, False, True]
    assert (counter.collects, counter.batches) == (1, 1)
    assert [r.sample is not None for r in results] == [True, True, False]


def test_registry_rules() -> None:
    rules = {r.name: r for r in default_rules()}
    assert rules["non_negative_total"].check_id == "A1"
    assert rules["checksum_unique"].check_id == "U1"
    assert rules["category_mismatch"].check_id == "V2"
    assert {n: r.severity for n, r in rules.items()} == {
        "non_negative_total": Severity.ERROR,
        "checksum_unique": Severity.ERROR,
        "category_mismatch": Severity.WARN,
    }
    assert len(default_dataset_checks()) == 5


def test_registered_rules_on_synthetic_data(tmp_path: Path) -> None:
    lf = synthetic_frame(tmp_path, 5)
    assert all(r.passed for r in run_checks(lf, dataset_checks=[]))

    # The synthetic rows carry a reseller, so they are food orders; A1 only looks at
    # rider-spend rows, so mark them as such.
    bad = lf.with_columns(
        is_rider_spend=pl.lit(True),
        order_total_amount=pl.when(pl.col("order_number").is_first_distinct())
        .then(pl.lit(-1.0))
        .otherwise(pl.col("order_total_amount")),
        checksum=pl.lit("same"),
    )
    by_name = {r.name: r for r in run_checks(bad, dataset_checks=[])}
    assert by_name["non_negative_total"].n_failed == 5
    assert by_name["checksum_unique"].n_failed == 5
    sample = by_name["checksum_unique"].sample
    assert sample is not None
    assert sample.columns == list(KEY_COLS)


def test_null_checksums_are_not_duplicates() -> None:
    lf = lazy({"checksum": [None, None, "a", "b"]})
    res = one(next(r for r in default_rules() if r.name == "checksum_unique"), lf)
    assert (res.n_failed, res.n_evaluated, res.passed) == (0, 4, True)


def test_structural_dataset_checks(tmp_path: Path) -> None:
    lf = synthetic_frame(tmp_path, 4)
    s4 = all_null_columns_check(lf)
    assert s4.severity is Severity.INFO
    assert s4.passed
    assert s4.details["columns"] == [
        "order_points",
        "order_shipping",
        "SKU",
        "item_id",
        "file_id",
    ]
    assert s4.n_failed == 5
    assert constant_columns_check(lf).n_failed == 4
    assert duplicate_rows_check(lf).passed
    dup = duplicate_rows_check(pl.concat([lf, lf.head(1)]))
    assert (dup.passed, dup.n_failed, dup.n_evaluated) == (False, 1, 5)


def test_cast_failures_check() -> None:
    assert cast_failures_check({"a": 0, "b": 0}).passed
    bad = cast_failures_check({"a": 0, "b": 7})
    assert not bad.passed
    assert bad.details["failures"] == {"b": 7}
    assert bad.threshold == 0.0


def test_dataset_check_results_are_included_in_order() -> None:
    lf = lazy({"x": [1]})
    marker = all_null_columns_check
    res = run_checks(lf, rules=[rule(pl.col("x") < 0)], dataset_checks=[marker])
    assert [r.name for r in res] == ["r", "all_null_columns"]


def test_importing_checks_does_not_load_polars() -> None:
    code = (
        "import sys\n"
        "import rideshare_data_pipeline.checks, rideshare_data_pipeline.checks.model\n"
        "sys.exit('polars' in sys.modules)\n"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
