# rideshare_data_pipeline

Quality-control and outlier checks for a consumer e-receipt panel sample covering two rideshare merchants (Uber and Lyft), built on [polars](https://pola.rs/).

## The problem

A data vendor parses the receipt emails that a panel of long-tenured, active consumers get from rideshare merchants, and delivers one row per receipt: who (a hashed panelist ID), when (trip, email and ingestion times), how much (totals, tax, discounts), what (a free-text product line), and vendor delivery metadata. There is no data dictionary, so the meaning of each field has to be inferred from the data itself. The goal is to design and run practical quality-control and outlier tests on that sample, and to state the assumptions behind them.

This first version validates the file's UTF-8 encoding and header, reads every column as text, casts the columns to their types while counting any value that doesn't convert, and reports structural problems: columns that are entirely null, columns that hold a single value, and exact duplicate rows. The [roadmap](#roadmap) lists the checks still to come.

## Data handling

- **Real data is never committed**: not the raw file, extracts of it, or anything derived from it. `.gitignore` blocks the usual data and archive formats everywhere except small synthetic fixtures under `tests/fixtures/`, and the tests build their synthetic files in a temporary directory.
- CI enforces this: `tools/check_no_data.py` fails the build if a tracked file is over 1 MB, if a data or archive file is tracked outside `tests/fixtures/`, or if a tracked CSV outside it has the input schema's header. Run `uv run python tools/check_no_data.py` locally, or use it as a pre-commit hook.
- Point the CLI at a file on your machine with `--input`. The opt-in real-data test reads the path in the `RIDESHARE_QC_DATA` environment variable.
- The CLI writes only to the `--out` directory you name. Archives are extracted to a temporary directory that is deleted when the run ends.

## Quick start

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.15 if you don't have it.

```sh
uv sync                                   # creates .venv with the package and the dev tools
uv run rideshare-qc --input /path/to/receipts.csv --out reports/
```

## CLI

```text
rideshare-qc --input PATH --out DIR [--drop-columns A,B]
```

- `--input PATH`: the CSV, or a `.tar.bz2`, `.tar.gz`, `.tar.xz`, `.tar` or `.zip` archive that holds exactly one CSV (macOS `._*` sidecar files are ignored).
- `--out DIR`: where to write `<input stem>.parquet` (zstd). The directory is created if it is missing.
- `--drop-columns A,B`: columns to leave out of the Parquet file. By default every column and every row is kept. The checks only report, and nothing is dropped unless you ask.

It prints a short summary, for example:

```text
rows:             3
columns:          33
all-null columns: 5 (order_points, order_shipping, SKU, item_id, file_id)
constant columns: 4 (order_total_qty, item_quantity, digital_transaction, product_category)
duplicate rows:   0
wrote:            reports/receipts.parquet (33 columns)

error  PASS  cast_failures                 0/20           0.0000%
error  PASS  non_negative_total            0/3            0.0000%
error  PASS  checksum_unique               0/3            0.0000%
info   PASS  all_null_columns              5/33          15.1515%
info   PASS  constant_columns              4/33          12.1212%
error  PASS  duplicate_rows                0/3            0.0000%
```

After the summary comes one line per check: severity, `PASS` or `FAIL`, name, failed/evaluated, and the failure rate. A row rule passes while its failure rate is at most its threshold (0 for the two registered rules: `non_negative_total`, `order_total_amount >= 0`, and `checksum_unique`); a null never counts as a failure. Info checks always pass. The check lines don't change the exit status yet.

A column is **all-null** when every value is null. It is **constant** when its non-null values have exactly one distinct value, so nulls are ignored. An all-null column is reported only as all-null. A file with only a header row has no rows to judge, so it reports neither.

| Exit status | Meaning |
|---:|---|
| 0 | Success. |
| 1 | Exact duplicate rows were found. The Parquet file is still written, with every row. |
| 2 | Input or parse error, invalid arguments, or the output can't be written. The reason is printed to stderr. Nothing is written when the input can't be read. |

Input errors include a missing or unsupported file, an archive that can't be extracted or doesn't hold exactly one CSV, an empty file, bytes that aren't valid UTF-8 (the message gives the byte offset), a header with missing, extra, duplicated or reordered columns (the message names them), and values that don't convert to their column's type (the message counts the failures per column). Values are never nulled silently. A UTF-8 BOM is accepted and stripped, and CRLF and LF files read identically; the summary reports both. Known gap: a row with too few fields is padded with nulls, not flagged. `python -m rideshare_data_pipeline` works the same way as `rideshare-qc`.

## Package layout

```text
src/rideshare_data_pipeline/
  __init__.py         __version__, read from the package metadata
  __main__.py         python -m rideshare_data_pipeline
  cli.py              the rideshare-qc command (argparse): summary, check lines, Parquet output, exit codes
  schema.py           the 33 expected columns, the raw all-String schema, the casts and their formats
  io.py               archive extraction (tar filter="data", zip), header and UTF-8 validation,
                      the raw scan, non-strict typing with per-column cast-failure
                      counts, CSV/Parquet writers
  checks/
    __init__.py       the registry (default_rules, default_dataset_checks) and run_checks:
                      every RowRule is counted in one lazy select, then the samples of
                      the failing rules are collected in one batch
    model.py          Severity, CheckResult, RowRule, DatasetCheck
    arithmetic.py     row rules: non_negative_total (A1)
    uniqueness.py     row rules: checksum_unique (U1)
    structural.py     all-null and constant columns (one lazy select), duplicate rows,
                      and their DatasetCheck wrappers
tests/
  conftest.py         shared fixtures
  synthetic.py        synthetic rows in the real column layout; tar/zip builders
  test_schema.py  test_io.py  test_checks_structural.py  test_checks_runner.py  test_cli.py
  test_real_data.py   opt-in (slow), runs only when RIDESHARE_QC_DATA is set
```

Importing the package doesn't import polars. The modules use PEP 810 `lazy import`, so `rideshare-qc --help` returns at once.

## Development

```sh
uv run pytest                    # tests (the real-data test is skipped)
RIDESHARE_QC_DATA=/path/to/receipts.csv uv run pytest -m slow   # opt-in real-data run
uv run ruff check .              # lint
uv run ruff format .             # format (CI runs: ruff format --check .)
uv run mypy .                    # type-check src/ and tests/ (strict)
uv build                         # sdist and wheel in dist/
```

## Roadmap

Planned work is tracked in the GitHub issues:

- Data hygiene and the input contract: guarding against committing data (#6), schema, header and encoding validation (#8), hardened extraction and a Parquet cache (#17).
- Check framework and reporting: the CheckResult model and runner (#9), the JSON and Markdown report with exit-code and threshold config (#11), and a synthetic fixture generator (#10).
- Receipt semantics: receipt kinds and subject tags (#12), duplicate receipts and the dedupe policy (#13), parsing distance and duration from product descriptions (#14), and sender country and currency (#15).
- Checks: structural and completeness (#16), arithmetic (#18), temporal (#19), fare outliers (#20), trip plausibility (#21), per-panelist behaviour (#26), volume and spend anomalies (#27), panel coverage (#28), and drift between delivery batches (#29).
- Outputs: a curated Parquet dataset with per-row QC flags (#22), privacy normalization (#30), and a stakeholder summary (#31).
- Tests and docs: edge-case inputs (#23), end-to-end and performance tests (#24), and the full README (#25).

## License

GNU General Public License v3.0 (`GPL-3.0-only`). See [LICENSE](LICENSE).
