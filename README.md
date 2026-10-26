# rideshare_data_pipeline

Quality-control and outlier checks for a consumer e-receipt panel sample covering two rideshare merchants (Uber and Lyft), built on [polars](https://pola.rs/).

## The problem

A data vendor parses the receipt emails that a panel of long-tenured, active consumers get from rideshare merchants, and delivers one row per receipt: who (a hashed panelist ID), when (trip, email and ingestion times), how much (totals, tax, discounts), what (a free-text product line), and vendor delivery metadata. There is no data dictionary, so the meaning of each field has to be inferred from the data itself. The goal is to design and run practical quality-control and outlier tests on that sample, and to state the assumptions behind them.

This first version validates the file's UTF-8 encoding and header, reads every column as text, casts the columns to their types while counting any value that doesn't convert, and reports structural problems: columns that are entirely null, columns that hold a single value, and exact duplicate rows. It then classifies every receipt by kind (a ride, a tip update, a driver payout, a pass, and so on; see [Receipt kinds](#receipt-kinds)). The [roadmap](#roadmap) lists the checks still to come.

## Data handling

- **Real data is never committed**: not the raw file, extracts of it, or anything derived from it. `.gitignore` blocks the usual data and archive formats everywhere except small synthetic fixtures under `tests/fixtures/`, and the tests build their synthetic files in a temporary directory.
- CI enforces this: `tools/check_no_data.py` fails the build if a tracked file is over 1 MB, if a data or archive file is tracked outside `tests/fixtures/`, or if a tracked CSV outside it has the input schema's header. Run `uv run python tools/check_no_data.py` locally, or use it as a pre-commit hook.
- Point the CLI at a file on your machine with `--input`. The opt-in real-data test reads the path in the `RIDESHARE_QC_DATA` environment variable.
- The CLI writes only to the `--out` directory you name. Archives are extracted to a temporary directory that is deleted when the run ends.
- Reports contain the input's base name and SHA-256, aggregate counts, and failing-row samples limited to the columns each rule chooses (hashed `user_id`, order number, times, amounts, receipt kind; no free-text subjects, and never the sender address, whose local part can be a person's). The one place subject text appears is the `unclassified_subject_templates` check, which lists at most 20 normalised templates (masks, digits, dates and weekdays replaced) of subjects no rule recognised. They still describe real data: write them to a directory outside the repository (`.gitignore` blocks `/reports/`), and share them like the data.

## Quick start

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.15 if you don't have it.

```sh
uv sync                                   # creates .venv with the package and the dev tools
uv run rideshare-qc run --input /path/to/receipts.csv --out ~/qc-reports/receipts
```

## CLI

```text
rideshare-qc run --input PATH --out DIR [--config FILE] [--fail-on {error,warn}] [--drop-columns A,B]
rideshare-qc --help | --version
```

- `--input PATH`: the CSV, or a `.tar.bz2`, `.tar.gz`, `.tar.xz`, `.tar` or `.zip` archive that holds exactly one CSV (macOS `._*` sidecar files are ignored).
- `--out DIR`: where everything is written (created if missing): `report.json`, `report.md`, `samples/<check>.parquet` (one per failed check that has sample rows), and `<input stem>.parquet` (zstd; the curated Parquet with per-row flags is planned in #22).
- `--config FILE`: a TOML file with thresholds, severities and on/off switches (see [Configuration](#configuration)).
- `--fail-on {error,warn}`: the lowest severity whose failure exits 1. Overrides the config's `fail_on`; the default is `error`.
- `--drop-columns A,B`: columns to leave out of the Parquet file. By default every column and every row is kept. The checks only report, and nothing is dropped unless you ask.

It prints a short summary, one line per check, and the paths written, for example:

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

errors failed: 0  warnings failed: 0  passed: 6/6  fail-on: error  exit: 0
wrote: reports/report.json
wrote: reports/report.md
```

A row rule passes while its failure rate is at most its threshold (`max_failed_rate`, 0 by default); a null never counts as a failure. Info checks always pass.

## Receipt kinds

The file mixes receipt types that are not rider spend, all tagged with the category `Travel`. `parse.py` derives, from the subject template, the description line, `product_reseller`, `UF-` order numbers and `order_pickup`, one of these kinds per row: `ride`, `tip_update`, `fare_correction`, `cancel_noshow`, `payment_failed`, `driver_earnings`, `pass_or_credit`, `micromobility`, `food_order`, `freight`, `rental`. The rules form an ordered chain, most specific first (see `_rules` in `parse.py`); a row no rule matches defaults to `ride` and is counted by the `unclassified_subject_templates` check. Mail-gateway prefixes (`(EXTERNAL SENDER)`, `{Disarmed}`) and leading bracket tags are stripped before matching, and a leading `[Business]` or `[Personal]` tag becomes `subject_profile`.

**Rider spend.** `is_rider_spend` is true for `ride`, `tip_update`, `fare_correction` and `cancel_noshow`. Other kinds stay in the data, flagged, and are left out of spend. A cancellation fee counts as spend under its own kind. A tip update or a fare correction supersedes the receipt of the ride it amends; the dedupe flags that original (see Duplicates and dedupe). Currency gating narrows this to USD spend (see [Currency](#currency-and-the-usd-gate)).

The derived columns (`receipt_kind`, `subject_profile`, `subject_unclassified`, `is_rider_spend`, `sender_country`, `currency_reason`, `currency_suspect`, and the trip columns) are computed once per run for the checks. They are not in the output Parquet yet; the curated dataset with flags is #22.

Checks added: `receipt_kinds` (K1, info: the kind counts and the merchant x kind matrix, shown in the report), `category_mismatch` (V2, warn: category `Travel` on a food order, freight, rental or driver earnings) and `unclassified_subject_templates` (K2, info). `non_negative_total` (A1) now applies to rider-spend receipts only: a refund on a bike or pass receipt isn't a rider-spend error.

A column is **all-null** when every value is null. It is **constant** when its non-null values have exactly one distinct value, so nulls are ignored. An all-null column is reported only as all-null. A file with only a header row has no rows to judge, so it reports neither.

| Exit status | Meaning |
|---:|---|
| 0 | Every check that gates passed. |
| 1 | An **error**-severity check failed (with `--fail-on warn`, a failed **warn** check too). Info checks never gate. `report.json`, `report.md`, the samples and the Parquet are still written. |
| 2 | Input or parse error, a bad config file (unknown key or check name, bad value), invalid arguments, or the output can't be written. The reason is printed to stderr. **Nothing is written** (no report either) when the input or config can't be read. |

Input errors include a missing or unsupported file, an archive that can't be extracted or doesn't hold exactly one CSV, an empty file, bytes that aren't valid UTF-8 (the message gives the byte offset), a header with missing, extra, duplicated or reordered columns (the message names them), and values that don't convert to their column's type (the message counts the failures per column). Values are never nulled silently. A UTF-8 BOM is accepted and stripped, and CRLF and LF files read identically; the summary reports both. Known gap: a row with too few fields is padded with nulls, not flagged. `python -m rideshare_data_pipeline run ...` works the same way as `rideshare-qc run ...`. The old flat invocation (`rideshare-qc --input ...`) is gone.

## Duplicates and dedupe

The legacy whole-row duplicate check could never fire: `checksum` is unique per row and the ingestion metadata differs between versions of a receipt. The vendor upserts updated receipts and the same receipt can reach two panelists, so duplicates are found on the order key `(merchant_name, order_number)`, over rows where both are non-null (a null is not a key: recurring pass charges have no order number and are legitimate).

**Dedupe policy** (`dedupe.py`). `supersede_flags` adds a boolean `is_superseded`; **rows are never dropped**, consumers filter on the flag (`deduplicated` is that view, used for spend metrics from #15 on). Within a key group one row wins and the rest are superseded. The winner is the first of: a `tip_update` or `fare_correction` (kind priority beats time, so a tip update supersedes its ride even when the ride's email is later); the latest `email_time`; the latest `update_time`; the greatest `checksum`. Rows without a key are never superseded, and dedupe is idempotent. The flag is computed once per run for the checks and is not in the output Parquet (the curated dataset with flags is #22).

| Check | Severity | Finds |
|---|---|---|
| `checksum_unique` (U1) | error | a repeated `checksum` |
| `duplicate_receipts` (U2) | warn | rows in repeated keys; details: groups, excess (superseded) rows, excess spend, breakdown by kind pair; shown in the report's "Duplicate receipts" section |
| `dedupe_key_unique` (U2b) | error | a repeated key among rows the dedupe kept; holds by construction and guards against a dedupe bug |
| `cross_panelist_duplicates` (U3) | warn | one key under more than one `user_id` |
| `near_duplicate_rides` (U4) | warn | rides identical on user, merchant, order date, total and description but with different or null order numbers (excess rows); flagged only, never deduped |

U2 is a warning because a repeated key is expected vendor behaviour (re-sent receipts, a receipt in two mailboxes) that the dedupe resolves; an error would make every real run exit 1. Excess spend is USD rider spend only; the currency-suspect part is reported apart as `excess_spend_suspect`.

## Currency and the USD gate

The file has no currency column, yet receipts from non-US merchant senders state amounts in local currency: about 0.5% of rows are about 37% of a naive spend total. `from_domain` is the **full sender address**, and it carries the evidence. `sender.py` derives `sender_country` from public receipt-sender patterns (`uber.<country>@` to an ISO code, `receipts.<city>@` to its country with US as the default city; generic senders such as `noreply@` give null; an unknown `uber.<token>@` is `unknown`, treated as non-US). `currency_reason` is the first of `non_us_sender`, `km_unit` (the description states kilometres) and `foreign_language` (a Spanish or Portuguese subject: `Tu ... del ...`, `por la mañana`, `Sua viagem`, `Error de pago`). `currency_suspect` is true where there is a reason. Rows are flagged, never dropped, and no exchange rate is applied (FX conversion is future work).

**`usd_rides(frame)`** (`money.py`) is THE population for every money statistic and outlier baseline: rider spend, not superseded, not currency-suspect, with a non-null `order_total_amount`. A later check that sums or ranks amounts starts from it.

| Check | Severity | Finds |
|---|---|---|
| `currency_suspect_rows` (CUR1) | warn | currency-suspect rows; details: counts and naive sums by reason and country (local currency, not summable as USD), the spend waterfall. A warning because the gate is the `usd_rides` exclusion; an error would fail every real run for an expected property of the data |
| `merchant_sender_consistent` (V1) | error | an Uber receipt not from an Uber domain, or a Lyft receipt not from lyft.com, lyftmail.com or a subdomain |
| `unknown_sender` (V3) | warn | a sender matching no known receipt-sender pattern (for example addresses of individual people on freight receipts; their samples never show the address) |

`report.md` has a "Spend" section, and `report.json` an optional `spend` object, with the waterfall: the naive total of `order_total_amount`, minus currency-suspect rows, minus non-rider-spend kinds, minus superseded duplicates, equals the clean USD rider spend, each with its row count.

## Configuration

Defaults live in code (`QCConfig` in `config.py`); a TOML file overrides them. [`qc.example.toml`](qc.example.toml) lists every check with its default:

```toml
sample_limit = 20          # failing rows kept per check
fail_on = "error"          # "error" or "warn"; --fail-on overrides it

[checks.non_negative_total]
max_failed_rate = 0.01     # passes while failed / evaluated <= 0.01
severity = "warn"          # "error", "warn" or "info"
enabled = true             # false removes the check from the run and the report
```

Unknown keys and unknown check names are errors (exit 2), so a typo can't silently do nothing. The effective thresholds and severities are recorded in the report. A `max_failed_rate` on a dataset check (for example `duplicate_rows`) replaces its threshold and re-judges it against its measured rate.

## Report

`report.md` has a title with the input name and the first 12 characters of its SHA-256, a one-line run summary (rows, panelists, window), the counts, a table of every check (sorted by severity, failures first) with its failed count, rate, threshold and rule, a column profile, and the failing-row samples (at most `sample_limit` rows per check, linked to the Parquet file). The "Composition by receipt kind" section is a merchant x kind table with a total column and a rider-spend marker per kind.

`report.json` is documented by the JSON Schema (draft 2020-12) in [`src/rideshare_data_pipeline/report.schema.json`](src/rideshare_data_pipeline/report.schema.json), which a test validates a real run against:

```text
{"schema_version": 1,
 "run": {tool_version, input_name (base name), input_sha256, input_size_bytes, bom, line_ending,
         rows, panelists (distinct user_id), window {start, end} (email_time, ISO), started, finished,
         elapsed_seconds, config {fail_on, sample_limit, checks {<name>: {threshold, severity}}}},
 "summary": {errors_failed, warnings_failed, info, passed, total},
 "composition": {rows, merchants, kinds: [{kind, rider_spend, total, by_merchant}]}  (optional),
 "spend": {naive_total, currency_suspect, non_rider_spend, superseded, clean_usd_rider_spend,
           suspect_rider_spend: {rows, amount}, note}  (optional),
 "checks": [{name, check_id, category, severity, passed, metric, threshold, n_failed, n_evaluated,
             predicate, details, sample_file (relative path or null)}],
 "exit_code": 0 | 1}
```

All numbers are finite, so the file is strict JSON.

## Description parsing

`product_description` states the product, distance and duration of a trip in several templates; `parse.py` reads them into four derived columns, each null on a line that is not a trip (`Increase Tip`, `Earnings`, `Subscription`, a null) and never 0 there. A trip that states zero is 0.

| Column | Type | Meaning |
|---|---|---|
| `distance_mi` | Float64 | distance in miles; kilometres are converted (x 0.621371) |
| `distance_unit` | Enum `mi`, `km` | the unit the description states (`miles` is `mi`; `kilometers` and `kilometres` are `km`) |
| `duration_s` | Int64 | duration in seconds |
| `product_family` | Enum | `uberx`, `lyft_standard`, `shared` (pool, shared, line), `express_pool`, `xl`, `black_lux` (black, lux, SUV, premier), `comfort` (select, plus, comfort), `vip`, `other` |

Templates: Lyft `Lyft fare (1.63mi, 12m 23s)` (also `45s` and `1h 2m 3s`) and `Base fare ... 7.06 mi` lines; Uber `UberX 9.93 mi | 32 min` (also `| 1 h 22 min`, `km`); older Uber `2.00 miles   00:15:14 Trip time   uberX Car`. The family comes from the product label, first match in this order: express, shared, vip, black/lux, xl, comfort, uberx, lyft; a trip with none (a Lyft `Base fare` line) is `other`. Like the other derived columns they are computed once per run and are not in the output Parquet (#22).

`description_parse_coverage` (PC1, warn) takes the ride and tip-update rows that are not superseded and passes while every merchant parses at least 98% of them (distance and duration both read). Its metric is the worst merchant's unparsed rate, and its details hold per-merchant `rows`, `parsed` and `coverage`, the unit counts, the family counts by merchant and the ten most common unparsed shapes (digits shown as `#`, never the text). The report's "Description parsing" section shows them.

## Package layout

```text
src/rideshare_data_pipeline/
  __init__.py         __version__, read from the package metadata
  __main__.py         python -m rideshare_data_pipeline
  cli.py              the rideshare-qc command (argparse, `run` subcommand): summary, check lines, reports, Parquet output, exit codes
  config.py           QCConfig and CheckOverride, TOML loading (tomllib), applying overrides to rules and results
  dedupe.py           the dedupe policy: supersede_flags, deduplicated
  parse.py            derived columns as pl.Expr factories: receipt kind, subject profile, rider-spend flag, currency reason and flag, distance, duration, unit, product family
  sender.py           sender_country and the known-sender patterns (from_domain is the full sender address)
  money.py            usd_rides (the population of every money statistic) and the spend waterfall
  report.py           report.json and report.md rendering, samples/<check>.parquet, exit-code policy
  report.schema.json  JSON Schema for report.json (package data)
  schema.py           the 33 expected columns, the raw all-String schema, the casts and their formats
  io.py               archive extraction (tar filter="data", zip), header and UTF-8 validation,
                      the raw scan (UTF-8 validation, SHA-256, line endings in one pass), non-strict typing with per-column cast-failure
                      counts, CSV/Parquet writers
  checks/
    __init__.py       the registry (default_rules, default_dataset_checks) and run_checks:
                      every RowRule is counted in one lazy select, then the samples of
                      the failing rules are collected in one batch
    model.py          Severity, CheckResult, RowRule, DatasetCheck
    arithmetic.py     row rules: non_negative_total (A1)
    uniqueness.py     checksum_unique (U1), duplicate_receipts (U2), dedupe_key_unique (U2b), cross_panelist_duplicates (U3), near_duplicate_rides (U4)
    kinds.py          category_mismatch (V2), receipt_kinds (K1) and unclassified_subject_templates
    parsing.py        description_parse_coverage (PC1)
    currency.py       currency_suspect_rows (CUR1), merchant_sender_consistent (V1), unknown_sender (V3)
    structural.py     all-null and constant columns (one lazy select), duplicate rows,
                      and their DatasetCheck wrappers
tests/
  conftest.py         shared fixtures
  synthetic.py        synthetic rows in the real column layout; tar/zip builders
  test_schema.py  test_io.py  test_checks_structural.py  test_checks_runner.py  test_cli.py
  test_config.py  test_report.py  test_parse.py  test_trip_parse.py  test_currency.py
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
- Check framework and reporting: the CheckResult model and runner (#9), a synthetic fixture generator (#10), and the JSON and Markdown report with exit codes and TOML thresholds (#11, done).
- Receipt semantics: receipt kinds and subject tags (#12, done), duplicate receipts and the dedupe policy (#13), parsing distance and duration from product descriptions (#14), and sender country and currency (#15, done).
- Checks: structural and completeness (#16), arithmetic (#18), temporal (#19), fare outliers (#20), trip plausibility (#21), per-panelist behaviour (#26), volume and spend anomalies (#27), panel coverage (#28), and drift between delivery batches (#29).
- Outputs: a curated Parquet dataset with per-row QC flags (#22), privacy normalization (#30), and a stakeholder summary (#31).
- Tests and docs: edge-case inputs (#23), end-to-end and performance tests (#24), and the full README (#25).

## License

GNU General Public License v3.0 (`GPL-3.0-only`). See [LICENSE](LICENSE).
