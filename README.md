# rideshare_data_pipeline

Quality-control and outlier checks for a consumer e-receipt panel sample covering two rideshare merchants (Uber and Lyft), built on [polars](https://pola.rs/).

## The problem

A data vendor parses the receipt emails that a panel of long-tenured, active consumers get from rideshare merchants, and delivers one row per receipt: who (a hashed panelist ID), when (trip, email and ingestion times), how much (totals, tax, discounts), what (a free-text product line), and vendor delivery metadata. There is no data dictionary, so the meaning of each field has to be inferred from the data itself. The goal is to design and run practical quality-control and outlier tests on that sample, and to state the assumptions behind them.

This first version validates the file's UTF-8 encoding and header, reads every column as text, casts the columns to their types while counting any value that doesn't convert, and reports structural and completeness problems (see [Structural and completeness checks](#structural-and-completeness-checks)): columns that are entirely null or constant, exact duplicate rows, null required fields, malformed IDs, incomplete rides and an out-of-band row count. It then classifies every receipt by kind (a ride, a tip update, a driver payout, a pass, and so on; see [Receipt kinds](#receipt-kinds)). The [roadmap](#roadmap) lists the checks still to come.

## Data handling

- **Real data is never committed**: not the raw file, extracts of it, or anything derived from it. `.gitignore` blocks the usual data and archive formats everywhere except small synthetic fixtures under `tests/fixtures/`, and the tests build their synthetic files in a temporary directory.
- CI enforces this: `tools/check_no_data.py` fails the build if a tracked file is over 1 MB, if a data or archive file is tracked outside `tests/fixtures/`, or if a tracked CSV outside it has the input schema's header. Run `uv run python tools/check_no_data.py` locally, or use it as a pre-commit hook.
- Point the CLI at a file on your machine with `--input`. The opt-in real-data test reads the path in the `RIDESHARE_QC_DATA` environment variable.
- The CLI writes only to the `--out` directory you name, plus the cache directory if (and only if) you opt in with `--cache-dir` or `RIDESHARE_QC_CACHE`. Archives are extracted to a temporary directory that is deleted when the run ends. The cache holds a typed copy of the real data as Parquet, so keep it outside the repository (`/.cache/` is gitignored, but a directory elsewhere is safer); see [Cache](#cache).
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
                  [--cache-dir DIR] [--refresh-cache]
rideshare-qc --help | --version
```

- `--input PATH`: the CSV, or a `.tar.bz2`, `.tar.gz`, `.tar.xz`, `.tar` or `.zip` archive that holds exactly one CSV (macOS `._*` sidecar files are ignored; see [Archives](#archives)).
- `--out DIR`: where everything is written (created if missing): `report.json`, `report.md`, `samples/<check>.parquet` (one per failed check that has sample rows), and `<input stem>.parquet` (zstd; the curated Parquet with per-row flags is planned in #22).
- `--config FILE`: a TOML file with thresholds, severities and on/off switches (see [Configuration](#configuration)).
- `--fail-on {error,warn}`: the lowest severity whose failure exits 1. Overrides the config's `fail_on`; the default is `error`.
- `--drop-columns A,B`: columns to leave out of the Parquet file. By default every column and every row is kept. The checks only report, and nothing is dropped unless you ask.
- `--cache-dir DIR`: keep (and reuse) a typed Parquet copy of the input in DIR. Also read from the environment variable `RIDESHARE_QC_CACHE` when the flag is absent; with neither, no cache is used. See [Cache](#cache).
- `--refresh-cache`: rebuild the cache entry for this input even if a valid one exists (needs a cache directory).

It prints a short summary, one line per check, and the paths written, for example:

```text
rows:             3
columns:          33
all-null columns: 5 (order_points, order_shipping, SKU, item_id, file_id)
constant columns: 4 (order_total_qty, item_quantity, digital_transaction, product_category)
duplicate rows:   0
wrote:            reports/receipts.parquet (33 columns)

error  PASS  header_valid                  0/1            0.0000%
error  PASS  utf8_valid                    0/1            0.0000%
error  PASS  cast_failures                 0/20           0.0000%
warn   PASS  row_count_band                0/1               3 rows
error  PASS  required_fields               0/3            0.0000%
warn   PASS  id_formats                    0/3            0.0000%
error  PASS  non_negative_total            0/3            0.0000%
error  PASS  checksum_unique               0/3            0.0000%
info   PASS  all_null_columns              5/33          15.1515%
info   PASS  constant_columns              4/33          12.1212%
error  PASS  duplicate_rows                0/3            0.0000%
warn   PASS  ride_completeness             0/3            0.0000%

errors failed: 0  warnings failed: 0  passed: 11/11  fail-on: error  exit: 0
wrote: reports/report.json
wrote: reports/report.md
```

A row rule passes while its failure rate is at most its threshold (`max_failed_rate`, 0 by default); a null never counts as a failure. Info checks always pass.

## Archives

An archive (`.tar`, `.tar.gz`, `.tar.bz2`, `.tar.xz` or `.zip`) must hold exactly one CSV. Only that member is extracted, streamed to `data.csv` in a temporary directory that is deleted when the run ends, so no member name ever reaches the file system and nothing else in the archive is written.

- Directories and macOS AppleDouble sidecars (`._*`, also under `__MACOSX/`) are ignored.
- No CSV, or more than one, raises `ArchiveError` (exit 2), and the message lists the members (at most 20, then "and N more").
- The CSV member is rejected if it is a symlink, a hardlink or any non-regular file, or if its name is absolute or has a `..` component.
- A tar is read in one sequential pass (a listing of a compressed tar means decompressing it, and seeking back would decompress it again).

## Cache

Decompressing the bz2 sample takes about 20 s per run. With a cache directory, the first run on an input saves the validated, typed frame as Parquet and later runs read that instead. It is opt-in: without `--cache-dir` (or `RIDESHARE_QC_CACHE`) nothing is cached and nothing is written but `--out`. The flag wins over the variable; an empty variable means no cache.

```text
<cache>/<sha256 of the input file as given>/data.parquet   the typed frame (zstd)
<cache>/<sha256 of the input file as given>/meta.json      format version, tool version, schema fingerprint,
                                                           input name/size/mtime_ns/sha256, the CSV's size,
                                                           sha256, BOM and line endings, row count, created
<cache>/index.json                                         resolved path -> size, mtime_ns, sha256
```

- **Validity**: an entry is used only if its format version and schema fingerprint match. The fingerprint hashes the 33 column names, the dtypes and the timestamp formats, so a change to the schema invalidates old entries and a release that leaves it alone does not (the tool version is recorded but is not part of the key). Anything else wrong with an entry (bad JSON, unreadable Parquet, a row count that disagrees) is a warning on stderr and a rebuild.
- **Fast lookup**: the input is statted and looked up in `index.json`. If the path, size and mtime match and the entry is valid, the input is not read at all. Otherwise the input (the archive, or the CSV) is hashed, which still finds an entry for a touched or copied file with the same bytes; failing that the run takes the normal path (extract, validate, read, cast) and writes the entry. The trade-off: an in-place edit that keeps both the size and the mtime would be missed. That is very unlikely; use `--refresh-cache` to rebuild regardless.
- **Only validated data is cached**: a run that fails with an input error (bad header, encoding or casts) writes no entry. Writes are atomic (a temporary directory, then a rename), so a crash or a concurrent run never leaves a half-written entry. A cache directory that can't be written is a warning; caching never fails a run.
- **Same results**: the report is the same as an uncached run's apart from the timings and `run.cache` (`"hit"`, `"miss"` or `"off"`).
- The same data given as a CSV and as an archive gives two entries (the key is the file as given).
- The cache holds real data, not a derived summary: keep it outside the repository and delete it when you are done.

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

## Structural and completeness checks

S1, S2 and S3 guard the input contract: a bad header, invalid UTF-8 or a value that doesn't cast is an input error (exit 2, nothing written), so on a run that produced a report they are recorded as passing error results (S1 `details`: the column count and `in_order`; S3: `has_bom` and `line_ending`). The rest report and never fix.

| Check | Severity | Rule (threshold) |
|---|---|---|
| `header_valid` (S1) | error | the 33 expected columns, in order |
| `utf8_valid` (S3) | error | the file is valid UTF-8; BOM and line endings recorded |
| `cast_failures` (S2) | error | every non-null raw value converts to its column's type |
| `row_count_band` | warn | `min_rows` <= rows <= `max_rows` (defaults: 1, no limit), so a header-only file fails; the metric is the row count |
| `required_fields` (C1) | error | `merchant_name`, `user_id`, `email_time`, `checksum`, `from_domain`, `email_subject` are never null (0); `details.by_field` has the null count per column |
| `id_formats` (S6) | warn | `user_id` is 64 lowercase hex digits, `checksum` 32, a Lyft order number 18-19 digits, an Uber one a lowercase UUID (0); nulls are not judged (C1 and C2 own them); `details.by_field` counts each. Freight `UF-` order numbers fail it |
| `ride_completeness` (C2) | warn | among rides that are not superseded, the null rate of `order_total_amount` is at most 0.01%, and of `order_number` and `order_time` at most 0.5% each. `n_failed` is the rides missing any of the three; `metric` and `threshold` are those of the field nearest its limit, `details` has all three |
| `all_null_columns` (S4), `constant_columns` (S5) | info | columns null in every row, or with one distinct non-null value; one shared aggregation |

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
min_rows = 1               # row_count_band: fewest rows (0 = no lower bound)
# max_rows = 5000000       # row_count_band: most rows (default: no limit)

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
         elapsed_seconds, cache (hit|miss|off), config {fail_on, sample_limit, checks {<name>: {threshold, severity}}}},
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
  cache.py            the opt-in Parquet cache: load_typed, index, atomic writes, schema fingerprint
  io.py               extract_member (exactly one safe CSV from a tar or zip), header and UTF-8 validation,
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
    structural.py     S1/S3 results from the file facts, row_count_band, C1 required_fields and
                      S6 id_formats (RowRules), C2 ride_completeness, S4/S5 (one shared
                      aggregation), duplicate rows
tests/
  conftest.py         shared fixtures
  synthetic.py        synthetic rows in the real column layout; tar/zip builders
  test_schema.py  test_io.py  test_cache.py  test_checks_structural.py  test_checks_runner.py  test_cli.py
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
- Checks: structural and completeness (#16, done), arithmetic (#18), temporal (#19), fare outliers (#20), trip plausibility (#21), per-panelist behaviour (#26), volume and spend anomalies (#27), panel coverage (#28), and drift between delivery batches (#29).
- Outputs: a curated Parquet dataset with per-row QC flags (#22), privacy normalization (#30), and a stakeholder summary (#31).
- Tests and docs: edge-case inputs (#23), end-to-end and performance tests (#24), and the full README (#25).

## License

GNU General Public License v3.0 (`GPL-3.0-only`). See [LICENSE](LICENSE).
