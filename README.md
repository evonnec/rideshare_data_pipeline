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
- `--out DIR`: where everything is written (created if missing): `report.json`, `report.md`, `samples/<check>.parquet` (one per failed check that has sample rows), and `<input stem>.curated.parquet` (zstd: every input row, the typed input columns, the derived columns and per-row QC flags; see [Curated dataset](#curated-dataset)). An input with no stem (a file named just `.csv`) writes `input.curated.parquet`, never a hidden file. See [Outputs](#outputs).
- `--config FILE`: a TOML file with thresholds, severities and on/off switches (see [Configuration](#configuration)).
- `--fail-on {error,warn}`: the lowest severity whose failure exits 1. Overrides the config's `fail_on`; the default is `error`.
- `--drop-columns A,B`: input columns to leave out of the curated Parquet. By default every column and every row is kept; nothing is dropped unless you ask, and the derived and flag columns are never dropped. The checks run on the whole frame first, so a dropped column's flags are still correct.
- `--cache-dir DIR`: keep (and reuse) a typed Parquet copy of the input in DIR. Also read from the environment variable `RIDESHARE_QC_CACHE` when the flag is absent; with neither, no cache is used. See [Cache](#cache).
- `--refresh-cache`: rebuild the cache entry for this input even if a valid one exists (needs a cache directory).

It prints a short summary, one line per check, and the paths written, for example:

```text
rows:             3
columns:          33
all-null columns: 5 (order_points, order_shipping, SKU, item_id, file_id)
constant columns: 4 (order_total_qty, item_quantity, digital_transaction, product_category)
duplicate rows:   0
wrote:            reports/receipts.curated.parquet (77 columns)

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

## Outputs

Every file the run writes (`report.json`, `report.md`, each `samples/<check>.parquet` and `<input stem>.curated.parquet`) is written to a temporary name beside its destination and then renamed into place, so a file is either the previous one or the new one and an interrupted run never leaves a partial file. The run deletes nothing in `--out` except the previous `samples/*.parquet`: **`--out` may hold outputs of other runs** (for example the Parquet of another input), and the report's `outputs` object (`{"parquet": "<name>", "curated": {"name", "rows", "columns"}, "samples": [...]}`) lists the files that belong to this run, so a stale file is easy to tell. Use a fresh directory per input if you want a clean folder.

## Curated dataset

`<input stem>.curated.parquet` (zstd) is the dataset to work from: a typed, strict superset of the delivered file. **Every input row is kept**; the dedupe policy is the `is_superseded` flag, not a deletion, so there is no `--no-dedupe` option: consumers filter on flags. Columns are never dropped silently (`--drop-columns` names input columns to leave out). It is written by a lazy plan (one `with_columns`, then `sink_parquet`), so the typed frame is not copied to write it. The contract is `curated.curated_schema(enabled_checks)`, asserted against the written file in the tests.

Columns, in this order:

| Group | Columns | Type | Meaning |
|---|---|---|---|
| Input | the 33 columns of the file, in file order | as `schema.py` casts them | the typed values; all-null and constant columns are kept and only reported |
| Derived | `receipt_kind` | Enum | the receipt kind (see [Receipt kinds](#receipt-kinds)) |
| | `subject_profile` | String | `business`, `personal` or null, from the subject's leading tag |
| | `subject_unclassified` | Boolean | no kind rule matched, so `receipt_kind` defaulted to `ride` |
| | `is_rider_spend` | Boolean | the kind is rider spend |
| | `sender_country` | String | country of the sender, or null |
| | `currency_reason` | Enum | why the amount may not be USD (`non_us_sender`, `km_unit`, `foreign_language`), else null |
| | `currency_suspect` | Boolean | `currency_reason` is set |
| | `foreign_subject` | Boolean | Spanish or Portuguese subject (an info tag; not a reason on its own) |
| | `distance_mi`, `distance_unit`, `duration_s`, `product_family` | Float64, Enum, Int64, Enum | the trip, parsed from the description (see [Description parsing](#description-parsing)) |
| Dedupe | `is_superseded`, `is_cross_panelist_copy` | Boolean | a later version of the receipt supersedes this row; the key was seen under more than one `user_id` (see [Duplicates and dedupe](#duplicates-and-dedupe)) |
| Spend | `is_usd_rider_spend` | Boolean | rider spend that is not currency-suspect (superseded or not) |
| Check flags | `qc_<check name>` | Boolean | **True means the row fails that check** (never null), one column per row-flagged check |
| Severity | `qc_error`, `qc_warn` | Boolean | the row fails at least one error- (warn-) severity flag, severities as configured |
| Names | `qc_flags` | List(String) | the names of the checks the row fails, in report order; an empty list, never null, on a clean row |

**Which checks have a flag.** Every row rule (its predicate, ANDed with `applies_to`, a null counting as not failed: the runner's own rule, so the column sums to the check's `n_failed`), plus the dataset checks that can point at rows: `duplicate_rows` (every exact copy after the first), `duplicate_receipts` (every row of a repeated key, as U2 counts), `cross_panelist_duplicates` (= `is_cross_panelist_copy`), `near_duplicate_rides` (the excess rows: the first ride of a group in file order is not flagged, every later one is, so the column sums to U4's `n_failed`), `currency_suspect_rows` (= `currency_suspect`), `description_parse_coverage` (the unparsed rows of PC1's population) and `fare_outlier_conditional` (O4's rows, from the same cached scores). Dataset-level findings have no column: `header_valid`, `utf8_valid`, `cast_failures`, `row_count_band`, `all_null_columns`, `constant_columns`, `receipt_kinds`, `unclassified_subject_templates`, `identity_share`, `date_only_order_time`, `fare_outlier_univariate` (info), `dedupe_key_unique` and `ride_completeness`. A check that is disabled in the config has no column. The invariant `sum(qc_<name>) == n_failed` is tested for every flag.

**Filtering.**

```python
import polars as pl

df = pl.scan_parquet("receipts.curated.parquet")
# clean USD rider spend
spend = df.filter(pl.col("is_usd_rider_spend") & ~pl.col("is_superseded"))
# the warn-severity review queue
review = df.filter(pl.col("qc_warn"))
# no error or warn flag
clean = df.filter(~pl.col("qc_error") & ~pl.col("qc_warn"))
# one check by name
outliers = df.filter(pl.col("qc_flags").list.contains("fare_outlier_conditional"))
```

`is_usd_rider_spend` does not look at `is_superseded` (a superseded row keeps its kind and currency), so add `~is_superseded` for totals, as above.

## Archives

An archive (`.tar`, `.tar.gz`, `.tar.bz2`, `.tar.xz` or `.zip`) must hold exactly one CSV. Only that member is extracted, streamed to `data.csv` in a temporary directory that is deleted when the run ends, so no member name ever reaches the file system and nothing else in the archive is written.

- Directories and macOS AppleDouble sidecars (`._*`, also under `__MACOSX/`) are ignored.
- No CSV, or more than one, raises `ArchiveError` (exit 2), and the message lists the members (at most 20, then "and N more").
- The CSV member is rejected if it is a symlink, a hardlink or any non-regular file, or if its name is absolute or has a `..` component.
- A truncated or corrupt archive, or an encrypted zip member, is also an `ArchiveError` (exit 2, never a crash): "archive is truncated or corrupt: ..." or "encrypted zip members are not supported".
- Error messages about the CSV name the member's path inside the archive, not the temporary file.
- A tar is read in one sequential pass (a listing of a compressed tar means decompressing it, and seeking back would decompress it again).

## Cache

Decompressing the bz2 sample takes about 20 s per run. With a cache directory, the first run on an input saves the validated, typed frame as Parquet and later runs read that instead. It is opt-in: without `--cache-dir` (or `RIDESHARE_QC_CACHE`) nothing is cached and nothing is written but `--out`. The flag wins over the variable; an empty variable means no cache.

```text
<cache>/<sha256 of the input file as given>/data.parquet   the typed frame (zstd)
<cache>/<sha256 of the input file as given>/meta.json      format version, tool version, schema fingerprint,
                                                           input name/size/mtime_ns/sha256, the CSV's member
                                                           name (null for a CSV input), size,
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

The derived columns (`receipt_kind`, `subject_profile`, `subject_unclassified`, `is_rider_spend`, `sender_country`, `currency_reason`, `currency_suspect`, `foreign_subject`, and the trip columns) are computed once per run for the checks. They are computed once per run for the checks and written to the [curated dataset](#curated-dataset).

Checks added: `receipt_kinds` (K1, info: the kind counts and the merchant x kind matrix, shown in the report), `category_mismatch` (V2, warn: category `Travel` on a food order, freight, rental or driver earnings) and `unclassified_subject_templates` (K2, info). `non_negative_total` (A1) now applies to rider-spend receipts only: a refund on a bike or pass receipt isn't a rider-spend error.

A column is **all-null** when every value is null. It is **constant** when its non-null values have exactly one distinct value, so nulls are ignored. An all-null column is reported only as all-null. A file with only a header row has no rows to judge, so it reports neither.

| Exit status | Meaning |
|---:|---|
| 0 | Every check that gates passed. |
| 1 | An **error**-severity check failed (with `--fail-on warn`, a failed **warn** check too). Info checks never gate. `report.json`, `report.md`, the samples and the Parquet are still written. |
| 2 | Input or parse error, a bad config file (unknown key or check name, bad value, no check left enabled), invalid arguments, or the output can't be written. The reason is printed to stderr. **Nothing is written** (no report either) when the input or config can't be read. |
| 3 | **Internal error**: an unexpected exception, i.e. a bug in this tool rather than a fault in the input or a verdict on the data. One line on stderr (`rideshare-qc: internal error: <type>: <message> (please report)`), no traceback. Ctrl-C is not caught (exit 130). |

Input errors include a missing or unsupported file, an archive that can't be extracted or doesn't hold exactly one CSV, an empty file, bytes that aren't valid UTF-8 (the message gives the byte offset), a header with missing, extra, duplicated or reordered columns (the message names them), and values that don't convert to their column's type (the message counts the failures per column). Values are never nulled silently: a value counts as a cast failure when it doesn't parse, when a number is not finite (`nan`, `inf`, `-inf`, `Infinity`, or a value such as `1e400` that overflows), or when a timestamp or date falls outside the years 1 to 9999 (`0000-01-01`). All of them exit 2 with nothing written.

**Blank lines.** A line that is empty (apart from a `\r`) is not a row: polars would read it as a row of nulls, so such rows are dropped on reading and counted. The count is `run.blank_lines` in `report.json`, `utf8_valid`'s `details.blank_lines`, and a `blank lines` line in the summary when it is not 0. A blank line is reported, not an error. (It is counted from the bytes, so a blank line inside a quoted field counts too.) Detecting a row with too few fields stays a separate known gap.

**`cast_failures` (S2)** records that the strict parse passed: any failure aborts the run with exit 2 before any check runs, so on a report it is always a pass. `enabled = false` for it is a config error, because it would only hide that line.

 A UTF-8 BOM is accepted and stripped, and CRLF and LF files read identically; the summary reports both. Known gap: a row with too few fields is padded with nulls, not flagged. `python -m rideshare_data_pipeline run ...` works the same way as `rideshare-qc run ...`. The old flat invocation (`rideshare-qc --input ...`) is gone.

## Duplicates and dedupe

The legacy whole-row duplicate check could never fire: `checksum` is unique per row and the ingestion metadata differs between versions of a receipt. The vendor upserts updated receipts and the same receipt can reach two panelists, so duplicates are found on the order key `(merchant_name, order_number)`, over rows where both are non-null (a null is not a key: recurring pass charges have no order number and are legitimate).

**Dedupe policy** (`dedupe.py`). `supersede_flags` adds a boolean `is_superseded`; **rows are never dropped**, consumers filter on the flag (`deduplicated` is that view, used for spend metrics from #15 on). Within a key group one row wins and the rest are superseded. The winner is the first of: a `tip_update` or `fare_correction` (kind priority beats time, so a tip update supersedes its ride even when the ride's email is later); the latest `email_time`; the latest `update_time`; the greatest `checksum`. Rows without a key are never superseded, and dedupe is idempotent. The flag is computed once per run for the checks and is a column of the [curated dataset](#curated-dataset).

**Cross-panelist copies.** The same receipt can reach two panelists' mailboxes. For panel totals the dedupe keeps one row per key, so the losing copy is superseded. `supersede_flags` also adds `is_cross_panelist_copy`, true for every row of a key group with more than one distinct non-null `user_id` (U3's `n_failed` is the number of rows with it). It is derived, not an input column, so the structural checks ignore it. A per-panelist statistic (checks from #26 on: a panelist's spend, trips or tenure) is about what each panelist received, so it should take the non-superseded rows **plus** the superseded rows that have this flag, which attributes a shared receipt to both panelists; panel-level totals keep using the deduplicated rows only.

**Tip updates count as full final fares.** A tip update is the final version of the ride it amends (kind priority beats time), and its `order_total_amount` is the whole fare including the tip, not just the tip, so it replaces the ride's amount in spend rather than adding to it. The ride's original receipt is superseded. This is deliberate; an `Increase Tip` stub line has no distance, which is why PC1 leaves those out.

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

The file has no currency column, yet receipts from non-US merchant senders state amounts in local currency: about 0.5% of rows are about 37% of a naive spend total. `from_domain` is the **full sender address**, and it carries the evidence. `sender.py` derives `sender_country` from public receipt-sender patterns (`uber.<country>@` to an ISO code, `receipts.<city>@` to its country with US as the default city; generic senders such as `noreply@` give null; an unknown `uber.<token>@` is `unknown`, treated as non-US). `currency_reason` is the first of `non_us_sender`, `km_unit` (the description states kilometres) and `foreign_language` (a Spanish or Portuguese subject: `Tu ... del ...`, `por la mañana`, `Sua viagem`, `Error de pago`, from a sender that is not a US one: in practice a generic sender with no country). `currency_suspect` is true where there is a reason.

**A Spanish or Portuguese subject from a US sender is not a currency reason.** In the full sample all 2,552 such rows come from US senders, state miles where they state a distance, and cost a median $2.38 per mile, in line with USD rides. The language stays findable as the info column `foreign_subject` (true for any such subject, suspect or not), and CUR1's `details.foreign_subject` counts the tagged rows that are not suspect. It gates nothing. Rows are flagged, never dropped, and no exchange rate is applied (FX conversion is future work).

**`usd_rides(frame)`** (`money.py`) is THE population for every money statistic and outlier baseline: rider spend, not superseded, not currency-suspect, with a non-null `order_total_amount`. A later check that sums or ranks amounts starts from it.

| Check | Severity | Finds |
|---|---|---|
| `currency_suspect_rows` (CUR1) | warn | currency-suspect rows; details: counts and naive sums by reason and country (local currency, not summable as USD), the spend waterfall. A warning because the gate is the `usd_rides` exclusion; an error would fail every real run for an expected property of the data |
| `merchant_sender_consistent` (V1) | error | an Uber receipt not from an Uber domain, or a Lyft receipt not from lyft.com, lyftmail.com or a subdomain |
| `unknown_sender` (V3) | warn | a sender matching no known receipt-sender pattern (for example addresses of individual people on freight receipts; their samples never show the address) |

`report.md` has a "Spend" section, and `report.json` an optional `spend` object, with the waterfall: the naive total of `order_total_amount`, minus currency-suspect rows, minus non-rider-spend kinds, minus superseded duplicates, equals the clean USD rider spend, each with its row count.

**Micromobility** (bikes and scooters) is not rider spend. It stays out of the population and has its own line in the Spend section (`spend.micromobility` in `report.json`: "micromobility (excluded; not rider spend)", rows and amount). It is a sub-line of the non-rider step, not a further deduction, so the waterfall still adds up exactly. The "superseded" line counts only superseded rows that have an amount and would otherwise be USD rider spend, so it differs from the dedupe's superseded count in "Duplicate receipts" (which flags any kind, in any currency, with or without an amount); the report says so.

**PC1** (`description_parse_coverage`) leaves out the `Increase Tip` / `Tip Increase` stub lines of tip updates (compared trimmed and case-insensitively): they name no trip, so they can never carry a distance. Its rate is the **worst merchant unparsed rate**, not `n_failed / n_evaluated`; `details.merchants` has every merchant's rate.

## Arithmetic checks

`total = subtotal + tax - discount` does not hold row by row: tips, tolls and fees are not itemized, and many credits are missing from `order_discount`. About a third of ride rows match it exactly, so it is reported (A7), not enforced. The narrower relations that do hold are the rules:

| id | check | severity | population | rule |
|---|---|---|---|---|
| A1 | `non_negative_total` | error | rider-spend receipts | total >= 0 |
| A2 | `item_price_equals_subtotal` | error | both values present | abs(item_price - subtotal) <= 0.01 |
| A3 | `zero_fare_rides` | warn | ride and tip-update receipts, not superseded | total is exactly 0 (per-merchant counts in `details.by_field`) |
| A4 | `discount_within_subtotal` | warn | both values present | discount <= subtotal + 0.01 |
| A5 | `tax_within_total` | warn | both values present | tax <= total + 0.01 |
| A6 | `unexplained_residual` | warn | USD rider-spend ride receipts, not superseded | residual <= max(20, subtotal) |
| A7 | `identity_share` | info | same as A6 | share of rows with abs(residual) <= 0.01; also the counts above, below and not computable, and the median gap above |

The residual is `total - (subtotal + tax - discount)`. Nulls: A2, A4 and A5 skip a row with a null operand. A6 and A7 treat a null tax or discount as 0 and call a null total or subtotal "not computable" (never a failure). Amounts are compared in whole cents, so a 0.01 tolerance does not depend on float rounding. A6 flags only a positive residual (a total far above the itemized lines); a large negative one is an unlisted credit. On the real file that reading gives 2,011 rows against 2,179 for the absolute value. The report has an "Arithmetic" section with the A7 breakdown.

Zero-total rides stay in spend sums, but they are not fares: per-ride averages and the outlier baselines (#20) must use `money.positive_usd_rides` (`order_total_amount > 0`).

## Temporal checks

They compare the receipt's timestamps with each other, on every row (superseded rows included: they describe the delivered data, not spend). A null operand is never a failure; a rule that needs two columns applies to the rows that have both.

| ID | Check | Severity | Rule |
| --- | --- | --- | --- |
| T1 | `ingestion_order` | error | `insert_time <= update_time <= source_dttimestamp <= dttimestamp` (per-link counts in `details.by_field`) |
| T2 | `order_before_receipt` | error | `date(order_time) <= date(email_time) + 1 day`; `details.by_field.order_after_email_no_tolerance` counts `order_time > email_time` |
| T3 | `late_receipts` | warn | the email date is at most 7 days after the order date; `details.by_field.over_30_days` counts those more than 30 days late |
| T4 | `order_before_window` | warn | `order_time` is not more than 7 days before the window start (the earliest `email_time` date of the frame) |
| T5 | `email_after_update` | warn | `email_time <= update_time` |
| T6 | `date_only_order_time` | info | per merchant, the share of `order_time` values at exactly 00:00:00.000 (`details.by_merchant`) |

The 7 and 30 day thresholds and the 1 day tolerance are constants in `checks/temporal.py`, not config keys. T4 is a row rule whose predicate holds the aggregate `email_time.min()`; inside the one select that is a scalar, so it shares the single scan. The report has a "Time" section: the email window, the T2 tolerance with its no-tolerance count, the T3 counts and the date-only share per merchant. On the full real file: T1 0, T2 0 (380 without the tolerance), T3 2,757 (968 over 30 days), T4 828, T5 19, and Uber `order_time` is date-only on about 99.6% of rows.

## Fare outliers

| ID | Check | Severity | Rule |
| --- | --- | --- | --- |
| O4 | `fare_outlier_conditional` | warn | on USD rides with `distance_mi >= 0.5`, `duration_s >= 60` and a product family: `\|rz(log $/mile)\| > k` AND `\|rz(log $/minute)\| > k` within merchant x product family (`k` defaults to 4) |
| O5 | `fare_outlier_univariate` | info | `\|rz(log total)\| > k` within merchant (`k` defaults to 3.5); context only, never fails |

**Method** (`checks/outliers.py`). Fares are heavy-tailed, roughly log-normal, and grow with distance and product, so a raw z-score or a Tukey fence flags 1-3% of rows and about 99.7% of the high flags have a normal price per mile: they are long trips. The checks therefore score within a **segment** (merchant x product family), on a **log scale** with a **robust** spread, and **conditional on the price drivers**:

- `robust_z(expr, by)` is `(x - median) / (1.4826 * MAD)` with both taken `.over(by)` (`MAD = median(|x - median|)`; 1.4826 makes it match a standard deviation under a normal). It is null, never infinite, when a group's MAD is 0 or the group has fewer than `MIN_GROUP` (30) values, and `log_positive` takes the log only where the value is positive, so a zero or negative input has no score.
- O4 needs both the price per mile and the price per minute to be extreme, so a long trip (high fare, ordinary $/mile) is not flagged, nor is a trip that is merely fast or slow (implausible speed is O1's). The sample shows the receipt columns plus `distance_mi`, `duration_s`, `product_family`, the two prices and the two scores, the most extreme first.
- Both checks use `money.positive_usd_rides` (USD rider spend, not superseded, not currency-suspect, total > 0) restricted to the kinds `ride` and `tip_update`; zero totals are A3's and negative ones A1's.
- The threshold `k` is a **review budget**, not a false-positive rate: the report shows the flag rate at the configured `k` and a table at k = 3, 3.5, 4 and 5, computed in the same pass. Both checks share one scoring pass.
- There is **no machine learning**: an isolation forest was tried and rejected (36 of 1,000 flags overlapped with these, and it gives no reason to read for a flag), while a log-scale robust z says why a row was flagged.

On the full real file, O4 at k = 4 flags about 0.115% of rides (1,988 of 1,730,233); nearly all are both-prices-low rides (near-zero fares). O5 flags about 0.47%, mostly long legitimate trips. The report has an "Outliers" section: k and the flag rate, the k table, the direction split (both prices high, both low, mixed), the top segments, the median $/mile and $/minute per product family, and a method note.

## Trip plausibility

| ID | Check | Severity | Rule |
| --- | --- | --- | --- |
| O1 | `implausible_speed` | warn | `duration_s >= min_seconds` AND `distance_mi / (duration_s / 3600) > max_mph` (defaults 120 s and 80 mph) |
| O2 | `zero_trip_metrics` | warn | `(distance_mi == 0 OR duration_s == 0)` AND `order_total_amount > 0`; `by_field` splits `zero_distance` (only), `zero_duration` (only) and `both` |
| O3 | `extreme_trip` | warn | `distance_mi > max_miles` OR `duration_s > max_seconds` (defaults 100 mi and 10,800 s); `by_field` counts `over_distance` and `over_duration` |

Cheap physical checks on the parsed trip metrics (`checks/plausibility.py`), with no baseline. The population of all three is ride and tip-update receipts that are not superseded: the question is about the trip, not the money, so the currency does not matter (a km distance is already in miles). A null distance or duration is never a failure (null means unparsed, which is PC1's), and the speed is computed only where `duration_s > 0`, so nothing is inf or NaN. The comparisons are strict: exactly 80 mph passes. O2 leaves zero-fare rides to A3. They are row rules in the runner's single select. The thresholds are `[checks.<name>.params]` (`max_mph` and `min_seconds` for O1, `max_miles` and `max_seconds` for O3; O2 has none), and the failing-row sample shows the receipt columns plus `distance_mi`, `duration_s` and `product_family` (the implied speed is `distance_mi / duration_s * 3600`).

On the full real file: O1 flags 85; O2 flags 1,494 (816 zero distance only, 176 zero duration only, 502 both, so 1,318 with a zero distance and 678 with a zero duration); O3 flags 450 (395 over distance, 135 over duration). The report has a "Trip plausibility" section with one row per check, its threshold and its breakdown.

## Assumptions

**Local time versus UTC.** `order_time` is the trip's local time (the rider's wall clock) and `email_time` is UTC. The evidence: the Lyft lag `email_time - order_time` is never below 4 h and peaks at 5 h and 7 h (US offsets plus a few minutes for the receipt), and every row whose `order_time` is after its `email_time` comes from a sender east of UTC (Asia-Pacific). The two columns are not on one clock, so the checks compare dates, not instants, and never a time of day. This was an owner decision, taken on that evidence.

**The 1 day tolerance on T2.** A trip on the 12th in UTC+10 can be on the 11th in UTC when its receipt is sent, so the UTC date of the receipt may be one day before the local order date and no more (offsets span at most 14 h). Without the tolerance 380 real rows fail; with it none do. The report keeps the no-tolerance count visible, as the evidence for the assumption.

**Upserts.** The vendor upserts receipts: an updated receipt (a tip added to a ride) is a new row that keeps the first version's `insert_time`. So `insert_time` can precede `email_time`, and that is normal. T1 orders only the vendor's own stamps and never involves `email_time`; T5 flags the opposite oddity, an email after the vendor's last update.

**Date-only Uber `order_time`.** Uber `order_time` is a date at midnight on about 99.6% of rows: the time of day was not recorded. Midnight here means "time unknown", which a polars `Datetime` cannot tell from a real 00:00:00.000, so T6 reports the share (an info check, it never fails) and no check relies on the time of day.

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

A check with numbers of its own takes a `[checks.<name>.params]` table. The fare-outlier checks accept `k` (a number greater than 0; the defaults are 4 for `fare_outlier_conditional` and 3.5 for `fare_outlier_univariate`):

```toml
[checks.fare_outlier_conditional.params]
k = 4.0                    # both |robust z| must exceed k; the report shows the effective k
```

The trip plausibility checks take `max_mph` and `min_seconds` (`implausible_speed`) and `max_miles` and `max_seconds` (`extreme_trip`); `zero_trip_metrics` takes none.

Unknown keys and unknown check names are errors (exit 2), so a typo can't silently do nothing. So is a param name the check doesn't know or a param that is not greater than 0, and so is a config that leaves no check enabled, and `enabled = false` for `cast_failures`. The effective thresholds and severities are recorded in the report. A `max_failed_rate` on a dataset check (for example `duplicate_rows`) replaces its threshold and re-judges it against its measured rate.

## Report

`report.md` has a title with the input name and the first 12 characters of its SHA-256 (of the archive, for an archive; a `member: <path> (<sha256[:12]>)` line for the CSV inside it follows), a one-line run summary (rows, panelists, window), the counts, a table of every check (sorted by severity, failures first) with its failed count, rate, threshold and rule, a column profile, and the failing-row samples (at most `sample_limit` rows per check, linked to the Parquet file). The "Composition by receipt kind" section is a merchant x kind table with a total column and a rider-spend marker per kind.

`report.json` is documented by the JSON Schema (draft 2020-12) in [`src/rideshare_data_pipeline/report.schema.json`](src/rideshare_data_pipeline/report.schema.json), which a test validates a real run against:

```text
{"schema_version": 1,
 "run": {tool_version, input_name (base name), input_sha256, input_size_bytes (of the file passed: the CSV or the archive),
         bom, line_ending (of the CSV read), member (null, or for an archive {name (path inside it), sha256, size_bytes, has_bom, line_ending}),
         rows, panelists (distinct user_id), window {start, end} (email_time, ISO), started, finished,
         elapsed_seconds, cache (hit|miss|off), blank_lines (optional), config {fail_on, sample_limit, checks {<name>: {threshold, severity}}}},
 "summary": {errors_failed, warnings_failed, info, passed, total},
 "composition": {rows, merchants, kinds: [{kind, rider_spend, total, by_merchant}]}  (optional),
 "spend": {naive_total, currency_suspect, non_rider_spend, superseded, clean_usd_rider_spend,
           micromobility (a sub-line of non_rider_spend), suspect_rider_spend: {rows, amount}, note}  (optional),
 "checks": [{name, check_id, category, severity, passed, metric, threshold, n_failed, n_evaluated,
             predicate, details, sample_file (relative path or null)}],
 "outputs": {parquet (file name of the curated Parquet), curated ({name, rows, columns} or null), samples [relative paths]}  (the files of this run),
 "exit_code": 0 | 1}   (2 and 3 are never written: those runs have no report)
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

Templates: Lyft `Lyft fare (1.63mi, 12m 23s)` (also `45s` and `1h 2m 3s`) and `Base fare ... 7.06 mi` lines; Uber `UberX 9.93 mi | 32 min` (also `| 1 h 22 min`, `km`); older Uber `2.00 miles   00:15:14 Trip time   uberX Car`. The family comes from the product label, first match in this order: express, shared, vip, black/lux, xl, comfort, uberx, lyft; a trip with none (a Lyft `Base fare` line) is `other`. Like the other derived columns they are computed once per run and are columns of the [curated dataset](#curated-dataset).

`description_parse_coverage` (PC1, warn) takes the ride and tip-update rows that are not superseded and passes while every merchant parses at least 98% of them (distance and duration both read). The population leaves out the `Increase Tip` / `Tip Increase` stub lines. Its metric is the worst merchant unparsed rate (`details.metric_label`; not `n_failed / n_evaluated`), and its details hold per-merchant `rows`, `parsed` and `coverage`, the unit counts, the family counts by merchant and the ten most common unparsed shapes (digits shown as `#`, never the text). The report's "Description parsing" section shows them.

## Package layout

```text
src/rideshare_data_pipeline/
  __init__.py         __version__, read from the package metadata
  __main__.py         python -m rideshare_data_pipeline
  cli.py              the rideshare-qc command (argparse, `run` subcommand): summary, check lines, reports, curated Parquet output, exit codes
  config.py           QCConfig and CheckOverride (with per-check params), TOML loading (tomllib), applying overrides to rules and results
  curated.py          the curated dataset: curated_schema (the column contract), row_flags, curated_plan, write_curated (sink_parquet)
  dedupe.py           the dedupe policy: supersede_flags (is_superseded, is_cross_panelist_copy), deduplicated
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
    model.py          Severity, CheckResult, RowRule, DatasetCheck, FlaggedCheck (a dataset check with a per-row flag)
    arithmetic.py     row rules A1-A6, dataset check A7
    temporal.py       row rules T1-T5 (timestamp order, local-vs-UTC tolerance), dataset check T6 (date-only order_time)
    uniqueness.py     checksum_unique (U1), duplicate_receipts (U2), dedupe_key_unique (U2b), cross_panelist_duplicates (U3), near_duplicate_rides (U4)
    kinds.py          category_mismatch (V2), receipt_kinds (K1) and unclassified_subject_templates
    parsing.py        description_parse_coverage (PC1)
    outliers.py       robust_z, fare_outlier_conditional (O4) and fare_outlier_univariate (O5), sharing one scoring pass
    plausibility.py   implausible_speed (O1), zero_trip_metrics (O2), extreme_trip (O3): row rules
    currency.py       currency_suspect_rows (CUR1), merchant_sender_consistent (V1), unknown_sender (V3)
    structural.py     S1/S3 results from the file facts, row_count_band, C1 required_fields and
                      S6 id_formats (RowRules), C2 ride_completeness, S4/S5 (one shared
                      aggregation), duplicate rows
tests/
  conftest.py         shared fixtures
  synthetic.py        synthetic rows in the real column layout; tar/zip builders
  test_schema.py  test_io.py  test_cache.py  test_checks_structural.py  test_checks_runner.py  test_cli.py
  test_checks_arithmetic.py  test_checks_temporal.py  test_checks_outliers.py  test_checks_plausibility.py  test_config.py  test_report.py  test_parse.py  test_trip_parse.py  test_currency.py  test_qa_fixes.py  test_curated.py
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
- Checks: structural and completeness (#16, done), arithmetic (#18, done), temporal (#19, done), fare outliers (#20, done), trip plausibility (#21, done), per-panelist behaviour (#26), volume and spend anomalies (#27), panel coverage (#28), and drift between delivery batches (#29).
- Outputs: a curated Parquet dataset with per-row QC flags (#22, done), privacy normalization (#30), and a stakeholder summary (#31).
- Tests and docs: edge-case inputs (#23), end-to-end and performance tests (#24), and the full README (#25).

## License

GNU General Public License v3.0 (`GPL-3.0-only`). See [LICENSE](LICENSE).
