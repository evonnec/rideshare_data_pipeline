"""Test helpers: synthetic rows in the real 33-column layout, and file builders.

Every value here is made up. Real data is only read by the opt-in ``slow``
test, from a local file named by RIDESHARE_QC_DATA. (pytest's importlib mode
can't import conftest from a test, so the helpers live in this module; the
pytest config puts tests/ on sys.path.)
"""

import csv
import hashlib
import io
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from rideshare_data_pipeline.io import read_raw, typed
from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

type Row = dict[str, str]

# How the synthetic frame looks once parsed: the columns that are empty in every
# row, and the ones holding the same value in every row. Every other column
# varies from row to row (with 3+ rows, merchant_name's first value equals its
# last, the pattern the legacy constant check got wrong).
ALL_NULL = ("order_points", "order_shipping", "SKU", "item_id", "file_id")
CONSTANT = (
    "order_total_qty",
    "item_quantity",
    "digital_transaction",
    "product_category",
)


def synthetic_row(i: int) -> Row:
    """Return row `i` (0 <= i < 28) of a synthetic CSV, as the raw text fields."""
    day = f"{i + 1:02d}"
    return {
        "merchant_name": ("Uber", "Lyft")[i % 2],
        "user_id": hashlib.sha256(f"synthetic-user-{i}".encode()).hexdigest(),
        "order_number": str(1_000_000_000_000_000_000 + i),
        "order_time": f"2019-01-{day} 08:00:00.000",
        "email_time": f"2019-01-{day} 09:00:00.000",
        "insert_time": f"2019-01-{day} 10:00:00.000",
        "update_time": f"2019-01-{day} 11:00:00.000",
        "order_total_amount": f"{10 + i}.50",
        "order_points": "",
        "order_shipping": "",
        "order_tax": f"{i}.25",
        "order_subtotal": f"{9 + i}.00",
        "order_total_qty": "1.0",
        "product_description": f"Ride {i + 1}.00 mi | {5 + i} min",
        "product_subtitle": f"subtitle {i}",
        "item_quantity": "1.0",
        "item_price": f"{9 + i}.00",
        "digital_transaction": "0",
        "checksum": hashlib.md5(f"synthetic-row-{i}".encode()).hexdigest(),
        "product_reseller": f"reseller {i}",
        # Not "Travel": these rows aren't receipt kinds that category could mismatch.
        "product_category": "Receipts",
        "order_discount": f"0.{i:02d}",
        "SKU": "",
        "item_id": "",
        "order_pickup": str(i % 5),
        "from_domain": ("noreply@uber.com", "no-reply@lyftmail.com")[i % 2],
        "email_subject": f"Your trip receipt {i}",
        "delivery_date": f"2019-02-{day} 00:00:00.000",
        "start_source_folder_date": f"2019-02-{day}",
        "end_source_folder_date": f"2019-02-{day}",
        "file_id": "",
        "source_dttimestamp": f"2019-02-{day} 01:00:00.000",
        "dttimestamp": f"2019-02-{day} 02:00:00.000",
    }


def synthetic_rows(n: int, **overrides: Mapping[int, str]) -> list[Row]:
    """Return n synthetic rows; ``order_time={1: "bad"}`` sets row 1's order_time."""
    rows = [synthetic_row(i) for i in range(n)]
    for column, values in overrides.items():
        for i, value in values.items():
            rows[i][column] = value
    return rows


def write_csv_text(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    header: Sequence[str] = EXPECTED_COLUMNS,
) -> Path:
    """Write rows under `header` as a CRLF CSV (like the real file) and return path."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows([row.get(col, "") for col in header] for row in rows)
    return path


def _open_tar_for_writing(path: Path) -> tarfile.TarFile:
    match path.suffix:
        case ".bz2":
            return tarfile.open(path, "w:bz2")
        case ".gz":
            return tarfile.open(path, "w:gz")
        case ".xz":
            return tarfile.open(path, "w:xz")
        case _:
            return tarfile.open(path, "w")


def make_tar(
    path: Path,
    members: Mapping[str, bytes],
    symlinks: Mapping[str, str] | None = None,
) -> Path:
    """Write a tar archive of members (name -> data) and symlinks (name -> target).

    The compression follows the suffix: .bz2, .gz, .xz, or none for plain .tar.
    """
    with _open_tar_for_writing(path) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return path


def make_zip(path: Path, members: Mapping[str, bytes]) -> Path:
    """Write a zip archive of members and return path."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def read_typed_frame(path: Path) -> pl.DataFrame:
    """Read a CSV through the raw -> typed path and collect it (no failure check)."""
    return typed(read_raw(path)).collect()
