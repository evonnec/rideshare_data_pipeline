"""The repository guard in tools/check_no_data.py, run against throwaway git repos."""

import subprocess
from pathlib import Path

import pytest
from check_no_data import find_problems, main

from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_HEADER = ",".join(EXPECTED_COLUMNS) + "\n"


def make_repo(root: Path, files: dict[str, bytes]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # -f: the repo's own .gitignore doesn't apply here, but be explicit that the
    # guard must catch files even when someone force-adds them.
    subprocess.run(["git", "-C", str(root), "add", "-f", "."], check=True)
    return root


def test_clean_repo_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"src/pkg.py": b"x = 1\n", "README.md": b"# hi\n"})
    assert find_problems(repo) == []
    assert main(["--repo", str(repo)]) == 0


def test_large_file_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"blob.bin": b"\0" * 2_000_000})
    assert main(["--repo", str(repo)]) == 1
    assert "blob.bin: 2,000,000 bytes exceeds" in capsys.readouterr().err


def test_large_fixture_still_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"tests/fixtures/big.csv": b"a,b\n" * 300_000})
    assert any("exceeds" in p for p in find_problems(repo))


@pytest.mark.parametrize("name", ["receipts.csv", "data/extract/sample.csv"])
def test_schema_csv_outside_fixtures_fails(tmp_path: Path, name: str) -> None:
    repo = make_repo(tmp_path, {name: SCHEMA_HEADER.encode() + b"Lyft,abc\n"})
    assert find_problems(repo) == [f"{name}: CSV header matches the input schema"]


def test_schema_csv_with_bom_and_subset_of_columns_fails(tmp_path: Path) -> None:
    header = "﻿" + ",".join(f'"{c}"' for c in EXPECTED_COLUMNS[:6]) + "\n"
    repo = make_repo(tmp_path, {"subset.csv": header.encode()})
    assert find_problems(repo) == ["subset.csv: CSV header matches the input schema"]


def test_schema_csv_in_fixtures_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"tests/fixtures/synthetic.csv": SCHEMA_HEADER.encode()})
    assert find_problems(repo) == []


def test_unrelated_csv_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"docs/thresholds.csv": b"check,threshold\nspeed,80\n"})
    assert find_problems(repo) == []


@pytest.mark.parametrize(
    "name", ["out.parquet", "cache/data.feather", "raw.tar.bz2", "raw.zip", "x.csv.gz"]
)
def test_data_or_archive_outside_fixtures_fails(tmp_path: Path, name: str) -> None:
    repo = make_repo(tmp_path, {name: b"not really data"})
    assert find_problems(repo) == [
        f"{name}: data or archive file outside tests/fixtures/"
    ]


@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="not a git checkout")
def test_this_repository_is_clean() -> None:
    assert find_problems(REPO_ROOT) == []
