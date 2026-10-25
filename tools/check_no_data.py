"""Fail if the repository tracks anything that looks like panel data.

Real data is never committed (see the README's "Data handling" section). CI runs this
guard on every push and pull request; it also works as a pre-commit hook:

    uv run python tools/check_no_data.py

It flags tracked files that are larger than the size limit, data or archive files
outside ``tests/fixtures/``, and CSVs outside ``tests/fixtures/`` whose header matches
the input schema.
"""

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

from rideshare_data_pipeline.schema import EXPECTED_COLUMNS

MAX_BYTES = 1_000_000
FIXTURES = PurePosixPath("tests/fixtures")
DATA_SUFFIXES = frozenset(
    {".parquet", ".feather", ".arrow", ".zip", ".tar", ".bz2", ".gz", ".xz"}
)
# A header sharing this many names with the input schema counts as a match, so a
# trimmed or reordered extract is still caught.
HEADER_MATCH = 5


def tracked_files(repo: Path) -> list[PurePosixPath]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [PurePosixPath(name) for name in out.decode().split("\0") if name]


def header_matches_schema(path: Path) -> bool:
    with path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline().lstrip("﻿")
    names = {name.strip().strip('"') for name in header.split(",")}
    return len(names & set(EXPECTED_COLUMNS)) >= HEADER_MATCH


def find_problems(repo: Path, max_bytes: int = MAX_BYTES) -> list[str]:
    problems = []
    for rel in tracked_files(repo):
        path = repo / rel
        if not path.is_file():  # tracked but deleted in the working tree
            continue
        if (size := path.stat().st_size) > max_bytes:
            problems.append(
                f"{rel}: {size:,} bytes exceeds the {max_bytes:,}-byte limit"
            )
        if rel.is_relative_to(FIXTURES):
            continue
        suffixes = {s.lower() for s in rel.suffixes}
        if suffixes & DATA_SUFFIXES:
            problems.append(f"{rel}: data or archive file outside {FIXTURES}/")
        if rel.suffix.lower() == ".csv" and header_matches_schema(path):
            problems.append(f"{rel}: CSV header matches the input schema")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--max-bytes", type=int, default=MAX_BYTES, help="size limit per file"
    )
    args = parser.parse_args(argv)

    problems = find_problems(args.repo, args.max_bytes)
    for problem in problems:
        print(f"check_no_data: {problem}", file=sys.stderr)
    if problems:
        print(
            "check_no_data: real data must never be committed; see the README's "
            '"Data handling" section',
            file=sys.stderr,
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
