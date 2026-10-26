"""Fail if the repository tracks anything that looks like panel data.

Real data is never committed (see the README's "Data handling" section). CI runs this
guard on every push and pull request; it also works as a pre-commit hook:

    uv run python tools/check_no_data.py

It flags tracked files that are larger than the size limit, data or archive files
outside ``tests/fixtures/``, and CSVs outside ``tests/fixtures/`` whose header matches
the input schema.

It also keeps names and local paths out of the (public) repository without writing
them down: a tracked text file fails if any word in it (a run of ``[a-z0-9]``,
lowercased) has a SHA-256 in ``DENYLIST_SHA256``, or if it holds an absolute home
directory path. Only hashes are stored, so this file does not contain the names.
Findings are reported as ``path:line``, never as the word.
"""

import argparse
import hashlib
import re
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

# SHA-256 of lowercase words that must not appear in any tracked text file.
DENYLIST_SHA256: frozenset[str] = frozenset(
    {
        "89d4952071d00f4639e4a2ede7293810f898074bd0964f56c8d710e73beaa376",
        "89f8a2f7cc79f23797fdcaadfc6ea4344a359dc8c41bfef756a37972c7f074aa",
        "50ddc0414312549a5b687be38f2049596d74cd2a878914dfbf424e6c48a9ef7c",
    }
)
# Files the name scan skips: generated, and full of hashes and package names.
NAME_SCAN_SKIP = frozenset({"uv.lock"})
_WORD = re.compile(r"[a-z0-9]+")
# An absolute home directory: the macOS or Linux form (the CI runner's is fine). The
# parts are joined so that this file does not match its own pattern.
_HOME_PATH = re.compile("/" + "Users" + "/|/" + "home" + "/(?!runner(?:/|$))[^/\\s]+/")


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


def read_text(path: Path) -> str | None:
    """The file's text, or None for a binary file (a NUL byte or not UTF-8)."""
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def find_forbidden(
    rel: PurePosixPath, text: str, denylist: frozenset[str] = DENYLIST_SHA256
) -> list[str]:
    """Findings (``path:line: ...``) for denylisted words and home paths in text."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _HOME_PATH.search(line):
            found.append(f"{rel}:{number}: absolute home-directory path")
        digests = (
            hashlib.sha256(word.encode()).hexdigest()
            for word in _WORD.findall(line.lower())
        )
        if any(digest in denylist for digest in digests):
            found.append(f"{rel}:{number}: denylisted word")
    return found


def find_problems(
    repo: Path,
    max_bytes: int = MAX_BYTES,
    denylist: frozenset[str] = DENYLIST_SHA256,
) -> list[str]:
    problems = []
    for rel in tracked_files(repo):
        path = repo / rel
        if not path.is_file():  # tracked but deleted in the working tree
            continue
        if (size := path.stat().st_size) > max_bytes:
            problems.append(
                f"{rel}: {size:,} bytes exceeds the {max_bytes:,}-byte limit"
            )
        if rel.name not in NAME_SCAN_SKIP and (text := read_text(path)) is not None:
            problems += find_forbidden(rel, text, denylist)
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
            "check_no_data: real data, names and local paths must never be "
            'committed; see the README\'s "Data handling" section',
            file=sys.stderr,
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
