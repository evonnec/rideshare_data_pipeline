"""The README stays true: the check catalog, the CLI flags, the quick start and links.

The catalog table is checked against what a real run reports (so against the
registry), the documented flags against ``rideshare-qc run --help``, the quick-start
command is run on the synthetic fixture, ``qc.example.toml`` must load, and every
table-of-contents link must point at a heading.
"""

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
from fixtures.make_synthetic import write_synthetic_csv

from rideshare_data_pipeline.cli import main
from rideshare_data_pipeline.config import KNOWN_CHECKS, load_config

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
# Without the fenced blocks, whose lines can look like headings (a report excerpt).
PROSE = re.sub(r"^```.*?^```", "", README, flags=re.M | re.S)

# Flags of other tools that the README's development commands use.
OTHER_TOOL_FLAGS = {"--check", "--locked"}


def section(heading: str) -> str:
    """The text under ``## heading`` up to the next ``## `` heading."""
    match = re.search(
        rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", README, re.M | re.S
    )
    assert match, f"no '## {heading}' section"
    return match.group(1)


def catalog_rows() -> list[tuple[str, str, str]]:
    """(ID, check name, severity) of each row of the check catalog table."""
    rows = []
    for line in section("Check catalog").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        is_row = line.startswith("|") and len(cells) == 5
        if is_row and cells[0] != "ID" and not set(cells[0]) <= {"-", ":"}:
            rows.append((cells[0], cells[1].strip("`"), cells[2]))
    return rows


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    work = tmp_path_factory.mktemp("readme")
    csv = write_synthetic_csv(work / "synthetic.csv")
    assert main(["run", "--input", str(csv), "--out", str(work / "out")]) == 1
    report: dict[str, Any] = json.loads((work / "out/report.json").read_text("utf-8"))
    return report


def test_catalog_names_match_the_registry() -> None:
    names = [name for _, name, _ in catalog_rows()]
    assert len(names) == len(set(names)), "a check appears twice in the catalog"
    assert set(names) == set(KNOWN_CHECKS)


def test_catalog_ids_and_severities_match_the_registry(report: dict[str, Any]) -> None:
    registry = {c["name"]: (c["check_id"], c["severity"]) for c in report["checks"]}
    documented = {name: (check_id, sev) for check_id, name, sev in catalog_rows()}
    assert documented == registry


def test_catalog_count_is_stated_correctly() -> None:
    words = {"Forty": 40}
    stated = re.search(
        r"^(\w+) checks, each shown once", section("Check catalog"), re.M
    )
    assert stated
    assert words[stated.group(1)] == len(KNOWN_CHECKS)


def run_help(capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as exit_info:
        main(["run", "--help"])
    assert exit_info.value.code == 0
    return capsys.readouterr().out


def test_every_flag_in_the_readme_exists(capsys: pytest.CaptureFixture[str]) -> None:
    documented = set(re.findall(r"(?<![-\w])--[a-z][a-z-]*", README))
    known = set(re.findall(r"(?<![-\w])--[a-z][a-z-]*", run_help(capsys)))
    assert documented - OTHER_TOOL_FLAGS <= known | {"--version"}
    # and every option of the command is documented
    assert known - {"--help"} <= documented


def test_the_quick_start_command_runs_on_the_synthetic_fixture(tmp_path: Path) -> None:
    block = section("Quick start").split("```sh\n")[1].split("```")[0]
    commands = [
        shlex.split(line.split("#")[0])
        for line in block.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    generate = next(c for c in commands if "make_synthetic.py" in " ".join(c))
    assert (ROOT / generate[-2]).is_file()
    csv = write_synthetic_csv(tmp_path / generate[-1])
    run = next(c for c in commands if c[:3] == ["uv", "run", "rideshare-qc"])
    args = run[3:]
    assert args[0] == "run"
    args[args.index("--input") + 1] = str(csv)
    args[args.index("--out") + 1] = str(tmp_path / "reports")
    assert main(args) == 1  # the fixture plants an error-severity failure
    assert (tmp_path / "reports/report.md").is_file()
    assert (tmp_path / "reports/synthetic.curated.parquet").is_file()


def test_the_example_config_loads_and_covers_every_check() -> None:
    config = load_config(ROOT / "qc.example.toml")
    assert set(config.checks) == set(KNOWN_CHECKS)


def test_environment_variables_in_the_readme_are_used_by_the_code() -> None:
    source = "\n".join(
        p.read_text("utf-8")
        for p in [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").glob("*.py")]
    )
    for name in set(re.findall(r"RIDESHARE_QC_[A-Z_]+", README)):
        assert source.count(name) > 1, f"{name} is documented but not used"


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, no punctuation, dashes for spaces."""
    return re.sub(r"[^a-z0-9 _-]", "", heading.lower()).replace(" ", "-")


def test_internal_links_point_at_headings() -> None:
    headings = {slug(h) for h in re.findall(r"^#{1,6} (.+)$", PROSE, re.M)}
    links = set(re.findall(r"\]\(#([^)]+)\)", README))
    assert links
    assert links <= headings, sorted(links - headings)


def test_the_contents_lists_every_top_level_section() -> None:
    contents = re.findall(r"^\d+\. \[(.+?)\]\(#", section("Contents"), re.M)
    sections = re.findall(r"^## (.+)$", PROSE, re.M)
    assert contents == [s for s in sections if s not in {"Contents", "License"}]
