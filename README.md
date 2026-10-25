# rideshare_data_pipeline

Data-quality checks for a rideshare transactions CSV, built on pandas.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.15+ (uv installs it if missing).

```sh
uv sync
```

This creates `.venv/` with the runtime dependencies and the `dev` group (pytest, mypy, ruff, pandas-stubs).

## Run the program

Put the data archive (`Data_Rideshare.tar.bz2`) in the repository root, then run:

```sh
uv run python rideshare_data_validation.py
```

## Development

```sh
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy .          # type-check
```
