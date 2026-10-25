"""``python -m rideshare_data_pipeline``: the same as the ``rideshare-qc`` command."""

from rideshare_data_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
