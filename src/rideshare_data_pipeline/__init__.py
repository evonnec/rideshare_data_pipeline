"""Quality-control checks for a rideshare e-receipt panel sample, built on polars."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rideshare-data-pipeline")
except PackageNotFoundError:  # imported from a source tree that isn't installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
