"""Customer churn ML package — training, evaluation, and serving utilities."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("churn-ml")
except PackageNotFoundError:
    __version__ = "1.0.1"
