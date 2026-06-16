"""Shared pytest configuration for the convexipy test suite."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"'). "
        "These exercise n=5 enumeration and take on the order of tens "
        "of seconds each.",
    )
