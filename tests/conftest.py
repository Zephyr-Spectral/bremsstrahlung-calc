"""Shared pytest fixtures for bremsstrahlung-calc tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def cu_params() -> dict[str, object]:
    """Copper material parameters (NASA reference material)."""
    return {"symbol": "Cu", "Z": 29, "A": 63.546, "density": 8.960}


@pytest.fixture
def al_params() -> dict[str, object]:
    """Aluminum material parameters (NASA reference material)."""
    return {"symbol": "Al", "Z": 13, "A": 26.982, "density": 2.699}


@pytest.fixture
def w_params() -> dict[str, object]:
    """Tungsten material parameters (high-Z NASA reference)."""
    return {"symbol": "W", "Z": 74, "A": 183.84, "density": 19.25}
