"""Shared input validation for physics modules."""

from __future__ import annotations


def require_positive_energy(value: float, label: str = "Energy") -> None:
    """Raise ValueError if energy value is not positive.

    Args:
        value: Energy value to validate.
        label: Human-readable label for the error message.
    """
    if value <= 0:
        msg = f"{label} must be positive, got {value} MeV"
        raise ValueError(msg)


def require_positive_z(z: int | float) -> None:
    """Raise ValueError if atomic number is not positive.

    Args:
        z: Atomic number to validate.
    """
    if z <= 0:
        msg = f"Atomic number must be positive, got Z={z}"
        raise ValueError(msg)
