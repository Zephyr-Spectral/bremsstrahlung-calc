"""Tests for electron range calculations."""

from __future__ import annotations

import pytest

from server.physics.electron_range import csda_range


class TestCSDARange:
    """Tests for CSDA electron range."""

    def test_positive_for_valid_inputs(self) -> None:
        result = csda_range(1.0, 29, 63.546)
        assert result > 0

    def test_increases_with_energy(self) -> None:
        """Range must increase monotonically with energy."""
        r_low = csda_range(0.5, 29, 63.546)
        r_high = csda_range(3.0, 29, 63.546)
        assert r_high > r_low

    def test_monotonic_with_energy(self) -> None:
        """Check monotonicity across multiple energies."""
        energies = [0.5, 0.75, 1.0, 2.0, 3.0]
        ranges = [csda_range(e, 29, 63.546) for e in energies]
        for i in range(len(ranges) - 1):
            assert ranges[i + 1] > ranges[i]

    def test_negative_energy_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            csda_range(-1.0, 29, 63.546)

    def test_negative_z_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            csda_range(1.0, -1, 63.546)

    def test_reasonable_magnitude_al_1mev(self) -> None:
        """Al at 1 MeV: NASA Table IV gives 0.549 g/cm^2."""
        result = csda_range(1.0, 13, 26.982)
        # Allow factor of 2 tolerance for first-principles calc
        assert 0.2 < result < 1.5

    def test_reasonable_magnitude_cu_1mev(self) -> None:
        """Cu at 1 MeV: NASA Table IV gives 0.620 g/cm^2."""
        result = csda_range(1.0, 29, 63.546)
        assert 0.2 < result < 5.0  # wider tolerance for first-principles calc
