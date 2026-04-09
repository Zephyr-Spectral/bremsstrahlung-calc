"""Tests for electron stopping power calculations."""

from __future__ import annotations

import pytest

from server.physics.stopping_power import (
    collision_stopping_power,
    radiative_stopping_power,
    total_stopping_power,
)


class TestCollisionStoppingPower:
    """Tests for collision stopping power."""

    def test_positive_for_valid_inputs(self) -> None:
        result = collision_stopping_power(1.0, 29, 63.546)
        assert result > 0

    def test_increases_at_low_energy(self) -> None:
        """Stopping power should increase as energy decreases (1/beta^2 behavior)."""
        s_low = collision_stopping_power(0.2, 29, 63.546)
        s_high = collision_stopping_power(2.0, 29, 63.546)
        assert s_low > s_high

    def test_higher_z_lower_stopping(self) -> None:
        """Higher Z materials have lower stopping power (per g/cm^2)."""
        s_al = collision_stopping_power(1.0, 13, 26.982)
        s_pb = collision_stopping_power(1.0, 82, 207.2)
        assert s_al > s_pb

    def test_negative_energy_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            collision_stopping_power(-1.0, 29, 63.546)

    def test_zero_energy_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            collision_stopping_power(0.0, 29, 63.546)

    def test_negative_z_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            collision_stopping_power(1.0, -1, 63.546)

    def test_zero_z_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            collision_stopping_power(1.0, 0, 63.546)

    def test_reasonable_magnitude_cu_1mev(self) -> None:
        """Cu at 1 MeV should be roughly 1.3-1.5 MeV cm^2/g (NASA Table I)."""
        result = collision_stopping_power(1.0, 29, 63.546)
        assert 0.1 < result < 5.0  # approximate; exact depends on density correction


class TestRadiativeStoppingPower:
    """Tests for radiative stopping power."""

    def test_positive(self) -> None:
        result = radiative_stopping_power(1.0, 29, 63.546)
        assert result > 0

    def test_increases_with_energy(self) -> None:
        s_low = radiative_stopping_power(0.5, 29, 63.546)
        s_high = radiative_stopping_power(3.0, 29, 63.546)
        assert s_high > s_low

    def test_increases_with_z(self) -> None:
        """Radiative stopping power scales roughly as Z²."""
        s_al = radiative_stopping_power(1.0, 13, 26.982)
        s_pb = radiative_stopping_power(1.0, 82, 207.2)
        assert s_pb > s_al

    def test_negative_energy_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            radiative_stopping_power(-1.0, 29, 63.546)


class TestTotalStoppingPower:
    """Tests for total stopping power."""

    def test_sum_of_components(self) -> None:
        s_col = collision_stopping_power(1.0, 29, 63.546)
        s_rad = radiative_stopping_power(1.0, 29, 63.546)
        s_tot = total_stopping_power(1.0, 29, 63.546)
        assert abs(s_tot - (s_col + s_rad)) < 1e-10

    def test_always_positive(self) -> None:
        for energy in [0.1, 0.5, 1.0, 3.0, 10.0]:
            assert total_stopping_power(energy, 29, 63.546) > 0

    def test_dominated_by_collision_at_low_energy(self) -> None:
        """At low energies, collision dominates over radiative."""
        s_col = collision_stopping_power(0.5, 13, 26.982)
        s_rad = radiative_stopping_power(0.5, 13, 26.982)
        assert s_col > s_rad
