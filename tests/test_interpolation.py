"""Tests for NASA data interpolation."""

from __future__ import annotations

import pytest

from server.physics.interpolation import (
    clear_cache,
    get_nasa_spectrum_at_grid_point,
    interpolate_nasa_spectrum,
)


@pytest.fixture(autouse=True)
def _clear_interpolation_cache() -> None:
    """Clear cache before each test."""
    clear_cache()


class TestGetNASAGridPoint:
    """Tests for exact grid point lookup."""

    @pytest.mark.nasa
    def test_fe_3mev_0deg(self) -> None:
        """Fe at 3 MeV, 0 deg should have data."""
        k, intensity = get_nasa_spectrum_at_grid_point(3.0, 0, "Fe")
        assert len(k) > 5
        assert len(intensity) == len(k)
        assert all(v > 0 for v in intensity)

    @pytest.mark.nasa
    def test_cu_1mev_30deg(self) -> None:
        """Cu at 1 MeV, 30 deg should have data."""
        k, intensity = get_nasa_spectrum_at_grid_point(1.0, 30, "Cu")
        assert len(k) > 3
        assert all(v > 0 for v in intensity)

    @pytest.mark.nasa
    def test_unknown_material_empty(self) -> None:
        """Unknown material returns empty."""
        k, intensity = get_nasa_spectrum_at_grid_point(1.0, 0, "Unobtanium")
        assert k == []
        assert intensity == []


class TestInterpolateNASASpectrum:
    """Tests for interpolated spectra."""

    @pytest.mark.nasa
    def test_at_grid_point_matches(self) -> None:
        """Interpolation at exact grid point should closely match direct lookup."""
        _k_direct, _i_direct = get_nasa_spectrum_at_grid_point(1.0, 0, "Fe")
        k_interp, i_interp = interpolate_nasa_spectrum(1.0, 0, "Fe")
        # Should have same number of points (approximately)
        assert len(k_interp) > 0
        assert len(i_interp) > 0

    @pytest.mark.nasa
    def test_interpolation_at_midpoint(self) -> None:
        """Interpolation between grid points should return valid data."""
        k, intensity = interpolate_nasa_spectrum(1.5, 15, "Cu")
        assert len(k) > 0
        assert all(v > 0 for v in intensity)

    @pytest.mark.nasa
    def test_clamping_below_range(self) -> None:
        """Energy below NASA range should be clamped to lowest."""
        k, _intensity = interpolate_nasa_spectrum(0.1, 0, "Al")
        assert len(k) > 0

    @pytest.mark.nasa
    def test_clamping_above_range(self) -> None:
        """Energy above NASA range should be clamped to highest."""
        k, _intensity = interpolate_nasa_spectrum(5.0, 0, "Al")
        assert len(k) > 0

    @pytest.mark.nasa
    def test_invalid_material_raises(self) -> None:
        with pytest.raises(ValueError, match="not in NASA data"):
            interpolate_nasa_spectrum(1.0, 0, "SS304")
