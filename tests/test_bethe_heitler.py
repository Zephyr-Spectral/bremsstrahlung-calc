"""Tests for Bethe-Heitler thin-target cross section."""

from __future__ import annotations

import math

import pytest

from server.physics.bethe_heitler import bethe_heitler_2bn, thin_target_spectrum


class TestBetheHeitler2BN:
    """Tests for the Koch & Motz 2BN cross section."""

    def test_positive_for_valid_inputs(self) -> None:
        result = bethe_heitler_2bn(1.0, 0.5, 0.0, 29)
        assert result > 0

    def test_zero_when_photon_exceeds_electron(self) -> None:
        """Cross section must be zero when k >= T0."""
        result = bethe_heitler_2bn(1.0, 1.5, 0.0, 29)
        assert result == 0.0

    def test_zero_at_endpoint(self) -> None:
        """Cross section is zero when photon energy equals electron energy."""
        result = bethe_heitler_2bn(1.0, 1.0, 0.0, 29)
        assert result == 0.0

    def test_forward_peaked(self) -> None:
        """Cross section should be higher at 0 degrees than 90 degrees."""
        forward = bethe_heitler_2bn(1.0, 0.5, 0.0, 29)
        side = bethe_heitler_2bn(1.0, 0.5, math.pi / 2, 29)
        assert forward > side

    def test_increases_with_z(self) -> None:
        """Cross section scales roughly as Z^2."""
        sigma_al = bethe_heitler_2bn(1.0, 0.5, 0.0, 13)
        sigma_pb = bethe_heitler_2bn(1.0, 0.5, 0.0, 82)
        assert sigma_pb > sigma_al

    def test_negative_energy_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bethe_heitler_2bn(-1.0, 0.5, 0.0, 29)

    def test_negative_photon_energy_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bethe_heitler_2bn(1.0, -0.5, 0.0, 29)

    def test_negative_z_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bethe_heitler_2bn(1.0, 0.5, 0.0, -1)

    def test_zero_z_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bethe_heitler_2bn(1.0, 0.5, 0.0, 0)

    def test_various_angles_no_crash(self) -> None:
        """No singularities at any angle."""
        for angle in [0.0, 0.1, math.pi / 4, math.pi / 2, math.pi]:
            result = bethe_heitler_2bn(1.0, 0.3, angle, 29)
            assert result >= 0.0

    def test_high_z_no_crash(self) -> None:
        """High-Z materials (Born approximation limit) should still work."""
        result = bethe_heitler_2bn(1.0, 0.5, 0.0, 82)
        assert result >= 0.0


class TestThinTargetSpectrum:
    """Tests for angle-integrated thin-target spectrum."""

    def test_returns_correct_length(self) -> None:
        k, sigma = thin_target_spectrum(1.0, 29, 63.546, n_points=50)
        assert len(k) == 50
        assert len(sigma) == 50

    def test_all_positive(self) -> None:
        _k, sigma = thin_target_spectrum(1.0, 29, 63.546, n_points=20)
        for s in sigma:
            assert s >= 0.0

    def test_custom_photon_energies(self) -> None:
        k_custom = [0.1, 0.3, 0.5, 0.7]
        k, sigma = thin_target_spectrum(1.0, 29, 63.546, photon_energies_mev=k_custom)
        assert k == k_custom
        assert len(sigma) == 4
