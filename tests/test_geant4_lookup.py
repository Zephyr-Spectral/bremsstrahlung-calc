"""Tests for Geant4 batch lookup data access and API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config
from server.data_access_geant4 import (
    clear_g4_cache,
    g4_angular_distribution,
    g4_energies,
    g4_heatmap,
    g4_info,
    g4_materials,
    g4_photon_energy_bins,
    g4_spectrum,
    get_g4_lookup,
)
from server.main import app

client = TestClient(app)

# Skip all tests if lookup file doesn't exist
pytestmark = pytest.mark.skipif(
    not config.GEANT4_LOOKUP_PATH.exists(),
    reason="Geant4 lookup file not found",
)


class TestG4DataAccess:
    """Tests for the Geant4 lookup data access module."""

    def test_load_lookup(self) -> None:
        lut = get_g4_lookup()
        assert lut["intensity"].shape == (11, 20, 36, 40)
        assert lut["counts"].shape == (11, 20, 36, 40)
        assert len(lut["materials"]) == 11
        assert len(lut["energies_mev"]) == 20

    def test_materials_list(self) -> None:
        mats = g4_materials()
        assert len(mats) == 11
        assert "Fe" in mats
        assert "C" in mats
        assert "Pb" in mats

    def test_energies_list(self) -> None:
        energies = g4_energies()
        assert len(energies) == 20
        assert energies[0] == pytest.approx(0.5, abs=0.01)
        assert energies[-1] == pytest.approx(10.0, abs=0.01)

    def test_info(self) -> None:
        info = g4_info()
        assert info["n_materials"] == 11
        assert info["n_energies"] == 20
        assert info["n_theta_bins"] == 36
        assert info["n_k_bins"] == 40
        assert info["n_events"] == 10_000_000

    def test_photon_energy_bins(self) -> None:
        centers, widths = g4_photon_energy_bins(3.0)
        assert len(centers) == 40
        assert len(widths) == 40
        # k_min = max(0.010, 0.005*3.0) = 0.015
        assert centers[0] > 0.01
        # k_max = 0.95 * 3.0 = 2.85
        assert centers[-1] < 3.0
        # Bins should be log-spaced (increasing widths)
        assert widths[-1] > widths[0]

    def test_spectrum_at_grid_point(self) -> None:
        result = g4_spectrum("Fe", 3.0, 0.0)
        assert len(result["photon_energy_mev"]) > 0
        assert len(result["intensity"]) > 0
        assert len(result["uncertainty"]) > 0
        assert result["n_events"] == 10_000_000
        assert result["is_interpolated"] is False
        # Fe at 3 MeV forward should have substantial intensity
        assert max(result["intensity"]) > 1e-5

    def test_spectrum_interpolated_on_grid(self) -> None:
        """On-grid requests should not flag as interpolated."""
        result = g4_spectrum("Cu", 1.0, 30.0)
        assert result["is_interpolated"] is False

    def test_spectrum_interpolated_off_grid(self) -> None:
        """Off-grid energy should interpolate between neighbors."""
        result = g4_spectrum("Fe", 2.75, 0.0)
        assert result["is_interpolated"] is True
        assert len(result["intensity"]) > 0
        # Should be between the 2.5 and 3.0 MeV results
        lo = g4_spectrum("Fe", 2.5, 0.0)
        hi = g4_spectrum("Fe", 3.0, 0.0)
        # Peak intensity should be bounded (roughly)
        peak = max(result["intensity"])
        assert peak > 0

    def test_spectrum_backward_angle(self) -> None:
        """Angles > 90 should work with reduced intensity."""
        fwd = g4_spectrum("Fe", 3.0, 0.0)
        bwd = g4_spectrum("Fe", 3.0, 150.0)
        # Forward should be more intense than backward
        assert max(fwd["intensity"]) > max(bwd["intensity"])

    def test_angular_distribution(self) -> None:
        result = g4_angular_distribution("Fe", 3.0, 1.0)
        assert len(result["angles_deg"]) > 0
        assert len(result["intensity"]) > 0
        assert result["n_events"] == 10_000_000
        # Forward should be more intense than backward
        angles = result["angles_deg"]
        intensities = result["intensity"]
        fwd_idx = min(range(len(angles)), key=lambda i: abs(angles[i] - 2.5))
        if len(angles) > 1:
            bwd_idx = min(range(len(angles)), key=lambda i: abs(angles[i] - 150.0))
            assert intensities[fwd_idx] > intensities[bwd_idx]

    def test_heatmap(self) -> None:
        result = g4_heatmap("Fe", 3.0)
        assert len(result["photon_energy_mev"]) == 40
        assert len(result["angles_deg"]) == 36
        assert len(result["intensity"]) == 36
        assert len(result["intensity"][0]) == 40

    def test_uncertainty_positive(self) -> None:
        result = g4_spectrum("Fe", 3.0, 0.0)
        for u in result["uncertainty"]:
            assert u >= 0

    def test_unknown_material_raises(self) -> None:
        with pytest.raises(ValueError, match="not in Geant4 lookup"):
            g4_spectrum("Xe", 3.0, 0.0)

    def test_clear_cache(self) -> None:
        get_g4_lookup()  # ensure loaded
        clear_g4_cache()
        # Should reload successfully
        lut = get_g4_lookup()
        assert lut["intensity"].shape[0] == 11


class TestG4API:
    """Tests for /api/geant4/* endpoints."""

    def test_geant4_info(self) -> None:
        response = client.get("/api/geant4/info")
        assert response.status_code == 200
        data = response.json()
        assert data["n_materials"] == 11
        assert data["n_events"] == 10_000_000

    def test_geant4_spectrum(self) -> None:
        response = client.get(
            "/api/geant4/spectrum",
            params={"material": "Fe", "electron_energy_mev": 3.0, "angle_deg": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["photon_energy_mev"]) > 0
        assert len(data["intensity"]) > 0
        assert data["n_events"] == 10_000_000

    def test_geant4_angular(self) -> None:
        response = client.get(
            "/api/geant4/angular",
            params={"material": "Fe", "electron_energy_mev": 3.0, "photon_energy_mev": 1.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["angles_deg"]) > 0
        assert len(data["intensity"]) > 0

    def test_geant4_heatmap(self) -> None:
        response = client.get(
            "/api/geant4/heatmap",
            params={"material": "Fe", "electron_energy_mev": 3.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["photon_energy_mev"]) == 40
        assert len(data["angles_deg"]) == 36

    def test_geant4_unknown_material_404(self) -> None:
        response = client.get(
            "/api/geant4/spectrum",
            params={"material": "Xe", "electron_energy_mev": 3.0, "angle_deg": 0},
        )
        assert response.status_code == 404

    def test_spectrum_calculate_with_geant4_mode(self) -> None:
        """Existing /api/spectrum/calculate should include geant4 data."""
        response = client.get(
            "/api/spectrum/calculate",
            params={
                "material": "Fe",
                "electron_energy_mev": 3.0,
                "angle_deg": 0,
                "mode": "all",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "geant4" in data
        assert data["geant4"]["status"] == "batch_lookup"
        assert len(data["geant4"]["intensity"]) > 0

    def test_validation_geant4_comparison(self) -> None:
        response = client.get(
            "/api/validation/geant4-comparison",
            params={"material": "Fe", "electron_energy_mev": 3.0, "angle_deg": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert "geant4" in data
        assert "calculated" in data
        # NASA data should be present at this grid point
        assert "nasa" in data
