"""Tests for API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health."""

    def test_health_returns_ok(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert isinstance(data["materials_count"], int)
        assert data["materials_count"] >= 10


class TestMaterialsAPI:
    """Tests for /api/materials."""

    def test_list_materials(self) -> None:
        response = client.get("/api/materials")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 10
        symbols = [m["symbol"] for m in data["materials"]]
        assert "Cu" in symbols
        assert "Al" in symbols

    def test_stopping_power_curve(self) -> None:
        response = client.get("/api/materials/Cu/stopping-power?n_points=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["electron_energy_mev"]) == 10
        assert len(data["stopping_power_mev_cm2_g"]) == 10
        assert all(v > 0 for v in data["stopping_power_mev_cm2_g"])

    def test_range_curve(self) -> None:
        response = client.get("/api/materials/Al/range?n_points=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["range_g_cm2"]) == 5
        assert all(v > 0 for v in data["range_g_cm2"])

    def test_unknown_material_404(self) -> None:
        response = client.get("/api/materials/Unobtanium/stopping-power")
        assert response.status_code == 404


class TestSpectrumAPI:
    """Tests for /api/spectrum."""

    def test_calculate_interpolated(self) -> None:
        response = client.get(
            "/api/spectrum/calculate",
            params={
                "material": "Cu",
                "electron_energy_mev": 1.0,
                "mode": "interpolated",
                "n_points": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "interpolated" in data

    @pytest.mark.nasa
    def test_nasa_data_endpoint(self) -> None:
        response = client.get(
            "/api/spectrum/nasa-data",
            params={"material": "Fe", "electron_energy_mev": 3.0, "angle_deg": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["photon_energy_mev"]) > 5
        assert all(v > 0 for v in data["intensity"])

    def test_unknown_material_404(self) -> None:
        response = client.get(
            "/api/spectrum/calculate",
            params={"material": "Unobtanium", "electron_energy_mev": 1.0},
        )
        assert response.status_code == 404

    def test_compare_materials(self) -> None:
        response = client.get(
            "/api/spectrum/compare",
            params={"materials": "Al,Cu", "electron_energy_mev": 1.0, "n_points": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert "Al" in data["spectra"]
        assert "Cu" in data["spectra"]

    def test_compare_too_many_materials(self) -> None:
        response = client.get(
            "/api/spectrum/compare",
            params={"materials": "Al,Cu,Fe,W,Au,Pb", "electron_energy_mev": 1.0},
        )
        assert response.status_code == 400


class TestValidationAPI:
    """Tests for /api/validation."""

    def test_non_nasa_material_404(self) -> None:
        response = client.get(
            "/api/validation/nasa-comparison",
            params={"material": "SS304", "electron_energy_mev": 1.0, "angle_deg": 0},
        )
        assert response.status_code == 404
