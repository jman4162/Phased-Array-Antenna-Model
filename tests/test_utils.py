"""Tests for phased_array.utils coordinate transforms and helpers."""

import numpy as np
import pytest

import phased_array as pa
from phased_array.utils import (
    azel_to_thetaphi,
    create_theta_phi_grid,
    create_uv_grid,
    db_to_linear,
    deg2rad,
    frequency_to_k,
    frequency_to_wavelength,
    is_visible_region,
    linear_to_db,
    normalize_pattern,
    rad2deg,
    theta_phi_to_uv,
    thetaphi_to_azel,
    uv_to_theta_phi,
    wavelength_to_k,
)


class TestAngleConversions:
    def test_deg2rad_known_values(self):
        assert np.isclose(deg2rad(180.0), np.pi)
        assert np.isclose(deg2rad(0.0), 0.0)

    def test_rad2deg_known_values(self):
        assert np.isclose(rad2deg(np.pi / 2), 90.0)

    def test_deg_rad_roundtrip(self):
        angles = np.linspace(-180, 180, 19)
        assert np.allclose(rad2deg(deg2rad(angles)), angles)


class TestAzElThetaPhi:
    def test_boresight(self):
        theta, phi = azel_to_thetaphi(0.0, 0.0)
        assert np.isclose(theta, 0.0)

    def test_roundtrip(self):
        az = np.deg2rad(np.array([10.0, -20.0, 30.0]))
        el = np.deg2rad(np.array([5.0, 15.0, -25.0]))
        theta, phi = azel_to_thetaphi(az, el)
        az2, el2 = thetaphi_to_azel(theta, phi)
        assert np.allclose(az2, az)
        assert np.allclose(el2, el)


class TestUVConversions:
    def test_boresight_is_origin(self):
        u, v = theta_phi_to_uv(0.0, 0.0)
        assert np.isclose(u, 0.0)
        assert np.isclose(v, 0.0)

    def test_known_value(self):
        u, v = theta_phi_to_uv(np.pi / 2, 0.0)
        assert np.isclose(u, 1.0)
        assert np.isclose(v, 0.0, atol=1e-12)

    def test_roundtrip(self):
        theta = np.deg2rad(np.array([10.0, 30.0, 60.0]))
        phi = np.deg2rad(np.array([0.0, 45.0, 200.0]))
        u, v = theta_phi_to_uv(theta, phi)
        theta2, phi2 = uv_to_theta_phi(u, v)
        u2, v2 = theta_phi_to_uv(theta2, phi2)
        # Angles may wrap; direction cosines must round-trip exactly
        assert np.allclose(u2, u)
        assert np.allclose(v2, v)

    def test_visible_region_boundary(self):
        assert is_visible_region(0.0, 0.0)
        assert is_visible_region(1.0, 0.0)
        assert not is_visible_region(0.8, 0.8)
        mask = is_visible_region(np.array([0.0, 0.9, 0.5]), np.array([0.0, 0.9, 0.5]))
        assert mask.tolist() == [True, False, True]


class TestWavenumberFrequency:
    def test_wavelength_to_k(self):
        assert np.isclose(wavelength_to_k(1.0), 2 * np.pi)
        assert np.isclose(wavelength_to_k(0.5), 4 * np.pi)

    def test_frequency_to_wavelength(self):
        assert np.isclose(frequency_to_wavelength(3e8), 1.0)
        assert np.isclose(frequency_to_wavelength(10e9, c=3e8), 0.03)

    def test_frequency_to_k_consistent(self):
        f = 10e9
        assert np.isclose(frequency_to_k(f), wavelength_to_k(frequency_to_wavelength(f)))


class TestDbConversions:
    def test_roundtrip(self):
        values = np.array([1e-3, 0.5, 1.0, 100.0])
        assert np.allclose(db_to_linear(linear_to_db(values)), values)

    def test_known_values(self):
        assert np.isclose(linear_to_db(10.0), 10.0)
        assert np.isclose(db_to_linear(3.0), 10 ** 0.3)

    def test_min_db_floor(self):
        assert linear_to_db(0.0) == -100.0
        assert linear_to_db(0.0, min_db=-60.0) == -60.0
        assert np.all(linear_to_db(np.array([0.0, 1e-30])) == -100.0)


class TestNormalizePattern:
    def test_peak_mode(self):
        pattern = np.array([1.0, 2.0, 4.0])
        normalized = normalize_pattern(pattern, mode="peak")
        assert np.isclose(np.max(normalized), 1.0)

    def test_power_mode(self):
        pattern = np.array([1.0, 2.0, 2.0])
        normalized = normalize_pattern(pattern, mode="power")
        assert np.isclose(np.sum(np.abs(normalized) ** 2), 1.0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            normalize_pattern(np.array([1.0]), mode="bogus")


class TestGrids:
    def test_theta_phi_grid_shapes(self):
        theta, phi, THETA, PHI = create_theta_phi_grid(n_theta=19, n_phi=37)
        assert theta.shape == (19,)
        assert phi.shape == (37,)
        assert THETA.shape == (19, 37)
        assert PHI.shape == (19, 37)

    def test_theta_phi_grid_ranges(self):
        theta, phi, _, _ = create_theta_phi_grid(
            theta_range=(0, np.pi / 2), phi_range=(0, np.pi), n_theta=10, n_phi=10
        )
        assert np.isclose(theta[0], 0.0)
        assert np.isclose(theta[-1], np.pi / 2)
        assert np.isclose(phi[-1], np.pi)

    def test_uv_grid_shapes(self):
        u, v, U, V = create_uv_grid(n_u=21, n_v=31)
        assert u.shape == (21,)
        assert v.shape == (31,)
        assert U.shape == (31, 21) or U.shape == (21, 31)
        assert U.shape == V.shape
