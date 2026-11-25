"""Tests for pattern_summary and the first-null beamwidth."""

import numpy as np
import pytest

import phased_array as pa

K = pa.wavelength_to_k(1.0)


class TestFirstNullBeamwidth:

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_uniform_linear_closed_form(self, n):
        geom = pa.create_rectangular_array(n, 1, 0.5, 0.5)
        angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, np.ones(n), K, n_points=36001)
        expected = 2 * np.rad2deg(np.arcsin(1 / (n * 0.5)))  # nulls at sin(theta) = 1/(N d)
        assert pa.compute_first_null_beamwidth(angles, cut) == pytest.approx(expected, abs=0.01)


class TestPatternSummary:

    def test_uniform_broadside(self):
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        s = pa.pattern_summary(geom.x, geom.y, np.ones(256), K)
        assert (s['theta0_deg'], s['phi0_deg']) == (0.0, 0.0)
        np.testing.assert_allclose(s['hpbw_deg'], 6.36, atol=0.01)
        np.testing.assert_allclose(s['fnbw_deg'], 2 * np.rad2deg(np.arcsin(0.125)), atol=0.01)
        np.testing.assert_allclose(s['sll_dB'], -13.15, atol=0.02)
        assert s['peak_sidelobe_dB'] == pytest.approx(-13.15, abs=0.1)
        assert s['grating_lobes'] == []

    def test_directivity_matches_direct_integration(self):
        geom = pa.create_rectangular_array(10, 10, 0.5, 0.5)
        w = pa.steering_vector(K, geom.x, geom.y, 20, 0)
        hemi = pa.pattern_summary(geom.x, geom.y, w, K, theta0_deg=20, phi0_deg=0)
        sphere = pa.pattern_summary(geom.x, geom.y, w, K, theta0_deg=20, phi0_deg=0, hemisphere=False)
        _, _, theta, phi = pa.create_theta_phi_grid((0, np.pi / 2), (0, 2 * np.pi), 181, 361)
        af = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, w, K)
        assert hemi['directivity_dBi'] == pytest.approx(
            10 * np.log10(pa.compute_directivity(theta, phi, af)), abs=1e-9)
        # Isotropic elements in the xy-plane radiate equally to both sides
        assert hemi['directivity_dBi'] - sphere['directivity_dBi'] == pytest.approx(
            10 * np.log10(2), abs=0.02)

    def test_beam_direction_found(self):
        geom = pa.create_rectangular_array(24, 24, 0.5, 0.5)
        w = pa.steering_vector(K, geom.x, geom.y, 37.3, 51.7)
        s = pa.pattern_summary(geom.x, geom.y, w, K)
        assert s['theta0_deg'] == pytest.approx(37.3, abs=0.02)
        assert s['phi0_deg'] == pytest.approx(51.7, abs=0.02)

    def test_off_axis_beam_hemisphere_sidelobes(self):
        # Separable Taylor -30 dB taper steered off the principal planes:
        # the diagonal cuts sit at -60 dB, the principal-plane sidelobes
        # (found by the hemisphere search) at -30 dB.
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        w = pa.steering_vector(K, geom.x, geom.y, 30, 45) * pa.taylor_taper_2d(16, 16, -30)
        s = pa.pattern_summary(geom.x, geom.y, w, K)
        assert max(s['sll_dB']) < -55
        assert s['peak_sidelobe_dB'] == pytest.approx(-30.0, abs=0.5)

    def test_grating_lobe_reported(self):
        # 0.8 wavelength spacing steered to 40 deg: lobe at u = sin40 - 1/0.8
        geom = pa.create_rectangular_array(12, 12, 0.8, 0.8)
        w = pa.steering_vector(K, geom.x, geom.y, 40, 0)
        s = pa.pattern_summary(geom.x, geom.y, w, K)
        expected_theta = np.rad2deg(np.arcsin(1 / 0.8 - np.sin(np.deg2rad(40))))
        assert len(s['grating_lobes']) == 1
        theta_gl, phi_gl, level = s['grating_lobes'][0]
        assert theta_gl == pytest.approx(expected_theta, abs=1.0)
        assert phi_gl == pytest.approx(180.0, abs=1.0)
        assert level > -1.0

    def test_element_pattern_kwargs(self):
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        w = pa.steering_vector(K, geom.x, geom.y, 40, 0)
        iso = pa.pattern_summary(geom.x, geom.y, w, K, theta0_deg=40, phi0_deg=0)
        cos = pa.pattern_summary(geom.x, geom.y, w, K, theta0_deg=40, phi0_deg=0,
                                 element_pattern_func=pa.element_pattern, cos_exp_theta=1.5)
        assert cos['directivity_dBi'] != pytest.approx(iso['directivity_dBi'], abs=0.1)
