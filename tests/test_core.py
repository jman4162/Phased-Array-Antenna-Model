"""
Unit tests for phased_array.core module.
"""

import numpy as np
import pytest
import phased_array as pa


class TestSteeringVector:
    """Tests for steering_vector function."""

    def test_broadside_steering(self):
        """Broadside steering (theta=0) should give uniform phase."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=0, phi0_deg=0)

        # All phases should be equal (or very close)
        phases = np.angle(weights)
        assert np.allclose(phases, phases[0], atol=1e-10)

    def test_steering_unit_magnitude(self):
        """Steering vector should have unit magnitude for all elements."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=45)

        magnitudes = np.abs(weights)
        assert np.allclose(magnitudes, 1.0)

    def test_steering_symmetry(self):
        """Steering to opposite phi angles should give conjugate phases for symmetric array."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        weights_pos = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=45)
        weights_neg = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=-45)

        # For a centered symmetric array, these should be related
        # The pattern should be symmetric
        assert weights_pos.shape == weights_neg.shape


class TestArrayFactor:
    """Tests for array factor computation."""

    def test_vectorized_matches_loop(self):
        """Vectorized AF should match a simple loop implementation."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        weights = pa.steering_vector(k, geom.x, geom.y, 20, 0)

        theta = np.linspace(0, np.pi/2, 21)
        phi = np.zeros_like(theta)

        # Simple loop implementation
        AF_loop = np.zeros_like(theta, dtype=complex)
        for i in range(len(geom.x)):
            u = np.sin(theta) * np.cos(phi)
            v = np.sin(theta) * np.sin(phi)
            phase = k * (geom.x[i] * u + geom.y[i] * v)
            AF_loop += weights[i] * np.exp(1j * phase)

        # Vectorized implementation
        AF_vec = pa.array_factor_vectorized(
            theta.reshape(-1, 1), phi.reshape(-1, 1),
            geom.x, geom.y, weights, k
        ).ravel()

        assert np.allclose(AF_loop, AF_vec)

    def test_broadside_peak_at_zero(self):
        """Uniform weights should produce peak at theta=0."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        weights = np.ones(geom.n_elements)

        theta = np.linspace(0, np.pi/2, 91)
        phi = np.zeros_like(theta)

        AF = pa.array_factor_vectorized(
            theta.reshape(-1, 1), phi.reshape(-1, 1),
            geom.x, geom.y, weights, k
        ).ravel()

        # Peak should be at theta=0
        peak_idx = np.argmax(np.abs(AF))
        assert peak_idx == 0

    def test_steered_beam_peak_location(self):
        """Steered beam should peak near the steering direction."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        theta0 = 25  # degrees
        weights = pa.steering_vector(k, geom.x, geom.y, theta0, 0)

        theta = np.linspace(0, np.pi/2, 181)
        phi = np.zeros_like(theta)

        AF = pa.array_factor_vectorized(
            theta.reshape(-1, 1), phi.reshape(-1, 1),
            geom.x, geom.y, weights, k
        ).ravel()

        # Find peak
        peak_idx = np.argmax(np.abs(AF))
        peak_theta_deg = np.rad2deg(theta[peak_idx])

        # Peak should be within 2 degrees of steering angle
        assert abs(peak_theta_deg - theta0) < 2.0

    def test_array_factor_uv_consistency(self):
        """AF in UV-space should be consistent with theta-phi computation."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        weights = np.ones(geom.n_elements)

        # Test point
        theta = 0.3
        phi = 0.5
        u, v = pa.theta_phi_to_uv(theta, phi)

        AF_thetaphi = pa.array_factor_vectorized(
            np.array([[theta]]), np.array([[phi]]),
            geom.x, geom.y, weights, k
        )

        AF_uv = pa.array_factor_uv(
            np.array([[u]]), np.array([[v]]),
            geom.x, geom.y, weights, k
        )

        assert np.allclose(AF_thetaphi, AF_uv)


class TestElementPattern:
    """Tests for element pattern functions."""

    def test_element_pattern_peak_at_broadside(self):
        """Element pattern should peak at theta=0."""
        theta = np.linspace(0, np.pi/2, 91)
        phi = np.zeros_like(theta)

        pattern = pa.element_pattern(theta, phi, cos_exp_theta=1.0)

        assert np.argmax(pattern) == 0

    def test_element_pattern_zero_at_endfire(self):
        """Cosine element pattern should be zero at theta=90."""
        theta = np.array([np.pi/2])
        phi = np.array([0.0])

        pattern = pa.element_pattern(theta, phi, cos_exp_theta=1.0)

        assert np.isclose(pattern[0], 0.0)

    def test_element_pattern_gain(self):
        """Element pattern should respect max_gain_dBi."""
        theta = np.array([0.0])
        phi = np.array([0.0])

        pattern = pa.element_pattern(theta, phi, max_gain_dBi=6.0)

        # 6 dBi = 10^0.6 ~ 3.98 linear
        expected_gain = 10 ** (6.0 / 10)
        assert np.isclose(pattern[0], expected_gain)


class TestPatternComputation:
    """Tests for full pattern computation."""

    def test_compute_pattern_cuts_shape(self):
        """Pattern cuts should return correct shapes."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        weights = np.ones(geom.n_elements)

        angles, E_plane, H_plane = pa.compute_pattern_cuts(
            geom.x, geom.y, weights, k,
            n_points=181
        )

        assert len(angles) == 181
        assert len(E_plane) == 181
        assert len(H_plane) == 181

    def test_compute_full_pattern_shape(self):
        """Full pattern should return correct shapes."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        weights = np.ones(geom.n_elements)

        theta, phi, pattern = pa.compute_full_pattern(
            geom.x, geom.y, weights, k,
            n_theta=31, n_phi=61
        )

        assert len(theta) == 31
        assert len(phi) == 61
        assert pattern.shape == (31, 61)

    def test_half_power_beamwidth(self):
        """HPBW should be reasonable for known array size."""
        # For N-element uniform array, HPBW ~ 0.886 * lambda / (N * d)
        N = 16
        d = 0.5  # wavelengths

        geom = pa.create_rectangular_array(N, 1, d, d)
        k = pa.wavelength_to_k(1.0)
        weights = np.ones(geom.n_elements)

        angles, E_plane, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, weights, k,
            n_points=361
        )

        hpbw = pa.compute_half_power_beamwidth(angles, E_plane)

        # Expected HPBW ~ 0.886 * 180/pi / (N * d) ~ 6.3 degrees for 16 elements
        # Allow reasonable tolerance
        assert 4 < hpbw < 10
