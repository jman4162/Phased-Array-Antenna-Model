"""
Unit tests for phased_array.beamforming module.
"""

import numpy as np
import pytest
import phased_array as pa


class TestAmplitudeTapers:
    """Tests for amplitude taper functions."""

    def test_taylor_taper_shape(self):
        """Taylor taper should have correct shape."""
        taper = pa.taylor_taper_1d(16, sidelobe_dB=-30)
        assert len(taper) == 16

    def test_taylor_taper_2d_shape(self):
        """2D Taylor taper should have correct shape."""
        taper = pa.taylor_taper_2d(8, 6, sidelobe_dB=-30)
        assert len(taper) == 48

    def test_taylor_taper_symmetry(self):
        """Taylor taper should be symmetric."""
        taper = pa.taylor_taper_1d(16, sidelobe_dB=-30)
        assert np.allclose(taper, taper[::-1])

    def test_taylor_taper_peak_at_center(self):
        """Taylor taper should peak at center."""
        taper = pa.taylor_taper_1d(17, sidelobe_dB=-30)
        center_idx = len(taper) // 2
        assert np.argmax(taper) == center_idx

    def test_chebyshev_taper_shape(self):
        """Chebyshev taper should have correct shape."""
        taper = pa.chebyshev_taper_1d(16, sidelobe_dB=-30)
        assert len(taper) == 16

    def test_chebyshev_taper_symmetry(self):
        """Chebyshev taper should be symmetric."""
        taper = pa.chebyshev_taper_1d(16, sidelobe_dB=-30)
        assert np.allclose(taper, taper[::-1])

    def test_hamming_taper_shape(self):
        """Hamming taper should have correct shape."""
        taper = pa.hamming_taper_2d(8, 8)
        assert len(taper) == 64

    def test_uniform_taper_efficiency(self):
        """Uniform taper should have 100% efficiency."""
        taper = np.ones(16)
        efficiency = pa.compute_taper_efficiency(taper)
        assert np.isclose(efficiency, 1.0)

    def test_tapered_efficiency_less_than_unity(self):
        """Tapered windows should have efficiency < 100%."""
        taper = pa.taylor_taper_1d(16, sidelobe_dB=-30)
        efficiency = pa.compute_taper_efficiency(taper)
        assert 0 < efficiency < 1.0

    def test_taper_efficiency_range(self):
        """Taper efficiency should be in valid range."""
        for taper_func in [pa.hamming_taper_1d, pa.hanning_taper_1d, pa.cosine_taper_1d]:
            taper = taper_func(16)
            efficiency = pa.compute_taper_efficiency(taper)
            assert 0.3 < efficiency < 1.0  # Typical range

    def test_directivity_loss_negative_or_zero(self):
        """Directivity loss should be <= 0 dB."""
        taper = pa.taylor_taper_1d(16, sidelobe_dB=-30)
        loss = pa.compute_taper_directivity_loss(taper)
        assert loss <= 0

    def test_gaussian_taper(self):
        """Gaussian taper should peak at center and decay."""
        taper = pa.gaussian_taper_1d(17, sigma=0.4)
        center_idx = len(taper) // 2

        assert np.argmax(taper) == center_idx
        assert taper[0] < taper[center_idx]
        assert taper[-1] < taper[center_idx]


class TestNullSteering:
    """Tests for null steering functions."""

    def test_null_depth(self):
        """Null steering should create deep null at specified direction."""
        geom = pa.create_rectangular_array(12, 12, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        null_dir = [(25, 0)]  # theta=25 deg, phi=0

        weights = pa.null_steering_projection(
            geom, k,
            theta_main_deg=0, phi_main_deg=0,
            null_directions=null_dir
        )

        depth = pa.compute_null_depth(weights, geom, k, 25, 0, 0, 0)

        # Null should be at least 20 dB below main beam
        assert depth < -20

    def test_null_steering_maintains_main_beam(self):
        """Null steering should maintain main beam direction."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        weights = pa.null_steering_projection(
            geom, k,
            theta_main_deg=0, phi_main_deg=0,
            null_directions=[(30, 0)]
        )

        # Check main beam is still near broadside
        theta = np.linspace(0, np.pi/2, 91)
        phi = np.zeros_like(theta)

        AF = pa.array_factor_vectorized(
            theta.reshape(-1, 1), phi.reshape(-1, 1),
            geom.x, geom.y, weights, k
        ).ravel()

        peak_idx = np.argmax(np.abs(AF))
        peak_theta = np.rad2deg(theta[peak_idx])

        # Peak should be within 5 degrees of broadside
        assert peak_theta < 5

    def test_lcmv_unity_constraint(self):
        """LCMV with unity gain constraint should achieve unity gain."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        # Unity gain at broadside
        constraints = [(0, 0, 1.0 + 0j)]

        weights = pa.null_steering_lcmv(geom, k, constraints)

        # Compute response at constraint direction
        theta = np.array([[0.0]])
        phi = np.array([[0.0]])
        AF = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, weights, k)

        # Should have approximately unity response
        assert np.isclose(np.abs(AF.item()), 1.0, atol=0.1)


class TestMultiBeam:
    """Tests for multiple beam functions."""

    def test_multi_beam_superposition(self):
        """Superposition multi-beam should create multiple peaks."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        beam_dirs = [(0, 0), (30, 0)]

        weights = pa.multi_beam_weights_superposition(geom, k, beam_dirs)

        # Compute pattern
        theta = np.linspace(0, np.pi/2, 91)
        phi = np.zeros_like(theta)

        AF = pa.array_factor_vectorized(
            theta.reshape(-1, 1), phi.reshape(-1, 1),
            geom.x, geom.y, weights, k
        ).ravel()

        pattern_dB = pa.linear_to_db(np.abs(AF)**2)
        pattern_dB -= np.max(pattern_dB)

        # Should have two peaks above -6 dB
        peaks_above_6dB = np.sum(pattern_dB > -6)
        assert peaks_above_6dB > 2

    def test_multi_beam_orthogonal_returns_list(self):
        """Orthogonal multi-beam should return list of weight vectors."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        beam_dirs = [(0, 0), (20, 0), (20, 90)]

        weight_list = pa.multi_beam_weights_orthogonal(geom, k, beam_dirs)

        assert len(weight_list) == 3
        for weights in weight_list:
            assert len(weights) == geom.n_elements

    def test_monopulse_sum_pattern(self):
        """Monopulse sum pattern should be normal steering."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        weights_sum = pa.monopulse_weights(geom, k, 0, 0, mode='sum')
        weights_steer = pa.steering_vector(k, geom.x, geom.y, 0, 0)

        assert np.allclose(weights_sum, weights_steer)

    def test_monopulse_delta_has_null_on_axis(self):
        """Monopulse delta pattern should have null on axis."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        weights_delta = pa.monopulse_weights(geom, k, 0, 0, mode='delta_az')

        # Response on axis
        theta = np.array([[0.0]])
        phi = np.array([[0.0]])
        AF = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, weights_delta, k)

        # Should be near zero on axis
        assert np.abs(AF.item()) < 0.1 * geom.n_elements
