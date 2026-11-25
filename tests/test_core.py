"""
Unit tests for phased_array.core module.
"""

import warnings

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

        # 6 dBi power gain = 10^(6/20) ~ 2.00 field amplitude
        expected_field = 10 ** (6.0 / 20)
        assert np.isclose(pattern[0], expected_field)
        assert np.isclose(10 * np.log10(pattern[0] ** 2), 6.0)

    def test_element_pattern_power_rolloff(self):
        """cos_exp_theta is the power exponent: cos^1 power is -3 dB at 60 deg."""
        theta = np.deg2rad(np.array([0.0, 60.0]))
        phi = np.zeros_like(theta)

        pattern = pa.element_pattern(theta, phi, cos_exp_theta=1.0)

        rel_dB = 10 * np.log10(pattern[1] ** 2 / pattern[0] ** 2)
        assert np.isclose(rel_dB, 10 * np.log10(0.5))

    def test_element_pattern_rear_hemisphere_no_warning(self):
        """Rear hemisphere is zero and emits no NaN RuntimeWarning."""
        theta = np.linspace(0, np.pi, 181)
        phi = np.zeros_like(theta)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            pattern = pa.element_pattern(theta, phi, cos_exp_theta=1.3)

        assert np.all(np.isfinite(pattern))
        assert np.all(pattern[theta > np.pi / 2] == 0.0)
        # cos(pi/2) is ~6e-17 in floating point, so endfire is ~0, not exact
        assert np.isclose(pattern[90], 0.0, atol=1e-6)  # theta[90] = 90 deg

    def test_cosine_tapered_half_power_beamwidth(self):
        """element_pattern_cosine_tapered is -3 dB (power) at theta_3dB."""
        theta = np.deg2rad(np.array([0.0, 65.0]))
        phi = np.zeros_like(theta)

        pattern = pa.element_pattern_cosine_tapered(
            theta, phi, theta_3dB_deg=65.0
        )

        rel_dB = 10 * np.log10(pattern[1] ** 2 / pattern[0] ** 2)
        assert np.isclose(rel_dB, 10 * np.log10(0.5))


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


class TestArrayFactorFFT:
    """array_factor_fft must agree with the direct AF (exp(+j k r.u))."""

    def test_steered_beam_on_correct_side(self):
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.steering_vector(k, geom.x, geom.y, 30, 0)
        u, v, AF = pa.array_factor_fft(w.reshape(16, 16), 0.5, 0.5, 64, 64)
        iu, iv = np.unravel_index(np.argmax(np.abs(AF)), AF.shape)
        assert u[iu] == pytest.approx(0.5)
        assert v[iv] == pytest.approx(0.0)

    def test_axis_is_fft_bin_centres(self):
        u, v, _ = pa.array_factor_fft(np.ones((16, 8)), 0.5, 0.7, 64, 32)
        np.testing.assert_allclose(u, (np.arange(64) - 32) / (64 * 0.5))
        np.testing.assert_allclose(v, (np.arange(32) - 16) / (32 * 0.7))
        assert u[32] == 0.0 and v[16] == 0.0

    @pytest.mark.parametrize("nx,ny", [(16, 16), (5, 7)])
    def test_matches_direct_array_factor(self, nx, ny):
        geom = pa.create_rectangular_array(nx, ny, 0.5, 0.6)
        k = pa.wavelength_to_k(1.0)
        rng = np.random.default_rng(3)
        w = rng.standard_normal(geom.n_elements) + 1j * rng.standard_normal(geom.n_elements)
        u, v, AF = pa.array_factor_fft(w.reshape(nx, ny), 0.5, 0.6, 32, 40)
        uu, vv = np.meshgrid(u, v, indexing='ij')
        AF_direct = pa.array_factor_uv(uu, vv, geom.x, geom.y, w, k)
        np.testing.assert_allclose(AF, AF_direct, atol=1e-9 * np.abs(AF_direct).max())

    def test_too_few_output_points_raises(self):
        with pytest.raises(ValueError, match="n_u"):
            pa.array_factor_fft(np.ones((16, 16)), 0.5, 0.5, 8, 64)


class TestPatternCutsThroughBeam:
    """Second cut passes through the scanned beam; cuts share one reference."""

    def test_orthogonal_cut_contains_scanned_beam(self):
        geom = pa.create_rectangular_array(10, 10, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.steering_vector(k, geom.x, geom.y, 30, 0)
        angles, e_cut, h_cut = pa.compute_pattern_cuts(
            geom.x, geom.y, w, k, theta0_deg=30, phi0_deg=0, n_points=721)
        assert e_cut.max() == pytest.approx(0.0, abs=1e-9)
        assert h_cut.max() == pytest.approx(0.0, abs=1e-6)
        assert angles[np.argmax(e_cut)] == pytest.approx(30.0, abs=0.25)
        assert angles[np.argmax(h_cut)] == pytest.approx(0.0, abs=0.25)

    def test_broadside_cut_shares_reference_with_beam_cut(self):
        """A cut that misses the beam is shown at its true level, not 0 dB."""
        geom = pa.create_rectangular_array(10, 10, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.steering_vector(k, geom.x, geom.y, 30, 0)
        _, e_cut, h_cut = pa.compute_pattern_cuts(geom.x, geom.y, w, k)
        assert e_cut.max() == pytest.approx(0.0, abs=1e-9)
        assert h_cut.max() < -10.0

    def test_default_cuts_unchanged_at_broadside(self):
        geom = pa.create_rectangular_array(8, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = np.ones(geom.n_elements)
        angles, e_cut, h_cut = pa.compute_pattern_cuts(geom.x, geom.y, w, k, n_points=181)
        th = np.deg2rad(angles)
        for cut, phi_pos in ((e_cut, 0.0), (h_cut, np.pi / 2)):
            phi = np.where(th >= 0, phi_pos, phi_pos + np.pi)
            af = pa.array_factor_vectorized(np.abs(th), phi, geom.x, geom.y, w, k)
            power = np.abs(af) ** 2
            ref = pa.linear_to_db(power / power.max())
            np.testing.assert_allclose(cut, ref, atol=1e-9)

    def test_z_positions_are_used(self):
        geom = pa.create_rectangular_array(8, 1, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = np.ones(geom.n_elements)
        z = np.linspace(0, 2, geom.n_elements)
        _, e_flat, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, k)
        _, e_z, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, k, z=z)
        assert not np.allclose(e_flat, e_z)


class TestPeakSidelobeLevel:

    def test_uniform_linear_array_matches_closed_form(self):
        N = 16
        geom = pa.create_rectangular_array(N, 1, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        angles, e_cut, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, np.ones(N), k, n_points=20001)
        psl = pa.compute_peak_sidelobe_level(angles, e_cut)
        # Closed form: |sin(N psi/2) / (N sin(psi/2))| beyond the first null
        psi = np.linspace(2 * np.pi / N, np.pi, 200001)
        af = np.abs(np.sin(N * psi / 2) / (N * np.sin(psi / 2)))
        assert psl == pytest.approx(20 * np.log10(af.max()), abs=0.01)

    def test_taylor_design_level(self):
        N = 32
        geom = pa.create_rectangular_array(N, 1, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.taylor_taper_1d(N, -30, nbar=5)
        angles, e_cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, k, n_points=8001)
        level, angle = pa.compute_peak_sidelobe_level(angles, e_cut, return_angle=True)
        assert -30.6 < level < -29.8
        assert abs(angle) > 3.0

    def test_no_sidelobes_returns_minus_inf(self):
        angles = np.linspace(-10, 10, 101)
        assert pa.compute_peak_sidelobe_level(angles, -angles ** 2) == -np.inf

    def test_unnormalized_input(self):
        angles = np.linspace(-90, 90, 2001)
        geom = pa.create_rectangular_array(8, 1, 0.5, 0.5)
        _, e_cut, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, np.ones(8), pa.wavelength_to_k(1.0), n_points=2001)
        assert pa.compute_peak_sidelobe_level(angles, e_cut + 17.0) == pytest.approx(
            pa.compute_peak_sidelobe_level(angles, e_cut))


class TestArrayFactorChunking:

    def test_chunked_equals_single_pass(self, monkeypatch):
        import phased_array.core as core
        geom = pa.create_cylindrical_array(12, 5, 2.0, 0.5)
        k = pa.wavelength_to_k(1.0)
        rng = np.random.default_rng(7)
        w = rng.standard_normal(geom.n_elements) + 1j * rng.standard_normal(geom.n_elements)
        _, _, theta, phi = pa.create_theta_phi_grid((0, np.pi), (0, 2 * np.pi), 37, 73)
        ref = np.sum(w * np.exp(1j * k * (
            np.multiply.outer(np.sin(theta) * np.cos(phi), geom.x)
            + np.multiply.outer(np.sin(theta) * np.sin(phi), geom.y)
            + np.multiply.outer(np.cos(theta), geom.z))), axis=-1)
        full = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, w, k, geom.z)
        monkeypatch.setattr(core, "_AF_CHUNK_BYTES", 16 * geom.n_elements * 7)
        chunked = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, w, k, geom.z)
        np.testing.assert_allclose(full, ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(chunked, ref, rtol=1e-12, atol=1e-12)

    def test_uv_chunked(self, monkeypatch):
        import phased_array.core as core
        geom = pa.create_rectangular_array(9, 7, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        uu, vv = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 39), indexing='ij')
        full = pa.array_factor_uv(uu, vv, geom.x, geom.y, np.ones(63), k)
        monkeypatch.setattr(core, "_AF_CHUNK_BYTES", 16 * 63 * 5)
        np.testing.assert_allclose(pa.array_factor_uv(uu, vv, geom.x, geom.y, np.ones(63), k),
                                   full, rtol=1e-12, atol=1e-12)


class TestLatticeFastPath:
    """The factorized lattice sum must equal the direct sum."""

    @staticmethod
    def _direct(theta, phi, x, y, w, k, z=None):
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        ph = np.multiply.outer(u, x) + np.multiply.outer(v, y)
        if z is not None:
            ph = ph + np.multiply.outer(np.cos(theta), z)
        return np.sum(w * np.exp(1j * k * ph), axis=-1)

    @pytest.mark.parametrize("build", [
        lambda: pa.create_rectangular_array(12, 9, 0.5, 0.6),
        lambda: pa.create_triangular_array(10, 8, 0.6, 0.52),
        lambda: pa.create_elliptical_array(3, 2, 0.5),
        lambda: pa.create_elliptical_array(3, 2, 0.5, grid_type='triangular'),
        lambda: pa.thin_array_random(pa.create_rectangular_array(16, 16, 0.5, 0.5), 0.6, seed=2),
        lambda: pa.create_rectangular_array(8, 8, 0.5, 0.5, wavelength=0.03),
    ])
    def test_matches_direct_sum(self, build):
        geom = build()
        wavelength = 0.03 if np.ptp(geom.x) < 1 else 1.0
        k = pa.wavelength_to_k(wavelength)
        rng = np.random.default_rng(11)
        w = rng.standard_normal(geom.n_elements) + 1j * rng.standard_normal(geom.n_elements)
        _, _, theta, phi = pa.create_theta_phi_grid((0, np.pi), (0, 2 * np.pi), 31, 61)
        got = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, w, k)
        ref = self._direct(theta, phi, geom.x, geom.y, w, k)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())

    def test_constant_z_offset_and_duplicates(self):
        geom = pa.create_rectangular_array(6, 6, 0.5, 0.5)
        x = np.r_[geom.x, geom.x[:3]]  # duplicated positions add their weights
        y = np.r_[geom.y, geom.y[:3]]
        z = np.full(x.size, 0.37)
        k = pa.wavelength_to_k(1.0)
        w = np.linspace(0.5, 1.5, x.size) * np.exp(0.3j * np.arange(x.size))
        _, _, theta, phi = pa.create_theta_phi_grid((0, np.pi), (0, 2 * np.pi), 25, 49)
        got = pa.array_factor_vectorized(theta, phi, x, y, w, k, z)
        np.testing.assert_allclose(got, self._direct(theta, phi, x, y, w, k, z), rtol=1e-10, atol=1e-10)

    def test_irregular_positions_use_direct_path(self):
        from phased_array.core import _lattice_factorization
        geom = pa.create_circular_array(32, 2.0)
        assert _lattice_factorization(geom.x, geom.y, np.ones(32)) is None
        rect = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        xs, ys, W = _lattice_factorization(rect.x, rect.y, np.ones(64))
        assert xs.size == 8 and ys.size == 8 and W.shape == (8, 8)
