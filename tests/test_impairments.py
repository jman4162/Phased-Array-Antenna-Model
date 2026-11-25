"""
Unit tests for phased_array.impairments module.
"""

import numpy as np
import pytest
import phased_array as pa


class TestMutualCoupling:
    """Tests for mutual coupling functions."""

    def test_coupling_matrix_diagonal(self):
        """Coupling matrix should have ones on diagonal."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        C = pa.mutual_coupling_matrix_theoretical(geom, k)

        diagonal = np.diag(C)
        assert np.allclose(diagonal, 1.0)

    def test_coupling_matrix_symmetric(self):
        """Coupling matrix should be symmetric for symmetric array."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_model='sinc')

        # Should be approximately symmetric
        assert np.allclose(C, C.T, atol=1e-10)

    def test_coupling_decreases_with_distance(self):
        """Coupling should decrease with element separation."""
        geom = pa.create_rectangular_array(4, 1, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)

        C = pa.mutual_coupling_matrix_theoretical(
            geom, k,
            coupling_model='exponential',
            coupling_coeff=0.3
        )

        # Coupling between adjacent elements should be stronger than distant
        coupling_adjacent = np.abs(C[0, 1])
        coupling_distant = np.abs(C[0, -1])

        assert coupling_adjacent > coupling_distant

    def test_apply_coupling_transmit(self):
        """Apply coupling should modify weights."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        weights = np.ones(geom.n_elements, dtype=complex)

        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.3)
        weights_coupled = pa.apply_mutual_coupling(weights, C, mode='transmit')

        # Weights should be modified
        assert not np.allclose(weights, weights_coupled)

    def test_coupling_compensation(self):
        """Compensation should approximately invert coupling effect."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        weights = np.ones(geom.n_elements, dtype=complex)

        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)

        # Compensate, then apply coupling
        weights_comp = pa.apply_mutual_coupling(weights, C, mode='compensate')
        weights_final = pa.apply_mutual_coupling(weights_comp, C, mode='transmit')

        # Should approximately recover original
        assert np.allclose(weights, weights_final, atol=1e-6)


class TestPhaseQuantization:
    """Tests for phase quantization functions."""

    def test_quantize_preserves_magnitude(self):
        """Phase quantization should preserve magnitude."""
        weights = np.array([1.0, 0.5, 2.0]) * np.exp(1j * np.array([0.1, 0.5, 1.2]))

        weights_q = pa.quantize_phase(weights, n_bits=4)

        assert np.allclose(np.abs(weights), np.abs(weights_q))

    def test_quantize_discrete_phases(self):
        """Quantized phases should be discrete levels."""
        n_bits = 3
        n_levels = 2 ** n_bits
        phase_step = 2 * np.pi / n_levels

        weights = np.exp(1j * np.linspace(0, 2*np.pi, 100))
        weights_q = pa.quantize_phase(weights, n_bits)

        phases = np.angle(weights_q)
        # All phases should be at quantization levels
        phase_levels = np.round(phases / phase_step) * phase_step
        assert np.allclose(phases, phase_levels, atol=1e-10)

    def test_rms_error_decreases_with_bits(self):
        """RMS error should decrease with more bits."""
        errors = [pa.quantization_rms_error(n) for n in [2, 3, 4, 5, 6]]

        # Each should be smaller than previous
        for i in range(1, len(errors)):
            assert errors[i] < errors[i-1]

    def test_rms_error_3bit(self):
        """3-bit quantization should have ~13 degree RMS error."""
        error = pa.quantization_rms_error(3)
        # 360/8 / sqrt(12) = 45 / 3.46 = 13 deg
        assert 12 < error < 14

    def test_analyze_quantization_returns_dict(self):
        """analyze_quantization_effect should return expected keys."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)

        results = pa.analyze_quantization_effect(weights, geom, k, n_bits=4)

        assert 'theta_deg' in results
        assert 'pattern_ideal_dB' in results
        assert 'pattern_quantized_dB' in results
        assert 'rms_error_deg' in results


class TestElementFailures:
    """Tests for element failure simulation."""

    def test_failure_off_mode(self):
        """'off' failure mode should zero out elements."""
        weights = np.ones(100, dtype=complex)

        weights_fail, mask = pa.simulate_element_failures(
            weights, failure_rate=0.2, mode='off', seed=42
        )

        # Failed elements should be zero
        assert np.allclose(weights_fail[mask], 0.0)
        # Non-failed elements should be unchanged
        assert np.allclose(weights_fail[~mask], weights[~mask])

    def test_failure_rate_approximate(self):
        """Failure rate should approximately match requested rate."""
        weights = np.ones(1000, dtype=complex)
        rate = 0.1

        _, mask = pa.simulate_element_failures(
            weights, failure_rate=rate, mode='off', seed=42
        )

        actual_rate = np.mean(mask)
        assert abs(actual_rate - rate) < 0.05  # Within 5%

    def test_failure_reproducible(self):
        """Same seed should give same failures."""
        weights = np.ones(100, dtype=complex)

        w1, m1 = pa.simulate_element_failures(weights, 0.2, 'off', seed=123)
        w2, m2 = pa.simulate_element_failures(weights, 0.2, 'off', seed=123)

        assert np.array_equal(m1, m2)
        assert np.allclose(w1, w2)

    def test_stuck_mode_random_phase(self):
        """'stuck' mode should have random phases but same magnitude."""
        weights = 2.0 * np.ones(100, dtype=complex)

        weights_fail, mask = pa.simulate_element_failures(
            weights, failure_rate=0.5, mode='stuck', seed=42
        )

        if np.any(mask):
            # Magnitudes should be preserved
            assert np.allclose(np.abs(weights_fail[mask]), 2.0)
            # Phases should vary (not all the same)
            phases = np.angle(weights_fail[mask])
            assert np.std(phases) > 0.1


class TestScanBlindness:
    """Tests for scan blindness functions."""

    def test_surface_wave_angle_range(self):
        """Scan blindness angles should be in valid range."""
        theta_E, theta_H = pa.surface_wave_scan_angle(
            dx=0.5, dy=0.5,
            substrate_er=4.0,
            substrate_h=0.1
        )

        assert 0 <= theta_E <= 90
        assert 0 <= theta_H <= 90

    def test_scan_blindness_model_returns_valid_factor(self):
        """Scan blindness model should return factor between 0 and 1."""
        theta = np.linspace(0, np.pi/2, 91)
        phi = np.zeros_like(theta)

        factor = pa.scan_blindness_model(
            theta, phi,
            theta_blind=60,
            null_width_deg=5.0,
            null_depth_dB=-30
        )

        assert np.all(factor >= 0)
        assert np.all(factor <= 1)

    def test_scan_blindness_null_at_blind_angle(self):
        """Scan blindness should create null near blind angle."""
        theta_blind = 50  # degrees
        theta = np.array([np.deg2rad(theta_blind)])
        phi = np.array([0.0])

        factor = pa.scan_blindness_model(
            theta, phi,
            theta_blind=theta_blind,
            null_width_deg=5.0,
            null_depth_dB=-30
        )

        # Factor should be low at blind angle
        assert factor[0] < 0.1

    def test_apply_scan_blindness(self):
        """apply_scan_blindness should reduce pattern at blind angle."""
        theta = np.linspace(0, np.pi/2, 91)
        phi = np.zeros_like(theta)
        pattern = np.ones_like(theta, dtype=complex)

        pattern_blind = pa.apply_scan_blindness(
            pattern, theta, phi,
            theta_blind_list=[45],
            null_depth_dB=-20
        )

        # Pattern at 45 deg should be reduced
        idx_45 = 45  # approximately
        assert np.abs(pattern_blind[idx_45]) < np.abs(pattern[idx_45])


class TestActiveImpedance:
    """Tests for active impedance and VSWR functions."""

    def test_active_reflection_coefficient_diagonal(self):
        """With identity coupling matrix, gamma should be 0."""
        n = 16
        C = np.eye(n, dtype=complex)
        weights = np.ones(n, dtype=complex)

        gamma = pa.active_reflection_coefficient(C, weights, element_idx=0)

        assert np.isclose(np.abs(gamma), 0.0, atol=1e-10)

    def test_active_reflection_coefficient_reasonable(self):
        """Active reflection should be reasonable for typical coupling."""
        geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
        k = pa.wavelength_to_k(1.0)
        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)
        weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)

        gamma = pa.active_reflection_coefficient(C, weights, element_idx=0)

        # Should be a reasonable value (not infinite, not >1 for passive array)
        assert np.isfinite(gamma)
        assert np.abs(gamma) < 2.0  # Allow some margin

    def test_active_impedance_positive_resistance(self):
        """Active impedance should have positive real part."""
        geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
        k = pa.wavelength_to_k(1.0)
        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)
        weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)

        Z = pa.active_impedance(C, weights, element_idx=0)

        # Real part should be positive (passive network)
        assert np.real(Z) > 0

    def test_vswr_vs_scan_shape(self):
        """VSWR vs scan should return correct shapes."""
        geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
        k = pa.wavelength_to_k(1.0)
        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)

        theta_deg, vswr_all, vswr_max = pa.vswr_vs_scan(
            geom, C, k, theta_range=(0, 45), n_angles=10
        )

        assert len(theta_deg) == 10
        assert vswr_all.shape == (10, geom.n_elements)
        assert len(vswr_max) == 10

    def test_vswr_minimum_value(self):
        """VSWR should always be >= 1."""
        geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
        k = pa.wavelength_to_k(1.0)
        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.1)

        _, vswr_all, vswr_max = pa.vswr_vs_scan(
            geom, C, k, theta_range=(0, 30), n_angles=5
        )

        # Filter out infinities for comparison
        finite_vswr = vswr_all[np.isfinite(vswr_all)]
        if len(finite_vswr) > 0:
            assert np.all(finite_vswr >= 1.0 - 1e-10)

    def test_mismatch_loss_perfect_match(self):
        """Perfect match should have zero mismatch loss."""
        loss = pa.mismatch_loss(0.0)
        assert np.isclose(loss, 0.0)

    def test_mismatch_loss_typical(self):
        """Typical VSWR should give reasonable loss."""
        # VSWR = 2:1 corresponds to gamma = 0.333
        gamma = 0.333
        loss = pa.mismatch_loss(gamma)

        # Should be between -1 and 0 dB
        assert -1.0 < loss < 0.0

    def test_mismatch_loss_high_reflection(self):
        """High reflection should give large loss."""
        gamma = 0.9
        loss = pa.mismatch_loss(gamma)

        # Should be significant negative value
        assert loss < -5.0

    def test_active_scan_impedance_matrix_shape(self):
        """Should return impedance for all elements."""
        geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
        k = pa.wavelength_to_k(1.0)
        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)

        Z = pa.active_scan_impedance_matrix(geom, C, k, theta_deg=30, phi_deg=0)

        assert Z.shape == (geom.n_elements,)

    def test_active_scan_impedance_matrix_positive_resistance(self):
        """All elements should have positive resistance."""
        geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
        k = pa.wavelength_to_k(1.0)
        C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)

        Z = pa.active_scan_impedance_matrix(geom, C, k, theta_deg=15, phi_deg=0)

        # All finite values should have positive real part
        finite_Z = Z[np.isfinite(Z)]
        assert np.all(np.real(finite_Z) > 0)


class TestSurfaceWaveBlindness:
    """Blind angle from the grounded-slab TM0 surface wave."""

    def test_pozar_schaubert_printed_dipole_benchmark(self):
        # er = 12.8, h = 0.06 lambda, d = 0.5 lambda: E-plane blindness near
        # 46 deg (Pozar & Schaubert, IEEE Trans. AP-32, 1984).
        theta_e, theta_h = pa.surface_wave_scan_angle(0.5, 0.5, 12.8, 0.06)
        assert theta_e == pytest.approx(45.6, abs=1.0)
        assert theta_h == pytest.approx(theta_e)

    def test_depends_on_substrate_height(self):
        angles = [pa.surface_wave_scan_angle(0.5, 0.5, 4.0, h)[0]
                  for h in (0.01, 0.05, 0.1, 0.2)]
        assert np.all(np.diff(angles) < 0)  # thicker slab -> blindness closer to broadside
        assert angles[0] > 80 and angles[-1] > 5

    def test_thin_slab_limit(self):
        # Thin slab: beta/k0 ~ 1 + 0.5 * (k0 h (er-1)/er)^2
        from phased_array.impairments import _tm0_surface_wave_beta
        er, h = 4.0, 0.005
        approx = 1 + 0.5 * (2 * np.pi * h * (er - 1) / er) ** 2
        assert _tm0_surface_wave_beta(er, h) == pytest.approx(approx, rel=1e-5)

    def test_dispersion_residual(self):
        from phased_array.impairments import _tm0_surface_wave_beta
        for er, h in [(2.2, 0.03), (4.0, 0.15), (10.2, 0.1), (12.8, 0.25)]:
            b = _tm0_surface_wave_beta(er, h)
            assert 1 < b < np.sqrt(er)
            kd = np.sqrt(er - b * b)
            resid = kd * np.tan(2 * np.pi * h * kd) - er * np.sqrt(b * b - 1)
            assert abs(resid) < 1e-8 * er

    def test_no_blindness_in_visible_space_returns_90(self):
        # Small spacing: |1/d - beta| > 1, no visible-space blind angle
        theta_e, _ = pa.surface_wave_scan_angle(0.3, 0.3, 2.2, 0.02)
        assert theta_e == 90.0


def _random_passive_s(n, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    S = (A + A.T) / 2  # reciprocal
    return scale * S / np.linalg.norm(S, 2)


class TestActiveReflectionFormulation:
    """Active Gamma must equal (S w)_n / w_n for either coupling form."""

    def test_impedance_form_default_matches_s_parameters(self):
        S = _random_passive_s(4)
        C = pa.mutual_coupling_matrix_measured(S)  # impedance form by default
        w = np.exp(1j * np.array([0.0, 0.7, 1.9, -0.4]))
        for n in range(4):
            expected = (S @ w)[n] / w[n]
            assert pa.active_reflection_coefficient(C, w, n) == pytest.approx(expected)

    def test_voltage_form(self):
        S = _random_passive_s(5, seed=2)
        C = pa.mutual_coupling_matrix_measured(S, formulation='voltage')
        w = np.ones(5, dtype=complex)
        for n in range(5):
            expected = (S @ w)[n] / w[n]
            got = pa.active_reflection_coefficient(C, w, n, formulation='voltage')
            assert got == pytest.approx(expected)

    def test_vswr_and_impedance_consistent_with_gamma(self):
        geom = pa.create_rectangular_array(3, 3, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        S = _random_passive_s(9, seed=5, scale=0.3)
        C = pa.mutual_coupling_matrix_measured(S)
        theta, vswr_all, _ = pa.vswr_vs_scan(geom, C, k, (0, 40), 5)
        Z = pa.active_scan_impedance_matrix(geom, C, k, 40, 0, Z0=50.0)
        w = pa.steering_vector(k, geom.x, geom.y, 40, 0, geom.z)
        gamma = (S @ w) / w
        np.testing.assert_allclose(vswr_all[-1], (1 + abs(gamma)) / (1 - abs(gamma)))
        np.testing.assert_allclose(Z, 50 * (1 + gamma) / (1 - gamma))

    def test_unknown_formulation_raises(self):
        with pytest.raises(ValueError, match="formulation"):
            pa.active_reflection_coefficient(np.eye(2), np.ones(2), 0, formulation='bogus')


class TestScanLossReference:

    def test_tapered_broadside_is_zero(self):
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.taylor_taper_2d(16, 16, -35) * pa.steering_vector(k, geom.x, geom.y, 0, 0)
        assert pa.compute_scan_loss(geom, w, k, 0, 0) == pytest.approx(0.0, abs=1e-9)

    def test_element_pattern_roll_off(self):
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.steering_vector(k, geom.x, geom.y, 60, 0) * pa.taylor_taper_2d(8, 8, -30)
        loss = pa.compute_scan_loss(geom, w, k, 60, 0, element_pattern_func=pa.element_pattern)
        assert loss == pytest.approx(10 * np.log10(np.cos(np.deg2rad(60))), abs=1e-9)

    def test_conformal_uses_z(self):
        geom = pa.create_cylindrical_array(16, 4, 3.0, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.steering_vector(k, geom.x, geom.y, 0, 0, geom.z)
        assert pa.compute_scan_loss(geom, w, k, 0, 0) == pytest.approx(0.0, abs=1e-9)


class TestGracefulDegradationSidelobes:

    def test_reference_sidelobe_is_first_sidelobe(self):
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        r = pa.analyze_graceful_degradation(np.ones(256), geom, k, [0.0], n_trials=2)
        assert r['sidelobe_ref_dB'] == pytest.approx(-13.15, abs=0.1)
        assert r['sidelobe_increase_mean_dB'][0] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("theta0", [0, 30])
    def test_failures_raise_low_sidelobes(self, theta0):
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.taylor_taper_2d(16, 16, -35) * pa.steering_vector(k, geom.x, geom.y, theta0, 0)
        np.random.seed(0)
        r = pa.analyze_graceful_degradation(w, geom, k, [0.0, 0.3], n_trials=10)
        assert r['sidelobe_increase_mean_dB'][1] > 5.0
        assert r['gain_loss_mean_dB'][1] > 1.0


class TestQuantizationSidelobeModels:

    def test_uncorrelated_average_sidelobe_level(self):
        # Haupt eq. 2.119 with sigma = 2 pi 2^-B / sqrt(12): N=100 uniform
        assert pa.quantization_sidelobe_level(3, 100) == pytest.approx(-32.78, abs=0.02)
        assert pa.quantization_sidelobe_level(4, 100) == pytest.approx(-38.88, abs=0.02)
        tapered = pa.quantization_sidelobe_level(4, 100, taper_efficiency=0.8)
        assert tapered == pytest.approx(-38.88 + 10 * np.log10(1 / 0.8), abs=0.02)

    def test_quantization_lobes_match_simulation(self):
        N, us, B = 512, 0.02, 3
        geom = pa.create_rectangular_array(N, 1, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.quantize_phase(pa.steering_vector(k, geom.x, geom.y,
                                                 np.rad2deg(np.arcsin(us)), 0), B)
        u_pos, levels = pa.quantization_lobes(B, us, max_order=2)
        np.testing.assert_allclose(sorted(u_pos), sorted([us * (1 + m * 8) for m in (-2, -1, 1, 2)]))
        peak = abs(np.sum(w * np.exp(1j * k * geom.x * us)))
        for u_m, lvl in zip(u_pos, levels):
            u = np.linspace(u_m - 4 / N, u_m + 4 / N, 401)
            af = np.abs(np.exp(1j * k * np.outer(u, geom.x)) @ w).max()
            assert 20 * np.log10(af / peak) == pytest.approx(lvl, abs=0.6)

    def test_old_function_deprecated(self):
        with pytest.warns(DeprecationWarning):
            pa.quantization_sidelobe_increase(4)


class TestQuantizationAndCouplingEdgeCases:

    def test_analyze_quantization_effect_uses_z(self):
        geom = pa.create_cylindrical_array(16, 2, 2.0, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.steering_vector(k, geom.x, geom.y, 0, 0, geom.z)
        r = pa.analyze_quantization_effect(w, geom, k, n_bits=8)
        # With z included the ideal pattern peaks at broadside
        assert r['theta_deg'][np.argmax(r['pattern_ideal_dB'])] == pytest.approx(0.0, abs=0.5)

    def test_unknown_coupling_model_raises(self):
        geom = pa.create_rectangular_array(2, 2, 0.5, 0.5)
        with pytest.raises(ValueError, match="coupling_model"):
            pa.mutual_coupling_matrix_theoretical(geom, 2 * np.pi, coupling_model='bogus')


class TestSurfaceWaveThickSubstrates:
    """TM0 solve stays on its branch for substrates of a wavelength or more."""

    @pytest.mark.parametrize("er, h", [(10, 3.0), (12.8, 3.0), (50, 1.0),
                                       (25, 1.2), (12.8, 2.0), (4, 50.0)])
    def test_root_in_range_and_satisfies_dispersion(self, er, h):
        from phased_array.impairments import _tm0_surface_wave_beta
        b = _tm0_surface_wave_beta(er, h)
        assert 1 < b < np.sqrt(er)
        kd = np.sqrt(er - b * b)
        x = 2 * np.pi * h * kd
        assert 0 < x < np.pi / 2
        # cos-multiplied form, well conditioned near the tan pole
        assert abs(kd * np.sin(x) - er * np.sqrt(b * b - 1) * np.cos(x)) < 1e-9 * er

    @pytest.mark.parametrize("er", [2.2, 4.0, 12.8, 50.0])
    def test_monotone_from_thin_to_thick(self, er):
        from phased_array.impairments import _tm0_surface_wave_beta
        b = np.array([_tm0_surface_wave_beta(er, h) for h in np.geomspace(1e-7, 50, 300)])
        assert np.all(np.diff(b) >= -1e-12)
        assert np.all((b >= 1) & (b <= np.sqrt(er)))
        assert b[-1] == pytest.approx(np.sqrt(er), rel=1e-5)


class TestGracefulDegradationOffPlaneBeam:

    def test_cut_passes_through_diagonal_beam(self):
        # Separable uniform array steered to (45, 45): the cut in the
        # phi = 45 plane has first sidelobes at 2 x -13.25 = -26.5 dB.
        geom = pa.create_rectangular_array(40, 40, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.steering_vector(k, geom.x, geom.y, 45, 45)
        r = pa.analyze_graceful_degradation(w, geom, k, [0.0], n_trials=1)
        assert r['sidelobe_ref_dB'] == pytest.approx(-26.5, abs=0.3)
        assert r['gain_loss_mean_dB'][0] == pytest.approx(0.0, abs=1e-9)

    def test_beam_direction_refined(self):
        from phased_array.impairments import _beam_direction
        geom = pa.create_rectangular_array(40, 40, 0.5, 0.5)
        k = pa.wavelength_to_k(1.0)
        w = pa.steering_vector(k, geom.x, geom.y, 37.3, 51.7)
        theta, phi, _ = _beam_direction(geom.x, geom.y, geom.z, w, k)
        assert np.rad2deg(theta) == pytest.approx(37.3, abs=0.01)
        assert np.rad2deg(phi) == pytest.approx(51.7, abs=0.01)
