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
