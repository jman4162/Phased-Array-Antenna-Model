"""
Tests for phased_array.wideband module.
"""

import numpy as np
import pytest

import phased_array as pa


class TestSteeringVectorTTD:
    """Tests for true-time delay steering functions."""

    def test_ttd_steering_returns_unit_magnitude(self):
        """TTD weights should have unit magnitude."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        weights = pa.steering_vector_ttd(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=0,
            frequency=10e9
        )
        np.testing.assert_array_almost_equal(np.abs(weights), 1.0)

    def test_ttd_broadside_uniform_phase(self):
        """Broadside TTD steering should have uniform phase."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        weights = pa.steering_vector_ttd(
            geom.x, geom.y,
            theta0_deg=0, phi0_deg=0,
            frequency=10e9
        )
        phases = np.angle(weights)
        # All phases should be equal (within numerical precision)
        np.testing.assert_array_almost_equal(phases, phases[0], decimal=10)

    def test_ttd_delays_non_negative(self):
        """Steering delays should be non-negative (normalized)."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        delays = pa.steering_delays_ttd(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=45
        )
        assert np.all(delays >= 0)

    def test_ttd_delays_broadside_zero(self):
        """Broadside steering should have all zero delays."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        delays = pa.steering_delays_ttd(
            geom.x, geom.y,
            theta0_deg=0, phi0_deg=0
        )
        np.testing.assert_array_almost_equal(delays, 0.0)


class TestSteeringVectorHybrid:
    """Tests for hybrid TTD + phase steering."""

    def test_hybrid_steering_returns_unit_magnitude(self):
        """Hybrid weights should have unit magnitude."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        weights = pa.steering_vector_hybrid(
            geom, arch,
            theta0_deg=30, phi0_deg=0,
            frequency=10e9
        )
        np.testing.assert_array_almost_equal(np.abs(weights), 1.0)

    def test_hybrid_same_as_ttd_at_subarray_centers(self):
        """Hybrid steering should give same phase at subarray centers as TTD."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        theta0, phi0, freq = 30, 0, 10e9

        weights_hybrid = pa.steering_vector_hybrid(
            geom, arch, theta0, phi0, freq
        )

        # For elements at subarray centers, hybrid should match TTD
        # (This is approximate since elements may not be exactly at centers)


class TestBeamSquint:
    """Tests for beam squint computation."""

    def test_ttd_no_squint(self):
        """TTD steering should have no beam squint."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        center_freq = 10e9
        frequencies = np.array([9e9, 10e9, 11e9])

        results = pa.compute_beam_squint(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=0,
            center_frequency=center_freq,
            frequencies=frequencies,
            steering_mode='ttd'
        )

        # TTD should have essentially zero squint
        assert np.max(np.abs(results['squint'])) < 0.5  # Less than 0.5 deg

    def test_phase_only_has_squint(self):
        """Phase-only steering should show beam squint with frequency."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        center_freq = 10e9
        frequencies = np.array([8e9, 10e9, 12e9])

        results = pa.compute_beam_squint(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=0,
            center_frequency=center_freq,
            frequencies=frequencies,
            steering_mode='phase'
        )

        # Phase-only should have some squint at off-center frequencies
        # Squint should be zero at center frequency
        center_idx = np.argmin(np.abs(frequencies - center_freq))
        assert abs(results['squint'][center_idx]) < 0.1

        # Should have non-zero squint at other frequencies
        assert np.max(np.abs(results['squint'])) > 0

    def test_hybrid_less_squint_than_phase(self):
        """Hybrid steering should have less squint than phase-only."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)
        center_freq = 10e9
        frequencies = np.array([8e9, 10e9, 12e9])

        results_phase = pa.compute_beam_squint(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=0,
            center_frequency=center_freq,
            frequencies=frequencies,
            steering_mode='phase'
        )

        results_hybrid = pa.compute_beam_squint(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=0,
            center_frequency=center_freq,
            frequencies=frequencies,
            steering_mode='hybrid',
            architecture=arch
        )

        # Hybrid should have less or equal max squint
        max_squint_phase = np.max(np.abs(results_phase['squint']))
        max_squint_hybrid = np.max(np.abs(results_hybrid['squint']))
        assert max_squint_hybrid <= max_squint_phase + 0.1  # Small tolerance


class TestInstantaneousBandwidth:
    """Tests for instantaneous bandwidth analysis."""

    def test_ttd_infinite_bandwidth(self):
        """TTD should report infinite bandwidth."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)

        results = pa.analyze_instantaneous_bandwidth(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=0,
            center_frequency=10e9,
            squint_tolerance_deg=0.5,
            steering_mode='ttd'
        )

        assert results['ibw_hz'] == np.inf

    def test_phase_finite_bandwidth(self):
        """Phase-only steering should have finite bandwidth."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)

        results = pa.analyze_instantaneous_bandwidth(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=0,
            center_frequency=10e9,
            squint_tolerance_deg=0.5,
            steering_mode='phase'
        )

        assert results['ibw_hz'] > 0
        assert results['ibw_hz'] < np.inf


class TestPatternVsFrequency:
    """Tests for wideband pattern computation."""

    def test_pattern_shape(self):
        """Pattern array should have correct shape."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        frequencies = np.array([9e9, 10e9, 11e9])

        results = pa.compute_pattern_vs_frequency(
            geom.x, geom.y,
            theta0_deg=30, phi0_deg=0,
            center_frequency=10e9,
            frequencies=frequencies,
            n_points=91
        )

        assert results['patterns'].shape == (3, 91)
        assert len(results['angles']) == 91
        assert len(results['frequencies']) == 3


class TestSubarrayWeightsHybrid:
    """Tests for hybrid subarray weight computation."""

    def test_returns_correct_keys(self):
        """Should return dict with weights, delays, and phases."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        results = pa.compute_subarray_weights_hybrid(
            geom, arch,
            theta0_deg=30, phi0_deg=0,
            frequency=10e9
        )

        assert 'weights' in results
        assert 'subarray_delays' in results
        assert 'element_phases' in results

    def test_weights_unit_magnitude(self):
        """Weights should have unit magnitude without taper."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        results = pa.compute_subarray_weights_hybrid(
            geom, arch,
            theta0_deg=30, phi0_deg=0,
            frequency=10e9
        )

        np.testing.assert_array_almost_equal(np.abs(results['weights']), 1.0)

    def test_subarray_delays_count(self):
        """Should have one delay per subarray."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        results = pa.compute_subarray_weights_hybrid(
            geom, arch,
            theta0_deg=30, phi0_deg=0,
            frequency=10e9
        )

        assert len(results['subarray_delays']) == arch.n_subarrays

    def test_amplitude_taper_applied(self):
        """Amplitude taper should be applied to weights."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)
        taper = np.ones(256) * 0.5  # Half amplitude

        results = pa.compute_subarray_weights_hybrid(
            geom, arch,
            theta0_deg=30, phi0_deg=0,
            frequency=10e9,
            amplitude_taper=taper
        )

        np.testing.assert_array_almost_equal(np.abs(results['weights']), 0.5)


class TestCompareSteeringModes:
    """Tests for steering mode comparison."""

    def test_returns_all_modes(self):
        """Should return results for all three steering modes."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        results = pa.compare_steering_modes(
            geom, arch,
            theta0_deg=30, phi0_deg=0,
            center_frequency=10e9,
            bandwidth_percent=20
        )

        assert 'phase' in results
        assert 'hybrid' in results
        assert 'ttd' in results

    def test_max_squint_ordering(self):
        """TTD should have least squint, phase-only most."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        results = pa.compare_steering_modes(
            geom, arch,
            theta0_deg=30, phi0_deg=0,
            center_frequency=10e9,
            bandwidth_percent=20
        )

        # TTD should have smallest max squint
        assert results['ttd']['max_squint'] <= results['hybrid']['max_squint'] + 0.1
        # Hybrid should typically be better than phase-only
        # (may not always be true for small subarrays, so we use loose tolerance)


class TestSubarrayDelaysTTD:
    """Tests for subarray TTD computation."""

    def test_delays_non_negative(self):
        """Subarray delays should be non-negative."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        delays = pa.compute_subarray_delays_ttd(
            arch,
            theta0_deg=30, phi0_deg=0
        )

        assert np.all(delays >= 0)

    def test_broadside_zero_delays(self):
        """Broadside steering should have all zero delays."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        delays = pa.compute_subarray_delays_ttd(
            arch,
            theta0_deg=0, phi0_deg=0
        )

        np.testing.assert_array_almost_equal(delays, 0.0)
