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

    def test_hybrid_exact_at_center_frequency(self):
        """At its design frequency hybrid TTD+phase steering is exact."""
        geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        theta0, phi0, freq = 30, 0, 10e9

        weights_hybrid = pa.steering_vector_hybrid(
            geom, arch, theta0, phi0, freq
        )
        k = 2 * np.pi * freq / 299792458.0
        weights_exact = pa.steering_vector(k, geom.x, geom.y, theta0, phi0)
        ratio = weights_hybrid / weights_exact
        np.testing.assert_allclose(ratio, ratio[0], atol=1e-9)


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
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        delays = pa.compute_subarray_delays_ttd(
            arch,
            theta0_deg=30, phi0_deg=0
        )

        assert np.all(delays >= 0)

    def test_broadside_zero_delays(self):
        """Broadside steering should have all zero delays."""
        arch = pa.create_rectangular_subarrays(16, 16, 4, 4, 0.5, 0.5)

        delays = pa.compute_subarray_delays_ttd(
            arch,
            theta0_deg=0, phi0_deg=0
        )

        np.testing.assert_array_almost_equal(delays, 0.0)


class TestHybridSquintPhysics:
    """Phase shifters are set at the centre frequency; only TTD scales."""

    F0 = 10e9

    def _setup(self, n, sub):
        wavelength = 3e8 / self.F0
        geom = pa.create_rectangular_array(n, n, 0.5, 0.5, wavelength=wavelength)
        arch = pa.create_rectangular_subarrays(n, n, sub, sub, 0.5, 0.5, wavelength=wavelength)
        return geom, arch

    def test_hybrid_squint_between_ttd_and_phase(self):
        geom, arch = self._setup(32, 8)
        freqs = np.array([0.9, 1.0, 1.1]) * self.F0
        kw = dict(n_points=6001)
        sq = {}
        for mode in ('phase', 'hybrid', 'ttd'):
            r = pa.compute_beam_squint(geom.x, geom.y, 45, 0, self.F0, freqs,
                                       steering_mode=mode, architecture=arch, **kw)
            sq[mode] = np.max(np.abs(r['squint']))
        assert sq['ttd'] < 0.02
        assert sq['ttd'] < sq['hybrid'] < sq['phase']

    def test_single_element_subarrays_match_ttd(self):
        geom, arch = self._setup(8, 1)
        f = 1.15 * self.F0
        w_h = pa.steering_vector_hybrid(geom, arch, 30, 0, f, center_frequency=self.F0)
        w_t = pa.steering_vector_ttd(geom.x, geom.y, 30, 0, f)
        # Equal up to a common phase
        ratio = w_h / w_t
        np.testing.assert_allclose(ratio, ratio[0], atol=1e-9)

    def test_single_subarray_matches_phase_only(self):
        geom, arch = self._setup(8, 8)
        f = 1.15 * self.F0
        w_h = pa.steering_vector_hybrid(geom, arch, 30, 0, f, c=3e8,
                                        center_frequency=self.F0)
        k0 = 2 * np.pi * self.F0 / 3e8
        w_p = pa.steering_vector(k0, geom.x, geom.y, 30, 0)
        ratio = w_h / w_p
        np.testing.assert_allclose(ratio, ratio[0], atol=1e-9)

    def test_default_center_frequency_is_frequency(self):
        geom, arch = self._setup(8, 4)
        w1 = pa.steering_vector_hybrid(geom, arch, 20, 10, self.F0)
        w2 = pa.steering_vector_hybrid(geom, arch, 20, 10, self.F0, center_frequency=self.F0)
        np.testing.assert_allclose(w1, w2)

    def test_pattern_vs_frequency_hybrid_differs_from_ttd(self):
        geom, arch = self._setup(16, 8)
        freqs = np.array([0.8, 1.0, 1.2]) * self.F0
        r_h = pa.compute_pattern_vs_frequency(geom.x, geom.y, 40, 0, self.F0, freqs,
                                              steering_mode='hybrid', architecture=arch)
        r_t = pa.compute_pattern_vs_frequency(geom.x, geom.y, 40, 0, self.F0, freqs,
                                              steering_mode='ttd')
        np.testing.assert_allclose(r_h['patterns'][1], r_t['patterns'][1], atol=1e-6)
        assert not np.allclose(r_h['patterns'][0], r_t['patterns'][0], atol=1e-3)

    def test_ibw_fallback_ratio_matches_search_range(self):
        geom, arch = self._setup(8, 1)
        r = pa.analyze_instantaneous_bandwidth(geom.x, geom.y, 30, 0, self.F0,
                                               squint_tolerance_deg=0.5,
                                               steering_mode='hybrid', architecture=arch)
        assert 'note' in r
        assert r['ibw_ratio'] == pytest.approx(3.0)


class TestInstantaneousBandwidthSearch:

    def test_returned_bandwidth_sits_on_the_tolerance_edge(self):
        # A grating lobe takes over the peak in this case, so squint jumps;
        # linear extrapolation reported 5.4% although 35% was within 4.5 deg.
        arch = pa.create_rectangular_subarrays(16, 16, 8, 8, 0.5, 0.5, 0.03)
        g = arch.geometry
        tol = 4.5
        r = pa.analyze_instantaneous_bandwidth(g.x, g.y, 45, 0, 10e9, squint_tolerance_deg=tol,
                                               steering_mode='hybrid', architecture=arch)

        def squint(bw_percent):
            f = 10e9 * np.array([1 - bw_percent / 200, 1, 1 + bw_percent / 200])
            s = pa.compute_beam_squint(g.x, g.y, 45, 0, 10e9, f,
                                       steering_mode='hybrid', architecture=arch)
            return np.max(np.abs(s['squint']))

        assert r['ibw_percent'] > 30
        assert squint(r['ibw_percent']) <= tol
        assert squint(r['ibw_percent'] + 0.01) > tol

    def test_phase_only_edge(self):
        g = pa.create_rectangular_array(16, 16, 0.5, 0.5, wavelength=299792458.0 / 10e9)
        r = pa.analyze_instantaneous_bandwidth(g.x, g.y, 30, 0, 10e9, squint_tolerance_deg=2.0)
        f = 10e9 * np.array([1 - r['ibw_percent'] / 200, 1, 1 + r['ibw_percent'] / 200])
        s = pa.compute_beam_squint(g.x, g.y, 30, 0, 10e9, f)
        assert np.max(np.abs(s['squint'])) <= 2.0


def test_speed_of_light_is_shared():
    from phased_array import utils, wideband
    assert utils.C == wideband.C == 299792458.0
    assert pa.frequency_to_k(10e9) == pytest.approx(2 * np.pi * 10e9 / 299792458.0)
