"""
Unit tests for phased_array.polarization module.
"""

import numpy as np
import pytest
import phased_array as pa


class TestJonesVector:
    """Tests for Jones vector creation."""

    def test_horizontal_polarization(self):
        """Horizontal polarization should have Ex=1, Ey=0."""
        j = pa.jones_vector(1.0, 0.0)
        assert np.allclose(j, [1, 0])

    def test_vertical_polarization(self):
        """Vertical polarization should have Ex=0, Ey=1."""
        j = pa.jones_vector(0.0, 1.0)
        assert np.allclose(j, [0, 1])

    def test_45_degree_linear(self):
        """45-degree linear should have equal Ex, Ey in phase."""
        j = pa.jones_vector(1.0, 1.0, phase_diff=0.0)
        assert np.isclose(np.abs(j[0]), np.abs(j[1]))
        assert np.isclose(np.angle(j[0]), np.angle(j[1]))

    def test_rhcp(self):
        """Right-hand circular should have 90-degree phase difference."""
        j = pa.jones_vector(1.0, 1.0, phase_diff=-np.pi/2)
        assert np.isclose(np.abs(j[0]), np.abs(j[1]))

    def test_lhcp(self):
        """Left-hand circular should have -90-degree phase difference."""
        j = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
        assert np.isclose(np.abs(j[0]), np.abs(j[1]))


class TestStokesParameters:
    """Tests for Stokes parameter calculation."""

    def test_horizontal_stokes(self):
        """Horizontal polarization: S1 = S0."""
        j = pa.jones_vector(1.0, 0.0)
        S0, S1, S2, S3 = pa.stokes_parameters(j)
        assert np.isclose(S1, S0)
        assert np.isclose(S2, 0.0)
        assert np.isclose(S3, 0.0)

    def test_vertical_stokes(self):
        """Vertical polarization: S1 = -S0."""
        j = pa.jones_vector(0.0, 1.0)
        S0, S1, S2, S3 = pa.stokes_parameters(j)
        assert np.isclose(S1, -S0)

    def test_45_degree_stokes(self):
        """45-degree linear: S2 = S0."""
        j = pa.jones_vector(1.0, 1.0, phase_diff=0.0)
        S0, S1, S2, S3 = pa.stokes_parameters(j)
        assert np.isclose(S2, S0, atol=1e-10)
        assert np.isclose(S1, 0.0, atol=1e-10)

    def test_rhcp_stokes(self):
        """RHCP: S3 = +S0 (IEEE convention with phase_diff=-pi/2)."""
        j = pa.jones_vector(1.0, 1.0, phase_diff=-np.pi/2)
        S0, S1, S2, S3 = pa.stokes_parameters(j)
        assert np.isclose(S3, S0, atol=1e-10)

    def test_lhcp_stokes(self):
        """LHCP: S3 = -S0 (IEEE convention with phase_diff=+pi/2)."""
        j = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
        S0, S1, S2, S3 = pa.stokes_parameters(j)
        assert np.isclose(S3, -S0, atol=1e-10)


class TestAxialRatio:
    """Tests for axial ratio calculation."""

    def test_circular_ar_unity(self):
        """Circular polarization has AR = 1."""
        j = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
        ar = pa.axial_ratio(j)
        assert np.isclose(ar, 1.0, atol=1e-10)

    def test_linear_ar_infinity(self):
        """Linear polarization has AR = infinity."""
        j = pa.jones_vector(1.0, 0.0)
        ar = pa.axial_ratio(j)
        assert np.isinf(ar)

    def test_elliptical_ar_between(self):
        """Elliptical polarization has 1 < AR < infinity."""
        # Create elliptical polarization
        j = pa.jones_vector(1.0, 0.5, phase_diff=np.pi/4)
        ar = pa.axial_ratio(j)
        assert ar > 1.0
        assert np.isfinite(ar)


class TestTiltAngle:
    """Tests for tilt angle calculation."""

    def test_horizontal_tilt_zero(self):
        """Horizontal polarization has tilt = 0."""
        j = pa.jones_vector(1.0, 0.0)
        tau = pa.tilt_angle(j)
        assert np.isclose(tau, 0.0, atol=1e-10)

    def test_vertical_tilt_90(self):
        """Vertical polarization has tilt = +/- 90 degrees."""
        j = pa.jones_vector(0.0, 1.0)
        tau = pa.tilt_angle(j)
        assert np.isclose(np.abs(tau), np.pi/2, atol=1e-10)

    def test_45_degree_tilt(self):
        """45-degree linear has tilt = 45 degrees."""
        j = pa.jones_vector(1.0, 1.0, phase_diff=0.0)
        tau = pa.tilt_angle(j)
        assert np.isclose(tau, np.pi/4, atol=1e-10)


class TestCrossPolDiscrimination:
    """Tests for cross-polarization discrimination."""

    def test_perfect_match(self):
        """Identical polarizations have high XPD."""
        j_ref = pa.jones_vector(1.0, 0.0)
        j_act = pa.jones_vector(1.0, 0.0)
        xpd = pa.cross_pol_discrimination(j_ref, j_act)
        assert xpd > 50  # Very high

    def test_orthogonal(self):
        """Orthogonal polarizations have very low XPD."""
        j_h = pa.jones_vector(1.0, 0.0)
        j_v = pa.jones_vector(0.0, 1.0)
        xpd = pa.cross_pol_discrimination(j_h, j_v)
        assert xpd < -50  # Very low


class TestPolarizationLossFactor:
    """Tests for polarization loss factor."""

    def test_matched_plf_unity(self):
        """Matched polarizations have PLF = 1."""
        j = pa.jones_vector(1.0, 0.0)
        plf = pa.polarization_loss_factor(j, j)
        assert np.isclose(plf, 1.0)

    def test_orthogonal_plf_zero(self):
        """Orthogonal polarizations have PLF = 0."""
        j_h = pa.jones_vector(1.0, 0.0)
        j_v = pa.jones_vector(0.0, 1.0)
        plf = pa.polarization_loss_factor(j_h, j_v)
        assert np.isclose(plf, 0.0, atol=1e-10)

    def test_circular_linear_plf_half(self):
        """Circular receiving linear has PLF = 0.5."""
        j_circ = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
        j_lin = pa.jones_vector(1.0, 0.0)
        plf = pa.polarization_loss_factor(j_circ, j_lin)
        assert np.isclose(plf, 0.5, atol=1e-10)


class TestLudwig3Decomposition:
    """Tests for Ludwig-3 co/cross-pol decomposition."""

    def test_phi_zero_copol(self):
        """At phi=0, E_theta is co-pol."""
        theta = np.array([np.pi/4])
        phi = np.array([0.0])
        E_theta = np.array([1.0 + 0j])
        E_phi = np.array([0.0 + 0j])

        E_co, E_cross = pa.ludwig3_decomposition(theta, phi, E_theta, E_phi)

        assert np.isclose(np.abs(E_co[0]), 1.0)
        assert np.isclose(np.abs(E_cross[0]), 0.0)

    def test_phi_90_copol(self):
        """At phi=90 deg, E_phi is co-pol (with sign)."""
        theta = np.array([np.pi/4])
        phi = np.array([np.pi/2])
        E_theta = np.array([0.0 + 0j])
        E_phi = np.array([1.0 + 0j])

        E_co, E_cross = pa.ludwig3_decomposition(theta, phi, E_theta, E_phi)

        # At phi=90, E_co = E_theta*cos(90) - E_phi*sin(90) = -E_phi
        assert np.isclose(np.abs(E_co[0]), 1.0)


class TestCoPolCrossPolPatterns:
    """Tests for co-pol and cross-pol pattern extraction."""

    def test_co_pol_shape(self):
        """co_pol_pattern should return correct shape."""
        theta = np.linspace(0, np.pi/2, 91)
        phi = np.zeros_like(theta)
        E_theta = np.cos(theta) + 0j
        E_phi = np.zeros_like(theta) + 0j

        E_co = pa.co_pol_pattern(theta, phi, E_theta, E_phi)
        assert E_co.shape == theta.shape

    def test_cross_pol_shape(self):
        """cross_pol_pattern should return correct shape."""
        theta = np.linspace(0, np.pi/2, 91)
        phi = np.zeros_like(theta)
        E_theta = np.cos(theta) + 0j
        E_phi = 0.1 * np.sin(theta) + 0j

        E_cross = pa.cross_pol_pattern(theta, phi, E_theta, E_phi)
        assert E_cross.shape == theta.shape

    def test_theta_reference(self):
        """reference_pol='theta' should return E_theta as co-pol."""
        theta = np.array([np.pi/4])
        phi = np.array([np.pi/4])
        E_theta = np.array([1.0 + 0j])
        E_phi = np.array([0.5 + 0j])

        E_co = pa.co_pol_pattern(theta, phi, E_theta, E_phi, reference_pol='theta')
        E_cross = pa.cross_pol_pattern(theta, phi, E_theta, E_phi, reference_pol='theta')

        assert np.allclose(E_co, E_theta)
        assert np.allclose(E_cross, E_phi)
