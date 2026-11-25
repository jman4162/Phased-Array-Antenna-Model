"""
Regression tests against textbook values.

References
----------
[Balanis] C. A. Balanis, *Antenna Theory: Analysis and Design*, 3rd ed.,
    Wiley, 2005.
[Haupt] R. L. Haupt, *Antenna Arrays: A Computational Approach*, Wiley-IEEE
    Press, 2010.

Where a book prints an approximation (large-array formulas, rounded
coefficients), the test compares against the exact closed form computed
here and checks the book value with a looser tolerance.
"""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import j0

import phased_array as pa

K = pa.wavelength_to_k(1.0)


def _linear_z_array_directivity(n, d, weights, n_theta=4001):
    """Directivity of a linear array along z (phi-symmetric pattern)."""
    _, _, theta, phi = pa.create_theta_phi_grid((0, np.pi), (0, 2 * np.pi), n_theta, 9)
    z = np.arange(n) * d
    zeros = np.zeros(n)
    af = pa.array_factor_vectorized(theta, phi, zeros, zeros, weights, K, z)
    return pa.compute_directivity(theta, phi, af)


def _uniform_linear_directivity_exact(n, kd, beta):
    """Exact directivity of a uniform isotropic linear array, phase beta.

    For AF = sum exp(j n (kd cos(theta) + beta)) the radiated power is
    proportional to N + 2 * sum_{m=1}^{N-1} (N-m) sin(m kd) cos(m beta) / (m kd),
    and D = max|AF|^2 divided by that. max|AF| = N only when the phase
    can be matched in visible space (not for Hansen-Woodyard).
    """
    m = np.arange(1, n)
    s = np.sum((n - m) * np.sin(m * kd) * np.cos(m * beta) / (m * kd))
    theta = np.linspace(0, np.pi, 200001)
    psi = kd * np.cos(theta)[:, None] + beta
    peak = np.max(np.abs(np.exp(1j * psi * np.arange(n)).sum(axis=1)) ** 2)
    return peak / (n + 2 * s)


class TestDolphChebyshev:

    def test_haupt_n6_minus20(self):
        # [Haupt] p126 (eq. 3.21): 1, 1.437, 1.85, 1.85, 1.437, 1; eta = 0.944
        w = pa.chebyshev_taper_1d(6, -20)
        np.testing.assert_allclose(w / w[0], [1, 1.437, 1.850, 1.850, 1.437, 1], atol=0.003)
        assert pa.compute_taper_efficiency(w) == pytest.approx(0.944, abs=0.001)

    def test_balanis_example_6_9_n10_26dB(self):
        # [Balanis] Ex. 6.9 prints 1, .890, .706, .485, .357 (rounded; reaches
        # only -25.96 dB). Exact T9 expansion: 1, .8950, .7106, .4894, .3611.
        w = pa.chebyshev_taper_1d(10, -26)
        half = w[5:] / w[5]
        np.testing.assert_allclose(half, [1, 0.8950, 0.7106, 0.4894, 0.3611], atol=2e-4)
        np.testing.assert_allclose(half, [1, 0.890, 0.706, 0.485, 0.357], atol=0.006)

        geom = pa.create_rectangular_array(10, 1, 0.5, 0.5)
        angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, K, n_points=36001)
        assert pa.compute_peak_sidelobe_level(angles, cut) == pytest.approx(-26.0, abs=0.01)

    def test_balanis_example_6_10_beamwidth_and_directivity(self):
        # Exact HPBW 12.35 deg and D = 8.928; Ex. 6.10's 10.97 deg / 9.18 come
        # from the beam-broadening approximation (6-78), not valid at N = 10.
        w = pa.chebyshev_taper_1d(10, -26)
        geom = pa.create_rectangular_array(10, 1, 0.5, 0.5)
        angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, K, n_points=36001)
        assert pa.compute_half_power_beamwidth(angles, cut) == pytest.approx(12.35, abs=0.02)
        assert _linear_z_array_directivity(10, 0.5, w) == pytest.approx(8.928, abs=0.005)

    @pytest.mark.parametrize("n", [7, 8, 9, 16, 17])
    @pytest.mark.parametrize("sll", [-20, -30, -40])
    def test_equiripple_sidelobes(self, n, sll):
        geom = pa.create_rectangular_array(n, 1, 0.5, 0.5)
        w = pa.chebyshev_taper_1d(n, sll)
        angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, K, n_points=20001)
        assert pa.compute_peak_sidelobe_level(angles, cut) == pytest.approx(sll, abs=0.02)


class TestTaylor:

    def test_haupt_n20_minus20_nbar5(self):
        # [Haupt] p129: edge to centre 0.667 ... 1, efficiency 0.965
        w = pa.taylor_taper_1d(20, -20, nbar=5)
        book = [0.667, 0.621, 0.589, 0.624, 0.718, 0.818, 0.888, 0.933, 0.972, 1.0]
        np.testing.assert_allclose(w[:10] / w[9], book, atol=0.004)
        assert pa.compute_taper_efficiency(w) == pytest.approx(0.965, abs=0.001)

    @pytest.mark.parametrize("n", [16, 32, 64, 128])
    @pytest.mark.parametrize("sll", [-25, -30, -35, -40])
    def test_peak_sidelobe_near_design(self, n, sll):
        if n == 16 and sll == -40:
            pytest.xfail("sampled Taylor line source: -39.25 dB at N=16 even "
                         "with nbar=7 (discretization error at deep sidelobes)")
        # nbar >= 2 A^2 + 1/2 keeps the design well behaved ([Haupt] 3.2.3)
        A = np.arccosh(10 ** (-sll / 20)) / np.pi
        nbar = max(5, int(np.ceil(2 * A * A + 0.5)))
        geom = pa.create_rectangular_array(n, 1, 0.5, 0.5)
        w = pa.taylor_taper_1d(n, sll, nbar=nbar)
        angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, K, n_points=40001)
        psl = pa.compute_peak_sidelobe_level(angles, cut)
        assert abs(psl - sll) < 0.35


class TestUniformLinearArray:
    """[Balanis] Sec. 6.3-6.4, N = 10 isotropic elements."""

    def test_broadside_half_wavelength(self):
        w = np.ones(10)
        assert _linear_z_array_directivity(10, 0.5, w) == pytest.approx(10.0, rel=1e-4)

    def test_broadside_quarter_wavelength(self):
        # Exact 5.166; [Balanis] approximation 2N d / lambda = 5
        d = _linear_z_array_directivity(10, 0.25, np.ones(10))
        assert d == pytest.approx(_uniform_linear_directivity_exact(10, np.pi / 2, 0), rel=1e-4)
        assert d == pytest.approx(5.166, abs=0.002)

    def test_ordinary_endfire_quarter_wavelength(self):
        # [Balanis] Ex. 6.4 prints 10.05 (quadrature error); exact is 10.000
        kd = np.pi / 2
        w = np.exp(-1j * kd * np.arange(10))
        assert _linear_z_array_directivity(10, 0.25, w) == pytest.approx(10.0, rel=1e-4)

    def test_hansen_woodyard_quarter_wavelength(self):
        # Exact 17.79; [Balanis] (6-48) approximation 1.789 * 4 N d = 17.89
        kd = np.pi / 2
        beta = -(kd + np.pi / 10)
        w = np.exp(1j * beta * np.arange(10))
        d = _linear_z_array_directivity(10, 0.25, w)
        assert d == pytest.approx(_uniform_linear_directivity_exact(10, kd, beta), rel=1e-4)
        assert d == pytest.approx(17.79, abs=0.01)

    @pytest.mark.parametrize("d, book", [(0.5, 10.16), (0.25, 20.40)])
    def test_hpbw(self, d, book):
        n = 10
        geom = pa.create_rectangular_array(n, 1, d, d)
        angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, np.ones(n), K, n_points=36001)
        hpbw = pa.compute_half_power_beamwidth(angles, cut)

        # Exact half-power point of |sin(N psi/2) / (N sin(psi/2))|^2
        kd = 2 * np.pi * d

        def f(t):
            psi = kd * np.sin(t)
            return (np.sin(n * psi / 2) / (n * np.sin(psi / 2))) ** 2 - 0.5

        exact = 2 * np.rad2deg(brentq(f, 1e-9, np.arcsin(2 * np.pi / (n * kd))))
        assert hpbw == pytest.approx(exact, abs=0.01)
        # [Balanis] (6-22a) large-array approximation
        assert hpbw == pytest.approx(book, abs=0.15)

    @pytest.mark.parametrize("n, d", [(8, 0.25), (8, 0.5), (8, 0.7), (16, 1.0)])
    @pytest.mark.parametrize("theta0", [0, 30, 60])
    def test_directivity_matches_closed_form_haupt_2_67(self, n, d, theta0):
        # Linear array along z steered to theta0 (measured from the axis,
        # so broadside is 90 deg): beta = -kd cos(theta_s)
        theta_s = np.deg2rad(90 - theta0)
        kd = 2 * np.pi * d
        beta = -kd * np.cos(theta_s)
        w = np.exp(1j * beta * np.arange(n))
        got = _linear_z_array_directivity(n, d, w)
        assert got == pytest.approx(_uniform_linear_directivity_exact(n, kd, beta), rel=2e-4)


class TestPlanarArrays:

    @pytest.mark.parametrize("theta0, phi0, expected", [(0, 0, 148.72), (30, 0, 126.03)])
    def test_balanis_10x10_full_sphere(self, theta0, phi0, expected):
        geom = pa.create_rectangular_array(10, 10, 0.5, 0.5)
        w = pa.steering_vector(K, geom.x, geom.y, theta0, phi0)
        _, _, theta, phi = pa.create_theta_phi_grid((0, np.pi), (0, 2 * np.pi), 721, 1441)
        af = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, w, K)
        assert pa.compute_directivity(theta, phi, af) == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("taper, expected_dB", [
        (None, 23.77),
        (np.array([0.541, 0.777, 1, 1, 0.777, 0.541]), 23.61),
    ])
    def test_haupt_6x10_hemisphere(self, taper, expected_dB):
        # [Haupt] p82: 6 x 10, dx = 0.5, dy = 0.7; book 23.8 / 23.6 dB
        geom = pa.create_rectangular_array(6, 10, 0.5, 0.7)
        w = np.ones(60) if taper is None else np.outer(taper, np.ones(10)).ravel()
        _, _, theta, phi = pa.create_theta_phi_grid((0, np.pi / 2), (0, 2 * np.pi), 181, 361)
        af = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, w, K)
        d_dB = 10 * np.log10(pa.compute_directivity(theta, phi, af))
        assert d_dB == pytest.approx(expected_dB, abs=0.01)


class TestCircularArray:

    def test_uniform_ring_is_bessel_j0(self):
        # [Balanis] (6-110)-(6-112): N |J0(k a sin(theta))| for large N
        n, a = 64, 2.0
        geom = pa.create_circular_array(n, a)
        theta = np.linspace(0, np.pi / 2, 181)
        for phi0 in (0.0, 0.3):
            af = pa.array_factor_vectorized(theta, np.full_like(theta, phi0),
                                            geom.x, geom.y, np.ones(n), K)
            np.testing.assert_allclose(np.abs(af) / n, np.abs(j0(K * a * np.sin(theta))),
                                       atol=1e-9)


class TestThinnedArray:

    def test_haupt_20_element_two_off(self):
        # [Haupt] p160: elements 2 and 19 off -> -15.74 dB (computed -15.53)
        w = np.ones(20)
        w[[1, 18]] = 0
        geom = pa.create_rectangular_array(20, 1, 0.5, 0.5)
        angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, K, n_points=36001)
        assert pa.compute_peak_sidelobe_level(angles, cut) == pytest.approx(-15.74, abs=0.3)
