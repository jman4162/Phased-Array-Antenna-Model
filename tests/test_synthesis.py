"""
Numerical validation of the array factor synthesis functions.

References
----------
[Haupt] R. L. Haupt, *Antenna Arrays: A Computational Approach*, Wiley-IEEE
    Press, 2010, Sec. 3.1 (Woodward-Lawson, least squares, Table 3.1) and
    Sec. 3.2.1 (binomial taper, Table 3.2).
[Skolnik] M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed.,
    McGraw-Hill, 2001: cosecant-squared antenna pattern,
    G(theta) = G(theta1) csc^2(theta) / csc^2(theta1) for
    theta1 <= theta <= theta2.
"""

from math import comb

import numpy as np
import pytest

import phased_array as pa

K = pa.wavelength_to_k(1.0)


def _ula_x(n, spacing):
    """Centered element positions (wavelengths = meters at wavelength 1)."""
    return (np.arange(n) - (n - 1) / 2) * spacing


def _af(u, w, spacing=0.5):
    """AF(u) of a centered uniform linear array along x."""
    u = np.asarray(u, dtype=float)
    n = len(w)
    return pa.array_factor_uv(u, np.zeros_like(u), _ula_x(n, spacing),
                              np.zeros(n), w, K)


# ---------------------------------------------------------------- binomial

# [Haupt] Table 3.2
HAUPT_TABLE_3_2 = {
    1: ([1], 1.000),
    2: ([1, 1], 1.000),
    3: ([1, 2, 1], 0.889),
    4: ([1, 3, 3, 1], 0.800),
    5: ([1, 4, 6, 4, 1], 0.731),
    6: ([1, 5, 10, 10, 5, 1], 0.667),
    7: ([1, 6, 15, 20, 15, 6, 1], 0.633),
    8: ([1, 7, 21, 35, 35, 21, 7, 1], 0.597),
    9: ([1, 8, 28, 56, 70, 56, 28, 8, 1], 0.566),
}


class TestBinomial:

    @pytest.mark.parametrize("n", sorted(HAUPT_TABLE_3_2))
    def test_haupt_table_3_2_weights(self, n):
        book, _ = HAUPT_TABLE_3_2[n]
        w = pa.binomial_taper_1d(n)
        assert w.max() == 1.0
        np.testing.assert_allclose(w * max(book), book, rtol=1e-14)

    @pytest.mark.parametrize("n", sorted(HAUPT_TABLE_3_2))
    def test_haupt_table_3_2_efficiency(self, n):
        # Exact: (sum C)^2 / (N sum C^2) = 4^(N-1) / (N C(2N-2, N-1))
        _, eta = HAUPT_TABLE_3_2[n]
        got = pa.compute_taper_efficiency(pa.binomial_taper_1d(n))
        assert got == pytest.approx(4 ** (n - 1) / (n * comb(2 * n - 2, n - 1)), abs=1e-14)
        if n == 6:
            # Table 3.2 prints 0.667; 32^2 / (6 * 252) = 0.6772 (digit misprint)
            assert got == pytest.approx(0.677, abs=5e-4)
        else:
            assert got == pytest.approx(eta, abs=5e-4)

    @pytest.mark.parametrize("n", [3, 5, 8, 9, 16])
    def test_no_sidelobes_at_half_wavelength(self, n):
        # [Haupt] eq. 3.10: AF = (1 + z)^(N-1), all zeros at u = +/-1
        u = np.linspace(-1, 1, 4001)
        a = np.abs(_af(u, pa.binomial_taper_1d(n)))
        a /= a.max()
        np.testing.assert_allclose(a, np.cos(np.pi * u / 2) ** (n - 1), atol=1e-12)
        peak = np.argmax(a)
        assert u[peak] == 0.0
        # Monotone falloff from the peak to the nulls at u = -1 and u = +1
        assert np.all(np.diff(a[peak:]) <= 1e-15)
        assert np.all(np.diff(a[:peak + 1]) >= -1e-15)
        assert a[0] < 1e-12 and a[-1] < 1e-12

        geom = pa.create_rectangular_array(n, 1, 0.5, 0.5)
        angles, cut, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, pa.binomial_taper_1d(n), K, n_points=3601)
        assert pa.compute_peak_sidelobe_level(angles, cut) == -np.inf

    def test_large_n_is_finite_and_symmetric(self):
        w = pa.binomial_taper_1d(1201)
        assert np.all(np.isfinite(w))
        assert w.max() == 1.0 and w[600] == 1.0
        np.testing.assert_array_equal(w, w[::-1])

    @pytest.mark.parametrize("n", [0, -3, 2.5, True])
    def test_invalid_n(self, n):
        with pytest.raises(ValueError):
            pa.binomial_taper_1d(n)


# ---------------------------------------------------------- Woodward-Lawson

def _sector(half_width):
    return lambda u: (np.abs(u) <= half_width + 1e-12).astype(float)


class TestWoodwardLawson:

    def test_haupt_table_3_1(self):
        # [Haupt] Table 3.1, 16 elements, AF = 1 for |u| <= 0.5; elements
        # numbered from the center outward
        book = [0.49291, 0.10149, -0.12688, -0.0096566,
                0.080463, -0.020572, -0.0576, 0.039841]
        w = pa.woodward_lawson_weights(_sector(0.5), 16, 0.5)
        np.testing.assert_allclose(w.real[8:], book, atol=6e-6)
        np.testing.assert_allclose(w.real[:8], book[::-1], atol=6e-6)
        assert np.max(np.abs(w.imag)) < 1e-15

    @pytest.mark.parametrize("n", [8, 15, 16, 21])
    @pytest.mark.parametrize("spacing", [0.25, 0.4, 0.5, 0.7])
    def test_pattern_equals_desired_at_samples(self, n, spacing):
        def desired(u):
            return np.exp(-3 * u ** 2) * np.exp(2j * u) + 0.2 * u

        w, u_m = pa.woodward_lawson_weights(desired, n, spacing, return_samples=True)
        np.testing.assert_allclose(_af(u_m, w, spacing), desired(u_m), atol=1e-9, rtol=0)

    @pytest.mark.parametrize("n, spacing, n_samples", [
        (16, 0.25, 9),     # |m| <= N d = 4
        (16, 0.4, 13),     # |m| <= floor(6.4)
        (15, 0.5, 15),     # m = -7..7
        (16, 0.5, 16),     # m = -8..7: u = -1 kept, u = +1 is the same beam
        (16, 0.7, 16),     # one period of the array factor
        (15, 0.9, 15),
    ])
    def test_sample_points(self, n, spacing, n_samples):
        _, u_m = pa.woodward_lawson_weights(lambda u: 1.0, n, spacing,
                                            return_samples=True)
        assert u_m.size == n_samples <= n
        m = u_m * n * spacing
        np.testing.assert_allclose(m, np.round(m), atol=1e-12)
        np.testing.assert_allclose(np.diff(m), 1.0)
        assert np.all(np.abs(u_m) <= 1 + 1e-12)

    @pytest.mark.parametrize("n", [11, 16])
    def test_matches_beam_sum_eq_3_5(self, n):
        # [Haupt] eq. 3.5: AF = sum_m b_m sin(N kd(u-u_m)/2) / (N sin(kd(u-u_m)/2))
        d = 0.5
        desired = _sector(0.35)
        w, u_m = pa.woodward_lawson_weights(desired, n, d, return_samples=True)
        u = np.linspace(-0.997, 0.997, 777)    # avoids every u_m
        psi = K * d * (u[:, None] - u_m[None, :])
        beams = np.sin(n * psi / 2) / (n * np.sin(psi / 2))
        np.testing.assert_allclose(_af(u, w, d), beams @ desired(u_m), atol=1e-12)

    def test_flat_top_sector_n20(self):
        # 20 elements at lambda/2, AF = 1 for |u| <= 0.3. Beam spacing is
        # lambda/(N d) = 0.1, so 7 samples fall in the sector and the first
        # zero-valued sample is at |u| = 0.4.
        # Measured: passband ripple 1.31 dB (+0.93 / -0.39 dB about the
        # desired level), out-of-sector peak for |u| >= 0.4 is -15.5 dB.
        w = pa.woodward_lawson_weights(_sector(0.3), 20, 0.5)
        u = np.linspace(-1, 1, 20001)
        level_dB = 20 * np.log10(np.abs(_af(u, w)) + 1e-300)
        passband = level_dB[np.abs(u) <= 0.3]
        ripple = passband.max() - passband.min()
        out_of_sector = level_dB[np.abs(u) >= 0.4].max()
        assert 0.5 < ripple < 2.0
        assert abs(passband).max() < 1.5
        assert out_of_sector < -10.0
        # A real, symmetric target gives real, symmetric weights
        assert np.max(np.abs(w.imag)) < 1e-15
        np.testing.assert_allclose(w, w[::-1], atol=1e-15)

    def test_least_squares_at_beam_points_equals_wl(self):
        # Square system at the orthogonal beam points: same solution
        desired = _sector(0.5)
        w_wl, u_m = pa.woodward_lawson_weights(desired, 16, 0.5, return_samples=True)
        w_ls = pa.least_squares_weights(u_m, desired(u_m), n_elements=16)
        np.testing.assert_allclose(w_ls, w_wl, atol=1e-12)

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            pa.woodward_lawson_weights(np.ones(8), 8)
        with pytest.raises(ValueError):
            pa.woodward_lawson_weights(_sector(0.3), 0)
        with pytest.raises(ValueError):
            pa.woodward_lawson_weights(_sector(0.3), 8, spacing=0.0)


# ------------------------------------------------------------ least squares

def _csc2_field(u, theta1_deg=10.0, theta2_deg=45.0):
    """Cosecant-squared field target [Skolnik].

    Power G ~ csc^2(theta) on [theta1, theta2] gives a constant two-way
    return from targets at constant altitude, so the field goes as
    csc(theta) / csc(theta1) = u1 / u there. The target is flat at the peak
    from broadside (u = 0) to u1 and zero outside [0, u2], with zero phase.
    """
    u = np.asarray(u, dtype=float)
    u1, u2 = np.sin(np.deg2rad([theta1_deg, theta2_deg]))
    d = np.zeros_like(u)
    d[(u >= 0) & (u <= u1)] = 1.0
    shaped = (u > u1) & (u <= u2)
    d[shaped] = u1 / u[shaped]
    return d


def _relative_rms(af, target):
    return np.sqrt(np.sum(np.abs(af - target) ** 2) / np.sum(np.abs(target) ** 2))


class TestLeastSquares:

    def test_haupt_table_3_1(self):
        # [Haupt] Table 3.1, least-squares column (center to edge). The text
        # says 20 equally spaced samples; the printed weights correspond to
        # 21 samples at u = 2m/21, m = -10..10 (cell midpoints of [-1, 1]).
        book = [0.46711, 0.13343, -0.10769, -0.047619,
                0.068812, 0.023733, -0.055073, -0.011761]
        u = 2 * np.arange(-10, 11) / 21
        w = pa.least_squares_weights(u, _sector(0.5)(u), n_elements=16)
        np.testing.assert_allclose(w.real[8:], book, atol=6e-6)
        assert np.max(np.abs(w.imag)) < 1e-12

    @pytest.mark.parametrize("spacing", [0.5, 0.7])
    def test_recovers_known_weights(self, spacing):
        rng = np.random.default_rng(1)
        n = 12
        w0 = rng.normal(size=n) + 1j * rng.normal(size=n)
        u = np.linspace(-1, 1, 401)
        target = _af(u, w0, spacing)
        w = pa.least_squares_weights(u, target, n_elements=n, spacing=spacing)
        np.testing.assert_allclose(w, w0, atol=1e-8, rtol=0)

    def test_recovers_known_weights_nonuniform(self):
        rng = np.random.default_rng(2)
        wavelength = 0.03                      # 10 GHz
        k = pa.wavelength_to_k(wavelength)
        x = np.sort(rng.uniform(-3, 3, size=10)) * wavelength
        w0 = rng.normal(size=10) + 1j * rng.normal(size=10)
        u = np.linspace(-1, 1, 401)
        target = pa.array_factor_uv(u, np.zeros_like(u), x, np.zeros(10), w0, k)
        w = pa.least_squares_weights(u, target, positions=x, k=k)
        np.testing.assert_allclose(w, w0, atol=1e-8, rtol=0)

    @pytest.mark.parametrize("regularization", [0.0, 0.05])
    def test_positions_path_matches_ula_path(self, regularization):
        n, spacing, wavelength = 14, 0.5, 0.03
        u = np.linspace(-1, 1, 301)
        target = _csc2_field(u)
        c = 1 + (np.abs(u) > 0.8)              # emphasize wide angles
        w_ula = pa.least_squares_weights(u, target, n_elements=n, spacing=spacing,
                                         weights=c, regularization=regularization)
        geom = pa.create_rectangular_array(n, 1, spacing, spacing, wavelength=wavelength)
        w_pos = pa.least_squares_weights(u, target, positions=geom.x,
                                         k=pa.wavelength_to_k(wavelength),
                                         weights=c, regularization=regularization)
        np.testing.assert_allclose(w_pos, w_ula, atol=1e-10, rtol=0)

    def test_cosecant_squared_error_falls_with_n(self):
        # Relative RMS error over 401 samples in [-1, 1], d = lambda/2,
        # theta1 = 10 deg, theta2 = 45 deg. Measured: N = 8: 0.323,
        # N = 16: 0.201, N = 32: 0.146. The jumps at u = 0 and u2 dominate.
        u = np.linspace(-1, 1, 401)
        target = _csc2_field(u)
        errors = []
        for n in (8, 16, 32):
            w = pa.least_squares_weights(u, target, n_elements=n)
            errors.append(_relative_rms(_af(u, w), target))
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 0.2
        assert errors[0] == pytest.approx(0.3225, abs=1e-3)
        assert errors[2] == pytest.approx(0.1459, abs=1e-3)

    def test_regularization_reduces_weight_norm(self):
        # At d = lambda/4 the visible region covers half the AF period; the
        # unregularized fit uses the invisible region and returns weights
        # with norm ~4e3 (superdirective). mu = 1e-3 cuts the norm to ~7.8
        # while the relative RMS error rises from 0.243 to 0.261.
        u = np.linspace(-1, 1, 401)
        target = _csc2_field(u)
        norms, errs = [], []
        for mu in (0.0, 1e-3, 1e-1, 10.0):
            w = pa.least_squares_weights(u, target, n_elements=16, spacing=0.25,
                                         regularization=mu)
            norms.append(np.linalg.norm(w))
            errs.append(_relative_rms(_af(u, w, 0.25), target))
        assert all(a > b for a, b in zip(norms, norms[1:]))
        assert all(a < b for a, b in zip(errs, errs[1:]))
        assert norms[0] > 1e3 and norms[1] < 10

    def test_matches_normal_equations(self):
        # Definition check: (A^H C A + mu I) w = A^H C D
        rng = np.random.default_rng(3)
        n, spacing, mu = 10, 0.5, 0.3
        u = np.linspace(-1, 1, 61)
        target = rng.normal(size=61) + 1j * rng.normal(size=61)
        c = rng.uniform(0.1, 2.0, size=61)
        A = np.exp(1j * K * np.outer(u, _ula_x(n, spacing)))
        AhC = A.conj().T * c
        expected = np.linalg.solve(AhC @ A + mu * np.eye(n), AhC @ target)
        w = pa.least_squares_weights(u, target, n_elements=n, spacing=spacing,
                                     weights=c, regularization=mu)
        np.testing.assert_allclose(w, expected, atol=1e-12, rtol=0)

    def test_zero_sample_weight_drops_sample(self):
        u = np.linspace(-1, 1, 81)
        target = _csc2_field(u)
        c = np.ones(81)
        c[::3] = 0.0
        w0 = pa.least_squares_weights(u, target, n_elements=9, weights=c)
        w1 = pa.least_squares_weights(u[c > 0], target[c > 0], n_elements=9)
        np.testing.assert_allclose(w0, w1, atol=1e-12)

    def test_invalid_inputs(self):
        u = np.linspace(-1, 1, 11)
        d = np.ones(11)
        with pytest.raises(ValueError, match="exactly one"):
            pa.least_squares_weights(u, d)
        with pytest.raises(ValueError, match="exactly one"):
            pa.least_squares_weights(u, d, n_elements=4, positions=np.zeros(4), k=K)
        with pytest.raises(ValueError, match="required"):
            pa.least_squares_weights(u, d, positions=np.arange(4.0))
        with pytest.raises(ValueError, match="only with positions"):
            pa.least_squares_weights(u, d, n_elements=4, k=K)
        with pytest.raises(ValueError, match="shape"):
            pa.least_squares_weights(u, np.ones(5), n_elements=4)
        with pytest.raises(ValueError, match="nonnegative"):
            pa.least_squares_weights(u, d, n_elements=4, weights=-np.ones(11))
        with pytest.raises(ValueError, match="nonnegative"):
            pa.least_squares_weights(u, d, n_elements=4, regularization=-1.0)
