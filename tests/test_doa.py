"""
Numerical validation of phased_array.doa.

Snapshots follow the package convention: a plane wave from u induces
``array_manifold(...) = exp(+j k r . u)``, so an estimator that conjugates
the data by mistake would find the mirror angle -theta instead of theta.
"""

import numpy as np
import pytest

import phased_array as pa

K_NORM = pa.wavelength_to_k(1.0)
SCAN = np.arange(-90.0, 90.01, 0.25)


def _ula(n=8, d=0.5):
    return pa.create_rectangular_array(n, 1, d, d)


def _spectra(R, geom, n_sources):
    return {
        "bartlett": pa.bartlett_spectrum(R, geom, K_NORM, SCAN, 0),
        "capon": pa.capon_spectrum(R, geom, K_NORM, SCAN, 0),
        "music": pa.music_spectrum(R, geom, K_NORM, SCAN, 0, n_sources),
    }


def _coherent_pair(geom, angles, snr_db, n_snap, rel_phase_deg, seed):
    """Two plane waves carrying the same waveform s(t) (fully coherent).

    The second source is ``exp(j rel_phase) * s(t)``: same amplitude, with
    the given phase offset relative to the first at the array center.
    """
    rng = np.random.default_rng(seed)
    s = 10 ** (snr_db / 20) * (rng.standard_normal(n_snap)
                               + 1j * rng.standard_normal(n_snap)) / np.sqrt(2)
    a1 = pa.array_manifold(K_NORM, geom.x, geom.y, angles[0], 0, geom.z)
    a2 = pa.array_manifold(K_NORM, geom.x, geom.y, angles[1], 0, geom.z)
    n = geom.n_elements
    noise = (rng.standard_normal((n_snap, n))
             + 1j * rng.standard_normal((n_snap, n))) / np.sqrt(2)
    return np.outer(s, a1 + np.exp(1j * np.deg2rad(rel_phase_deg)) * a2) + noise


class TestSnapshots:

    def test_shape_and_power(self):
        geom = _ula(8)
        X = pa.generate_snapshots(geom, K_NORM, [(20, 0), (-35, 0)], [10, 0], 4000, seed=0)
        assert X.shape == (4000, 8)
        # Per-element power: unit noise + 10 + 1
        power = np.mean(np.abs(X) ** 2, axis=0)
        np.testing.assert_allclose(power, 12.0, rtol=0.08)

    def test_noise_only(self):
        geom = _ula(4)
        X = pa.generate_snapshots(geom, K_NORM, [], 10, 5000, seed=1)
        np.testing.assert_allclose(pa.sample_covariance(X), np.eye(4), atol=0.06)

    def test_rejects_bare_direction(self):
        with pytest.raises(ValueError, match="pairs"):
            pa.generate_snapshots(_ula(4), K_NORM, (20, 0), 10, 8)

    def test_seed_handling(self):
        geom = _ula(4)
        args = (geom, K_NORM, [(10, 0)], 5, 16)
        np.testing.assert_array_equal(pa.generate_snapshots(*args, seed=3),
                                      pa.generate_snapshots(*args, seed=3))
        g1 = pa.generate_snapshots(*args, seed=np.random.default_rng(3))
        g2 = pa.generate_snapshots(*args, seed=np.random.default_rng(3))
        np.testing.assert_array_equal(g1, g2)

    def test_principal_eigenvector_is_manifold_not_conjugate(self):
        geom = _ula(8)
        X = pa.generate_snapshots(geom, K_NORM, [(30, 0)], 30, 500, seed=2)
        R = pa.sample_covariance(X)
        np.testing.assert_allclose(R, X.T @ X.conj() / 500)
        v = np.linalg.eigh(R)[1][:, -1]
        a = pa.array_manifold(K_NORM, geom.x, geom.y, 30, 0)
        assert abs(np.vdot(a, v)) ** 2 / 8 > 0.99
        assert abs(np.vdot(np.conj(a), v)) ** 2 / 8 < 0.1


class TestSpectra:

    @pytest.mark.parametrize("n", [8, 16])
    def test_single_source_peaks_at_true_angle_not_mirror(self, n):
        geom = _ula(n)
        X = pa.generate_snapshots(geom, K_NORM, [(25, 0)], 10, 200, seed=0)
        R = pa.sample_covariance(X)
        for name, P in _spectra(R, geom, 1).items():
            assert SCAN[np.argmax(P)] == pytest.approx(25, abs=0.5), name
            mirror = P[np.argmin(np.abs(SCAN + 25))]
            assert 10 * np.log10(P.max() / mirror) > 10, name

    @pytest.mark.parametrize("n", [8, 16])
    def test_two_separated_sources(self, n):
        geom = _ula(n)
        truth = np.array([-10.0, 30.0])
        X = pa.generate_snapshots(geom, K_NORM, [(t, 0) for t in truth], 10, 200, seed=1)
        R = pa.sample_covariance(X)
        for name, P in _spectra(R, geom, 2).items():
            peaks = pa.find_spectrum_peaks(P, SCAN, 2)
            np.testing.assert_allclose(peaks, truth, atol=1.0, err_msg=name)

    def test_resolution_inside_bartlett_beamwidth(self):
        """Sources 10 deg apart, inside the 8-element beamwidth (~12.8 deg)."""
        geom = _ula(8)
        truth = np.array([0.0, 10.0])
        X = pa.generate_snapshots(geom, K_NORM, [(t, 0) for t in truth], 10, 200, seed=2)
        R = pa.sample_covariance(X)
        spectra = _spectra(R, geom, 2)

        for name in ("capon", "music"):
            peaks = pa.find_spectrum_peaks(spectra[name], SCAN, 2)
            np.testing.assert_allclose(peaks, truth, atol=1.0, err_msg=name)
            mid = spectra[name][np.argmin(np.abs(SCAN - 5))]
            assert mid < 0.5 * spectra[name].max(), name

        # Bartlett: one merged lobe between the sources
        P = spectra["bartlett"]
        window = (SCAN > -10) & (SCAN < 20)
        local = pa.find_spectrum_peaks(P[window], SCAN[window], 5)
        assert local.size == 1
        assert local[0] == pytest.approx(5.0, abs=1.5)

    def test_shapes_and_broadcasting(self):
        geom = pa.create_rectangular_array(3, 3, 0.5, 0.5)
        X = pa.generate_snapshots(geom, K_NORM, [(20, 45)], 10, 50, seed=0)
        R = pa.sample_covariance(X)
        th, ph = np.meshgrid(np.arange(0, 91, 10.0), np.arange(0, 360, 30.0), indexing="ij")
        for P in (pa.bartlett_spectrum(R, geom, K_NORM, th, ph),
                  pa.capon_spectrum(R, geom, K_NORM, th, ph),
                  pa.music_spectrum(R, geom, K_NORM, th, ph, 1)):
            assert P.shape == th.shape
            assert np.isrealobj(P) and np.all(P > 0)
        assert pa.bartlett_spectrum(R, geom, K_NORM, 20.0, 45.0).shape == ()
        assert pa.music_spectrum(R, geom, K_NORM, SCAN, 0, 1).shape == SCAN.shape

    def test_chunked_evaluation_matches(self, monkeypatch):
        import phased_array.doa as doa
        geom = _ula(8)
        X = pa.generate_snapshots(geom, K_NORM, [(10, 0)], 10, 50, seed=0)
        R = pa.sample_covariance(X)
        full = pa.capon_spectrum(R, geom, K_NORM, SCAN, 0)
        monkeypatch.setattr(doa, "_CHUNK_BYTES", 16 * 8 * 7)  # 7 angles/chunk
        np.testing.assert_allclose(pa.capon_spectrum(R, geom, K_NORM, SCAN, 0), full)

    def test_spectrum_levels(self):
        """Bartlett gives p N^2 + N and Capon p + 1/N for an exact R."""
        geom = _ula(8)
        p = 4.0
        a = pa.array_manifold(K_NORM, geom.x, geom.y, 20, 0)
        R = p * np.outer(a, a.conj()) + np.eye(8)
        assert pa.bartlett_spectrum(R, geom, K_NORM, 20, 0) == pytest.approx(p * 64 + 8)
        assert pa.capon_spectrum(R, geom, K_NORM, 20, 0) == pytest.approx(p + 1 / 8)

    def test_capon_diagonal_loading_with_few_snapshots(self):
        geom = _ula(8)
        X = pa.generate_snapshots(geom, K_NORM, [(15, 0)], 20, 4, seed=0)
        R = pa.sample_covariance(X)  # rank 4 < 8
        P = pa.capon_spectrum(R, geom, K_NORM, SCAN, 0, diagonal_loading=1.0)
        assert np.all(np.isfinite(P))
        assert SCAN[np.argmax(P)] == pytest.approx(15, abs=2)

    def test_input_validation(self):
        geom = _ula(8)
        with pytest.raises(ValueError, match="geometry has"):
            pa.bartlett_spectrum(np.eye(4), geom, K_NORM, SCAN, 0)
        with pytest.raises(ValueError, match="n_sources"):
            pa.music_spectrum(np.eye(8), geom, K_NORM, SCAN, 0, 8)
        with pytest.raises(ValueError, match="2-D"):
            pa.sample_covariance(np.ones(8))


class TestRootMusicEsprit:

    @pytest.mark.parametrize("func", [pa.root_music, pa.esprit])
    def test_two_sources(self, func):
        geom = _ula(10)
        truth = np.array([-20.0, 15.0])
        X = pa.generate_snapshots(geom, K_NORM, [(t, 0) for t in truth], 20, 200, seed=4)
        est = func(pa.sample_covariance(X), 2, spacing=0.5)
        np.testing.assert_allclose(est, truth, atol=0.2)

    @pytest.mark.parametrize("func", [pa.root_music, pa.esprit])
    def test_physical_units_and_phi_180(self, func):
        """10 GHz, 15 mm spacing; theta=25 at phi=180 is -25 deg."""
        wl = 0.03
        geom = pa.create_rectangular_array(8, 1, 0.5, 0.5, wavelength=wl)
        k = pa.wavelength_to_k(wl)
        X = pa.generate_snapshots(geom, k, [(25, 180), (40, 0)], 20, 200, seed=5)
        est = func(pa.sample_covariance(X), 2, spacing=0.015, wavelength=wl)
        np.testing.assert_allclose(est, [-25.0, 40.0], atol=0.3)

    @pytest.mark.parametrize("func", [pa.root_music, pa.esprit])
    def test_reversed_element_order_mirrors_angles(self, func):
        geom = _ula(8)
        X = pa.generate_snapshots(geom, K_NORM, [(12, 0)], 20, 200, seed=6)
        R = pa.sample_covariance(X)
        assert func(R, 1, 0.5)[0] == pytest.approx(12, abs=0.2)
        assert func(R[::-1, ::-1], 1, 0.5)[0] == pytest.approx(-12, abs=0.2)

    def test_rmse_near_crb(self):
        """30 dB, 200 snapshots, 200 trials, N = 8, theta = 10 deg."""
        n, snr, n_snap, theta = 8, 30.0, 200, 10.0
        geom = _ula(n)
        rng = np.random.default_rng(1)
        err_rm, err_es = [], []
        for _ in range(200):
            X = pa.generate_snapshots(geom, K_NORM, [(theta, 0)], snr, n_snap, seed=rng)
            R = pa.sample_covariance(X)
            err_rm.append(pa.root_music(R, 1, 0.5)[0] - theta)
            err_es.append(pa.esprit(R, 1, 0.5)[0] - theta)
        crb = pa.crb_ula_single_source(n, 0.5, snr, n_snap, theta)
        ratio_rm = np.sqrt(np.mean(np.square(err_rm))) / crb
        ratio_es = np.sqrt(np.mean(np.square(err_es))) / crb
        # root-MUSIC is efficient for one source at high SNR
        assert 0.85 < ratio_rm < 1.2
        # LS-ESPRIT: std ratio sqrt(N (N+1) / (6 (N-1))) = 1.31 for N = 8
        assert 1.0 < ratio_es < 1.5
        assert ratio_es == pytest.approx(np.sqrt(n * (n + 1) / (6 * (n - 1))), rel=0.15)


class TestCRB:

    def test_scaling(self):
        base = pa.crb_ula_single_source(8, 0.5, 10, 100, 20)
        assert pa.crb_ula_single_source(8, 0.5, 10, 400, 20) == pytest.approx(base / 2)
        assert pa.crb_ula_single_source(8, 0.5, 20, 100, 20) == pytest.approx(base / np.sqrt(10))
        assert pa.crb_ula_single_source(8, 0.5, 10, 100, 0) < base

    def test_matches_general_deterministic_crb(self):
        """CRB = 1 / (2 K SNR d^H P_perp d) with d = da/dtheta (Stoica-Nehorai)."""
        n, d, snr_db, n_snap, theta = 12, 0.4, 5.0, 64, 35.0
        geom = _ula(n, d)
        h = 1e-6
        a = pa.array_manifold(K_NORM, geom.x, geom.y, theta, 0)
        da = (pa.array_manifold(K_NORM, geom.x, geom.y, theta + h, 0)
              - pa.array_manifold(K_NORM, geom.x, geom.y, theta - h, 0)) / (2 * np.deg2rad(h))
        p_perp = np.eye(n) - np.outer(a, a.conj()) / n
        fisher = 2 * n_snap * 10 ** (snr_db / 10) * np.real(da.conj() @ p_perp @ da)
        expected = np.rad2deg(1 / np.sqrt(fisher))
        assert pa.crb_ula_single_source(n, d, snr_db, n_snap, theta) == pytest.approx(expected, rel=1e-6)


class TestModelOrder:

    @pytest.mark.parametrize("truth", [[], [(-10, 0), (25, 0)], [(-30, 0), (0, 0), (20, 0)]])
    def test_mdl_counts_sources(self, truth):
        geom = _ula(8)
        for seed in range(5):
            X = pa.generate_snapshots(geom, K_NORM, truth, 5, 200, seed=seed)
            assert pa.estimate_num_sources(pa.sample_covariance(X), 200) == len(truth)

    def test_aic_and_validation(self):
        geom = _ula(8)
        X = pa.generate_snapshots(geom, K_NORM, [(-10, 0), (25, 0)], 10, 200, seed=0)
        R = pa.sample_covariance(X)
        assert pa.estimate_num_sources(R, 200, method="AIC") == 2
        with pytest.raises(ValueError, match="method"):
            pa.estimate_num_sources(R, 200, method="bic")


class TestForwardBackward:

    TRUTH = np.array([0.0, 10.0])

    def test_decorrelates_coherent_sources(self):
        geom = _ula(8)
        X = _coherent_pair(geom, self.TRUTH, 10, 200, rel_phase_deg=90, seed=0)
        R = pa.sample_covariance(X)
        R_fb = pa.sample_covariance(X, forward_backward=True)

        # Plain covariance: rank-1 signal part, second eigenvalue at the noise floor
        ev = np.sort(np.linalg.eigvalsh(R))[::-1]
        ev_fb = np.sort(np.linalg.eigvalsh(R_fb))[::-1]
        assert ev[1] < 2.0
        assert ev_fb[1] > 10.0
        assert pa.estimate_num_sources(R, 200) == 1
        assert pa.estimate_num_sources(R_fb, 200) == 2

        # Plain MUSIC merges the pair; FB MUSIC resolves it
        plain = pa.find_spectrum_peaks(pa.music_spectrum(R, geom, K_NORM, SCAN, 0, 2), SCAN, 2)
        assert np.min(np.abs(plain[:, None] - self.TRUTH[None, :]), axis=0).max() > 2.0
        fb = pa.find_spectrum_peaks(pa.music_spectrum(R_fb, geom, K_NORM, SCAN, 0, 2), SCAN, 2)
        np.testing.assert_allclose(fb, self.TRUTH, atol=1.0)

    def test_in_phase_at_center_stays_coherent(self):
        """With a real source covariance the FB average adds nothing."""
        geom = _ula(8)
        X = _coherent_pair(geom, self.TRUTH, 10, 200, rel_phase_deg=0, seed=0)
        ev_fb = np.sort(np.linalg.eigvalsh(pa.sample_covariance(X, True)))[::-1]
        assert ev_fb[1] < 2.0

    def test_fb_of_exact_covariance(self):
        """For a centered ULA, J a* = a, so FB leaves an incoherent R unchanged."""
        geom = _ula(6)
        A = np.stack([pa.array_manifold(K_NORM, geom.x, geom.y, t, 0) for t in (-20, 15)], 1)
        R = A @ np.diag([3.0, 1.0]) @ A.conj().T + np.eye(6)
        X = np.sqrt(6) * np.linalg.cholesky(R).T  # 6 rows with X.T X* / 6 = R
        np.testing.assert_allclose(pa.sample_covariance(X), R, atol=1e-12)
        np.testing.assert_allclose(pa.sample_covariance(X, True), R, atol=1e-12)


class TestPeaks:

    def test_sorted_by_angle_and_edges_excluded(self):
        angles = np.linspace(-90, 90, 181)
        spec = np.exp(-(angles - 30) ** 2 / 20) + 0.5 * np.exp(-(angles + 40) ** 2 / 20)
        spec += 2.0 * (angles == 90)  # edge sample is not a local maximum
        assert pa.find_spectrum_peaks(spec, angles, 2).tolist() == [-40.0, 30.0]
        assert pa.find_spectrum_peaks(spec, angles, 1).tolist() == [30.0]
        assert pa.find_spectrum_peaks(spec, angles, 5).size == 2


def test_music_planar_array_2d():
    """MUSIC on a 4 x 4 planar array finds (theta, phi) = (35, 120) on a 1-deg grid."""
    geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
    X = pa.generate_snapshots(geom, K_NORM, [(35, 120)], 10, 100, seed=7)
    R = pa.sample_covariance(X)
    th, ph = np.meshgrid(np.arange(0, 90.5, 1.0), np.arange(0, 360, 1.0), indexing="ij")
    P = pa.music_spectrum(R, geom, K_NORM, th, ph, 1)
    i = np.unravel_index(np.argmax(P), P.shape)
    assert th[i] == pytest.approx(35, abs=1)
    assert ph[i] == pytest.approx(120, abs=2)
