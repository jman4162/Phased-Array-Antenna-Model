"""
Direction-of-arrival (DOA) estimation.

Synthetic snapshots, sample covariance, spectral estimators (Bartlett,
Capon/MVDR, MUSIC), polynomial and subspace-rotation estimators for uniform
linear arrays (root-MUSIC, ESPRIT), model-order selection (MDL, AIC) and
the Cramer-Rao bound for a single source on a ULA.

All functions follow the package conventions: positions in meters,
``k = 2*pi/wavelength``, snapshots stored as the rows of an
``(n_snapshots, n_elements)`` array, and a plane wave from direction ``u``
inducing ``array_manifold(k, x, y, theta_deg, phi_deg, z)`` =
``exp(+j k r . u)`` at the elements. Covariances are ``R = E[x x^H]``.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import find_peaks

from .core import array_manifold
from .geometry import ArrayGeometry
from .utils import SeedLike, _make_rng

# Upper bound on the (angles x elements) manifold block built per chunk
_CHUNK_BYTES = 32 * 2**20


def _manifold_rows(
    geometry: ArrayGeometry,
    k: float,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    """Manifold vectors as rows, shape (n_angles, n_elements).

    Calls :func:`array_manifold` with angle columns against element rows,
    so NumPy broadcasting builds the whole matrix in one call.
    """
    z = None if geometry.z is None else np.asarray(geometry.z)[None, :]
    th_col = np.asarray(theta_deg, dtype=float)[:, None]
    ph_col = np.asarray(phi_deg, dtype=float)[:, None]
    return array_manifold(
        k,
        np.asarray(geometry.x)[None, :],
        np.asarray(geometry.y)[None, :],
        th_col,  # type: ignore[arg-type]
        ph_col,  # type: ignore[arg-type]
        z,
    )


def _check_covariance(R: np.ndarray, n_elements: Optional[int] = None) -> np.ndarray:
    R = np.asarray(R, dtype=complex)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be a square matrix, got shape {R.shape}")
    if n_elements is not None and R.shape[0] != n_elements:
        raise ValueError(
            f"R is {R.shape[0]}x{R.shape[0]} but the geometry has "
            f"{n_elements} elements"
        )
    return R


def _check_n_sources(n_sources: int, n_elements: int) -> int:
    n = int(n_sources)
    if n != n_sources or not 0 <= n < n_elements:
        raise ValueError(
            f"n_sources must be an integer in [0, {n_elements - 1}] for "
            f"{n_elements} elements, got {n_sources}"
        )
    return n


def _check_n_snapshots(n_snapshots: int) -> int:
    n = int(n_snapshots)
    if n != n_snapshots or n < 1:
        raise ValueError(
            f"n_snapshots must be a positive integer, got {n_snapshots}"
        )
    return n


def _check_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def _spectrum(geometry, k, theta_deg, phi_deg, quadratic):
    """Evaluate ``quadratic(A)`` over angle chunks; A holds manifold rows."""
    th, ph = np.broadcast_arrays(
        np.asarray(theta_deg, dtype=float), np.asarray(phi_deg, dtype=float)
    )
    shape = th.shape
    th = th.ravel()
    ph = ph.ravel()
    out = np.empty(th.size)
    step = max(1, _CHUNK_BYTES // (16 * max(geometry.n_elements, 1)))
    for start in range(0, th.size, step):
        sl = slice(start, start + step)
        out[sl] = quadratic(_manifold_rows(geometry, k, th[sl], ph[sl]))
    return out.reshape(shape)


def _sorted_eigh(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors of the Hermitian part of R."""
    R_h = 0.5 * (R + R.conj().T)
    vals, vecs = np.linalg.eigh(R_h)
    return vals[::-1], vecs[:, ::-1]


def generate_snapshots(
    geometry: ArrayGeometry,
    k: float,
    sources: Sequence[Tuple[float, float]],
    snr_db: Union[float, Sequence[float]],
    n_snapshots: int,
    seed: SeedLike = None,
) -> np.ndarray:
    """
    Simulate received array snapshots for plane-wave sources in white noise.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry (positions in meters).
    k : float
        Wavenumber (2*pi/wavelength) in rad/m.
    sources : sequence of (float, float)
        Arrival directions ``(theta_deg, phi_deg)``, one per source. An empty
        sequence gives noise only.
    snr_db : float or sequence of float
        Per-element signal-to-noise ratio of each source in dB. A scalar
        applies to every source.
    n_snapshots : int
        Number of snapshots K.
    seed : int, Generator, RandomState or None, optional
        Random source. ``None`` uses NumPy's global generator; an int gives
        a repeatable local ``RandomState``.

    Returns
    -------
    snapshots : ndarray
        Complex array of shape ``(n_snapshots, n_elements)``; each row is
        one snapshot.

    Notes
    -----
    Snapshot ``t`` is

    ``x(t) = sum_i sqrt(SNR_i) s_i(t) a(theta_i, phi_i) + n(t)``

    where ``a = array_manifold(k, x, y, theta, phi, z)``, the ``s_i(t)`` are
    independent zero-mean circular complex Gaussian with unit power, and
    ``n(t)`` is circular complex Gaussian noise with unit power per element.
    The signal draws are taken before the noise draws.

    Examples
    --------
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> X = pa.generate_snapshots(geom, k, [(20, 0)], snr_db=10,
    ...                           n_snapshots=100, seed=0)
    >>> X.shape
    (100, 8)
    """
    n_snapshots = _check_n_snapshots(n_snapshots)
    directions = np.asarray(list(sources), dtype=float)
    if directions.size == 0:
        directions = directions.reshape(0, 2)
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError(
            "sources must be a sequence of (theta_deg, phi_deg) pairs, "
            f"got an array of shape {directions.shape}"
        )
    n_src = directions.shape[0]
    snr = np.asarray(snr_db, dtype=float)
    if snr.ndim == 0:
        snr = np.full(n_src, float(snr))
    elif snr.shape != (n_src,):
        raise ValueError(
            "snr_db must be a scalar or one value per source: got "
            f"{snr.size} snr_db values for {n_src} sources"
        )
    amp = np.sqrt(10.0 ** (snr / 10.0))

    rng = _make_rng(seed)
    n_el = geometry.n_elements
    shape_s = (n_snapshots, n_src)
    shape_n = (n_snapshots, n_el)
    s = (rng.standard_normal(shape_s) + 1j * rng.standard_normal(shape_s)) / np.sqrt(2)
    noise = (rng.standard_normal(shape_n) + 1j * rng.standard_normal(shape_n)) / np.sqrt(2)
    if n_src == 0:
        return noise
    A = _manifold_rows(geometry, k, directions[:, 0], directions[:, 1])  # (n_src, n_el)
    return (s * amp) @ A + noise


def sample_covariance(
    snapshots: np.ndarray,
    forward_backward: bool = False,
) -> np.ndarray:
    """
    Sample covariance matrix ``R = E[x x^H]`` from row snapshots.

    Parameters
    ----------
    snapshots : ndarray
        Complex array of shape ``(n_snapshots, n_elements)``.
    forward_backward : bool, optional
        If True, return the forward-backward average
        ``(R + J R* J) / 2`` with ``J`` the exchange (reversal) matrix.

    Returns
    -------
    R : ndarray
        Hermitian matrix of shape ``(n_elements, n_elements)``,
        ``R = X.T @ X.conj() / K``.

    Notes
    -----
    Forward-backward averaging is valid only for arrays that are
    centro-symmetric in the element order: element ``N-1-n`` sits at the
    reflection of element ``n`` through the array center, as in a ULA or a
    grid from :func:`create_rectangular_array`. For such an array
    ``J a* = c a`` for every direction with ``|c| = 1``, so the average keeps
    the manifold and replaces the source covariance ``S`` by
    ``(S + D S* D*) / 2`` for a diagonal unit-modulus ``D``. That partly
    decorrelates coherent sources. Two coherent sources whose relative phase
    at the array center is 0 or 180 deg stay coherent, because their source
    covariance is then real.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> X = pa.generate_snapshots(geom, k, [(20, 0)], 10, 200, seed=0)
    >>> R = pa.sample_covariance(X)
    >>> R.shape
    (8, 8)
    >>> bool(np.allclose(R, R.conj().T))
    True
    """
    X = np.asarray(snapshots)
    if X.ndim != 2:
        raise ValueError(
            "snapshots must be 2-D with shape (n_snapshots, n_elements), "
            f"got shape {X.shape}"
        )
    R = X.T @ X.conj() / X.shape[0]
    if forward_backward:
        R = 0.5 * (R + R.conj()[::-1, ::-1])
    return R


def bartlett_spectrum(
    R: np.ndarray,
    geometry: ArrayGeometry,
    k: float,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    """
    Bartlett (conventional beamformer) spatial spectrum ``a^H R a``.

    Parameters
    ----------
    R : ndarray
        Covariance matrix ``E[x x^H]``, shape ``(n_elements, n_elements)``.
    geometry : ArrayGeometry
        Array geometry.
    k : float
        Wavenumber (2*pi/wavelength) in rad/m.
    theta_deg, phi_deg : array_like
        Scan directions in degrees. They are broadcast against each other,
        so equal-shape grids or an array with a scalar both work.

    Returns
    -------
    spectrum : ndarray
        Real, non-negative linear power with the broadcast shape of
        ``theta_deg`` and ``phi_deg``. It is not normalized: a source of
        power ``p`` in unit white noise gives ``p N^2 + N`` in its
        direction for N elements.

    Notes
    -----
    ``a = array_manifold(k, x, y, theta, phi, z)``. The resolution is set
    by the array beamwidth: sources closer than about one beamwidth merge
    into one peak.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> X = pa.generate_snapshots(geom, k, [(20, 0)], 10, 200, seed=0)
    >>> R = pa.sample_covariance(X)
    >>> theta = np.arange(-90.0, 90.5, 0.5)
    >>> P = pa.bartlett_spectrum(R, geom, k, theta, 0)
    >>> float(theta[np.argmax(P)])
    20.0
    """
    R = _check_covariance(R, geometry.n_elements)

    def quad(A):
        return np.real(np.sum(A.conj() * (A @ R.T), axis=1))

    return _spectrum(geometry, k, theta_deg, phi_deg, quad)


def capon_spectrum(
    R: np.ndarray,
    geometry: ArrayGeometry,
    k: float,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    diagonal_loading: float = 0.0,
) -> np.ndarray:
    """
    Capon (MVDR) spatial spectrum ``1 / (a^H R^-1 a)``.

    Parameters
    ----------
    R : ndarray
        Covariance matrix ``E[x x^H]``, shape ``(n_elements, n_elements)``.
    geometry : ArrayGeometry
        Array geometry.
    k : float
        Wavenumber (2*pi/wavelength) in rad/m.
    theta_deg, phi_deg : array_like
        Scan directions in degrees, broadcast against each other.
    diagonal_loading : float, optional
        Value ``delta`` added to the diagonal, ``R + delta I``, before
        inversion. Needed when R is singular, for example a sample
        covariance from fewer snapshots than elements (rank K < N). A value
        near the noise power is a common choice.

    Returns
    -------
    spectrum : ndarray
        Real linear power with the broadcast shape of the angle inputs. It
        is the output power of the MVDR beamformer steered to each
        direction and is not normalized: in a source direction it
        approaches ``p + 1/N`` for a source of power ``p`` in unit noise.

    Raises
    ------
    ValueError
        If ``R + delta I`` is singular or numerically rank-deficient.

    Notes
    -----
    Capon resolves sources inside the Bartlett beamwidth when the SNR is
    high enough; its peaks broaden as the SNR drops. With a sample
    covariance from K Gaussian snapshots the expected level is the true
    level times ``(K - N + 1) / K``.

    ``R + delta I`` must have full numerical rank
    (:func:`numpy.linalg.matrix_rank`, tolerance ``N * eps`` times the
    largest singular value). The inverse of a near-singular matrix gives a
    meaningless spectrum, so the function raises instead of returning one.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> X = pa.generate_snapshots(geom, k, [(20, 0)], 10, 200, seed=0)
    >>> R = pa.sample_covariance(X)
    >>> theta = np.arange(-90.0, 90.5, 0.5)
    >>> P = pa.capon_spectrum(R, geom, k, theta, 0)
    >>> float(theta[np.argmax(P)])
    20.0
    """
    R = _check_covariance(R, geometry.n_elements)
    n_el = R.shape[0]
    if diagonal_loading:
        R = R + diagonal_loading * np.eye(n_el)
    rank = np.linalg.matrix_rank(R)
    if rank < n_el:
        raise ValueError(
            f"R is singular or ill-conditioned (numerical rank {rank} of "
            f"{n_el}; a sample covariance from K < N snapshots has rank K). "
            "Pass diagonal_loading > 0, for example a value near the noise "
            "power."
        )
    R_inv = np.linalg.inv(R)

    def quad(A):
        denom = np.real(np.sum(A.conj() * (A @ R_inv.T), axis=1))
        return 1.0 / np.maximum(denom, np.finfo(float).tiny)

    return _spectrum(geometry, k, theta_deg, phi_deg, quad)


def music_spectrum(
    R: np.ndarray,
    geometry: ArrayGeometry,
    k: float,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    n_sources: int,
) -> np.ndarray:
    """
    MUSIC pseudo-spectrum ``1 / (a^H E_n E_n^H a)``.

    Parameters
    ----------
    R : ndarray
        Covariance matrix ``E[x x^H]``, shape ``(n_elements, n_elements)``.
    geometry : ArrayGeometry
        Array geometry.
    k : float
        Wavenumber (2*pi/wavelength) in rad/m.
    theta_deg, phi_deg : array_like
        Scan directions in degrees, broadcast against each other.
    n_sources : int
        Number of sources M, ``0 <= M < N``. The noise subspace ``E_n`` is
        spanned by the eigenvectors of the ``N - M`` smallest eigenvalues.

    Returns
    -------
    spectrum : ndarray
        Real positive pseudo-spectrum with the broadcast shape of the angle
        inputs. It is not a power: peak heights depend on SNR and snapshot
        count, and only peak locations carry information.

    Notes
    -----
    MUSIC needs the sources to be non-coherent (full-rank source
    covariance). For coherent sources apply forward-backward averaging
    (``sample_covariance(X, forward_backward=True)``) or spatial smoothing
    (not provided here) first. :func:`estimate_num_sources` estimates M.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> X = pa.generate_snapshots(geom, k, [(-10, 0), (25, 0)], 10, 200,
    ...                           seed=0)
    >>> R = pa.sample_covariance(X)
    >>> theta = np.arange(-90.0, 90.5, 0.5)
    >>> P = pa.music_spectrum(R, geom, k, theta, 0, n_sources=2)
    >>> pa.find_spectrum_peaks(P, theta, 2).tolist()
    [-10.0, 25.0]
    """
    R = _check_covariance(R, geometry.n_elements)
    m = _check_n_sources(n_sources, R.shape[0])
    _, vecs = _sorted_eigh(R)
    E_n = vecs[:, m:]

    def quad(A):
        denom = np.sum(np.abs(A @ E_n.conj()) ** 2, axis=1)
        return 1.0 / np.maximum(denom, np.finfo(float).tiny)

    return _spectrum(geometry, k, theta_deg, phi_deg, quad)


def _ula_angles(phase: np.ndarray, spacing: float, wavelength: float) -> np.ndarray:
    """Electrical angle psi = k d sin(theta) to sorted theta in degrees."""
    k = 2 * np.pi / wavelength
    sin_theta = np.clip(np.asarray(phase) / (k * spacing), -1.0, 1.0)
    return np.sort(np.rad2deg(np.arcsin(sin_theta)))


def root_music(
    R: np.ndarray,
    n_sources: int,
    spacing: float,
    wavelength: float = 1.0,
) -> np.ndarray:
    """
    Root-MUSIC arrival angles for a uniform linear array along x.

    Parameters
    ----------
    R : ndarray
        Covariance matrix ``E[x x^H]``, shape ``(N, N)``, with the elements
        ordered by increasing x.
    n_sources : int
        Number of sources M, ``0 <= M < N``; 0 returns an empty array.
    spacing : float
        Element spacing d in meters.
    wavelength : float, optional
        Wavelength in meters (default 1.0, so spacing is in wavelengths).

    Returns
    -------
    angles_deg : ndarray
        M arrival angles in degrees, sorted ascending, measured from
        broadside (+z) in the x-z plane. A positive angle is ``theta`` at
        ``phi = 0``; a negative angle ``-theta`` is ``theta`` at
        ``phi = 180``.

    Notes
    -----
    With ``E_n`` the noise subspace and ``C = E_n E_n^H``, the MUSIC
    denominator on the unit circle is the polynomial
    ``sum_l c_l z^l`` with ``c_l`` the sum of the l-th diagonal of C and
    ``z = exp(j k d sin(theta))``. Of its 2N-2 roots, the M inside the unit
    circle and closest to it give ``sin(theta) = angle(z) / (k d)``.

    The ULA model assumes element ``n`` sits at ``x_0 + n d``, so the rows
    and columns of R must follow increasing x; ``create_rectangular_array(N,
    1, ...)`` produces this order. Reversing the order mirrors every angle.
    Spacing above half a wavelength makes ``sin(theta)`` ambiguous.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> X = pa.generate_snapshots(geom, k, [(-10, 0), (25, 0)], 20, 200,
    ...                           seed=0)
    >>> R = pa.sample_covariance(X)
    >>> np.round(pa.root_music(R, 2, spacing=0.5), 1).tolist()
    [-10.0, 25.0]
    """
    R = _check_covariance(R)
    spacing = _check_positive("spacing", spacing)
    wavelength = _check_positive("wavelength", wavelength)
    n_el = R.shape[0]
    m = _check_n_sources(n_sources, n_el)
    if m == 0:
        return np.empty(0)
    _, vecs = _sorted_eigh(R)
    E_n = vecs[:, m:]
    C = E_n @ E_n.conj().T
    # Coefficients of z^(N-1) * sum_l c_l z^l, highest power first
    coeffs = np.array([np.trace(C, offset=l) for l in range(n_el - 1, -n_el, -1)])
    roots = np.roots(coeffs)
    inside = roots[np.abs(roots) < 1.0]
    if inside.size < m:
        inside = roots
    closest = inside[np.argsort(np.abs(np.abs(inside) - 1.0))[:m]]
    return _ula_angles(np.angle(closest), spacing, wavelength)


def esprit(
    R: np.ndarray,
    n_sources: int,
    spacing: float,
    wavelength: float = 1.0,
) -> np.ndarray:
    """
    Least-squares ESPRIT arrival angles for a uniform linear array along x.

    Parameters
    ----------
    R : ndarray
        Covariance matrix ``E[x x^H]``, shape ``(N, N)``, with the elements
        ordered by increasing x.
    n_sources : int
        Number of sources M, ``0 <= M < N``; 0 returns an empty array.
    spacing : float
        Element spacing d in meters.
    wavelength : float, optional
        Wavelength in meters (default 1.0).

    Returns
    -------
    angles_deg : ndarray
        M arrival angles in degrees, sorted ascending, from broadside in the
        x-z plane, with the sign convention of :func:`root_music`.

    Notes
    -----
    The signal subspace ``E_s`` (eigenvectors of the M largest
    eigenvalues) is split into the first and last ``N - 1`` rows,
    ``E_1`` and ``E_2``. Shift invariance of the ULA manifold gives
    ``E_2 = E_1 Phi`` with eigenvalues ``exp(j k d sin(theta_i))``; Phi is
    the least-squares solution. As in :func:`root_music`, the rows of R
    must follow increasing x and spacing above half a wavelength is
    ambiguous.

    LS-ESPRIT does not reach the Cramer-Rao bound: for one source at high
    SNR its variance is about ``N (N + 1) / (6 (N - 1))`` times the bound
    (1.7 for N = 8).

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> X = pa.generate_snapshots(geom, k, [(-10, 0), (25, 0)], 20, 200,
    ...                           seed=0)
    >>> R = pa.sample_covariance(X)
    >>> np.round(pa.esprit(R, 2, spacing=0.5), 1).tolist()
    [-10.0, 25.0]
    """
    R = _check_covariance(R)
    spacing = _check_positive("spacing", spacing)
    wavelength = _check_positive("wavelength", wavelength)
    m = _check_n_sources(n_sources, R.shape[0])
    if m == 0:
        return np.empty(0)
    _, vecs = _sorted_eigh(R)
    E_s = vecs[:, :m]
    phi, *_ = np.linalg.lstsq(E_s[:-1], E_s[1:], rcond=None)
    return _ula_angles(np.angle(np.linalg.eigvals(phi)), spacing, wavelength)


def estimate_num_sources(
    R: np.ndarray,
    n_snapshots: int,
    method: str = "mdl",
) -> int:
    """
    Estimate the number of sources with MDL or AIC (Wax and Kailath, 1985).

    Parameters
    ----------
    R : ndarray
        Sample covariance, shape ``(N, N)``.
    n_snapshots : int
        Number of snapshots K used to form R.
    method : {'mdl', 'aic'}, optional
        Information criterion. MDL is consistent (correct as K grows); AIC
        tends to overestimate at large K.

    Returns
    -------
    n_sources : int
        The value of ``m`` in ``0 .. N-1`` that minimizes the criterion.

    Raises
    ------
    ValueError
        If ``n_snapshots <= N``, or R is numerically rank-deficient or not
        positive definite.

    Notes
    -----
    With eigenvalues ``l_1 >= ... >= l_N`` of R and
    ``L(m) = K (N - m) log(a_m / g_m)``, where ``a_m`` and ``g_m`` are the
    arithmetic and geometric means of ``l_(m+1) .. l_N``:

    - ``AIC(m) = 2 L(m) + 2 m (2N - m)``
    - ``MDL(m) = L(m) + m (2N - m) log(K) / 2``

    Both assume white noise of nonzero power and K > N, so a ValueError is
    raised for ``n_snapshots <= N`` or a numerically rank-deficient R
    (with K <= N the smallest ``N - K`` eigenvalues are zero and the
    criteria return a meaningless count). Coherent sources reduce the rank
    of the signal covariance and are undercounted.

    References: M. Wax and T. Kailath, "Detection of signals by information
    theoretic criteria," IEEE Trans. ASSP, 33(2), 1985.

    Examples
    --------
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> X = pa.generate_snapshots(geom, k, [(-10, 0), (25, 0)], 10, 200,
    ...                           seed=0)
    >>> pa.estimate_num_sources(pa.sample_covariance(X), 200)
    2
    """
    R = _check_covariance(R)
    method = method.lower()
    if method not in ("mdl", "aic"):
        raise ValueError(f"method must be 'mdl' or 'aic', got {method!r}")
    n_snapshots = _check_n_snapshots(n_snapshots)
    n_el = R.shape[0]
    if n_snapshots <= n_el:
        raise ValueError(
            f"MDL/AIC need more snapshots than elements: n_snapshots = "
            f"{n_snapshots}, n_elements = {n_el}"
        )
    vals, _ = _sorted_eigh(R)
    # Same tolerance as numpy.linalg.matrix_rank; also catches indefinite R
    if vals[-1] <= n_el * np.finfo(float).eps * abs(vals[0]):
        rank = int(np.sum(vals > n_el * np.finfo(float).eps * abs(vals[0])))
        raise ValueError(
            f"R is rank-deficient or not positive definite (numerical rank "
            f"{rank} of {n_el}); MDL and AIC assume white noise of nonzero "
            "power"
        )

    scores = np.empty(n_el)
    for m in range(n_el):
        tail = vals[m:]
        log_ratio = np.log(np.mean(tail)) - np.mean(np.log(tail))
        L = n_snapshots * (n_el - m) * log_ratio
        n_free = m * (2 * n_el - m)
        if method == "mdl":
            scores[m] = L + 0.5 * n_free * np.log(n_snapshots)
        else:
            scores[m] = 2 * L + 2 * n_free
    return int(np.argmin(scores))


def find_spectrum_peaks(
    spectrum: np.ndarray,
    angles_deg: np.ndarray,
    n_sources: int,
) -> np.ndarray:
    """
    Angles of the largest local maxima of a 1-D spectrum.

    Parameters
    ----------
    spectrum : ndarray
        1-D spectrum (linear or dB; only the ordering matters).
    angles_deg : ndarray
        Angles of the spectrum samples, same length.
    n_sources : int
        Number of peaks to return.

    Returns
    -------
    peak_angles_deg : ndarray
        Angles of the ``n_sources`` highest local maxima, sorted by angle.
        Fewer are returned if the spectrum has fewer local maxima.

    Notes
    -----
    A local maximum is an interior sample higher than its neighbors
    (:func:`scipy.signal.find_peaks`; a flat top counts once, at its
    middle). The first and last samples are never peaks, so a source at the
    edge of the scan range is missed; extend the scan past it. Accuracy is
    limited to the grid step.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> angles = np.linspace(-90, 90, 181)
    >>> spectrum = np.exp(-(angles - 30) ** 2 / 20) + 0.5 * np.exp(-(angles + 40) ** 2 / 20)
    >>> pa.find_spectrum_peaks(spectrum, angles, 2).tolist()
    [-40.0, 30.0]
    """
    spectrum = np.asarray(spectrum, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if spectrum.ndim != 1 or spectrum.shape != angles_deg.shape:
        raise ValueError(
            "spectrum and angles_deg must be 1-D arrays of the same length, "
            f"got shapes {spectrum.shape} and {angles_deg.shape}"
        )
    n = int(n_sources)
    if n < 0:
        raise ValueError("n_sources must be non-negative")
    idx, _ = find_peaks(spectrum)
    top = idx[np.argsort(spectrum[idx])[::-1][:n]]
    return np.sort(angles_deg[top])


def crb_ula_single_source(
    n_elements: int,
    spacing: float,
    snr_db: float,
    n_snapshots: int,
    theta_deg: float,
    wavelength: float = 1.0,
) -> float:
    """
    Cramer-Rao bound on the angle of one source seen by a ULA.

    Parameters
    ----------
    n_elements : int
        Number of elements N (at least 2).
    spacing : float
        Element spacing d in meters.
    snr_db : float
        Per-element, per-snapshot SNR in dB (signal power over noise power
        at one element), as in :func:`generate_snapshots`.
    n_snapshots : int
        Number of snapshots K.
    theta_deg : float
        Arrival angle from broadside in degrees.
    wavelength : float, optional
        Wavelength in meters (default 1.0).

    Returns
    -------
    std_deg : float
        Square root of the bound on ``var(theta)``, in degrees.

    Notes
    -----
    Model: K snapshots ``x(t) = s(t) a(psi) + n(t)`` with
    ``a_n = exp(j n psi)``, ``psi = k d sin(theta)``, unknown deterministic
    amplitudes ``s(t)`` and white circular Gaussian noise. With
    ``SNR = mean |s(t)|^2 / sigma^2``,

    ``var(psi) >= 6 / (K SNR N (N^2 - 1))``

    and ``var(theta) = var(psi) / (k d cos(theta))^2``. This is the
    deterministic (conditional) bound. The stochastic bound, for Gaussian
    amplitudes, is larger by ``1 + 1 / (N SNR)``, which is negligible at
    high SNR. The bound grows without limit toward endfire.

    Examples
    --------
    >>> import phased_array as pa
    >>> round(pa.crb_ula_single_source(8, 0.5, 30, 200, 10), 4)
    0.0045
    """
    n = int(n_elements)
    if n != n_elements or n < 2:
        raise ValueError(f"n_elements must be an integer >= 2, got {n_elements}")
    spacing = _check_positive("spacing", spacing)
    wavelength = _check_positive("wavelength", wavelength)
    n_snapshots = _check_n_snapshots(n_snapshots)
    snr = 10.0 ** (snr_db / 10.0)
    var_psi = 6.0 / (n_snapshots * snr * n * (n**2 - 1))
    k = 2 * np.pi / wavelength
    dpsi_dtheta = k * spacing * np.cos(np.deg2rad(theta_deg))
    return float(np.rad2deg(np.sqrt(var_psi) / abs(dpsi_dtheta)))
