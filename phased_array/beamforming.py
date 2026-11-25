"""
Beamforming functions for phased arrays.

Includes amplitude tapering, null steering, and multiple simultaneous beams.
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal

from .core import steering_vector
from .geometry import ArrayGeometry

ArrayLike = Union[np.ndarray, float]


# ============== Amplitude Tapering ==============

def taylor_taper_1d(
    n: int,
    sidelobe_dB: float = -30.0,
    nbar: int = 4
) -> np.ndarray:
    """
    Generate 1D Taylor window for sidelobe control.

    Parameters
    ----------
    n : int
        Number of elements
    sidelobe_dB : float
        Desired peak sidelobe level in dB. Either sign is accepted
        (-30 and 30 both mean sidelobes 30 dB below the peak).
    nbar : int
        Number of nearly equal-level sidelobes

    Returns
    -------
    taper : ndarray
        Amplitude taper weights (n,)

    Notes
    -----
    The weights sample the continuous Taylor line-source distribution
    (``scipy.signal.windows.taylor``), so the achieved peak sidelobe level
    is close to, but not exactly, the design value. Choose
    ``nbar >= 2*A**2 + 0.5`` with ``A = arccosh(10**(|sidelobe_dB|/20))/pi``
    (about 7 for -40 dB). Small arrays with deep sidelobes fall short:
    N = 16 designed for -40 dB reaches about -39.3 dB with nbar = 7.
    """
    # Use scipy's Taylor window
    return signal.windows.taylor(n, nbar=nbar, sll=abs(sidelobe_dB), norm=True)


def taylor_taper_2d(
    Nx: int,
    Ny: int,
    sidelobe_dB: float = -30.0,
    nbar: int = 4
) -> np.ndarray:
    """
    Generate 2D Taylor window (separable product).

    Parameters
    ----------
    Nx, Ny : int
        Number of elements in x and y
    sidelobe_dB : float
        Desired peak sidelobe level in dB
    nbar : int
        Number of nearly equal-level sidelobes

    Returns
    -------
    taper : ndarray
        2D amplitude taper (Nx, Ny), flattened row-major

    Examples
    --------
    Apply Taylor taper for -30 dB sidelobes:

    >>> import phased_array as pa
    >>> taper = pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)
    >>> taper.shape
    (256,)

    Combine with steering weights:

    >>> geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
    >>> weights_tapered = weights * pa.taylor_taper_2d(16, 16, sidelobe_dB=-35)

    Compare taper efficiency:

    >>> taper = pa.taylor_taper_2d(16, 16, sidelobe_dB=-40)
    >>> efficiency = pa.compute_taper_efficiency(taper)
    >>> efficiency < 1.0  # Tapering reduces efficiency
    True
    """
    tx = taylor_taper_1d(Nx, sidelobe_dB, nbar)
    ty = taylor_taper_1d(Ny, sidelobe_dB, nbar)
    taper_2d = np.outer(tx, ty)
    return taper_2d.ravel()


def chebyshev_taper_1d(
    n: int,
    sidelobe_dB: float = -30.0
) -> np.ndarray:
    """
    Generate 1D Dolph-Chebyshev window for equi-ripple sidelobes.

    Parameters
    ----------
    n : int
        Number of elements
    sidelobe_dB : float
        Desired sidelobe level in dB. Either sign is accepted.

    Returns
    -------
    taper : ndarray
        Amplitude taper weights

    Notes
    -----
    For large arrays with modest sidelobe levels the edge elements carry
    noticeably more weight than their neighbors (Haupt Fig. 3.11); prefer
    a Taylor taper there.
    """
    with warnings.catch_warnings():
        # SciPy warns that chebwin is unsuitable for spectral analysis
        # above -45 dB; that concern does not apply to array tapers.
        warnings.filterwarnings(
            "ignore", message=".*not suitable for spectral analysis.*",
            category=UserWarning,
        )
        return signal.windows.chebwin(n, at=abs(sidelobe_dB))


def chebyshev_taper_2d(
    Nx: int,
    Ny: int,
    sidelobe_dB: float = -30.0
) -> np.ndarray:
    """
    Generate 2D Chebyshev window (separable product).

    Parameters
    ----------
    Nx, Ny : int
        Number of elements
    sidelobe_dB : float
        Desired sidelobe level in dB

    Returns
    -------
    taper : ndarray
        2D taper, flattened
    """
    tx = chebyshev_taper_1d(Nx, sidelobe_dB)
    ty = chebyshev_taper_1d(Ny, sidelobe_dB)
    return np.outer(tx, ty).ravel()


def _bayliss_parameters(sidelobe_dB: float) -> Tuple[float, np.ndarray, float]:
    """Bayliss A, the first four zero parameters and the peak position.

    Polynomial fits in the sidelobe level (positive dB), Haupt, *Antenna
    Arrays*, eqs. 3.34-3.35 (after Bayliss 1968 and Elliott).
    """
    s = abs(float(sidelobe_dB))
    A = 0.3038753 + s * (0.05042922 + s * (-2.7989e-4 + s * (3.43e-6 - 2e-8 * s)))
    xi = np.array([
        0.9858302 + s * (0.0333885 + s * (1.4064e-4 + s * (-1.9e-6 + 1e-8 * s))),
        2.00337487 + s * (0.01141548 + s * (4.159e-4 + s * (-3.73e-6 + 1e-8 * s))),
        3.00636321 + s * (0.00683394 + s * (2.9281e-4 + s * (-1.61e-6))),
        4.00518423 + s * (0.00501795 + s * (2.1735e-4 + s * (-8.8e-7))),
    ])
    peak = 0.4797212 + s * (0.01456692 + s * (-1.8739e-4 + s * (2.18e-6 - 1e-8 * s)))
    return A, xi, peak


def bayliss_taper_1d(
    n: int,
    sidelobe_dB: float = -30.0,
    nbar: int = 5
) -> np.ndarray:
    """
    Bayliss difference-pattern taper.

    The difference-pattern counterpart of the Taylor taper: an
    antisymmetric distribution whose pattern has an exact null on
    boresight, ``nbar - 1`` nearly equal sidelobes at ``sidelobe_dB``
    next to the two difference lobes, and sidelobes that decay beyond
    them.

    Parameters
    ----------
    n : int
        Number of elements
    sidelobe_dB : float
        Design sidelobe level relative to the difference-pattern peak, in
        dB. Either sign is accepted.
    nbar : int
        Number of nearly equal sidelobes plus one (at least 2). As for the
        Taylor taper, deeper designs need larger values; for -35 dB use 6
        or more.

    Returns
    -------
    taper : ndarray
        Antisymmetric real weights (n,), ``taper[::-1] == -taper``, scaled
        to a peak magnitude of 1. Multiply by steering weights for a
        steered difference beam.

    Notes
    -----
    The weights sample the Bayliss line-source distribution
    ``g(p) = sum_m B_m sin((m + 1/2) pi p)``, ``-1 <= p <= 1``, at the
    element centers, with pattern zeros from Haupt, *Antenna Arrays*,
    eqs. 3.33-3.34 (Bayliss 1968). For 32 to 128 elements the achieved
    peak sidelobe is within about 0.4 dB of the design when ``nbar`` is
    large enough. The difference peak lies near
    ``z = L u / lambda`` given by Haupt eq. 3.35; the sampled designs put
    it 2-6 percent farther out (6 percent at -20 dB, 2 percent at
    -35 dB). (The worked example on Haupt p. 141
    uses the Taylor parameter A = 0.95 instead of Bayliss's B, a
    misprint.)

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> w = pa.bayliss_taper_1d(64, -30, nbar=5)
    >>> bool(np.allclose(w[::-1], -w))
    True
    >>> geom = pa.create_rectangular_array(64, 1, 0.5, 0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, w, k, n_points=8001)
    >>> right = angles >= 0
    >>> peak = cut[right].max()
    >>> i = np.argmax(cut[right]); r = cut[right][i:]
    >>> j = np.argmax(np.diff(r) > 0)          # first null beyond the lobe
    >>> round(float(r[j:].max() - peak), 1)    # peak sidelobe in dB
    -30.3
    """
    if nbar < 2:
        raise ValueError(f"nbar must be at least 2, got {nbar}")
    A, xi, _ = _bayliss_parameters(sidelobe_dB)

    zeros = []
    for m in range(1, nbar):
        if m <= 4:
            zeros.append((nbar + 0.5) * xi[m - 1] / np.sqrt(A**2 + nbar**2))
        else:
            zeros.append((nbar + 0.5) * np.sqrt((A**2 + m**2) / (A**2 + nbar**2)))
    zeros = np.asarray(zeros)

    coeffs = []
    for m in range(nbar):
        mu = m + 0.5
        numerator = mu**2 * np.prod(1 - mu**2 / zeros**2)
        denominator = np.prod([1 - mu**2 / (q + 0.5)**2 for q in range(nbar) if q != m])
        coeffs.append((-1) ** m * numerator / denominator)

    p = (2 * np.arange(n) - (n - 1)) / n
    g = np.zeros(n)
    for m, b in enumerate(coeffs):
        g += b * np.sin((m + 0.5) * np.pi * p)
    return g / np.max(np.abs(g))


def bayliss_taper_2d(
    Nx: int,
    Ny: int,
    sidelobe_dB: float = -30.0,
    nbar: int = 5,
    axis: str = 'x',
    sum_sidelobe_dB: Optional[float] = None,
    sum_nbar: int = 5
) -> np.ndarray:
    """
    Separable 2D difference taper: Bayliss along one axis, sum taper along
    the other.

    Parameters
    ----------
    Nx, Ny : int
        Number of elements in x and y
    sidelobe_dB : float
        Bayliss design sidelobe level (dB) along the difference axis
    nbar : int
        Bayliss ``nbar``
    axis : str
        'x' for an azimuth (x) difference pattern, 'y' for elevation
    sum_sidelobe_dB : float, optional
        Taylor sidelobe level for the other axis; uniform if omitted
    sum_nbar : int
        Taylor ``nbar`` for the other axis

    Returns
    -------
    taper : ndarray
        Weights flattened in the order of ``create_rectangular_array``
        (x index major), shape (Nx * Ny,)

    Examples
    --------
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> w_delta = pa.steering_vector(k, geom.x, geom.y, 10, 0) * \\
    ...     pa.bayliss_taper_2d(16, 16, -30, axis='x', sum_sidelobe_dB=-30)
    >>> w_delta.shape
    (256,)
    """
    if axis not in ('x', 'y'):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    n_diff, n_sum = (Nx, Ny) if axis == 'x' else (Ny, Nx)
    diff = bayliss_taper_1d(n_diff, sidelobe_dB, nbar)
    if sum_sidelobe_dB is None:
        other = np.ones(n_sum)
    else:
        other = taylor_taper_1d(n_sum, sum_sidelobe_dB, sum_nbar)
    tx, ty = (diff, other) if axis == 'x' else (other, diff)
    return np.outer(tx, ty).ravel()


def hamming_taper_1d(n: int) -> np.ndarray:
    """Generate 1D Hamming window."""
    return np.hamming(n)


def hamming_taper_2d(Nx: int, Ny: int) -> np.ndarray:
    """Generate 2D Hamming window."""
    return np.outer(np.hamming(Nx), np.hamming(Ny)).ravel()


def hanning_taper_1d(n: int) -> np.ndarray:
    """Generate 1D Hanning (Hann) window."""
    return np.hanning(n)


def hanning_taper_2d(Nx: int, Ny: int) -> np.ndarray:
    """Generate 2D Hanning window."""
    return np.outer(np.hanning(Nx), np.hanning(Ny)).ravel()


def cosine_taper_1d(n: int) -> np.ndarray:
    """Generate 1D cosine (sine) window."""
    return np.sin(np.pi * np.arange(n) / (n - 1))


def cosine_taper_2d(Nx: int, Ny: int) -> np.ndarray:
    """Generate 2D cosine window."""
    return np.outer(cosine_taper_1d(Nx), cosine_taper_1d(Ny)).ravel()


def cosine_on_pedestal_taper_1d(
    n: int,
    pedestal: float = 0.1
) -> np.ndarray:
    """
    Generate 1D cosine-on-pedestal window.

    Parameters
    ----------
    n : int
        Number of elements
    pedestal : float
        Minimum amplitude at edges (0 to 1)

    Returns
    -------
    taper : ndarray
        Amplitude taper
    """
    t = np.linspace(0, np.pi, n)
    return pedestal + (1 - pedestal) * np.sin(t)


def cosine_on_pedestal_taper_2d(
    Nx: int,
    Ny: int,
    pedestal: float = 0.1
) -> np.ndarray:
    """Generate 2D cosine-on-pedestal window."""
    tx = cosine_on_pedestal_taper_1d(Nx, pedestal)
    ty = cosine_on_pedestal_taper_1d(Ny, pedestal)
    return np.outer(tx, ty).ravel()


def gaussian_taper_1d(n: int, sigma: float = 0.4) -> np.ndarray:
    """
    Generate 1D Gaussian window.

    Parameters
    ----------
    n : int
        Number of elements
    sigma : float
        Standard deviation as fraction of array (typical: 0.3-0.5)

    Returns
    -------
    taper : ndarray
    """
    x = np.linspace(-0.5, 0.5, n)
    return np.exp(-0.5 * (x / sigma) ** 2)


def gaussian_taper_2d(Nx: int, Ny: int, sigma: float = 0.4) -> np.ndarray:
    """Generate 2D Gaussian window."""
    return np.outer(gaussian_taper_1d(Nx, sigma), gaussian_taper_1d(Ny, sigma)).ravel()


def compute_taper_efficiency(taper: np.ndarray) -> float:
    """
    Compute aperture efficiency for a taper.

    Efficiency = (sum of weights)^2 / (N * sum of weights^2)
    Uniform illumination gives 100% efficiency.

    Parameters
    ----------
    taper : ndarray
        Amplitude taper weights

    Returns
    -------
    efficiency : float
        Aperture efficiency (0 to 1)
    """
    n = len(taper)
    return (np.sum(taper) ** 2) / (n * np.sum(taper ** 2))


def compute_taper_directivity_loss(taper: np.ndarray) -> float:
    """
    Compute directivity loss due to tapering in dB.

    Parameters
    ----------
    taper : ndarray
        Amplitude taper weights

    Returns
    -------
    loss_dB : float
        Directivity loss relative to uniform illumination (negative or zero)
    """
    efficiency = compute_taper_efficiency(taper)
    return 10 * np.log10(efficiency)


def apply_taper_to_geometry(
    geometry: ArrayGeometry,
    taper_func: str = 'taylor',
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    **taper_kwargs
) -> np.ndarray:
    """
    Apply an amplitude taper based on element positions.

    For non-rectangular arrays, uses radial distance from center.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    taper_func : str
        'taylor', 'chebyshev', 'hamming', 'hanning', 'gaussian', 'cosine'
    Nx, Ny : int, optional
        For rectangular arrays, specify dimensions
    **taper_kwargs
        Additional arguments for the taper function

    Returns
    -------
    taper : ndarray
        Amplitude weights for each element
    """
    n = geometry.n_elements

    # If Nx, Ny provided, use 2D separable taper
    if Nx is not None and Ny is not None:
        if taper_func == 'taylor':
            return taylor_taper_2d(Nx, Ny, **taper_kwargs)
        elif taper_func == 'chebyshev':
            return chebyshev_taper_2d(Nx, Ny, **taper_kwargs)
        elif taper_func == 'hamming':
            return hamming_taper_2d(Nx, Ny)
        elif taper_func == 'hanning':
            return hanning_taper_2d(Nx, Ny)
        elif taper_func == 'gaussian':
            sigma = taper_kwargs.get('sigma', 0.4)
            return gaussian_taper_2d(Nx, Ny, sigma)
        elif taper_func == 'cosine':
            return cosine_taper_2d(Nx, Ny)

    # For irregular arrays, use radial taper
    r = np.sqrt(geometry.x**2 + geometry.y**2)
    r_max = np.max(r) if np.max(r) > 0 else 1.0
    r_norm = r / r_max  # 0 at center, 1 at edge

    if taper_func == 'gaussian':
        sigma = taper_kwargs.get('sigma', 0.4)
        return np.exp(-0.5 * (r_norm / sigma) ** 2)
    elif taper_func == 'cosine':
        return np.cos(np.pi * r_norm / 2)
    elif taper_func == 'taylor':
        # Approximate Taylor with a polynomial
        sll = -taper_kwargs.get('sidelobe_dB', -30.0)
        # Simple approximation: cosine-squared on pedestal
        pedestal = 10 ** (-sll / 40)
        return pedestal + (1 - pedestal) * np.cos(np.pi * r_norm / 2) ** 2
    else:
        # Default to uniform
        return np.ones(n)


# ============== Null Steering ==============

def null_steering_projection(
    geometry: ArrayGeometry,
    k: float,
    theta_main_deg: float,
    phi_main_deg: float,
    null_directions: List[Tuple[float, float]],
    initial_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute weights with nulls using orthogonal projection.

    Projects the desired weight vector onto the null-space of the
    steering vectors for the null directions.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    theta_main_deg : float
        Main beam theta in degrees
    phi_main_deg : float
        Main beam phi in degrees
    null_directions : list of (theta_deg, phi_deg) tuples
        Directions for placing nulls
    initial_weights : ndarray, optional
        Starting weights (default: uniform with steering)

    Returns
    -------
    weights : ndarray
        Complex weights with nulls placed

    Examples
    --------
    Place nulls at specific interference directions:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> # Main beam at 30 deg, nulls at 45 and 60 degrees
    >>> null_dirs = [(45, 0), (60, 0)]
    >>> weights = pa.null_steering_projection(
    ...     geom, k, theta_main_deg=30, phi_main_deg=0,
    ...     null_directions=null_dirs
    ... )
    >>> weights.shape
    (256,)

    Verify null depth:

    >>> null_depth = pa.compute_null_depth(
    ...     weights, geom, k, theta_deg=45, phi_deg=0,
    ...     theta_main_deg=30, phi_main_deg=0
    ... )
    >>> null_depth < -30  # Deep null achieved
    True
    """
    n = geometry.n_elements

    # Initial steering weights
    if initial_weights is None:
        initial_weights = steering_vector(
            k, geometry.x, geometry.y,
            theta_main_deg, phi_main_deg,
            geometry.z
        )

    if len(null_directions) == 0:
        return initial_weights

    # Build constraint matrix (columns are steering vectors for null directions)
    C = np.zeros((n, len(null_directions)), dtype=complex)
    for i, (theta_null, phi_null) in enumerate(null_directions):
        C[:, i] = steering_vector(
            k, geometry.x, geometry.y,
            theta_null, phi_null,
            geometry.z
        )

    # Project initial weights onto null space of C
    # P_null = I - C(C^H C)^-1 C^H
    try:
        CTC_inv = np.linalg.inv(C.conj().T @ C)
        P_null = np.eye(n) - C @ CTC_inv @ C.conj().T
        weights = P_null @ initial_weights
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudoinverse
        P_null = np.eye(n) - C @ np.linalg.pinv(C)
        weights = P_null @ initial_weights

    return weights


def null_steering_lcmv(
    geometry: ArrayGeometry,
    k: float,
    constraints: List[Tuple[float, float, complex]],
    noise_covariance: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    LCMV (Linearly Constrained Minimum Variance) beamformer.

    Minimizes output power subject to linear constraints on response
    in specified directions.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    constraints : list of (theta_deg, phi_deg, response) tuples
        Each constraint specifies (direction, desired complex response)
        Use response=1+0j for unity gain, response=0 for null
    noise_covariance : ndarray, optional
        N x N noise covariance matrix (default: identity)

    Returns
    -------
    weights : ndarray
        Complex LCMV weights
    """
    n = geometry.n_elements

    if noise_covariance is None:
        R = np.eye(n, dtype=complex)
    else:
        R = noise_covariance

    # Build constraint matrix and response vector
    n_constraints = len(constraints)
    C = np.zeros((n, n_constraints), dtype=complex)
    f = np.zeros(n_constraints, dtype=complex)

    for i, (theta, phi, response) in enumerate(constraints):
        C[:, i] = steering_vector(
            k, geometry.x, geometry.y,
            theta, phi, geometry.z
        )
        f[i] = response

    # LCMV solution: w = R^-1 C (C^H R^-1 C)^-1 f
    try:
        R_inv = np.linalg.inv(R)
        R_inv_C = R_inv @ C
        weights = R_inv_C @ np.linalg.inv(C.conj().T @ R_inv_C) @ f
    except np.linalg.LinAlgError:
        # Use pseudoinverse if singular
        R_inv = np.linalg.pinv(R)
        R_inv_C = R_inv @ C
        weights = R_inv_C @ np.linalg.pinv(C.conj().T @ R_inv_C) @ f

    return weights


def compute_null_depth(
    weights: np.ndarray,
    geometry: ArrayGeometry,
    k: float,
    theta_deg: float,
    phi_deg: float,
    theta_main_deg: float,
    phi_main_deg: float
) -> float:
    """
    Compute null depth relative to main beam in dB.

    Parameters
    ----------
    weights : ndarray
        Array weights
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    theta_deg, phi_deg : float
        Null direction
    theta_main_deg, phi_main_deg : float
        Main beam direction

    Returns
    -------
    depth_dB : float
        Null depth (negative, lower is deeper)
    """
    from .core import array_factor_vectorized

    # Response at null
    theta_null = np.array([[np.deg2rad(theta_deg)]])
    phi_null = np.array([[np.deg2rad(phi_deg)]])
    AF_null = array_factor_vectorized(
        theta_null, phi_null,
        geometry.x, geometry.y, weights, k, geometry.z
    )

    # Response at main beam
    theta_main = np.array([[np.deg2rad(theta_main_deg)]])
    phi_main = np.array([[np.deg2rad(phi_main_deg)]])
    AF_main = array_factor_vectorized(
        theta_main, phi_main,
        geometry.x, geometry.y, weights, k, geometry.z
    )

    ratio = np.abs(AF_null) / np.abs(AF_main)
    if ratio > 0:
        return 20 * np.log10(ratio.item())
    else:
        return -200.0  # Very deep null


# ============== Multiple Simultaneous Beams ==============

def multi_beam_weights_superposition(
    geometry: ArrayGeometry,
    k: float,
    beam_directions: List[Tuple[float, float]],
    amplitudes: Optional[List[float]] = None,
    tapers: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Compute weights for multiple simultaneous beams via superposition.

    The weights are a weighted sum of individual steering vectors.
    This creates multiple beams but with reduced gain per beam.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    beam_directions : list of (theta_deg, phi_deg) tuples
        Directions for each beam
    amplitudes : list of float, optional
        Relative amplitude for each beam (default: equal)
    tapers : list of ndarray, optional
        Amplitude taper for each beam (default: uniform)

    Returns
    -------
    weights : ndarray
        Combined complex weights
    """
    n_beams = len(beam_directions)

    if amplitudes is None:
        amplitudes = [1.0] * n_beams

    weights = np.zeros(geometry.n_elements, dtype=complex)

    for i, (theta, phi) in enumerate(beam_directions):
        sv = steering_vector(
            k, geometry.x, geometry.y,
            theta, phi, geometry.z
        )

        if tapers is not None and i < len(tapers):
            sv = sv * tapers[i]

        weights += amplitudes[i] * sv

    return weights


def multi_beam_weights_orthogonal(
    geometry: ArrayGeometry,
    k: float,
    beam_directions: List[Tuple[float, float]]
) -> List[np.ndarray]:
    """
    Compute separate weight vectors for orthogonal digital beamforming.

    Each beam has its own weight vector, allowing digital combination
    with no loss per beam (requires N receive channels).

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    beam_directions : list of (theta_deg, phi_deg) tuples
        Directions for each beam

    Returns
    -------
    weight_vectors : list of ndarray
        List of complex weight vectors, one per beam
    """
    weight_vectors = []

    for theta, phi in beam_directions:
        weights = steering_vector(
            k, geometry.x, geometry.y,
            theta, phi, geometry.z
        )
        weight_vectors.append(weights)

    return weight_vectors


def compute_beam_isolation(
    weights_list: List[np.ndarray],
    geometry: ArrayGeometry,
    k: float,
    beam_directions: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Compute isolation between multiple beams.

    Parameters
    ----------
    weights_list : list of ndarray
        Weight vector for each beam
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    beam_directions : list of (theta_deg, phi_deg)
        Direction of each beam

    Returns
    -------
    isolation_matrix : ndarray
        N_beams x N_beams matrix of isolation in dB
        Diagonal is 0 dB, off-diagonal is negative (isolation)
    """
    from .core import array_factor_vectorized

    n_beams = len(weights_list)
    isolation = np.zeros((n_beams, n_beams))

    # Compute response of each beam's weights at each beam's direction
    for i in range(n_beams):
        for j in range(n_beams):
            theta = np.array([[np.deg2rad(beam_directions[j][0])]])
            phi = np.array([[np.deg2rad(beam_directions[j][1])]])

            AF = array_factor_vectorized(
                theta, phi,
                geometry.x, geometry.y,
                weights_list[i], k, geometry.z
            )

            AF_main = array_factor_vectorized(
                np.array([[np.deg2rad(beam_directions[i][0])]]),
                np.array([[np.deg2rad(beam_directions[i][1])]]),
                geometry.x, geometry.y,
                weights_list[i], k, geometry.z
            )

            if np.abs(AF_main) > 0:
                ratio = np.abs(AF) / np.abs(AF_main)
                isolation[i, j] = 20 * np.log10(ratio.item()) if ratio > 0 else -200
            else:
                isolation[i, j] = -200

    return isolation


def monopulse_weights(
    geometry: ArrayGeometry,
    k: float,
    theta0_deg: float,
    phi0_deg: float,
    mode: str = 'sum'
) -> np.ndarray:
    """
    Compute monopulse tracking weights.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    theta0_deg : float
        Nominal beam direction theta
    phi0_deg : float
        Nominal beam direction phi
    mode : str
        'sum' - sum pattern (normal beam)
        'delta_az' - azimuth difference pattern (split in x)
        'delta_el' - elevation difference pattern (split in y)

    Returns
    -------
    weights : ndarray
        Complex weights for the specified mode

    Notes
    -----
    The difference halves are split about the array centroid, and
    elements lying on the split line (the center column or row of an
    odd-sized array) get zero weight, giving an exact boresight null.
    The difference weights are amplitude-uniform; multiply by a taper
    for lower difference-pattern sidelobes.
    """
    # Base steering vector
    sv = steering_vector(
        k, geometry.x, geometry.y,
        theta0_deg, phi0_deg, geometry.z
    )

    if mode == 'sum':
        return sv

    elif mode in ('delta_az', 'delta_el'):
        # Difference pattern: opposite signs on the two halves of the
        # aperture, split about the array centroid. Elements on the split
        # line get zero weight so the halves cancel exactly on boresight.
        coord = np.asarray(geometry.x if mode == 'delta_az' else geometry.y,
                           dtype=float)
        if np.ptp(coord) == 0:
            axis = 'x' if mode == 'delta_az' else 'y'
            raise ValueError(
                f"{mode} needs element positions that vary in {axis}; this "
                "array has no extent in that direction, so it has no "
                "difference pattern in that plane"
            )
        offset = coord - coord.mean()
        tol = 1e-9 * max(float(np.ptp(coord)), 1e-30)
        sign = np.where(np.abs(offset) <= tol, 0.0, np.sign(offset))
        return sv * sign

    else:
        raise ValueError(f"Unknown monopulse mode: {mode}")


# ============== Beam Spoiling ==============

def quadratic_phase_spoil(
    geometry: ArrayGeometry,
    k: float,
    theta0_deg: float,
    phi0_deg: float,
    spoil_factor: float,
    axis: str = 'both'
) -> np.ndarray:
    """
    Compute beam spoiling weights using quadratic phase distribution.

    Quadratic phase across the aperture broadens the beam by introducing
    a defocusing effect, similar to moving away from the focal plane.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber (2*pi/wavelength)
    theta0_deg : float
        Steering direction theta in degrees
    phi0_deg : float
        Steering direction phi in degrees
    spoil_factor : float
        Spoiling factor controlling beam broadening.
        Higher values = broader beam. Typical range: 0.5 to 5.0
    axis : str
        'both' - spoil in both x and y
        'x' - spoil only in x direction
        'y' - spoil only in y direction

    Returns
    -------
    weights : ndarray
        Complex weights with steering and quadratic phase spoiling

    Examples
    --------
    Create a spoiled beam for search mode:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> weights = pa.quadratic_phase_spoil(
    ...     geom, k, theta0_deg=0, phi0_deg=0, spoil_factor=2.0
    ... )
    >>> weights.shape
    (256,)

    Compute expected beamwidth:

    >>> bw_unspoiled = 6.0  # degrees (approximate for 16x16 at lambda/2)
    >>> bw_spoiled = pa.spoiled_beamwidth(bw_unspoiled, spoil_factor=2.0)
    >>> bw_spoiled > bw_unspoiled
    True

    Notes
    -----
    The quadratic phase distribution is:
        phi_quad = spoil_factor * (x^2 + y^2) / aperture^2

    This creates a beam that is approximately sqrt(1 + spoil_factor^2)
    times wider than the unspoiled beam.
    """
    # Get steering weights first
    weights = steering_vector(
        k, geometry.x, geometry.y,
        theta0_deg, phi0_deg, geometry.z
    )

    # Compute aperture size
    x_span = np.max(geometry.x) - np.min(geometry.x)
    y_span = np.max(geometry.y) - np.min(geometry.y)

    # Normalize positions to [-1, 1]
    x_center = (np.max(geometry.x) + np.min(geometry.x)) / 2
    y_center = (np.max(geometry.y) + np.min(geometry.y)) / 2

    if x_span > 0:
        x_norm = (geometry.x - x_center) / (x_span / 2)
    else:
        x_norm = np.zeros_like(geometry.x)

    if y_span > 0:
        y_norm = (geometry.y - y_center) / (y_span / 2)
    else:
        y_norm = np.zeros_like(geometry.y)

    # Compute quadratic phase
    if axis == 'both':
        quad_phase = spoil_factor * np.pi * (x_norm**2 + y_norm**2)
    elif axis == 'x':
        quad_phase = spoil_factor * np.pi * x_norm**2
    elif axis == 'y':
        quad_phase = spoil_factor * np.pi * y_norm**2
    else:
        raise ValueError(f"Unknown axis: {axis}. Use 'both', 'x', or 'y'.")

    # Apply quadratic phase
    weights = weights * np.exp(1j * quad_phase)

    return weights


def compute_spoil_factor(
    geometry: ArrayGeometry,
    desired_beamwidth_deg: float,
    unspoiled_beamwidth_deg: float
) -> float:
    """
    Compute the spoil factor needed to achieve a desired beamwidth.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry (used to validate reasonableness)
    desired_beamwidth_deg : float
        Target beamwidth in degrees
    unspoiled_beamwidth_deg : float
        Natural (unspoiled) beamwidth in degrees

    Returns
    -------
    spoil_factor : float
        Spoil factor to use with quadratic_phase_spoil

    Examples
    --------
    Double the beamwidth:

    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
    >>> sf = pa.compute_spoil_factor(geom, desired_beamwidth_deg=12.0,
    ...                              unspoiled_beamwidth_deg=6.0)
    >>> sf > 0
    True

    Notes
    -----
    The relationship between spoil factor and beamwidth broadening is:
        BW_spoiled / BW_unspoiled ~ sqrt(1 + spoil_factor^2)

    This is an approximation that works well for moderate spoiling.
    """
    if desired_beamwidth_deg <= unspoiled_beamwidth_deg:
        return 0.0

    # BW_spoiled / BW_unspoiled = sqrt(1 + sf^2)
    # (BW_spoiled / BW_unspoiled)^2 = 1 + sf^2
    # sf^2 = (BW_spoiled / BW_unspoiled)^2 - 1
    ratio = desired_beamwidth_deg / unspoiled_beamwidth_deg
    sf_squared = ratio**2 - 1

    if sf_squared < 0:
        return 0.0

    return np.sqrt(sf_squared)


def spoiled_beam_gain(
    n_elements: int,
    element_gain_dBi: float,
    spoil_factor: float,
    taper_efficiency: float = 1.0
) -> float:
    """
    Estimate gain of a spoiled beam.

    Beam spoiling reduces peak gain because power is spread over a
    wider solid angle.

    Parameters
    ----------
    n_elements : int
        Number of array elements
    element_gain_dBi : float
        Single element gain in dBi
    spoil_factor : float
        Spoiling factor used
    taper_efficiency : float
        Aperture taper efficiency (0 to 1)

    Returns
    -------
    gain_dBi : float
        Estimated peak gain in dBi

    Examples
    --------
    >>> import phased_array as pa
    >>> gain_unspoiled = 10 * np.log10(256) + 5  # 256 elements, 5 dBi each
    >>> gain_spoiled = pa.spoiled_beam_gain(256, 5.0, spoil_factor=2.0)
    >>> gain_spoiled < gain_unspoiled  # Spoiling reduces gain
    True

    Notes
    -----
    Gain loss due to spoiling is approximately:
        Loss_dB ~ 10 * log10(1 + spoil_factor^2)

    This corresponds to the beam area increase.
    """
    # Array gain without spoiling
    array_factor_dB = 10 * np.log10(n_elements)
    efficiency_dB = 10 * np.log10(taper_efficiency) if taper_efficiency > 0 else -100

    # Gain loss from spoiling
    spoil_loss_dB = 10 * np.log10(1 + spoil_factor**2)

    # Total gain
    gain_dBi = element_gain_dBi + array_factor_dB + efficiency_dB - spoil_loss_dB

    return gain_dBi


def spoiled_beamwidth(
    unspoiled_beamwidth_deg: float,
    spoil_factor: float
) -> float:
    """
    Estimate the beamwidth of a spoiled beam.

    Parameters
    ----------
    unspoiled_beamwidth_deg : float
        Natural (unspoiled) beamwidth in degrees
    spoil_factor : float
        Spoiling factor used

    Returns
    -------
    beamwidth_deg : float
        Estimated beamwidth of spoiled beam in degrees

    Examples
    --------
    >>> import phased_array as pa
    >>> bw = pa.spoiled_beamwidth(6.0, spoil_factor=2.0)
    >>> round(bw, 1)
    13.4

    Notes
    -----
    The beamwidth broadening factor is approximately:
        BW_spoiled / BW_unspoiled = sqrt(1 + spoil_factor^2)
    """
    broadening = np.sqrt(1 + spoil_factor**2)
    return unspoiled_beamwidth_deg * broadening


# ============== Adaptive Beamforming (SMI/GSC) ==============

def adaptive_weights_smi(
    geometry: ArrayGeometry,
    k: float,
    theta_desired_deg: float,
    phi_desired_deg: float,
    interference_data: np.ndarray,
    diagonal_loading: float = 0.0
) -> np.ndarray:
    """
    Compute adaptive weights using Sample Matrix Inversion (SMI).

    SMI is a direct computation of the Minimum Variance Distortionless
    Response (MVDR) beamformer weights from sample data.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    theta_desired_deg : float
        Desired signal direction theta in degrees
    phi_desired_deg : float
        Desired signal direction phi in degrees
    interference_data : ndarray
        Interference-plus-noise data matrix (n_snapshots x n_elements).
        Each row is a snapshot of received signals. A plane wave from
        direction u contributes ``amplitude * array_manifold(...)``
        (``exp(+j k r . u)``), not ``steering_vector``.
    diagonal_loading : float
        Diagonal loading factor for robustness.
        Adds diagonal_loading * I to the covariance matrix.

    Returns
    -------
    weights : ndarray
        Adaptive complex weights, applied without conjugation: the array
        output is ``weights @ x`` and the pattern is
        ``array_factor_vectorized(..., weights, ...)``. The look-direction
        response ``weights @ array_manifold(look)`` equals 1.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> # Simulate interference data (normally from receiver)
    >>> n_snapshots = 100
    >>> rng = np.random.default_rng(0)
    >>> interference = rng.standard_normal((n_snapshots, geom.n_elements)) + \\
    ...                1j * rng.standard_normal((n_snapshots, geom.n_elements))
    >>> weights = pa.adaptive_weights_smi(
    ...     geom, k, theta_desired_deg=0, phi_desired_deg=0,
    ...     interference_data=interference, diagonal_loading=0.01
    ... )
    >>> weights.shape
    (64,)

    Notes
    -----
    The SMI weight vector is:
        w = R~^(-1) @ s / (s^H @ R~^(-1) @ s)

    where ``s = steering_vector(...)`` and ``R~ = X^H X / K`` for the
    snapshot matrix X (rows are snapshots). ``R~`` is the conjugate of the
    usual covariance ``E[x x^H]``, so ``w`` is the conjugate of the
    textbook MVDR weight and is applied as ``w @ x``.
    Diagonal loading improves robustness when snapshots are limited.
    """
    n = geometry.n_elements

    # Steering vector for desired signal
    s = steering_vector(
        k, geometry.x, geometry.y,
        theta_desired_deg, phi_desired_deg, geometry.z
    )

    # Estimate covariance matrix from data
    interference_data = np.asarray(interference_data, dtype=complex)
    n_snapshots = interference_data.shape[0]

    R = (interference_data.conj().T @ interference_data) / n_snapshots

    # Apply diagonal loading
    if diagonal_loading > 0:
        R = R + diagonal_loading * np.eye(n)

    # Compute MVDR weights: w = R^(-1) @ s / (s^H @ R^(-1) @ s)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    R_inv_s = R_inv @ s
    denominator = s.conj().T @ R_inv_s

    if np.abs(denominator) < 1e-15:
        denominator = 1e-15

    weights = R_inv_s / denominator

    return weights


def adaptive_weights_gsc(
    geometry: ArrayGeometry,
    k: float,
    theta_desired_deg: float,
    phi_desired_deg: float,
    interference_data: np.ndarray,
    n_blocking_vectors: Optional[int] = None,
    mu: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute adaptive weights using Generalized Sidelobe Canceller (GSC).

    GSC provides a constrained adaptive filter structure that maintains
    distortionless response in the look direction while minimizing
    interference and noise. The adaptive branch uses normalized LMS
    (NLMS) over one pass through the snapshots.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    theta_desired_deg : float
        Desired signal direction theta in degrees
    phi_desired_deg : float
        Desired signal direction phi in degrees
    interference_data : ndarray
        Interference-plus-noise data (n_snapshots x n_elements). A plane
        wave from direction u contributes ``amplitude * array_manifold``.
    n_blocking_vectors : int, optional
        Number of blocking matrix columns. Default: n_elements - 1
    mu : float
        Normalized LMS step size, stable for ``0 < mu < 2``. Each update
        is scaled by ``1 / ||x_b||^2``, so the step does not depend on the
        input power. Larger values adapt faster with more misadjustment.

    Returns
    -------
    weights : ndarray
        Adaptive weights (n_elements,), applied without conjugation like
        :func:`adaptive_weights_smi`; the look-direction response is 1.
    blocking_matrix : ndarray
        Blocking matrix used (n_elements x n_blocking_vectors), with
        ``B.T @ array_manifold(look) = 0``

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> rng = np.random.default_rng(0)
    >>> interference = rng.standard_normal((50, geom.n_elements)) + \\
    ...                1j * rng.standard_normal((50, geom.n_elements))
    >>> weights, B = pa.adaptive_weights_gsc(
    ...     geom, k, theta_desired_deg=0, phi_desired_deg=0,
    ...     interference_data=interference
    ... )
    >>> weights.shape
    (64,)

    Notes
    -----
    The GSC structure is:
        w = w_q - B @ w_a

    where:
        - ``w_q = s / ||s||^2`` is the quiescent steering weight (unit
          look-direction response), with ``s = steering_vector(...)``
        - B is a blocking matrix with ``s^H @ B = 0``, equivalently
          ``B.T @ a0 = 0`` for the look-direction manifold ``a0``
        - ``w_a`` minimizes the output power ``|w.T @ x|^2``

    With ``y_q = w_q @ x``, ``x_b = B.T @ x`` and ``e = y_q - w_a @ x_b``,
    the NLMS update is ``w_a += mu * e * conj(x_b) / (eps + ||x_b||^2)``.
    """
    n = geometry.n_elements

    if n_blocking_vectors is None:
        n_blocking_vectors = n - 1

    # Quiescent steering vector, scaled for unit look-direction response
    s = steering_vector(
        k, geometry.x, geometry.y,
        theta_desired_deg, phi_desired_deg, geometry.z
    )
    w_q = s / np.sum(np.abs(s)**2)

    # Blocking matrix: orthonormal columns orthogonal to s (s^H B = 0),
    # which is B.T @ conj(s) = 0, i.e. the look-direction manifold is
    # rejected by the w.T convention used here.
    s_col = (s / np.linalg.norm(s)).reshape(-1, 1)
    Q, _ = np.linalg.qr(np.hstack([s_col, np.eye(n)[:, :n_blocking_vectors]]))
    B = Q[:, 1:n_blocking_vectors + 1]

    interference_data = np.asarray(interference_data, dtype=complex)
    n_snapshots = interference_data.shape[0]

    w_a = np.zeros(n_blocking_vectors, dtype=complex)
    eps = 1e-12

    # NLMS adaptation, output y = w.T @ x (no conjugation)
    for snapshot_idx in range(n_snapshots):
        x = interference_data[snapshot_idx, :]

        y_q = w_q @ x            # upper (quiescent) branch
        x_b = B.T @ x            # blocked branch
        e = y_q - w_a @ x_b      # GSC output

        w_a = w_a + (mu / (eps + np.vdot(x_b, x_b).real)) * e * x_b.conj()

    weights = w_q - B @ w_a

    return weights, B


def compute_sinr_improvement(
    weights_before: np.ndarray,
    weights_after: np.ndarray,
    geometry: ArrayGeometry,
    k: float,
    signal_direction: Tuple[float, float],
    interference_directions: List[Tuple[float, float]],
    signal_power: float,
    interference_powers: List[float],
    noise_power: float
) -> Tuple[float, float, float]:
    """
    Compute SINR improvement from adaptive beamforming.

    Parameters
    ----------
    weights_before : ndarray
        Weights before adaptation
    weights_after : ndarray
        Weights after adaptation
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    signal_direction : tuple
        (theta_deg, phi_deg) of desired signal
    interference_directions : list
        List of (theta_deg, phi_deg) for each interferer
    signal_power : float
        Signal power (linear)
    interference_powers : list
        Power of each interferer (linear)
    noise_power : float
        Thermal noise power (linear)

    Returns
    -------
    sinr_before : float
        SINR before adaptation in dB
    sinr_after : float
        SINR after adaptation in dB
    improvement_dB : float
        SINR improvement in dB

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> w_q = pa.steering_vector(k, geom.x, geom.y, 0, 0)
    >>> # Create adapted weights that null an interferer in a sidelobe
    >>> w_adapted = pa.null_steering_projection(
    ...     geom, k, theta_main_deg=0, phi_main_deg=0,
    ...     null_directions=[(20, 0)]
    ... )
    >>> sinr_b, sinr_a, imp = pa.compute_sinr_improvement(
    ...     w_q, w_adapted, geom, k,
    ...     signal_direction=(0, 0),
    ...     interference_directions=[(20, 0)],
    ...     signal_power=1.0, interference_powers=[10.0], noise_power=0.1
    ... )
    >>> imp > 0  # Should show improvement
    True
    """
    from .core import array_factor_vectorized

    def compute_sinr(weights):
        # Signal response
        theta_s = np.array([[np.deg2rad(signal_direction[0])]])
        phi_s = np.array([[np.deg2rad(signal_direction[1])]])
        AF_signal = array_factor_vectorized(
            theta_s, phi_s,
            geometry.x, geometry.y, weights, k, geometry.z
        )
        signal_out = signal_power * np.abs(AF_signal.item())**2

        # Interference response
        interference_out = 0.0
        for (theta_i, phi_i), power_i in zip(interference_directions, interference_powers):
            theta_int = np.array([[np.deg2rad(theta_i)]])
            phi_int = np.array([[np.deg2rad(phi_i)]])
            AF_int = array_factor_vectorized(
                theta_int, phi_int,
                geometry.x, geometry.y, weights, k, geometry.z
            )
            interference_out += power_i * np.abs(AF_int.item())**2

        # Noise output (proportional to weight norm squared)
        noise_out = noise_power * np.sum(np.abs(weights)**2)

        # SINR
        sinr = signal_out / (interference_out + noise_out + 1e-20)
        return 10 * np.log10(sinr) if sinr > 0 else -100.0

    sinr_before = compute_sinr(weights_before)
    sinr_after = compute_sinr(weights_after)
    improvement_dB = sinr_after - sinr_before

    return sinr_before, sinr_after, improvement_dB


def plot_adapted_pattern(
    geometry: ArrayGeometry,
    k: float,
    weights_quiescent: np.ndarray,
    weights_adapted: np.ndarray,
    interference_directions: List[Tuple[float, float]],
    title: str = "Adapted Pattern Comparison",
    phi_cut_deg: float = 0.0,
    n_points: int = 361
):
    """
    Plot comparison of quiescent and adapted antenna patterns.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    weights_quiescent : ndarray
        Non-adaptive (quiescent) weights
    weights_adapted : ndarray
        Adaptive weights
    interference_directions : list
        List of (theta_deg, phi_deg) tuples for interferers
    title : str
        Plot title
    phi_cut_deg : float
        Phi angle for the pattern cut in degrees
    n_points : int
        Number of points in pattern

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes object

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> w_q = pa.steering_vector(k, geom.x, geom.y, 0, 0)
    >>> w_a = pa.null_steering_projection(
    ...     geom, k, theta_main_deg=0, phi_main_deg=0,
    ...     null_directions=[(25, 0)]
    ... )
    >>> ax = pa.plot_adapted_pattern(
    ...     geom, k, w_q, w_a,
    ...     interference_directions=[(25, 0)],
    ...     title="Null at 25 degrees"
    ... )  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    from .core import array_factor_vectorized
    from .utils import linear_to_db

    # Compute patterns
    theta = np.linspace(-np.pi/2, np.pi/2, n_points)
    # Negative theta is the opposite side of the cut plane (phi + 180)
    phi = np.where(theta >= 0, np.deg2rad(phi_cut_deg),
                   np.deg2rad(phi_cut_deg) + np.pi)

    theta_grid = np.abs(theta).reshape(-1, 1)
    phi_grid = phi.reshape(-1, 1)

    AF_quiescent = array_factor_vectorized(
        theta_grid, phi_grid,
        geometry.x, geometry.y, weights_quiescent, k, geometry.z
    ).ravel()

    AF_adapted = array_factor_vectorized(
        theta_grid, phi_grid,
        geometry.x, geometry.y, weights_adapted, k, geometry.z
    ).ravel()

    # Convert to dB
    pattern_q = linear_to_db(np.abs(AF_quiescent)**2)
    pattern_a = linear_to_db(np.abs(AF_adapted)**2)

    # Normalize to quiescent peak
    peak_q = np.max(pattern_q)
    pattern_q -= peak_q
    pattern_a -= peak_q

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    theta_deg = np.rad2deg(theta)
    ax.plot(theta_deg, pattern_q, 'b-', linewidth=2, label='Quiescent')
    ax.plot(theta_deg, pattern_a, 'r--', linewidth=2, label='Adapted')

    # Mark interference directions
    for theta_i, phi_i in interference_directions:
        ax.axvline(x=theta_i, color='g', linestyle=':', linewidth=1.5,
                   label=f'Interferer @ {theta_i:.1f}°')

    ax.set_xlabel('Theta (degrees)', fontsize=12)
    ax.set_ylabel('Normalized Pattern (dB)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()

    return ax
