"""
Beamforming functions for phased arrays.

Includes amplitude tapering, null steering, and multiple simultaneous beams.
"""

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
        Desired peak sidelobe level in dB (negative)
    nbar : int
        Number of nearly equal-level sidelobes

    Returns
    -------
    taper : ndarray
        Amplitude taper weights (n,)
    """
    # Use scipy's Taylor window
    return signal.windows.taylor(n, nbar=nbar, sll=-sidelobe_dB, norm=True)


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
        Desired sidelobe level in dB (negative)

    Returns
    -------
    taper : ndarray
        Amplitude taper weights
    """
    return signal.windows.chebwin(n, at=-sidelobe_dB)


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

    >>> null_depth = pa.compute_null_depth(geom, k, weights, null_dirs[0])
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
        'delta_az' - azimuth difference pattern
        'delta_el' - elevation difference pattern

    Returns
    -------
    weights : ndarray
        Complex weights for the specified mode
    """
    # Base steering vector
    sv = steering_vector(
        k, geometry.x, geometry.y,
        theta0_deg, phi0_deg, geometry.z
    )

    if mode == 'sum':
        return sv

    elif mode == 'delta_az':
        # Difference in azimuth: flip sign for half the array in x
        sign = np.sign(geometry.x)
        sign[sign == 0] = 1  # Handle elements at x=0
        return sv * sign

    elif mode == 'delta_el':
        # Difference in elevation: flip sign for half the array in y
        sign = np.sign(geometry.y)
        sign[sign == 0] = 1
        return sv * sign

    else:
        raise ValueError(f"Unknown monopulse mode: {mode}")
