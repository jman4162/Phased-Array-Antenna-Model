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
        Interference-plus-noise data matrix (n_snapshots x n_elements)
        Each row is a snapshot of received signals
    diagonal_loading : float
        Diagonal loading factor for robustness.
        Adds diagonal_loading * I to the covariance matrix.

    Returns
    -------
    weights : ndarray
        Adaptive complex weights

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> # Simulate interference data (normally from receiver)
    >>> n_snapshots = 100
    >>> interference = np.random.randn(n_snapshots, geom.n_elements) + \\
    ...                1j * np.random.randn(n_snapshots, geom.n_elements)
    >>> weights = pa.adaptive_weights_smi(
    ...     geom, k, theta_desired_deg=0, phi_desired_deg=0,
    ...     interference_data=interference, diagonal_loading=0.01
    ... )
    >>> weights.shape
    (64,)

    Notes
    -----
    The SMI weight vector is:
        w = R^(-1) @ s / (s^H @ R^(-1) @ s)

    where R is the sample covariance matrix and s is the steering vector.
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
    mu: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute adaptive weights using Generalized Sidelobe Canceller (GSC).

    GSC provides a constrained adaptive filter structure that maintains
    distortionless response in the look direction while minimizing
    interference and noise.

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
        Interference-plus-noise data (n_snapshots x n_elements)
    n_blocking_vectors : int, optional
        Number of blocking matrix columns. Default: n_elements - 1
    mu : float
        Adaptation step size for LMS algorithm

    Returns
    -------
    weights : ndarray
        Adaptive weights (n_elements,)
    blocking_matrix : ndarray
        Blocking matrix used (n_elements x n_blocking_vectors)

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> interference = np.random.randn(50, geom.n_elements) + \\
    ...                1j * np.random.randn(50, geom.n_elements)
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
        - w_q is the quiescent (non-adaptive) steering weight
        - B is a blocking matrix orthogonal to the steering vector
        - w_a are adaptive weights that minimize output power

    The blocking matrix satisfies: s^H @ B = 0 (nulls the look direction)
    """
    n = geometry.n_elements

    if n_blocking_vectors is None:
        n_blocking_vectors = n - 1

    # Quiescent steering vector
    s = steering_vector(
        k, geometry.x, geometry.y,
        theta_desired_deg, phi_desired_deg, geometry.z
    )
    w_q = s / np.sqrt(np.sum(np.abs(s)**2))  # Normalize

    # Construct blocking matrix B orthogonal to s
    # Use SVD of s to get orthogonal complement
    s_normalized = s / np.linalg.norm(s)
    s_col = s_normalized.reshape(-1, 1)

    # Create orthonormal basis for null space of s^H
    # Using Householder or Gram-Schmidt
    Q, _ = np.linalg.qr(np.hstack([s_col, np.eye(n)[:, :n_blocking_vectors]]))
    B = Q[:, 1:n_blocking_vectors + 1]  # Take columns orthogonal to s

    # Process data through GSC
    interference_data = np.asarray(interference_data, dtype=complex)
    n_snapshots = interference_data.shape[0]

    # Initialize adaptive weights
    w_a = np.zeros(n_blocking_vectors, dtype=complex)

    # LMS adaptation through data
    for snapshot_idx in range(n_snapshots):
        x = interference_data[snapshot_idx, :]

        # Upper path output (reference)
        y_q = w_q.conj() @ x

        # Lower path (blocking matrix output)
        x_b = B.conj().T @ x

        # Error signal
        y_a = w_a.conj() @ x_b
        e = y_q - y_a

        # LMS update
        w_a = w_a + mu * e.conj() * x_b

    # Final weights
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
    >>> # Create adapted weights that null an interferer
    >>> w_adapted = pa.null_steering_projection(
    ...     geom, k, theta_main_deg=0, phi_main_deg=0,
    ...     null_directions=[(30, 0)]
    ... )
    >>> sinr_b, sinr_a, imp = pa.compute_sinr_improvement(
    ...     w_q, w_adapted, geom, k,
    ...     signal_direction=(0, 0),
    ...     interference_directions=[(30, 0)],
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
            geometry.x, geometry.y, weights, k
        )
        signal_out = signal_power * np.abs(AF_signal.item())**2

        # Interference response
        interference_out = 0.0
        for (theta_i, phi_i), power_i in zip(interference_directions, interference_powers):
            theta_int = np.array([[np.deg2rad(theta_i)]])
            phi_int = np.array([[np.deg2rad(phi_i)]])
            AF_int = array_factor_vectorized(
                theta_int, phi_int,
                geometry.x, geometry.y, weights, k
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
    phi = np.full_like(theta, np.deg2rad(phi_cut_deg))

    theta_grid = theta.reshape(-1, 1)
    phi_grid = phi.reshape(-1, 1)

    AF_quiescent = array_factor_vectorized(
        theta_grid, phi_grid,
        geometry.x, geometry.y, weights_quiescent, k
    ).ravel()

    AF_adapted = array_factor_vectorized(
        theta_grid, phi_grid,
        geometry.x, geometry.y, weights_adapted, k
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
