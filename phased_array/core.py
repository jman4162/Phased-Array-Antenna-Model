"""
Core computation functions for phased array antennas.

Includes vectorized array factor, FFT-based computation, steering vectors,
and element patterns.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy.integrate import trapezoid

from .utils import create_theta_phi_grid, linear_to_db, theta_phi_to_uv

ArrayLike = Union[np.ndarray, float]


def steering_vector(
    k: float,
    x: np.ndarray,
    y: np.ndarray,
    theta0_deg: float,
    phi0_deg: float,
    z: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute steering vector (phase weights) for beam pointing.

    Parameters
    ----------
    k : float
        Wavenumber (2*pi/wavelength) in rad/m
    x : ndarray
        Element x-positions in meters (flattened)
    y : ndarray
        Element y-positions in meters (flattened)
    theta0_deg : float
        Desired beam steering angle theta in degrees (from z-axis)
    phi0_deg : float
        Desired beam steering angle phi in degrees (azimuth)
    z : ndarray, optional
        Element z-positions in meters (for 3D arrays)

    Returns
    -------
    weights : ndarray
        Complex steering weights (unit magnitude, appropriate phases)

    Examples
    --------
    Create steering weights for a 4x4 array pointing at 30 degrees:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
    >>> weights.shape
    (16,)
    >>> np.abs(weights[0])  # All weights have unit magnitude
    1.0

    Steer to azimuth with 3D array positions:

    >>> z = np.zeros(16)  # Planar array
    >>> weights_3d = pa.steering_vector(k, geom.x, geom.y, 20, 45, z=z)
    """
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    # Direction cosines for steering direction
    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)
    w0 = np.cos(theta0)

    # Phase shift to steer beam
    if z is None:
        phase = k * (x * u0 + y * v0)
    else:
        phase = k * (x * u0 + y * v0 + z * w0)

    return np.exp(-1j * phase)


def array_factor_vectorized(
    theta: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float,
    z: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute array factor using vectorized NumPy operations.

    This is 50-100x faster than nested loop implementations for typical
    grid sizes (181x181 angular points).

    Parameters
    ----------
    theta : ndarray
        Polar angles in radians, shape (n_theta, n_phi) or (n_points,)
    phi : ndarray
        Azimuthal angles in radians, same shape as theta
    x : ndarray
        Element x-positions in meters, shape (n_elements,)
    y : ndarray
        Element y-positions in meters, shape (n_elements,)
    weights : ndarray
        Complex element weights, shape (n_elements,)
    k : float
        Wavenumber (2*pi/wavelength) in rad/m
    z : ndarray, optional
        Element z-positions in meters, shape (n_elements,)

    Returns
    -------
    AF : ndarray
        Complex array factor, same shape as theta/phi

    Examples
    --------
    Compute array factor for a 4x4 array at broadside:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> weights = np.ones(16)  # Uniform weights
    >>> theta = np.array([[0.0]])  # Broadside
    >>> phi = np.array([[0.0]])
    >>> AF = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, weights, k)
    >>> np.abs(AF[0, 0])  # Peak at broadside equals number of elements
    16.0

    Compute over a grid of angles:

    >>> theta_grid, phi_grid = np.meshgrid(
    ...     np.linspace(0, np.pi/2, 91),
    ...     np.linspace(0, 2*np.pi, 181),
    ...     indexing='ij'
    ... )
    >>> AF_grid = pa.array_factor_vectorized(
    ...     theta_grid, phi_grid, geom.x, geom.y, weights, k
    ... )
    >>> AF_grid.shape
    (91, 181)
    """
    # Flatten inputs for computation
    original_shape = theta.shape
    theta_flat = theta.ravel()
    phi_flat = phi.ravel()

    # Direction cosines for all observation angles: shape (n_angles,)
    u = np.sin(theta_flat) * np.cos(phi_flat)
    v = np.sin(theta_flat) * np.sin(phi_flat)

    # Element positions: shape (n_elements,)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    weights = np.asarray(weights).ravel()

    # Phase contributions: shape (n_angles, n_elements)
    # Using broadcasting: (n_angles, 1) * (1, n_elements)
    phase = k * (np.outer(u, x) + np.outer(v, y))

    if z is not None:
        w = np.cos(theta_flat)
        z = np.asarray(z).ravel()
        phase += k * np.outer(w, z)

    # Array factor: sum over elements
    AF = np.sum(weights * np.exp(1j * phase), axis=1)

    return AF.reshape(original_shape)


def array_factor_uv(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float
) -> np.ndarray:
    """
    Compute array factor directly in UV-space.

    Parameters
    ----------
    u : ndarray
        Direction cosine u, shape (n_u, n_v) or (n_points,)
    v : ndarray
        Direction cosine v, same shape as u
    x : ndarray
        Element x-positions in meters
    y : ndarray
        Element y-positions in meters
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber in rad/m

    Returns
    -------
    AF : ndarray
        Complex array factor, same shape as u/v
    """
    original_shape = u.shape
    u_flat = u.ravel()
    v_flat = v.ravel()

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    weights = np.asarray(weights).ravel()

    # Phase: (n_angles, n_elements)
    phase = k * (np.outer(u_flat, x) + np.outer(v_flat, y))

    AF = np.sum(weights * np.exp(1j * phase), axis=1)

    return AF.reshape(original_shape)


def array_factor_fft(
    weights_2d: np.ndarray,
    dx: float,
    dy: float,
    n_u: int = 512,
    n_v: int = 512,
    wavelength: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute array factor using 2D FFT for uniform rectangular arrays.

    This is O(N log N) instead of O(N * M) and is fastest for large
    uniform rectangular arrays.

    Parameters
    ----------
    weights_2d : ndarray
        Complex weights on 2D grid, shape (Nx, Ny)
    dx : float
        Element spacing in x (in wavelengths)
    dy : float
        Element spacing in y (in wavelengths)
    n_u : int
        Number of output points in u direction
    n_v : int
        Number of output points in v direction
    wavelength : float
        Wavelength (for normalizing spacing, default=1 means dx, dy in wavelengths)

    Returns
    -------
    u : ndarray
        Direction cosine u values, shape (n_u,)
    v : ndarray
        Direction cosine v values, shape (n_v,)
    AF : ndarray
        Complex array factor, shape (n_u, n_v)
    """
    Nx, Ny = weights_2d.shape

    # Zero-pad for desired resolution
    pad_x = max(0, n_u - Nx)
    pad_y = max(0, n_v - Ny)

    weights_padded = np.pad(
        weights_2d,
        ((pad_x // 2, pad_x - pad_x // 2), (pad_y // 2, pad_y - pad_y // 2)),
        mode='constant'
    )

    # 2D FFT
    AF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(weights_padded)))

    # UV coordinates from FFT frequencies
    # For element spacing d, maximum unambiguous u is lambda/(2*d)
    # FFT gives spatial frequencies, need to scale to direction cosines
    n_u_actual, n_v_actual = weights_padded.shape

    # Spatial frequency to direction cosine: u = lambda * f_x
    # FFT frequency spacing: df = 1/(N*d) where d is element spacing
    # So u spacing = lambda * df = lambda / (N * d)
    u_max = wavelength / (2 * dx) if dx > 0 else 1.0
    v_max = wavelength / (2 * dy) if dy > 0 else 1.0

    u = np.linspace(-u_max, u_max, n_u_actual)
    v = np.linspace(-v_max, v_max, n_v_actual)

    return u, v, AF


def element_pattern(
    theta: np.ndarray,
    phi: np.ndarray,
    cos_exp_theta: float = 1.0,
    cos_exp_phi: float = 1.0,
    max_gain_dBi: float = 0.0
) -> np.ndarray:
    """
    Compute element pattern using raised cosine model.

    Parameters
    ----------
    theta : ndarray
        Polar angle in radians
    phi : ndarray
        Azimuthal angle in radians (not used; the model is phi-symmetric)
    cos_exp_theta : float
        Cosine exponent for theta dependence (1.0 = simple cosine)
    cos_exp_phi : float
        Deprecated and unused. The basic model has no phi dependence;
        for polarized/phi-dependent element models see the v1.4 vector
        pattern API.
    max_gain_dBi : float
        Peak element gain in dBi

    Returns
    -------
    pattern : ndarray
        Element pattern (linear scale, same shape as theta)
    """
    if cos_exp_phi != 1.0:
        warnings.warn(
            "cos_exp_phi is unused: element_pattern is phi-symmetric. "
            "The parameter is deprecated and will be removed in a future "
            "release; use the vector element models for phi-dependent "
            "patterns.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Convert max gain to linear
    max_gain_linear = 10 ** (max_gain_dBi / 10)

    # Raised cosine pattern (only valid for forward hemisphere)
    cos_theta = np.cos(theta)
    pattern = np.where(
        cos_theta > 0,
        max_gain_linear * (cos_theta ** cos_exp_theta),
        0.0
    )

    return pattern


def element_pattern_cosine_tapered(
    theta: np.ndarray,
    phi: np.ndarray,
    theta_3dB_deg: float = 65.0,
    max_gain_dBi: float = 5.0
) -> np.ndarray:
    """
    Compute element pattern with specified 3dB beamwidth.

    The cosine exponent is computed to achieve the desired beamwidth.

    Parameters
    ----------
    theta : ndarray
        Polar angle in radians
    phi : ndarray
        Azimuthal angle in radians
    theta_3dB_deg : float
        Half-power beamwidth in degrees
    max_gain_dBi : float
        Peak element gain in dBi

    Returns
    -------
    pattern : ndarray
        Element pattern (linear scale)
    """
    # Compute cosine exponent for desired beamwidth
    # At theta_3dB, pattern = 0.5 * max
    # cos(theta_3dB)^n = 0.5
    # n = log(0.5) / log(cos(theta_3dB))
    theta_3dB = np.deg2rad(theta_3dB_deg)
    if np.cos(theta_3dB) > 0:
        cos_exp = np.log(0.5) / np.log(np.cos(theta_3dB))
    else:
        cos_exp = 1.0

    return element_pattern(theta, phi, cos_exp, 1.0, max_gain_dBi)


def total_pattern(
    theta: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float,
    element_pattern_func: Optional[callable] = None,
    z: Optional[np.ndarray] = None,
    **element_kwargs
) -> np.ndarray:
    r"""
    Compute total radiation pattern (element pattern * array factor).

    Parameters
    ----------
    theta : ndarray
        Polar angles in radians
    phi : ndarray
        Azimuthal angles in radians
    x : ndarray
        Element x-positions in meters
    y : ndarray
        Element y-positions in meters
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber in rad/m
    element_pattern_func : callable, optional
        Function to compute element pattern. If None, uses isotropic elements.
        Should have signature: func(theta, phi, \*\*kwargs) -> ndarray
    z : ndarray, optional
        Element z-positions for 3D arrays
    element_kwargs : dict
        Additional keyword arguments passed to element_pattern_func

    Returns
    -------
    pattern : ndarray
        Total radiation pattern (complex)
    """
    # Compute array factor
    AF = array_factor_vectorized(theta, phi, x, y, weights, k, z)

    # Apply element pattern
    if element_pattern_func is not None:
        EP = element_pattern_func(theta, phi, **element_kwargs)
        pattern = EP * AF
    else:
        pattern = AF

    return pattern


def compute_pattern_cuts(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float,
    theta0_deg: float = 0.0,
    phi0_deg: float = 0.0,
    n_points: int = 361,
    theta_range_deg: Tuple[float, float] = (-90, 90),
    element_pattern_func: Optional[callable] = None,
    **element_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute principal plane pattern cuts (E-plane and H-plane).

    Parameters
    ----------
    x, y : ndarray
        Element positions in meters
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber in rad/m
    theta0_deg : float
        Beam steering theta angle
    phi0_deg : float
        Beam steering phi angle
    n_points : int
        Number of points in each cut
    theta_range_deg : tuple
        Angular range in degrees
    element_pattern_func : callable, optional
        Element pattern function
    **element_kwargs
        Arguments for element pattern

    Returns
    -------
    theta_deg : ndarray
        Angle values in degrees
    E_plane_dB : ndarray
        E-plane pattern in dB (phi = phi0)
    H_plane_dB : ndarray
        H-plane pattern in dB (phi = phi0 + 90)
    """
    theta_deg = np.linspace(theta_range_deg[0], theta_range_deg[1], n_points)
    theta_rad = np.deg2rad(theta_deg)

    # For theta range including negative values, map to standard coordinates
    # theta < 0 means scanning on opposite side
    theta_positive = np.abs(theta_rad)
    phi_e = np.where(theta_rad >= 0, np.deg2rad(phi0_deg), np.deg2rad(phi0_deg + 180))
    phi_h = np.where(theta_rad >= 0, np.deg2rad(phi0_deg + 90), np.deg2rad(phi0_deg + 270))

    # E-plane cut
    pattern_e = total_pattern(
        theta_positive, phi_e, x, y, weights, k,
        element_pattern_func, **element_kwargs
    )

    # H-plane cut
    pattern_h = total_pattern(
        theta_positive, phi_h, x, y, weights, k,
        element_pattern_func, **element_kwargs
    )

    # Convert to dB
    E_plane_dB = linear_to_db(np.abs(pattern_e)**2)
    H_plane_dB = linear_to_db(np.abs(pattern_h)**2)

    # Normalize to peak
    E_plane_dB -= np.max(E_plane_dB)
    H_plane_dB -= np.max(H_plane_dB)

    return theta_deg, E_plane_dB, H_plane_dB


def compute_full_pattern(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float,
    n_theta: int = 181,
    n_phi: int = 361,
    theta_range: Tuple[float, float] = (0, np.pi/2),
    phi_range: Tuple[float, float] = (0, 2*np.pi),
    element_pattern_func: Optional[callable] = None,
    z: Optional[np.ndarray] = None,
    **element_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute full 2D radiation pattern.

    Parameters
    ----------
    x, y : ndarray
        Element positions in meters
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber in rad/m
    n_theta, n_phi : int
        Number of angular points
    theta_range, phi_range : tuple
        Angular ranges in radians
    element_pattern_func : callable, optional
        Element pattern function
    z : ndarray, optional
        Element z-positions
    **element_kwargs
        Arguments for element pattern

    Returns
    -------
    theta : ndarray
        Theta values in radians, shape (n_theta,)
    phi : ndarray
        Phi values in radians, shape (n_phi,)
    pattern_dB : ndarray
        Pattern in dB, shape (n_theta, n_phi)

    Examples
    --------
    Compute a full pattern for a 16x16 array steered to 30 degrees:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
    >>> theta, phi, pattern_dB = pa.compute_full_pattern(
    ...     geom.x, geom.y, weights, k, n_theta=91, n_phi=181
    ... )
    >>> theta.shape, phi.shape, pattern_dB.shape
    ((91,), (181,), (91, 181))
    >>> pattern_dB.max()  # Normalized to 0 dB peak
    0.0

    Include element pattern (cosine model):

    >>> theta, phi, pattern_dB = pa.compute_full_pattern(
    ...     geom.x, geom.y, weights, k,
    ...     element_pattern_func=pa.element_pattern,
    ...     cos_exp_theta=1.3
    ... )
    """
    theta_1d, phi_1d, theta_grid, phi_grid = create_theta_phi_grid(
        theta_range, phi_range, n_theta, n_phi
    )

    pattern = total_pattern(
        theta_grid, phi_grid, x, y, weights, k,
        element_pattern_func, z, **element_kwargs
    )

    pattern_dB = linear_to_db(np.abs(pattern)**2)
    pattern_dB -= np.max(pattern_dB)

    return theta_1d, phi_1d, pattern_dB


def _angle_axes(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate a (theta, phi) meshgrid and return its 1D axes.

    Parameters
    ----------
    theta, phi : ndarray
        2D angle grids in radians, as produced by
        :func:`~phased_array.utils.create_theta_phi_grid`
        (``np.meshgrid(..., indexing='ij')``)
    pattern : ndarray
        Pattern sampled on the same grid

    Returns
    -------
    theta_1d : ndarray
        Theta samples, shape (n_theta,)
    phi_1d : ndarray
        Phi samples, shape (n_phi,)

    Raises
    ------
    ValueError
        If the arrays are not 2D and identically shaped, contain non-finite
        values, have fewer than two samples on an axis, are not separable in
        the ``indexing='ij'`` sense, are not strictly increasing with uniform
        spacing, or sample theta outside [0, pi].
    """
    theta, phi, pattern = (np.asarray(a) for a in (theta, phi, pattern))

    for name, arr in (("theta", theta), ("phi", phi), ("pattern", pattern)):
        if np.ndim(arr) != 2:
            raise ValueError(
                f"{name} must be a 2D grid, got ndim={np.ndim(arr)}. Pass the "
                "theta_grid/phi_grid outputs of create_theta_phi_grid(), not "
                "the 1D axes."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (NaN or inf)")

    if theta.shape != phi.shape or theta.shape != pattern.shape:
        raise ValueError(
            "theta, phi and pattern must have the same shape, got "
            f"{theta.shape}, {phi.shape} and {pattern.shape}"
        )

    n_theta, n_phi = theta.shape
    if n_theta < 2 or n_phi < 2:
        raise ValueError(
            "at least two samples are required on each axis, got shape "
            f"{theta.shape}. A single-sample axis carries no integration span."
        )

    # Separability: theta must vary along axis 0 only, phi along axis 1 only.
    if not np.allclose(theta, theta[:, :1]):
        raise ValueError(
            "theta must be constant along axis 1. The grid looks transposed "
            "or built with indexing='xy'; use indexing='ij'."
        )
    if not np.allclose(phi, phi[:1, :]):
        raise ValueError(
            "phi must be constant along axis 0. The grid looks transposed "
            "or built with indexing='xy'; use indexing='ij'."
        )

    axes = []
    tolerances = []
    for name, samples in (("theta", theta[:, 0]), ("phi", phi[0, :])):
        axis = samples.astype(np.float64)
        # Compare positions to a uniform axis rather than successive
        # differences, which amplify rounding error. The tolerance scales
        # with the axis span, not the array dtype, so a float32 grid and
        # the same grid upcast to float64 are judged identically.
        span = float(axis[-1] - axis[0])
        tol = max(1e-12, 1e-6 * abs(span))
        if not np.all(np.diff(axis) > 0):
            raise ValueError(
                f"{name} must be strictly increasing, got samples from "
                f"{axis[0]:.6g} to {axis[-1]:.6g} rad"
            )
        uniform = np.linspace(axis[0], axis[-1], axis.size)
        deviation = float(np.max(np.abs(axis - uniform)))
        if deviation > tol:
            raise ValueError(
                f"{name} must be uniformly spaced; samples deviate from a "
                f"uniform axis by up to {deviation:.3g} rad, tolerance "
                f"{tol:.3g} rad"
            )
        axes.append(axis)
        tolerances.append(tol)

    theta_1d, phi_1d = axes
    theta_tol, phi_tol = tolerances
    if theta_1d[0] < -theta_tol or theta_1d[-1] > np.pi + theta_tol:
        raise ValueError(
            "theta must lie within [0, pi] radians, got samples from "
            f"{theta_1d[0]:.6g} to {theta_1d[-1]:.6g} rad. Angles in "
            "degrees are a common cause."
        )

    # A span wider than a full turn counts some directions more than once.
    phi_span = float(phi_1d[-1] - phi_1d[0])
    if phi_span > 2 * np.pi + phi_tol:
        raise ValueError(
            f"phi must span at most 2*pi radians, got {phi_span:.6g} rad "
            f"from {phi_1d[0]:.6g} to {phi_1d[-1]:.6g}. Angles in degrees "
            "are a common cause."
        )

    # float32(pi) lies slightly above pi; clamp only permitted roundoff.
    return np.clip(theta_1d, 0, np.pi), phi_1d


def compute_directivity(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern: np.ndarray
) -> float:
    """
    Compute peak directivity by integrating a pattern over solid angle.

    Power is treated as constant over each theta cell. Cell boundaries are
    the midpoints of adjacent theta samples, clipped at the first and last
    samples. The weight of row i is

        w_i = cos(theta_edge_i) - cos(theta_edge_{i+1})

    so that ``sum(w_i) * (phi span)`` reproduces the sampled solid angle
    within floating-point error. Azimuth is integrated with the trapezoidal
    rule. Midpoint boundaries keep neighboring cells contiguous even when
    single-precision coordinates have small rounding differences.

    The spherical Jacobian sin(theta) correctly vanishes at the poles,
    which individually have zero solid angle. Numerical error arises from
    approximating power over finite angular regions. Cell weighting can
    reduce that error for pole-peaked patterns and exactly integrates an
    isotropic pattern on a full sphere, but does not eliminate quadrature
    bias generally. For smooth patterns that vanish at the poles, the
    previous sin(theta) trapezoidal rule can be more accurate. Both rules
    are generally second order, with higher-order convergence possible for
    special patterns. Refine the angular grid to check convergence.

    Parameters
    ----------
    theta : ndarray
        2D theta grid in radians, shape (n_theta, n_phi). Must be uniformly
        spaced, strictly increasing along axis 0, constant along axis 1,
        and contained in [0, pi] - the ``theta_grid`` output of
        :func:`~phased_array.utils.create_theta_phi_grid`.
    phi : ndarray
        2D phi grid in radians, same shape. Must be uniformly spaced,
        strictly increasing along axis 1, constant along axis 0, and span
        at most 2*pi.
    pattern : ndarray
        Complex or magnitude (amplitude) pattern on the same grid. Power is
        taken as ``|pattern|**2``, so pass amplitude - not power, and not
        dB.

    Returns
    -------
    directivity : float
        Peak directivity in linear scale (not dB)

    Raises
    ------
    ValueError
        If theta, phi and pattern are not 2D arrays of matching shape, hold
        non-finite values, have fewer than two samples on an axis, are not
        separable in the ``indexing='ij'`` sense, are not strictly increasing
        with uniform spacing, sample theta outside [0, pi], or span more
        than 2*pi in phi

    Notes
    -----
    The grid need not cover the full sphere. A partial grid integrates only
    the solid angle it samples, which is equivalent to assuming the pattern
    radiates no power outside that region. A hemisphere grid
    (``theta_range=(0, np.pi/2)``) therefore gives the expected answer for a
    ground-plane backed array.

    :func:`compute_full_pattern` output cannot be passed here directly: it
    returns 1D theta and phi axes and a normalized dB pattern. Build the
    grid with :func:`~phased_array.utils.create_theta_phi_grid` and evaluate
    amplitude on it with :func:`total_pattern` or
    :func:`array_factor_vectorized` instead. A dB-valued pattern is not
    detectable here and integrates to a meaningless result.

    Examples
    --------
    An isotropic pattern has unit directivity:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> _, _, theta, phi = pa.create_theta_phi_grid()
    >>> d = pa.compute_directivity(theta, phi, np.ones_like(theta))
    >>> f"{d:.9f}"
    '1.000000000'

    A short dipole, amplitude sin(theta), has directivity 3/2:

    >>> d = pa.compute_directivity(theta, phi, np.sin(theta))
    >>> f"{d:.4f}"
    '1.5000'
    """
    theta_1d, phi_1d = _angle_axes(theta, phi, pattern)

    power = np.abs(pattern)**2
    peak_power = np.max(power)

    # Shared midpoint boundaries avoid gaps/overlaps from rounded spacing.
    edges = np.concatenate((theta_1d[:1],
                            (theta_1d[:-1] + theta_1d[1:]) / 2,
                            theta_1d[-1:]))
    lower, upper = edges[:-1], edges[1:]
    # Equivalent to cos(lower) - cos(upper), without polar cancellation.
    theta_weights = 2 * np.sin((upper + lower) / 2) * np.sin((upper - lower) / 2)

    # trapezoid() rather than np.trapz(), which NumPy 2.4 removed, or
    # np.trapezoid(), which NumPy 1.x does not have.
    azimuth_integral = trapezoid(power, x=phi_1d, axis=1)
    total_power = float(np.sum(theta_weights * azimuth_integral))

    if total_power > 0:
        directivity = 4 * np.pi * peak_power / total_power
    else:
        directivity = 1.0

    return float(directivity)


def compute_half_power_beamwidth(
    angles_deg: np.ndarray,
    pattern_dB: np.ndarray
) -> float:
    """
    Compute half-power beamwidth from a 1D pattern cut.

    Parameters
    ----------
    angles_deg : ndarray
        Angle values in degrees
    pattern_dB : ndarray
        Normalized pattern in dB (0 dB at peak)

    Returns
    -------
    hpbw : float
        Half-power beamwidth in degrees
    """
    # Find points at -3 dB
    above_3dB = pattern_dB >= -3.0

    if not np.any(above_3dB):
        return 180.0

    # Find contiguous region around peak
    peak_idx = np.argmax(pattern_dB)
    left_idx = peak_idx
    right_idx = peak_idx

    while left_idx > 0 and above_3dB[left_idx - 1]:
        left_idx -= 1

    while right_idx < len(pattern_dB) - 1 and above_3dB[right_idx + 1]:
        right_idx += 1

    # Linearly interpolate the -3 dB crossings; fall back to the sample
    # position when the crossing lies outside the sampled range
    left_angle = angles_deg[left_idx]
    if left_idx > 0 and pattern_dB[left_idx - 1] < -3.0:
        a0, a1 = angles_deg[left_idx - 1], angles_deg[left_idx]
        p0, p1 = pattern_dB[left_idx - 1], pattern_dB[left_idx]
        left_angle = a0 + (-3.0 - p0) * (a1 - a0) / (p1 - p0)

    right_angle = angles_deg[right_idx]
    if right_idx < len(pattern_dB) - 1 and pattern_dB[right_idx + 1] < -3.0:
        a0, a1 = angles_deg[right_idx], angles_deg[right_idx + 1]
        p0, p1 = pattern_dB[right_idx], pattern_dB[right_idx + 1]
        right_angle = a0 + (-3.0 - p0) * (a1 - a0) / (p1 - p0)

    return abs(right_angle - left_angle)
