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

# Exact half-power level, 10*log10(0.5) ~ -3.0103 dB
_HALF_POWER_DB = float(10 * np.log10(0.5))


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

    Notes
    -----
    The array factor convention is ``AF(u) = sum_n w_n exp(+j k r_n . u)``
    (see :func:`array_factor_vectorized`), so the weights are
    ``exp(-j k r_n . u0)``, the conjugate of the array manifold returned
    by :func:`array_manifold`. A plane wave arriving from ``u0`` produces
    element signals proportional to the manifold, not to these weights;
    build received snapshots with :func:`array_manifold`.

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


def array_manifold(
    k: float,
    x: np.ndarray,
    y: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    z: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Array manifold (received signal vector) for a plane wave.

    A plane wave arriving from direction ``(theta, phi)`` produces the
    element signals ``a_n = exp(+j k r_n . u)``, with ``u`` the unit vector
    toward the source. In this package the array output for weights ``w``
    is ``w @ a`` (no conjugation), which equals
    :func:`array_factor_vectorized` evaluated in that direction. Use this
    function to synthesize snapshots for :func:`adaptive_weights_smi` and
    :func:`adaptive_weights_gsc`.

    Parameters
    ----------
    k : float
        Wavenumber (2*pi/wavelength) in rad/m
    x, y : ndarray
        Element positions in meters
    theta_deg : float
        Arrival direction theta in degrees (from z-axis)
    phi_deg : float
        Arrival direction phi in degrees
    z : ndarray, optional
        Element z-positions in meters

    Returns
    -------
    a : ndarray
        Complex manifold vector, ``np.conj(steering_vector(...))``

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 1, 0.5, 0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> w = pa.steering_vector(k, geom.x, geom.y, 30, 0)
    >>> a = pa.array_manifold(k, geom.x, geom.y, 30, 0)
    >>> round(abs(w @ a), 6)  # matched beam: full array gain N
    8.0
    """
    return np.conj(steering_vector(k, x, y, theta_deg, phi_deg, z))


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
    uniform rectangular arrays. The result equals
    :func:`array_factor_uv` evaluated on the returned (u, v) grid for an
    array centered at the origin, as built by
    ``create_rectangular_array(Nx, Ny, dx, dy, center=True)`` with weights
    reshaped to ``(Nx, Ny)``.

    Parameters
    ----------
    weights_2d : ndarray
        Complex weights on 2D grid, shape (Nx, Ny); axis 0 runs along x
    dx : float
        Element spacing in x, in the same length unit as ``wavelength``
        (with the default ``wavelength=1`` this is spacing in wavelengths)
    dy : float
        Element spacing in y, same unit as ``dx``
    n_u : int
        Number of output points in u (FFT length along x); must be at
        least Nx
    n_v : int
        Number of output points in v (FFT length along y); must be at
        least Ny
    wavelength : float
        Wavelength, in the same unit as ``dx`` and ``dy``

    Returns
    -------
    u : ndarray
        Direction cosine u at the FFT bin centers, shape (n_u,).
        ``u[n_u // 2] == 0``; the step is ``wavelength / (n_u * dx)``.
    v : ndarray
        Direction cosine v at the FFT bin centers, shape (n_v,)
    AF : ndarray
        Complex array factor, shape (n_u, n_v). Points with
        ``u**2 + v**2 > 1`` lie in the invisible region.

    Raises
    ------
    ValueError
        If ``weights_2d`` is not 2D, a spacing is not positive, or
        ``n_u < Nx`` or ``n_v < Ny``.

    Notes
    -----
    The array factor convention is ``AF(u, v) = sum w_n exp(+j k (x_n u +
    y_n v))``, the same as :func:`array_factor_vectorized`, so the inverse
    DFT kernel applies: ``AF(u_m) = M * ifft(w)[m]`` at
    ``u_m = m * wavelength / (M * dx)``. The phase is referenced to the
    array center.
    """
    weights_2d = np.asarray(weights_2d)
    if weights_2d.ndim != 2:
        raise ValueError(
            f"weights_2d must be 2D (Nx, Ny), got shape {weights_2d.shape}"
        )
    if dx <= 0 or dy <= 0:
        raise ValueError(f"dx and dy must be positive, got {dx} and {dy}")
    Nx, Ny = weights_2d.shape
    if n_u < Nx or n_v < Ny:
        raise ValueError(
            f"n_u and n_v must be at least the array size ({Nx}, {Ny}), got "
            f"({n_u}, {n_v}). The FFT length sets the number of output points."
        )

    # Inverse DFT matches the exp(+j k r.u) array-factor sign convention.
    AF = np.fft.fftshift(np.fft.ifft2(weights_2d, s=(n_u, n_v))) * (n_u * n_v)

    u = (np.arange(n_u) - n_u // 2) * wavelength / (n_u * dx)
    v = (np.arange(n_v) - n_v // 2) * wavelength / (n_v * dy)

    # Element 0 sits at index 0 of the FFT input; shift the phase reference
    # to the array center, where create_rectangular_array places the origin.
    k = 2 * np.pi / wavelength
    cx = (Nx - 1) / 2 * dx
    cy = (Ny - 1) / 2 * dy
    AF = AF * np.exp(-1j * k * cx * u)[:, None] * np.exp(-1j * k * cy * v)[None, :]

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

    The model is specified as a power pattern,
    ``G(theta) = G0 * cos(theta)**cos_exp_theta``, but returned as the
    corresponding field amplitude ``sqrt(G(theta))`` so it can be
    multiplied directly into the (field) array factor. Pattern utilities
    then form power as ``|EP * AF|**2``, which recovers ``G(theta)``.

    Parameters
    ----------
    theta : ndarray
        Polar angle in radians
    phi : ndarray
        Azimuthal angle in radians (not used; the model is phi-symmetric)
    cos_exp_theta : float
        Cosine exponent of the power pattern (1.0 = cos(theta) power
        roll-off). The returned field uses half this exponent.
    cos_exp_phi : float
        Deprecated and unused. The basic model has no phi dependence;
        for polarized/phi-dependent element models see the v1.4 vector
        pattern API.
    max_gain_dBi : float
        Peak element gain in dBi, applied as the field scale factor
        ``10**(max_gain_dBi / 20)``.

    Returns
    -------
    pattern : ndarray
        Element field amplitude (linear scale, same shape as theta). Zero
        behind the element (theta > 90 degrees). Square it to get the
        power pattern.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> ep = pa.element_pattern(np.array([0.0]), np.array([0.0]),
    ...                         max_gain_dBi=6.0)
    >>> round(float(10 * np.log10(ep[0] ** 2)), 6)
    6.0
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

    # Peak gain (power) -> peak field amplitude
    max_field_linear = 10 ** (max_gain_dBi / 20)

    # Raised cosine pattern (only valid for forward hemisphere). Clip the
    # base before exponentiating: np.where evaluates both branches, and a
    # negative base with a fractional exponent would emit a NaN warning.
    cos_theta = np.cos(theta)
    cos_forward = np.clip(cos_theta, 0.0, None)
    pattern = np.where(
        cos_theta > 0,
        max_field_linear * cos_forward ** (cos_exp_theta / 2.0),
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
    # At theta_3dB, power pattern = 0.5 * max (element_pattern takes the
    # power exponent and returns the field, so |EP|^2 is -3 dB here)
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
    z: Optional[np.ndarray] = None,
    **element_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute two orthogonal principal cuts through the beam direction.

    The first cut lies in the ``phi = phi0`` plane and is swept in theta
    (negative angles fall at ``phi0 + 180``). It contains the beam at
    ``(theta0, phi0)``. The second cut is the great circle through the
    beam direction perpendicular to the first: the direction at angle
    ``t`` is ``cos(t) * b + sin(t) * phi_hat``, where ``b`` is the beam
    direction and ``phi_hat`` the azimuthal unit vector there. At
    ``theta0 = 0`` it reduces to the ``phi0 + 90`` plane.

    The returned names keep the historical E-plane/H-plane labels. They
    are E and H planes only when the element polarization lines up with
    ``phi0``.

    Parameters
    ----------
    x, y : ndarray
        Element positions in meters
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber in rad/m
    theta0_deg : float
        Beam direction theta in degrees. Pass the steering angle so that
        the second cut passes through the beam peak.
    phi0_deg : float
        Beam direction phi in degrees; sets the plane of the first cut
    n_points : int
        Number of points in each cut
    theta_range_deg : tuple
        Angular range in degrees
    element_pattern_func : callable, optional
        Element pattern function (field amplitude)
    z : ndarray, optional
        Element z-positions in meters, for 3D arrays
    **element_kwargs
        Arguments for element pattern

    Returns
    -------
    theta_deg : ndarray
        Cut angle in degrees. For the first cut this is theta (signed);
        for the second it is the angle from the beam direction along the
        orthogonal great circle.
    E_plane_dB : ndarray
        First cut (``phi = phi0`` plane) in dB
    H_plane_dB : ndarray
        Second cut (orthogonal great circle through the beam) in dB

    Notes
    -----
    Both cuts are power patterns normalized to a shared reference, the
    larger of the two cut maxima. A cut that misses the beam therefore
    shows its true level instead of being lifted to 0 dB.
    """
    theta_deg = np.linspace(theta_range_deg[0], theta_range_deg[1], n_points)
    t = np.deg2rad(theta_deg)
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    # First cut: phi0 plane; theta < 0 means the opposite side (phi0 + 180)
    theta_e = np.abs(t)
    phi_e = np.where(t >= 0, phi0, phi0 + np.pi)

    # Second cut: great circle through the beam direction b, orthogonal to
    # the first cut, swept by the angle t from b toward phi_hat(theta0, phi0)
    b = np.array([np.sin(theta0) * np.cos(phi0),
                  np.sin(theta0) * np.sin(phi0),
                  np.cos(theta0)])
    phi_hat = np.array([-np.sin(phi0), np.cos(phi0), 0.0])
    dirs = np.outer(b, np.cos(t)) + np.outer(phi_hat, np.sin(t))
    theta_h = np.arccos(np.clip(dirs[2], -1.0, 1.0))
    phi_h = np.mod(np.arctan2(dirs[1], dirs[0]), 2 * np.pi)

    pattern_e = total_pattern(
        theta_e, phi_e, x, y, weights, k,
        element_pattern_func, z, **element_kwargs
    )
    pattern_h = total_pattern(
        theta_h, phi_h, x, y, weights, k,
        element_pattern_func, z, **element_kwargs
    )

    power_e = np.abs(pattern_e)**2
    power_h = np.abs(pattern_h)**2
    peak = max(float(np.max(power_e)), float(np.max(power_h)))
    if peak <= 0:
        peak = 1.0

    E_plane_dB = linear_to_db(power_e / peak)
    H_plane_dB = linear_to_db(power_h / peak)

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
        Half-power beamwidth in degrees, measured between the
        ``10*log10(0.5)`` (about -3.0103 dB) crossings
    """
    half_power_dB = _HALF_POWER_DB
    above_3dB = pattern_dB >= half_power_dB

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
    if left_idx > 0 and pattern_dB[left_idx - 1] < half_power_dB:
        a0, a1 = angles_deg[left_idx - 1], angles_deg[left_idx]
        p0, p1 = pattern_dB[left_idx - 1], pattern_dB[left_idx]
        left_angle = a0 + (half_power_dB - p0) * (a1 - a0) / (p1 - p0)

    right_angle = angles_deg[right_idx]
    if right_idx < len(pattern_dB) - 1 and pattern_dB[right_idx + 1] < half_power_dB:
        a0, a1 = angles_deg[right_idx], angles_deg[right_idx + 1]
        p0, p1 = pattern_dB[right_idx], pattern_dB[right_idx + 1]
        right_angle = a0 + (half_power_dB - p0) * (a1 - a0) / (p1 - p0)

    return abs(right_angle - left_angle)


def compute_peak_sidelobe_level(
    angles_deg: np.ndarray,
    pattern_dB: np.ndarray,
    return_angle: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Compute the peak sidelobe level of a 1D pattern cut.

    The main lobe extends from the pattern maximum out to the first local
    minimum (null) on each side. The peak sidelobe is the largest value
    outside that region, relative to the maximum. Grating lobes count as
    sidelobes, so a grating lobe as tall as the main beam gives 0 dB.

    Parameters
    ----------
    angles_deg : ndarray
        Cut angles in degrees, shape (n,)
    pattern_dB : ndarray
        Power pattern in dB on the same samples; it need not be
        normalized
    return_angle : bool
        If True, also return the angle of the peak sidelobe

    Returns
    -------
    level_dB : float
        Peak sidelobe level relative to the main-beam peak in dB, or
        ``-inf`` if the cut has no sidelobe
    angle_deg : float
        Angle of the peak sidelobe (only if ``return_angle``); NaN when
        there is no sidelobe

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(32, 1, 0.5, 0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> angles, cut, _ = pa.compute_pattern_cuts(
    ...     geom.x, geom.y, np.ones(32), k, n_points=4001)
    >>> round(pa.compute_peak_sidelobe_level(angles, cut), 1)
    -13.2
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    pattern_dB = np.asarray(pattern_dB, dtype=float)
    if angles_deg.ndim != 1 or angles_deg.shape != pattern_dB.shape:
        raise ValueError(
            "angles_deg and pattern_dB must be 1D arrays of the same length, "
            f"got shapes {angles_deg.shape} and {pattern_dB.shape}"
        )

    n = pattern_dB.size
    peak_idx = int(np.argmax(pattern_dB))

    left = peak_idx
    while left > 0 and pattern_dB[left - 1] <= pattern_dB[left]:
        left -= 1
    right = peak_idx
    while right < n - 1 and pattern_dB[right + 1] <= pattern_dB[right]:
        right += 1

    outside = np.r_[np.arange(0, left), np.arange(right + 1, n)].astype(int)
    if outside.size == 0:
        return (-np.inf, np.nan) if return_angle else -np.inf

    sl_idx = outside[np.argmax(pattern_dB[outside])]
    level = float(pattern_dB[sl_idx] - pattern_dB[peak_idx])
    if return_angle:
        return level, float(angles_deg[sl_idx])
    return level
