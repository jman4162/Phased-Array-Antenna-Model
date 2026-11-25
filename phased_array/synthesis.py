"""
Array factor synthesis for linear arrays.

Each function returns element weights that produce a specified array
factor for a linear array along x, under the package convention

    AF(u) = sum_n w_n exp(+j k x_n u),   u = sin(theta) in the x-z plane.

- :func:`binomial_taper_1d`: binomial amplitude taper, no sidelobes at
  half-wavelength spacing.
- :func:`woodward_lawson_weights`: superposition of orthogonal uniform
  beams weighted by samples of the desired pattern.
- :func:`least_squares_weights`: weights that minimize the (weighted)
  squared error to complex pattern samples, for uniform or arbitrary
  element positions.

The uniform-spacing functions return weights for elements at
``x_n = (n - (N - 1) / 2) * spacing`` wavelengths, the x positions of
``create_rectangular_array(N, 1, spacing, spacing, wavelength)`` divided
by ``wavelength``. A separable planar design is the outer product of two
linear designs: ``np.outer(wx, wy).ravel()`` matches the element order of
``create_rectangular_array(Nx, Ny, ...)``.

Reference: R. L. Haupt, *Antenna Arrays: A Computational Approach*,
Wiley-IEEE Press, 2010, Sections 3.1 and 3.2.1.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np


def _ula_positions(n_elements: int, spacing: float) -> np.ndarray:
    """Element x positions in wavelengths, centered at the origin."""
    return (np.arange(n_elements) - (n_elements - 1) / 2) * spacing


def _check_n_elements(n: int, name: str = "n_elements") -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"{name} must be a positive integer, got {n!r}")
    return int(n)


def _check_spacing(spacing: float) -> float:
    spacing = float(spacing)
    if not (np.isfinite(spacing) and spacing > 0):
        raise ValueError(
            f"spacing must be a finite positive number, got {spacing!r}"
        )
    return spacing


def binomial_taper_1d(n: int) -> np.ndarray:
    """
    Binomial amplitude taper for a linear array.

    The weights are the binomial coefficients ``C(n - 1, i)``,
    ``i = 0 ... n - 1``, divided by the largest one.

    Parameters
    ----------
    n : int
        Number of elements (at least 1)

    Returns
    -------
    taper : ndarray
        Real amplitude weights, shape (n,), maximum 1

    Notes
    -----
    With the integer coefficients and ``z = exp(j k d u)`` the array
    factor is ``(1 + z)**(n - 1)`` (Haupt eq. 3.10): all ``n - 1`` zeros
    sit at ``k d u = pi``, so ``|AF| = 2**(n - 1) |cos(k d u / 2)|**(n - 1)``.
    At ``d = lambda/2``
    the pattern falls monotonically from broadside to the null at
    ``u = +/-1`` and has no sidelobes. For ``d < lambda/2`` the null lies
    outside visible space; for ``d > lambda/2`` the pattern rises again
    past the null toward a grating lobe.

    The taper efficiency is low: 0.731 for ``n = 5`` and 0.566 for
    ``n = 9`` (Haupt Table 3.2), and the main beam is wide. Coefficients
    are computed in exact integer arithmetic, so large ``n`` gives finite
    weights; the edge weights underflow to 0 from ``n = 1082``.

    Reference: R. L. Haupt, *Antenna Arrays: A Computational Approach*,
    Wiley-IEEE Press, 2010, Sec. 3.2.1, Table 3.2.

    Examples
    --------
    >>> import phased_array as pa
    >>> print(pa.binomial_taper_1d(5) * 6)
    [1. 4. 6. 4. 1.]
    >>> round(pa.compute_taper_efficiency(pa.binomial_taper_1d(5)), 3)
    0.731
    """
    n = _check_n_elements(n, "n")
    coeffs = [1]
    for i in range(n - 1):
        coeffs.append(coeffs[-1] * (n - 1 - i) // (i + 1))
    peak = coeffs[(n - 1) // 2]
    # int / int true division is correctly rounded even for huge integers
    return np.array([c / peak for c in coeffs], dtype=float)


def woodward_lawson_weights(
    desired: Callable[[np.ndarray], np.ndarray],
    n_elements: int,
    spacing: float = 0.5,
    return_samples: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Woodward-Lawson synthesis of a linear array factor.

    The desired pattern is sampled at the peaks of orthogonal uniform
    beams, ``u_m = m * lambda / (N d)``, and the weights superpose those
    beams with the sampled amplitudes ``b_m = desired(u_m)``.

    Parameters
    ----------
    desired : callable
        Function of a 1D array of direction cosines ``u`` returning the
        desired field amplitude (real or complex) at each ``u``
    n_elements : int
        Number of elements N
    spacing : float
        Element spacing d in wavelengths
    return_samples : bool
        If True, also return the points where ``desired`` was sampled

    Returns
    -------
    weights : ndarray
        Complex weights, shape (N,), for elements at
        ``x_n = (n - (N - 1) / 2) * spacing`` wavelengths
    u_samples : ndarray
        Points where ``desired`` was sampled, in increasing order (only
        if ``return_samples``). N + 1 points for even N with
        ``d >= lambda/2``, otherwise at most N (see Notes).

    Notes
    -----
    The weights are (Haupt eq. 3.8, with the array centered)

    .. math::

        w_n = \\frac{1}{N} \\sum_m b_m \\exp(-j k x_n u_m),

    and the array factor is the sum of shifted uniform beams (eq. 3.5)

    .. math::

        AF(u) = \\sum_m b_m
        \\frac{\\sin[N k d (u - u_m) / 2]}{N \\sin[k d (u - u_m) / 2]}.

    Each beam peaks at its own ``u_m`` and is zero at every other sample
    point, so the synthesized pattern equals ``desired`` exactly at every
    ``u_m``, with one exception for even N described below. Between
    samples it interpolates, with Gibbs-type ripple near discontinuities
    of ``desired``.

    Sample points are the ``u_m`` inside visible space, ``|u_m| <= 1``.
    For ``d < lambda/2`` that gives ``2 floor(N d / lambda) + 1 <= N``
    samples and the pattern in invisible space is left uncontrolled.

    For ``d >= lambda/2`` the array factor of a centered array satisfies
    ``AF(u + lambda/d) = (-1)**(N - 1) AF(u)``, so visible space holds a
    full period and only the N beams of one period,
    ``-lambda/(2d) <= u_m <= lambda/(2d)``, are independent. For odd N
    these are N interior samples. For even N both period ends
    ``u = -e`` and ``u = +e`` (``e = lambda/(2d)``) are visible, and a
    single beam serves both with ``AF(+e) = -AF(-e)``. Its coefficient
    is the least-squares compromise ``b = (D(-e) - D(+e)) / 2``, so
    ``AF(-e) = b`` and ``AF(+e) = -b``; both points are returned as
    samples. The result matches ``desired`` at both ends only when
    ``D(+e) = -D(-e)``. A target symmetric in ``u`` gives a null at both
    ends: an even-N array with symmetric weights cannot radiate there.
    :func:`least_squares_weights` evaluated at the returned samples
    gives the same weights in every case.

    Reference: R. L. Haupt, *Antenna Arrays: A Computational Approach*,
    Wiley-IEEE Press, 2010, Sec. 3.1.2, eqs. (3.5)-(3.8), Table 3.1.

    Examples
    --------
    Haupt's 16-element sector beam, ``|AF| = 1`` for ``|u| <= 0.5``:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> w = pa.woodward_lawson_weights(lambda u: np.abs(u) <= 0.5, 16)
    >>> np.round(w.real[8:], 4)  # center to edge, Haupt Table 3.1
    array([ 0.4929,  0.1015, -0.1269, -0.0097,  0.0805, -0.0206, -0.0576,
            0.0398])

    The pattern equals the desired value at the sample points:

    >>> geom = pa.create_rectangular_array(16, 1, 0.5, 0.5)
    >>> u = np.array([0.0, 0.25, 0.75])
    >>> af = pa.array_factor_uv(u, np.zeros(3), geom.x, geom.y, w,
    ...                         pa.wavelength_to_k(1.0))
    >>> np.round(np.abs(af), 9)
    array([1., 1., 0.])
    """
    if not callable(desired):
        raise TypeError("desired must be a callable of u")
    n = _check_n_elements(n_elements)
    spacing = _check_spacing(spacing)

    # Beams |m| <= N d fit in visible space. The tolerance keeps u_m = 1
    # when N d is an integer that rounds low; u_m is clipped below.
    nd = n * spacing + 1e-9
    shared_edge = False
    if nd >= (n + 1) // 2:
        # More than N beams fit: visible space holds a full period, so
        # use the N beams of one period
        m = np.arange(-(n // 2), (n - 1) // 2 + 1)
        # Even N: the beam at u = -e also serves u = +e
        shared_edge = n % 2 == 0
    else:
        m_max = int(np.floor(nd))
        m = np.arange(-m_max, m_max + 1)
    m_sampled = np.append(m, n // 2) if shared_edge else m
    u_samples = np.clip(m_sampled / (n * spacing), -1.0, 1.0)

    d = np.asarray(desired(u_samples), dtype=complex)
    d = np.broadcast_to(d, u_samples.shape)
    b = d[:m.size].copy()
    if shared_edge:
        # Least-squares fit of AF(-e) = b and AF(+e) = -b to D(-e), D(+e)
        b[0] = (d[0] - d[-1]) / 2

    # k x_n u_m = 2 pi (n - (N-1)/2) m / N, independent of the spacing
    n_idx = np.arange(n) - (n - 1) / 2
    weights = np.exp(-2j * np.pi * np.outer(n_idx, m) / n) @ b / n
    if return_samples:
        return weights, u_samples
    return weights


def least_squares_weights(
    u_samples: np.ndarray,
    desired: np.ndarray,
    n_elements: Optional[int] = None,
    spacing: float = 0.5,
    positions: Optional[np.ndarray] = None,
    k: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    regularization: float = 0.0,
) -> np.ndarray:
    """
    Least-squares synthesis of a linear array factor.

    Finds complex element weights ``w`` that minimize

    .. math::

        \\sum_i c_i \\left| AF(u_i) - D_i \\right|^2
        + \\mu \\lVert w \\rVert^2,

    where ``AF(u) = sum_n w_n exp(+j k x_n u)``, ``D_i`` is ``desired``,
    ``c_i`` is ``weights`` and ``mu`` is ``regularization``.

    Give either ``n_elements`` (a uniform array with ``spacing`` in
    wavelengths) or ``positions`` with ``k``.

    Parameters
    ----------
    u_samples : ndarray
        Direction cosines ``u_i`` of the sample points, shape (M,)
    desired : ndarray
        Desired complex array factor ``D_i`` at ``u_samples``, shape (M,).
        The phase is part of the target; use zero phase for a real
        amplitude target.
    n_elements : int, optional
        Number of elements of a uniform linear array centered at the
        origin, ``x_n = (n - (N - 1) / 2) * spacing`` wavelengths
    spacing : float
        Element spacing in wavelengths for the uniform array; ignored
        when ``positions`` is given
    positions : ndarray, optional
        Element x positions in meters, shape (N,), for an arbitrary
        (for example nonuniform) linear array
    k : float, optional
        Wavenumber ``2*pi/wavelength`` in rad/m, finite and positive.
        Required with ``positions``; must be omitted with ``n_elements``
    weights : ndarray, optional
        Nonnegative weight ``c_i`` of each sample point, shape (M,).
        Default is 1 for every sample.
    regularization : float
        Nonnegative Tikhonov factor ``mu`` on the weight norm. Larger
        values trade pattern error for smaller weights.

    Returns
    -------
    w : ndarray
        Complex element weights, shape (N,)

    Notes
    -----
    Each sample gives one row of the linear system ``A w = D`` with
    ``A[i, n] = exp(j k x_n u_i)`` (Haupt eq. 3.9). The weighted problem
    is solved with :func:`numpy.linalg.lstsq` on the rows scaled by
    ``sqrt(c_i)``; with ``mu > 0`` the rows ``sqrt(mu) * I`` (target 0)
    are appended, which gives the same solution as the normal equations
    ``(A^H C A + mu I) w = A^H C D`` without squaring the condition
    number. With fewer samples than elements and ``mu = 0`` the result
    is the minimum-norm solution.

    The fit matches complex values, not magnitudes, so a magnitude-only
    target needs a chosen phase. The method works for any element
    positions, but the weights it returns can vary strongly from element
    to element. At the sample points returned by
    ``woodward_lawson_weights(..., return_samples=True)`` the solution
    equals the Woodward-Lawson weights.

    Reference: R. L. Haupt, *Antenna Arrays: A Computational Approach*,
    Wiley-IEEE Press, 2010, Sec. 3.1.3, eq. (3.9), Table 3.1.

    Examples
    --------
    Recover a Taylor taper from samples of its own array factor:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(10, 1, 0.5, 0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> w0 = pa.taylor_taper_1d(10, -30)
    >>> u = np.linspace(-1, 1, 201)
    >>> target = pa.array_factor_uv(u, np.zeros_like(u), geom.x, geom.y, w0, k)
    >>> w = pa.least_squares_weights(u, target, n_elements=10)
    >>> bool(np.allclose(w, w0))
    True

    The same fit with explicit positions in meters:

    >>> w = pa.least_squares_weights(u, target, positions=geom.x, k=k)
    >>> bool(np.allclose(w, w0))
    True
    """
    u_samples = np.asarray(u_samples, dtype=float)
    if u_samples.ndim != 1 or u_samples.size == 0:
        raise ValueError("u_samples must be a non-empty 1D array")
    if not np.all(np.isfinite(u_samples)):
        raise ValueError("u_samples must be finite")
    m = u_samples.size
    desired = np.asarray(desired, dtype=complex)
    if desired.shape not in ((), (m,)):
        raise ValueError(
            f"desired must have shape ({m},) to match u_samples, "
            f"got {desired.shape}"
        )
    if not np.all(np.isfinite(desired)):
        raise ValueError("desired must be finite")
    desired = np.broadcast_to(desired, (m,))

    if (n_elements is None) == (positions is None):
        raise ValueError("give exactly one of n_elements or positions")
    if positions is None:
        n = _check_n_elements(n_elements)
        if k is not None:
            raise ValueError(
                "k applies only with positions; a uniform array uses "
                "spacing in wavelengths"
            )
        x = _ula_positions(n, _check_spacing(spacing))
        kk = 2 * np.pi
    else:
        x = np.asarray(positions, dtype=float)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("positions must be a non-empty 1D array")
        if not np.all(np.isfinite(x)):
            raise ValueError("positions must be finite")
        if k is None:
            raise ValueError("k (rad/m) is required with positions in meters")
        kk = float(k)
        if not (np.isfinite(kk) and kk > 0):
            raise ValueError(f"k must be a finite positive number, got {k!r}")
        n = x.size

    if weights is None:
        c = np.ones(m)
    else:
        c = np.asarray(weights, dtype=float)
        if c.shape != (m,):
            raise ValueError(
                f"weights must have shape ({m},) to match u_samples, "
                f"got {c.shape}"
            )
        if np.any(c < 0) or not np.all(np.isfinite(c)):
            raise ValueError("weights must be finite and nonnegative")
    regularization = float(regularization)
    if not (np.isfinite(regularization) and regularization >= 0):
        raise ValueError(
            "regularization must be finite and nonnegative, "
            f"got {regularization!r}"
        )

    sqrt_c = np.sqrt(c)
    A = np.exp(1j * kk * np.outer(u_samples, x)) * sqrt_c[:, None]
    rhs = desired * sqrt_c
    if regularization > 0:
        A = np.vstack([A, np.sqrt(regularization) * np.eye(n)])
        rhs = np.concatenate([rhs, np.zeros(n, dtype=complex)])
    w, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return w
