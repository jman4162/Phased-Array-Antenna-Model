"""
Vector (polarized) pattern computation for phased array antennas.

Bridges the scalar array-factor engine in `core` with the polarization
math in `polarization`: polarized element models produce (E_theta, E_phi)
field components, which multiply the array factor to give full vector
patterns, co/cross-polar decompositions, axial-ratio maps, and
polarization-correct conformal array patterns.

A polarized element pattern is any callable

    f(theta, phi, **kwargs) -> (E_theta, E_phi)

returning the complex field components in the spherical basis of the
frame it is evaluated in, for boresight along +z. The factories in this
module (`dipole_element`, `ideal_patch_element`,
`cos_q_polarized_element`, `crossed_dipole_element`) and
`GriddedElementPattern` all satisfy this protocol.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .core import array_factor_vectorized
from .geometry import ArrayGeometry
from .polarization import axial_ratio, ludwig3_decomposition
from .utils import create_theta_phi_grid, linear_to_db

# ============== Pattern container ==============

@dataclass
class VectorPattern:
    """
    Full vector radiation pattern on a theta/phi grid.

    Attributes
    ----------
    theta : ndarray
        Theta grid in radians, shape (n_theta, n_phi)
    phi : ndarray
        Phi grid in radians, shape (n_theta, n_phi)
    E_theta : ndarray
        Complex theta field component, shape (n_theta, n_phi)
    E_phi : ndarray
        Complex phi field component, shape (n_theta, n_phi)
    """
    theta: np.ndarray
    phi: np.ndarray
    E_theta: np.ndarray
    E_phi: np.ndarray

    @property
    def power(self) -> np.ndarray:
        """Total radiated power density ``|E_theta|^2 + |E_phi|^2``."""
        return np.abs(self.E_theta) ** 2 + np.abs(self.E_phi) ** 2

    def power_dB(self, normalize: bool = True) -> np.ndarray:
        """
        Total power pattern in dB.

        Parameters
        ----------
        normalize : bool
            If True (default), normalize so the peak is 0 dB.
        """
        power = self.power
        if normalize:
            peak = np.max(power)
            if peak > 0:
                power = power / peak
        return linear_to_db(power)

    def co_cross(
        self, reference_pol: str = 'ludwig3'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Co- and cross-polar field components.

        Parameters
        ----------
        reference_pol : str
            'ludwig3' (x-polarized reference) or 'ludwig3-y'
            (y-polarized reference).

        Returns
        -------
        E_co, E_cross : ndarray
            Complex co- and cross-polar components, same shape as
            `E_theta`.
        """
        if reference_pol == 'ludwig3':
            return ludwig3_decomposition(
                self.theta, self.phi, self.E_theta, self.E_phi
            )
        elif reference_pol == 'ludwig3-y':
            # Rotate the reference plane by 90 degrees; the rotated
            # decomposition returns (co_y, cross_y) directly
            return ludwig3_decomposition(
                self.theta, self.phi - np.pi / 2, self.E_theta, self.E_phi
            )
        else:
            raise ValueError(
                f"reference_pol must be 'ludwig3' or 'ludwig3-y', "
                f"got {reference_pol!r}"
            )

    def axial_ratio_map(self) -> np.ndarray:
        """
        Axial ratio at every grid point (1 = circular, inf = linear),
        computed from the Ludwig-3 co/cross components.
        """
        E_co, E_cross = self.co_cross()
        return axial_ratio(np.stack([E_co, E_cross]))

    def xpd_map(self, reference_pol: str = 'ludwig3') -> np.ndarray:
        """
        Cross-polar discrimination map in dB: ``|E_co|^2 / |E_cross|^2``.
        """
        E_co, E_cross = self.co_cross(reference_pol)
        co_power = np.abs(E_co) ** 2
        cross_power = np.abs(E_cross) ** 2
        return linear_to_db(co_power / np.maximum(cross_power, 1e-30))


# ============== Polarized element models ==============

def dipole_element(orientation: str = 'x') -> Callable:
    """
    Ideal short (Hertzian) dipole element pattern.

    Parameters
    ----------
    orientation : str
        Dipole axis: 'x', 'y', or 'z'.

    Returns
    -------
    element_func : callable
        f(theta, phi) -> (E_theta, E_phi), normalized so the peak field
        magnitude is 1.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> f = pa.dipole_element('x')
    >>> E_theta, E_phi = f(np.array([0.0]), np.array([0.0]))
    >>> np.isclose(abs(E_theta[0]), 1.0)  # boresight, co-pol
    True
    """
    if orientation not in ('x', 'y', 'z'):
        raise ValueError(f"orientation must be 'x', 'y', or 'z', got {orientation!r}")

    def element_func(theta, phi, **kwargs):
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        if orientation == 'x':
            E_theta = np.cos(theta) * np.cos(phi)
            E_phi = -np.sin(phi) * np.ones_like(theta)
        elif orientation == 'y':
            E_theta = np.cos(theta) * np.sin(phi)
            E_phi = np.cos(phi) * np.ones_like(theta)
        else:  # 'z'
            E_theta = -np.sin(theta)
            E_phi = np.zeros_like(theta)
        return E_theta.astype(complex), E_phi.astype(complex)

    return element_func


def ideal_patch_element(pol: str = 'x', cos_exp: float = 1.0) -> Callable:
    """
    Ideal Ludwig-3 element: cos^n(theta) co-polar magnitude with zero
    cross-polarization by construction.

    Parameters
    ----------
    pol : str
        Co-polar direction, 'x' or 'y'.
    cos_exp : float
        Cosine exponent of the theta roll-off.

    Returns
    -------
    element_func : callable
        f(theta, phi) -> (E_theta, E_phi). Zero behind the +z hemisphere.
    """
    if pol not in ('x', 'y'):
        raise ValueError(f"pol must be 'x' or 'y', got {pol!r}")

    def element_func(theta, phi, **kwargs):
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        cos_theta = np.cos(theta)
        magnitude = np.where(
            cos_theta > 0, np.maximum(cos_theta, 0.0) ** cos_exp, 0.0
        )
        if pol == 'x':
            # Inverse Ludwig-3 with E_co = magnitude, E_cross = 0
            E_theta = magnitude * np.cos(phi)
            E_phi = -magnitude * np.sin(phi)
        else:
            E_theta = magnitude * np.sin(phi)
            E_phi = magnitude * np.cos(phi)
        return E_theta.astype(complex), E_phi.astype(complex)

    return element_func


def cos_q_polarized_element(
    q_theta: float = 1.0,
    jones: Optional[np.ndarray] = None,
    max_gain_dBi: float = 0.0
) -> Callable:
    """
    cos^q(theta) element with an arbitrary polarization state.

    Successor to the deprecated ``cos_exp_phi`` parameter of
    `core.element_pattern`: the scalar cos^q magnitude is mapped onto
    the Ludwig-3 basis of the given Jones vector.

    Parameters
    ----------
    q_theta : float
        Cosine exponent for the theta roll-off.
    jones : ndarray, optional
        Jones vector [E_x, E_y] defining the polarization state
        (normalized internally). Default [1, 0] (x-linear).
    max_gain_dBi : float
        Peak element gain in dBi (applied as a field scale factor).

    Returns
    -------
    element_func : callable
        f(theta, phi) -> (E_theta, E_phi).
    """
    if jones is None:
        jones = np.array([1.0, 0.0], dtype=complex)
    jones = np.asarray(jones, dtype=complex)
    norm = np.sqrt(np.abs(jones[0]) ** 2 + np.abs(jones[1]) ** 2)
    if norm == 0:
        raise ValueError("jones vector must be nonzero")
    jones = jones / norm
    field_scale = 10 ** (max_gain_dBi / 20)

    x_pol = ideal_patch_element('x', cos_exp=q_theta)
    y_pol = ideal_patch_element('y', cos_exp=q_theta)

    def element_func(theta, phi, **kwargs):
        Ex_theta, Ex_phi = x_pol(theta, phi)
        Ey_theta, Ey_phi = y_pol(theta, phi)
        E_theta = field_scale * (jones[0] * Ex_theta + jones[1] * Ey_theta)
        E_phi = field_scale * (jones[0] * Ex_phi + jones[1] * Ey_phi)
        return E_theta, E_phi

    return element_func


def crossed_dipole_element(phase_diff: float = -np.pi / 2) -> Callable:
    """
    Crossed-dipole (turnstile) element for circular polarization.

    An x-dipole and a y-dipole fed with a relative phase of
    ``phase_diff``: -pi/2 gives RHCP at boresight, +pi/2 LHCP.

    Returns
    -------
    element_func : callable
        f(theta, phi) -> (E_theta, E_phi), normalized so the boresight
        co-polar magnitude is 1.
    """
    x_dip = dipole_element('x')
    y_dip = dipole_element('y')
    phase = np.exp(1j * phase_diff)

    def element_func(theta, phi, **kwargs):
        Ex_theta, Ex_phi = x_dip(theta, phi)
        Ey_theta, Ey_phi = y_dip(theta, phi)
        E_theta = (Ex_theta + phase * Ey_theta) / np.sqrt(2)
        E_phi = (Ex_phi + phase * Ey_phi) / np.sqrt(2)
        return E_theta, E_phi

    return element_func


class GriddedElementPattern:
    """
    Polarized element pattern interpolated from gridded data
    (e.g. a full-wave simulation or measurement export).

    Satisfies the element protocol: instances are callables
    f(theta, phi) -> (E_theta, E_phi).

    Parameters
    ----------
    theta_grid : ndarray
        Monotonic 1D theta sample points in radians
    phi_grid : ndarray
        Monotonic 1D phi sample points in radians
    E_theta : ndarray
        Complex theta component, shape (len(theta_grid), len(phi_grid))
    E_phi : ndarray
        Complex phi component, same shape
    method : str
        Interpolation method for `scipy.interpolate.RegularGridInterpolator`
    """

    def __init__(
        self,
        theta_grid: np.ndarray,
        phi_grid: np.ndarray,
        E_theta: np.ndarray,
        E_phi: np.ndarray,
        method: str = 'linear'
    ):
        from scipy.interpolate import RegularGridInterpolator

        points = (np.asarray(theta_grid, dtype=float),
                  np.asarray(phi_grid, dtype=float))
        common = dict(method=method, bounds_error=False, fill_value=0.0)
        self._interp_theta = RegularGridInterpolator(
            points, np.asarray(E_theta, dtype=complex), **common
        )
        self._interp_phi = RegularGridInterpolator(
            points, np.asarray(E_phi, dtype=complex), **common
        )

    def __call__(self, theta, phi, **kwargs):
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        pts = np.stack([theta.ravel(), np.mod(phi.ravel(), 2 * np.pi)], axis=-1)
        E_theta = self._interp_theta(pts).reshape(theta.shape)
        E_phi = self._interp_phi(pts).reshape(theta.shape)
        return E_theta, E_phi

    @classmethod
    def from_scalar(
        cls,
        theta_grid: np.ndarray,
        phi_grid: np.ndarray,
        magnitude: np.ndarray,
        jones: Optional[np.ndarray] = None,
        method: str = 'linear'
    ) -> 'GriddedElementPattern':
        """
        Build a polarized gridded pattern from a scalar magnitude grid
        and a Jones vector (default x-linear), mapping the magnitude
        onto the Ludwig-3 basis like `cos_q_polarized_element`.
        """
        if jones is None:
            jones = np.array([1.0, 0.0], dtype=complex)
        jones = np.asarray(jones, dtype=complex)
        jones = jones / np.sqrt(np.abs(jones[0]) ** 2 + np.abs(jones[1]) ** 2)

        THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')
        magnitude = np.asarray(magnitude, dtype=complex)
        cos_phi = np.cos(PHI)
        sin_phi = np.sin(PHI)
        E_theta = magnitude * (jones[0] * cos_phi + jones[1] * sin_phi)
        E_phi = magnitude * (-jones[0] * sin_phi + jones[1] * cos_phi)
        return cls(theta_grid, phi_grid, E_theta, E_phi, method=method)


# ============== Vector pattern computation ==============

def vector_total_pattern(
    theta: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float,
    element_func: Callable,
    z: Optional[np.ndarray] = None,
    **element_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vector total pattern for identical, identically-oriented elements:
    (E_theta, E_phi) = AF * element field components.

    Parameters
    ----------
    theta, phi : ndarray
        Observation angles in radians (any matching shapes)
    x, y : ndarray
        Element positions in meters
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber in rad/m
    element_func : callable
        Polarized element pattern ``f(theta, phi, **kwargs) -> (E_theta, E_phi)``
    z : ndarray, optional
        Element z-positions

    Returns
    -------
    E_theta, E_phi : ndarray
        Complex field components, same shape as theta.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> w = np.ones(16)
    >>> f = pa.ideal_patch_element('x')
    >>> E_theta, E_phi = pa.vector_total_pattern(
    ...     np.array([0.0]), np.array([0.0]), geom.x, geom.y, w, k, f
    ... )
    >>> np.isclose(abs(E_theta[0]), 16.0)  # AF peak x unit co-pol field
    True
    """
    AF = array_factor_vectorized(theta, phi, x, y, weights, k, z)
    E_theta_e, E_phi_e = element_func(theta, phi, **element_kwargs)
    return AF * E_theta_e, AF * E_phi_e


def compute_full_vector_pattern(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float,
    element_func: Optional[Callable] = None,
    n_theta: int = 181,
    n_phi: int = 361,
    theta_range: Tuple[float, float] = (0, np.pi / 2),
    phi_range: Tuple[float, float] = (0, 2 * np.pi),
    z: Optional[np.ndarray] = None,
    **element_kwargs
) -> VectorPattern:
    """
    Full vector pattern over a theta/phi grid.

    Mirrors `core.compute_full_pattern` grid conventions but returns a
    `VectorPattern` with complex (E_theta, E_phi) instead of a scalar
    dB pattern.

    Parameters
    ----------
    element_func : callable, optional
        Polarized element pattern; default `ideal_patch_element('x')`.
    (other parameters as in `core.compute_full_pattern`)
    """
    if element_func is None:
        element_func = ideal_patch_element('x')

    _, _, THETA, PHI = create_theta_phi_grid(
        theta_range=theta_range, phi_range=phi_range,
        n_theta=n_theta, n_phi=n_phi
    )
    E_theta, E_phi = vector_total_pattern(
        THETA, PHI, x, y, weights, k, element_func, z=z, **element_kwargs
    )
    return VectorPattern(theta=THETA, phi=PHI, E_theta=E_theta, E_phi=E_phi)


def compute_co_cross_pattern_cuts(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float,
    element_func: Callable,
    phi_cut_deg: float = 0.0,
    theta_range_deg: Tuple[float, float] = (-90, 90),
    n_points: int = 361,
    reference_pol: str = 'ludwig3',
    **element_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Co- and cross-polar 1D pattern cuts through a phi plane.

    Negative theta values are mapped to the phi + 180 deg half-plane.
    Both cuts are normalized to the co-polar peak.

    Returns
    -------
    theta_deg : ndarray
        Cut angles in degrees
    co_dB, cross_dB : ndarray
        Co- and cross-polar patterns in dB relative to the co-pol peak
    """
    theta_deg = np.linspace(theta_range_deg[0], theta_range_deg[1], n_points)
    theta = np.deg2rad(np.abs(theta_deg))
    phi_cut = np.deg2rad(phi_cut_deg)
    phi = np.where(theta_deg >= 0, phi_cut, phi_cut + np.pi)

    E_theta, E_phi = vector_total_pattern(
        theta, phi, x, y, weights, k, element_func, **element_kwargs
    )
    pattern = VectorPattern(theta=theta, phi=phi, E_theta=E_theta, E_phi=E_phi)
    E_co, E_cross = pattern.co_cross(reference_pol)

    co_power = np.abs(E_co) ** 2
    cross_power = np.abs(E_cross) ** 2
    peak = np.max(co_power)
    if peak > 0:
        co_power = co_power / peak
        cross_power = cross_power / peak
    return theta_deg, linear_to_db(co_power), linear_to_db(cross_power)


def dual_pol_weights(
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    jones_goal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale two orthogonal feed weight sets to synthesize a target
    polarization state.

    Parameters
    ----------
    weights_x, weights_y : ndarray
        Complex weights of the x- and y-polarized feeds
    jones_goal : ndarray
        Target Jones vector [E_x, E_y] (normalized internally)

    Returns
    -------
    weights_x_scaled, weights_y_scaled : ndarray
    """
    jones_goal = np.asarray(jones_goal, dtype=complex)
    norm = np.sqrt(np.abs(jones_goal[0]) ** 2 + np.abs(jones_goal[1]) ** 2)
    if norm == 0:
        raise ValueError("jones_goal must be nonzero")
    jones_goal = jones_goal / norm
    return jones_goal[0] * weights_x, jones_goal[1] * weights_y


# ============== Conformal vector patterns ==============

def element_rotation_matrices(geometry: ArrayGeometry) -> np.ndarray:
    """
    Per-element rotation matrices from the global frame to each
    element's local frame.

    The local z-axis is the element normal. The local x-axis is the
    element tangent (`tx`, `ty`, `tz`) if the geometry provides one,
    otherwise normalize(z_hat x normal), falling back to the global
    x-axis when the normal is parallel to z_hat. The local y-axis
    completes the right-handed triad.

    Returns
    -------
    R : ndarray, shape (n_elements, 3, 3)
        Rows of R[i] are the local basis vectors in global coordinates,
        so v_local = R[i] @ v_global.
    """
    n = geometry.n_elements
    if geometry.nx is None or geometry.ny is None or geometry.nz is None:
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    else:
        normals = np.stack([geometry.nx, geometry.ny, geometry.nz], axis=1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-30)

    tangents = getattr(geometry, 'tx', None)
    if tangents is not None and geometry.ty is not None and geometry.tz is not None:
        x_local = np.stack([geometry.tx, geometry.ty, geometry.tz], axis=1)
        x_local = x_local / np.maximum(
            np.linalg.norm(x_local, axis=1, keepdims=True), 1e-30
        )
    else:
        z_hat = np.array([0.0, 0.0, 1.0])
        x_local = np.cross(np.tile(z_hat, (n, 1)), normals)
        lengths = np.linalg.norm(x_local, axis=1, keepdims=True)
        degenerate = lengths[:, 0] < 1e-9
        x_local = np.where(
            degenerate[:, None], np.array([1.0, 0.0, 0.0]), x_local
        )
        lengths = np.where(degenerate[:, None], 1.0, lengths)
        x_local = x_local / lengths

    y_local = np.cross(normals, x_local)

    R = np.stack([x_local, y_local, normals], axis=1)
    return R


def global_to_local_angles(
    R: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Observation angles in each element's local frame.

    Parameters
    ----------
    R : ndarray, shape (3, 3) or (n_elements, 3, 3)
        Rotation matrices from `element_rotation_matrices`
    theta, phi : ndarray
        Global observation angles in radians (flattened internally)

    Returns
    -------
    local_theta, local_phi : ndarray
        Local angles; shape (n_angles,) for a single matrix, else
        (n_elements, n_angles).
    """
    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    d = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])  # (3, n_angles)

    single = (R.ndim == 2)
    R3 = R[None, :, :] if single else R
    d_local = np.einsum('nij,jk->nik', R3, d)  # (n_elem, 3, n_angles)

    local_theta = np.arccos(np.clip(d_local[:, 2, :], -1.0, 1.0))
    local_phi = np.arctan2(d_local[:, 1, :], d_local[:, 0, :])
    if single:
        return local_theta[0], local_phi[0]
    return local_theta, local_phi


def _spherical_basis(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit vectors theta_hat, phi_hat as arrays of shape (3, n)."""
    theta_hat = np.stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta),
    ])
    phi_hat = np.stack([
        -np.sin(phi),
        np.cos(phi),
        np.zeros_like(phi),
    ])
    return theta_hat, phi_hat


def vector_array_factor_conformal(
    theta: np.ndarray,
    phi: np.ndarray,
    geometry: ArrayGeometry,
    weights: np.ndarray,
    k: float,
    element_func: Optional[Callable] = None,
    per_element_funcs: Optional[Sequence[Callable]] = None,
    **element_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vector pattern for a conformal array with per-element orientation.

    Each element's polarized pattern is evaluated in its local frame
    (local z = element normal), the resulting field is rotated back to
    global Cartesian coordinates, and projected onto the global
    spherical basis at the observation angle before summation.

    Parameters
    ----------
    theta, phi : ndarray
        Global observation angles in radians
    geometry : ArrayGeometry
        Geometry; normals default to +z when absent
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber in rad/m
    element_func : callable, optional
        Polarized element pattern used for every element; default
        `ideal_patch_element('x')`
    per_element_funcs : sequence of callables, optional
        Per-element patterns (e.g. embedded element patterns);
        overrides `element_func`

    Returns
    -------
    E_theta, E_phi : ndarray
        Complex global field components, same shape as theta.
    """
    if element_func is None:
        element_func = ideal_patch_element('x')
    if per_element_funcs is not None and len(per_element_funcs) != geometry.n_elements:
        raise ValueError(
            f"per_element_funcs has {len(per_element_funcs)} entries "
            f"for {geometry.n_elements} elements"
        )

    original_shape = np.asarray(theta).shape
    theta_flat = np.asarray(theta, dtype=float).ravel()
    phi_flat = np.asarray(phi, dtype=float).ravel()
    n_angles = len(theta_flat)

    obs = np.stack([
        np.sin(theta_flat) * np.cos(phi_flat),
        np.sin(theta_flat) * np.sin(phi_flat),
        np.cos(theta_flat),
    ])  # (3, n_angles)
    theta_hat, phi_hat = _spherical_basis(theta_flat, phi_flat)

    R = element_rotation_matrices(geometry)
    z = geometry.z if geometry.z is not None else np.zeros_like(geometry.x)

    E_theta = np.zeros(n_angles, dtype=complex)
    E_phi = np.zeros(n_angles, dtype=complex)

    for i in range(geometry.n_elements):
        phase = k * (geometry.x[i] * obs[0] + geometry.y[i] * obs[1]
                     + z[i] * obs[2])

        local_theta, local_phi = global_to_local_angles(R[i], theta_flat, phi_flat)
        func = per_element_funcs[i] if per_element_funcs is not None else element_func
        E_theta_l, E_phi_l = func(local_theta, local_phi, **element_kwargs)

        # Field vector in the element's local Cartesian frame
        theta_hat_l, phi_hat_l = _spherical_basis(local_theta, local_phi)
        E_cart_local = E_theta_l * theta_hat_l + E_phi_l * phi_hat_l

        # Rotate to global Cartesian: v_global = R^T v_local
        E_cart_global = R[i].T @ E_cart_local

        # Project onto the global spherical basis and accumulate
        w_phase = weights[i] * np.exp(1j * phase)
        E_theta += w_phase * np.sum(theta_hat * E_cart_global, axis=0)
        E_phi += w_phase * np.sum(phi_hat * E_cart_global, axis=0)

    return E_theta.reshape(original_shape), E_phi.reshape(original_shape)
