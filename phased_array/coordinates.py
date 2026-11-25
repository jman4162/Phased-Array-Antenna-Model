"""
Coordinate system transformations for phased array antennas.

Includes conversions between antenna, radar, and cone/clock coordinate systems,
as well as rotation matrices for pattern rotation.
"""

from typing import Tuple, Union

import numpy as np
from scipy import interpolate

ArrayLike = Union[np.ndarray, float]


def antenna_to_radar(
    theta_ant: ArrayLike,
    phi_ant: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert antenna coordinates (theta/phi) to radar coordinates (az/el).

    Antenna coordinates:
        - theta: angle from boresight (z-axis), 0 at boresight
        - phi: azimuthal angle in x-y plane, 0 along x-axis

    Radar coordinates (array face pointing at zenith):
        - az: azimuth in the array (x-y) plane, measured from the x-axis,
          ``atan2(v, u)``
        - el: elevation above the array plane, ``90 deg - theta``; the
          array boresight (+z) is at el = 90 deg

    For az/el angles centred on boresight (boresight at az = el = 0), use
    :func:`~phased_array.utils.thetaphi_to_azel` instead.

    Parameters
    ----------
    theta_ant : array_like
        Theta angle in radians (from boresight)
    phi_ant : array_like
        Phi angle in radians (azimuthal)

    Returns
    -------
    az : ndarray
        Azimuth angle in radians
    el : ndarray
        Elevation angle in radians

    Examples
    --------
    Array boresight maps to the zenith (el = 90 deg):

    >>> import numpy as np
    >>> import phased_array as pa
    >>> az, el = pa.antenna_to_radar(0.0, 0.0)
    >>> np.isclose(az, 0.0) and np.isclose(el, np.pi/2)
    True

    Off-boresight conversion:

    >>> az, el = pa.antenna_to_radar(np.pi/6, 0.0)  # 30 deg in x-z plane
    >>> np.isclose(np.rad2deg(az), 0.0, atol=1e-10)
    True
    """
    theta_ant = np.asarray(theta_ant)
    phi_ant = np.asarray(phi_ant)

    # Convert to direction cosines
    u = np.sin(theta_ant) * np.cos(phi_ant)
    v = np.sin(theta_ant) * np.sin(phi_ant)
    w = np.cos(theta_ant)

    # Radar coordinates: az measured from x-axis in x-y plane
    # el measured from x-y plane toward z-axis
    az = np.arctan2(v, u)
    el = np.arcsin(np.clip(w, -1, 1))

    return az, el


def radar_to_antenna(
    az: ArrayLike,
    el: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert radar coordinates (az/el) to antenna coordinates (theta/phi).

    Parameters
    ----------
    az : array_like
        Azimuth angle in radians
    el : array_like
        Elevation angle in radians (from horizon)

    Returns
    -------
    theta : ndarray
        Theta angle in radians (from boresight)
    phi : ndarray
        Phi angle in radians (azimuthal)

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> theta, phi = pa.radar_to_antenna(0.0, np.pi/2)  # Boresight
    >>> np.isclose(theta, 0.0, atol=1e-10)
    True

    Round-trip conversion:

    >>> theta_orig, phi_orig = np.pi/4, np.pi/3
    >>> az, el = pa.antenna_to_radar(theta_orig, phi_orig)
    >>> theta, phi = pa.radar_to_antenna(az, el)
    >>> np.isclose(theta, theta_orig, atol=1e-10)
    True
    """
    az = np.asarray(az)
    el = np.asarray(el)

    # Direction cosines from radar coords
    u = np.cos(el) * np.cos(az)
    v = np.cos(el) * np.sin(az)
    w = np.sin(el)

    # Convert to antenna theta/phi
    theta = np.arccos(np.clip(w, -1, 1))
    phi = np.arctan2(v, u)

    return theta, phi


def antenna_to_cone(
    theta: ArrayLike,
    phi: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert antenna coordinates (theta/phi) to cone/clock coordinates.

    Cone/clock coordinates are useful for describing patterns on
    aircraft radomes or for visualizing scan limits.

    Parameters
    ----------
    theta : array_like
        Theta angle in radians (from boresight)
    phi : array_like
        Phi angle in radians (azimuthal)

    Returns
    -------
    cone : ndarray
        Cone angle in radians (distance from boresight, same as theta)
    clock : ndarray
        Clock angle in radians (azimuthal position, same as phi)

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> cone, clock = pa.antenna_to_cone(np.pi/6, np.pi/4)
    >>> np.isclose(cone, np.pi/6) and np.isclose(clock, np.pi/4)
    True
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # Cone angle is the same as theta (angle from boresight)
    cone = theta

    # Clock angle is the same as phi (azimuthal)
    clock = phi

    return cone, clock


def cone_to_antenna(
    cone: ArrayLike,
    clock: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert cone/clock coordinates to antenna coordinates (theta/phi).

    Parameters
    ----------
    cone : array_like
        Cone angle in radians
    clock : array_like
        Clock angle in radians

    Returns
    -------
    theta : ndarray
        Theta angle in radians
    phi : ndarray
        Phi angle in radians

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> theta, phi = pa.cone_to_antenna(np.pi/6, np.pi/4)
    >>> np.isclose(theta, np.pi/6) and np.isclose(phi, np.pi/4)
    True
    """
    cone = np.asarray(cone)
    clock = np.asarray(clock)

    theta = cone
    phi = clock

    return theta, phi


def rotation_matrix_roll(angle: float) -> np.ndarray:
    """
    Create 3x3 rotation matrix for roll (rotation about x-axis).

    Parameters
    ----------
    angle : float
        Roll angle in radians (positive = right wing down)

    Returns
    -------
    R : ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> R = pa.rotation_matrix_roll(0.0)
    >>> np.allclose(R, np.eye(3))
    True

    90 degree roll:

    >>> R = pa.rotation_matrix_roll(np.pi/2)
    >>> v = np.array([0, 1, 0])  # y-axis
    >>> v_rot = R @ v
    >>> np.allclose(v_rot, [0, 0, 1], atol=1e-10)  # Rotates to z-axis
    True
    """
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_pitch(angle: float) -> np.ndarray:
    """
    Create 3x3 rotation matrix for pitch (rotation about y-axis).

    Parameters
    ----------
    angle : float
        Pitch angle in radians (positive = nose up)

    Returns
    -------
    R : ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> R = pa.rotation_matrix_pitch(0.0)
    >>> np.allclose(R, np.eye(3))
    True

    90 degree pitch:

    >>> R = pa.rotation_matrix_pitch(np.pi/2)
    >>> v = np.array([0, 0, 1])  # z-axis (boresight)
    >>> v_rot = R @ v
    >>> np.allclose(v_rot, [1, 0, 0], atol=1e-10)  # Rotates to x-axis
    True
    """
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_yaw(angle: float) -> np.ndarray:
    """
    Create 3x3 rotation matrix for yaw (rotation about z-axis).

    Parameters
    ----------
    angle : float
        Yaw angle in radians (positive = nose left)

    Returns
    -------
    R : ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> R = pa.rotation_matrix_yaw(0.0)
    >>> np.allclose(R, np.eye(3))
    True

    90 degree yaw:

    >>> R = pa.rotation_matrix_yaw(np.pi/2)
    >>> v = np.array([1, 0, 0])  # x-axis
    >>> v_rot = R @ v
    >>> np.allclose(v_rot, [0, 1, 0], atol=1e-10)  # Rotates to y-axis
    True
    """
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotate_pattern(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern: np.ndarray,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate a radiation pattern by specified Euler angles.

    The rotation order is: yaw -> pitch -> roll (intrinsic rotations),
    which corresponds to standard aerospace convention.

    Parameters
    ----------
    theta : ndarray
        Original theta angles in radians (1D or 2D grid)
    phi : ndarray
        Original phi angles in radians (same shape as theta)
    pattern : ndarray
        Pattern values (complex or magnitude, same shape as theta)
    roll_deg : float
        Roll angle in degrees
    pitch_deg : float
        Pitch angle in degrees
    yaw_deg : float
        Yaw angle in degrees

    Returns
    -------
    theta_out : ndarray
        Theta grid of the output, equal to the input ``theta``
    phi_out : ndarray
        Phi grid of the output, equal to the input ``phi``
    pattern_interp : ndarray
        Rotated pattern sampled on ``(theta_out, phi_out)``: the value at
        each grid direction is the original pattern at the direction that
        the rotation maps onto it

    Raises
    ------
    ValueError
        If ``theta``, ``phi`` and ``pattern`` are not 2D grids of the same
        shape (``np.meshgrid(..., indexing='ij')``)

    Examples
    --------
    Rotate pattern by 30 degrees in yaw:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> theta = np.linspace(0, np.pi/2, 46)
    >>> phi = np.linspace(0, 2*np.pi, 73)
    >>> theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    >>> pattern = np.cos(theta_grid)  # Simple cosine pattern
    >>> theta_r, phi_r, pattern_r = pa.rotate_pattern(
    ...     theta_grid, phi_grid, pattern,
    ...     roll_deg=0, pitch_deg=0, yaw_deg=30
    ... )
    >>> pattern_r.shape == pattern.shape
    True

    Notes
    -----
    Directions whose source lies outside the original grid are filled
    with zero. Before v1.6.0 the first two outputs were the input points
    rotated forward, which did not correspond to ``pattern_interp``.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    pattern = np.asarray(pattern)
    if theta.ndim != 2 or theta.shape != phi.shape or theta.shape != pattern.shape:
        raise ValueError(
            "theta, phi and pattern must be 2D grids of the same shape "
            "(np.meshgrid(..., indexing='ij')), got shapes "
            f"{theta.shape}, {phi.shape} and {pattern.shape}"
        )

    # Create rotation matrix (yaw * pitch * roll)
    R_roll = rotation_matrix_roll(np.deg2rad(roll_deg))
    R_pitch = rotation_matrix_pitch(np.deg2rad(pitch_deg))
    R_yaw = rotation_matrix_yaw(np.deg2rad(yaw_deg))
    R = R_yaw @ R_pitch @ R_roll

    # Each output grid direction takes the value of the original pattern at
    # the direction that R maps onto it, i.e. R^T applied to the output.
    theta_flat = theta.ravel()
    phi_flat = phi.ravel()
    vectors_out = np.vstack([np.sin(theta_flat) * np.cos(phi_flat),
                             np.sin(theta_flat) * np.sin(phi_flat),
                             np.cos(theta_flat)])
    vectors_src = R.T @ vectors_out

    theta_1d = theta[:, 0]
    phi_1d = phi[0, :]

    # Wrap source phi into the grid's own range, and close an open
    # full-turn grid ([0, 2*pi) without the endpoint) at the seam.
    two_pi = 2 * np.pi
    phi_min = float(phi_1d[0])
    grid_phi = phi_1d
    close_seam = False
    if phi_1d.size > 1:
        span = float(phi_1d[-1] - phi_1d[0])
        step = float(np.median(np.diff(phi_1d)))
        tol = 1e-6 * two_pi
        close_seam = span < two_pi - tol and span + step >= two_pi - tol
        if close_seam:
            grid_phi = np.append(phi_1d, phi_min + two_pi)

    theta_src = np.arccos(np.clip(vectors_src[2], -1, 1))
    phi_src = phi_min + np.mod(
        np.arctan2(vectors_src[1], vectors_src[0]) - phi_min, two_pi)
    points = np.column_stack([theta_src, phi_src])

    def sample(values):
        if close_seam:
            values = np.concatenate([values, values[:, :1]], axis=1)
        interp = interpolate.RegularGridInterpolator(
            (theta_1d, grid_phi), values,
            method='linear', bounds_error=False, fill_value=0
        )
        return interp(points)

    if np.iscomplexobj(pattern):
        pattern_interp = sample(np.real(pattern)) + 1j * sample(np.imag(pattern))
    else:
        pattern_interp = sample(pattern)
    pattern_interp = pattern_interp.reshape(theta.shape)

    return theta.copy(), phi.copy(), pattern_interp
