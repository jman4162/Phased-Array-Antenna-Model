"""
Utility functions for phased array antenna computations.

Includes coordinate transforms and helper functions.
"""

import numpy as np
from typing import Tuple, Union

ArrayLike = Union[np.ndarray, float]


def deg2rad(degrees: ArrayLike) -> ArrayLike:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)


def rad2deg(radians: ArrayLike) -> ArrayLike:
    """Convert radians to degrees."""
    return np.rad2deg(radians)


def azel_to_thetaphi(az: ArrayLike, el: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Convert azimuth/elevation to theta/phi (spherical coordinates).

    Parameters
    ----------
    az : array_like
        Azimuth angle in radians (0 = boresight, positive = right)
    el : array_like
        Elevation angle in radians (0 = horizon, positive = up)

    Returns
    -------
    theta : array_like
        Polar angle from z-axis in radians (0 = zenith)
    phi : array_like
        Azimuthal angle in radians
    """
    az = np.asarray(az)
    el = np.asarray(el)

    theta = np.arccos(np.cos(az) * np.cos(el))
    phi = np.arctan2(np.sin(az), np.tan(el))

    return theta, phi


def thetaphi_to_azel(theta: ArrayLike, phi: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Convert theta/phi (spherical coordinates) to azimuth/elevation.

    Parameters
    ----------
    theta : array_like
        Polar angle from z-axis in radians (0 = zenith)
    phi : array_like
        Azimuthal angle in radians

    Returns
    -------
    az : array_like
        Azimuth angle in radians
    el : array_like
        Elevation angle in radians
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    el = np.arcsin(np.cos(theta))
    az = np.arctan2(np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi))

    return az, el


def theta_phi_to_uv(theta: ArrayLike, phi: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Convert theta/phi angles to UV-space (direction cosines).

    Parameters
    ----------
    theta : array_like
        Polar angle from z-axis in radians
    phi : array_like
        Azimuthal angle in radians

    Returns
    -------
    u : array_like
        Direction cosine u = sin(theta) * cos(phi)
    v : array_like
        Direction cosine v = sin(theta) * sin(phi)
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)

    return u, v


def uv_to_theta_phi(u: ArrayLike, v: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Convert UV-space (direction cosines) to theta/phi angles.

    Parameters
    ----------
    u : array_like
        Direction cosine u = sin(theta) * cos(phi)
    v : array_like
        Direction cosine v = sin(theta) * sin(phi)

    Returns
    -------
    theta : array_like
        Polar angle from z-axis in radians
    phi : array_like
        Azimuthal angle in radians

    Notes
    -----
    Points outside the visible region (u^2 + v^2 > 1) will have
    theta values computed from the magnitude, which may be complex
    or undefined. Use is_visible_region() to check validity.
    """
    u = np.asarray(u)
    v = np.asarray(v)

    r = np.sqrt(u**2 + v**2)
    theta = np.arcsin(np.clip(r, -1, 1))
    phi = np.arctan2(v, u)

    return theta, phi


def is_visible_region(u: ArrayLike, v: ArrayLike) -> np.ndarray:
    """
    Check if UV coordinates are within the visible region.

    Parameters
    ----------
    u, v : array_like
        Direction cosines

    Returns
    -------
    visible : ndarray
        Boolean array, True where u^2 + v^2 <= 1
    """
    u = np.asarray(u)
    v = np.asarray(v)
    return (u**2 + v**2) <= 1.0


def wavelength_to_k(wavelength: float) -> float:
    """
    Convert wavelength to wavenumber.

    Parameters
    ----------
    wavelength : float
        Wavelength in meters

    Returns
    -------
    k : float
        Wavenumber (2*pi/wavelength) in rad/m
    """
    return 2.0 * np.pi / wavelength


def frequency_to_wavelength(frequency: float, c: float = 3e8) -> float:
    """
    Convert frequency to wavelength.

    Parameters
    ----------
    frequency : float
        Frequency in Hz
    c : float, optional
        Speed of light in m/s (default: 3e8)

    Returns
    -------
    wavelength : float
        Wavelength in meters
    """
    return c / frequency


def frequency_to_k(frequency: float, c: float = 3e8) -> float:
    """
    Convert frequency to wavenumber.

    Parameters
    ----------
    frequency : float
        Frequency in Hz
    c : float, optional
        Speed of light in m/s (default: 3e8)

    Returns
    -------
    k : float
        Wavenumber in rad/m
    """
    return 2.0 * np.pi * frequency / c


def db_to_linear(db: ArrayLike) -> ArrayLike:
    """Convert decibels to linear scale (power)."""
    return 10.0 ** (np.asarray(db) / 10.0)


def linear_to_db(linear: ArrayLike, min_db: float = -100.0) -> ArrayLike:
    """
    Convert linear scale (power) to decibels.

    Parameters
    ----------
    linear : array_like
        Linear power values
    min_db : float, optional
        Minimum dB value to return (clips zeros/negatives)

    Returns
    -------
    db : array_like
        Power in decibels
    """
    linear = np.asarray(linear)
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 10.0 * np.log10(linear)
    db = np.where(np.isfinite(db), db, min_db)
    return np.maximum(db, min_db)


def normalize_pattern(pattern: np.ndarray, mode: str = 'peak') -> np.ndarray:
    """
    Normalize a radiation pattern.

    Parameters
    ----------
    pattern : ndarray
        Complex or magnitude pattern values
    mode : str
        'peak' - normalize to peak value
        'power' - normalize to total radiated power

    Returns
    -------
    normalized : ndarray
        Normalized pattern (same type as input)
    """
    mag = np.abs(pattern)

    if mode == 'peak':
        max_val = np.max(mag)
        if max_val > 0:
            return pattern / max_val
        return pattern
    elif mode == 'power':
        total_power = np.sum(mag**2)
        if total_power > 0:
            return pattern / np.sqrt(total_power)
        return pattern
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def create_theta_phi_grid(
    theta_range: Tuple[float, float] = (0, np.pi),
    phi_range: Tuple[float, float] = (0, 2*np.pi),
    n_theta: int = 181,
    n_phi: int = 361
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a theta/phi grid for pattern computation.

    Parameters
    ----------
    theta_range : tuple
        (theta_min, theta_max) in radians
    phi_range : tuple
        (phi_min, phi_max) in radians
    n_theta : int
        Number of theta points
    n_phi : int
        Number of phi points

    Returns
    -------
    theta_1d : ndarray
        1D array of theta values
    phi_1d : ndarray
        1D array of phi values
    theta_grid : ndarray
        2D meshgrid of theta values
    phi_grid : ndarray
        2D meshgrid of phi values
    """
    theta_1d = np.linspace(theta_range[0], theta_range[1], n_theta)
    phi_1d = np.linspace(phi_range[0], phi_range[1], n_phi)
    theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    return theta_1d, phi_1d, theta_grid, phi_grid


def create_uv_grid(
    u_range: Tuple[float, float] = (-1, 1),
    v_range: Tuple[float, float] = (-1, 1),
    n_u: int = 201,
    n_v: int = 201
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a UV-space grid for pattern computation.

    Parameters
    ----------
    u_range : tuple
        (u_min, u_max) direction cosine range
    v_range : tuple
        (v_min, v_max) direction cosine range
    n_u : int
        Number of u points
    n_v : int
        Number of v points

    Returns
    -------
    u_1d : ndarray
        1D array of u values
    v_1d : ndarray
        1D array of v values
    u_grid : ndarray
        2D meshgrid of u values
    v_grid : ndarray
        2D meshgrid of v values
    """
    u_1d = np.linspace(u_range[0], u_range[1], n_u)
    v_1d = np.linspace(v_range[0], v_range[1], n_v)
    u_grid, v_grid = np.meshgrid(u_1d, v_1d, indexing='ij')

    return u_1d, v_1d, u_grid, v_grid
