"""
Polarization analysis and manipulation for phased array antennas.

Includes Jones vectors, Stokes parameters, axial ratio calculations,
and Ludwig-3 co/cross-pol decomposition.
"""

from typing import Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, float]


def jones_vector(
    Ex: ArrayLike,
    Ey: ArrayLike,
    phase_diff: float = 0.0
) -> np.ndarray:
    """
    Create a Jones vector representing polarization state.

    The Jones vector describes the amplitude and phase of the electric
    field components in a plane transverse to propagation.

    Parameters
    ----------
    Ex : array_like
        Amplitude of x-component (horizontal)
    Ey : array_like
        Amplitude of y-component (vertical)
    phase_diff : float
        Phase difference between Ey and Ex in radians

    Returns
    -------
    jones : ndarray
        Complex Jones vector [Ex, Ey * exp(j * phase_diff)]

    Examples
    --------
    Linear horizontal polarization:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> j = pa.jones_vector(1.0, 0.0)
    >>> np.allclose(j, [1, 0])
    True

    Right-hand circular polarization:

    >>> j = pa.jones_vector(1.0, 1.0, phase_diff=-np.pi/2)
    >>> ar = pa.axial_ratio(j)
    >>> np.isclose(ar, 1.0, atol=1e-10)  # AR = 1 for circular
    True

    Left-hand circular polarization:

    >>> j = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
    >>> ar = pa.axial_ratio(j)
    >>> np.isclose(ar, 1.0, atol=1e-10)
    True
    """
    Ex = np.asarray(Ex, dtype=complex)
    Ey = np.asarray(Ey, dtype=complex)
    return np.array([Ex, Ey * np.exp(1j * phase_diff)])


def stokes_parameters(jones: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute Stokes parameters from a Jones vector.

    The Stokes parameters provide a complete description of the
    polarization state, including partially polarized light.

    Parameters
    ----------
    jones : ndarray
        Jones vector [Ex, Ey] (complex, shape (2,) or (2, N))

    Returns
    -------
    S0 : float or ndarray
        Total intensity: |Ex|^2 + |Ey|^2
    S1 : float or ndarray
        Linear horizontal-vertical: |Ex|^2 - |Ey|^2
    S2 : float or ndarray
        Linear +45/-45: 2*Re(Ex*Ey*)
    S3 : float or ndarray
        Circular right-left: 2*Im(Ex*Ey*)

    Examples
    --------
    Horizontal linear polarization (S1 = S0):

    >>> import numpy as np
    >>> import phased_array as pa
    >>> j = pa.jones_vector(1.0, 0.0)
    >>> S0, S1, S2, S3 = pa.stokes_parameters(j)
    >>> np.isclose(S1, S0)
    True

    Circular polarization (S3 = ±S0):

    >>> j = pa.jones_vector(1.0, 1.0, phase_diff=-np.pi/2)  # RHCP
    >>> S0, S1, S2, S3 = pa.stokes_parameters(j)
    >>> np.isclose(S3, -S0, atol=1e-10)
    True
    """
    jones = np.asarray(jones, dtype=complex)
    Ex = jones[0]
    Ey = jones[1]

    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = 2 * np.imag(Ex * np.conj(Ey))

    return S0, S1, S2, S3


def axial_ratio(jones: np.ndarray) -> np.ndarray:
    """
    Compute the axial ratio from a Jones vector.

    The axial ratio is the ratio of the major to minor axes of the
    polarization ellipse. AR = 1 for circular, AR -> infinity for linear.

    Parameters
    ----------
    jones : ndarray
        Jones vector [Ex, Ey] (complex)

    Returns
    -------
    ar : float or ndarray
        Axial ratio (>= 1). Returns infinity for perfect linear polarization.

    Examples
    --------
    Circular polarization has AR = 1:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> j_circ = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
    >>> ar = pa.axial_ratio(j_circ)
    >>> np.isclose(ar, 1.0, atol=1e-10)
    True

    Linear polarization has AR = infinity:

    >>> j_lin = pa.jones_vector(1.0, 0.0)
    >>> ar = pa.axial_ratio(j_lin)
    >>> np.isinf(ar)
    True
    """
    jones = np.asarray(jones, dtype=complex)
    Ex = jones[0]
    Ey = jones[1]

    # Compute parameters of polarization ellipse
    a = np.abs(Ex)**2
    b = np.abs(Ey)**2
    c = 2 * np.real(Ex * np.conj(Ey))
    d = 2 * np.imag(Ex * np.conj(Ey))

    # Semi-major and semi-minor axes
    sum_ab = a + b
    diff = np.sqrt((a - b)**2 + c**2)

    major = np.sqrt(0.5 * (sum_ab + diff))
    minor = np.sqrt(np.maximum(0.5 * (sum_ab - diff), 0))

    # Handle linear polarization (minor = 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ar = np.where(minor > 1e-15, major / minor, np.inf)

    return ar


def tilt_angle(jones: np.ndarray) -> np.ndarray:
    """
    Compute the tilt angle of the polarization ellipse.

    The tilt angle is the orientation of the major axis of the
    polarization ellipse with respect to the x-axis (horizontal).

    Parameters
    ----------
    jones : ndarray
        Jones vector [Ex, Ey] (complex)

    Returns
    -------
    tau : float or ndarray
        Tilt angle in radians (-pi/2 to pi/2)

    Examples
    --------
    Horizontal polarization has tilt = 0:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> j = pa.jones_vector(1.0, 0.0)
    >>> tau = pa.tilt_angle(j)
    >>> np.isclose(tau, 0.0, atol=1e-10)
    True

    45-degree linear polarization:

    >>> j = pa.jones_vector(1.0, 1.0, phase_diff=0.0)
    >>> tau = pa.tilt_angle(j)
    >>> np.isclose(tau, np.pi/4, atol=1e-10)
    True
    """
    jones = np.asarray(jones, dtype=complex)
    Ex = jones[0]
    Ey = jones[1]

    # Tilt angle from Stokes parameters
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * np.conj(Ey))

    tau = 0.5 * np.arctan2(S2, S1)

    return tau


def cross_pol_discrimination(
    jones_desired: np.ndarray,
    jones_actual: np.ndarray
) -> np.ndarray:
    """
    Compute cross-polarization discrimination (XPD).

    XPD is the ratio of co-polarized to cross-polarized power,
    measuring polarization purity.

    Parameters
    ----------
    jones_desired : ndarray
        Desired (reference) Jones vector
    jones_actual : ndarray
        Actual measured Jones vector

    Returns
    -------
    xpd_dB : float or ndarray
        Cross-polarization discrimination in dB

    Examples
    --------
    Perfect match has infinite XPD:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> j_ref = pa.jones_vector(1.0, 0.0)
    >>> j_act = pa.jones_vector(1.0, 0.0)
    >>> xpd = pa.cross_pol_discrimination(j_ref, j_act)
    >>> xpd > 50  # Very high XPD
    True

    Orthogonal polarizations have XPD = -infinity:

    >>> j_h = pa.jones_vector(1.0, 0.0)
    >>> j_v = pa.jones_vector(0.0, 1.0)
    >>> xpd = pa.cross_pol_discrimination(j_h, j_v)
    >>> xpd < -50  # Very low (negative) XPD
    True
    """
    jones_desired = np.asarray(jones_desired, dtype=complex)
    jones_actual = np.asarray(jones_actual, dtype=complex)

    # Normalize reference vector
    norm_desired = np.sqrt(np.abs(jones_desired[0])**2 + np.abs(jones_desired[1])**2)
    if norm_desired < 1e-15:
        return np.array(-np.inf)
    jones_desired_norm = jones_desired / norm_desired

    # Create orthogonal (cross-pol) reference
    jones_cross = np.array([-np.conj(jones_desired_norm[1]), np.conj(jones_desired_norm[0])])

    # Project actual onto co-pol and cross-pol
    co_pol = np.abs(np.sum(jones_actual * np.conj(jones_desired_norm)))**2
    cross_pol = np.abs(np.sum(jones_actual * np.conj(jones_cross)))**2

    # Compute XPD in dB
    with np.errstate(divide='ignore', invalid='ignore'):
        xpd = 10 * np.log10(co_pol / cross_pol) if cross_pol > 1e-20 else 100.0

    return xpd


def polarization_loss_factor(
    jones_antenna: np.ndarray,
    jones_incident: np.ndarray
) -> np.ndarray:
    """
    Compute polarization loss factor (PLF).

    The PLF is the fraction of incident power that couples to the
    antenna due to polarization mismatch.

    Parameters
    ----------
    jones_antenna : ndarray
        Antenna polarization (Jones vector)
    jones_incident : ndarray
        Incident wave polarization (Jones vector)

    Returns
    -------
    plf : float or ndarray
        Polarization loss factor (0 to 1)

    Examples
    --------
    Matched polarizations have PLF = 1:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> j_ant = pa.jones_vector(1.0, 0.0)  # H-pol antenna
    >>> j_inc = pa.jones_vector(1.0, 0.0)  # H-pol wave
    >>> plf = pa.polarization_loss_factor(j_ant, j_inc)
    >>> np.isclose(plf, 1.0)
    True

    Orthogonal polarizations have PLF = 0:

    >>> j_ant = pa.jones_vector(1.0, 0.0)  # H-pol antenna
    >>> j_inc = pa.jones_vector(0.0, 1.0)  # V-pol wave
    >>> plf = pa.polarization_loss_factor(j_ant, j_inc)
    >>> np.isclose(plf, 0.0)
    True

    Circular antenna receiving linear has PLF = 0.5:

    >>> j_circ = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
    >>> j_lin = pa.jones_vector(1.0, 0.0)
    >>> plf = pa.polarization_loss_factor(j_circ, j_lin)
    >>> np.isclose(plf, 0.5)
    True
    """
    jones_antenna = np.asarray(jones_antenna, dtype=complex)
    jones_incident = np.asarray(jones_incident, dtype=complex)

    # Normalize both vectors
    norm_ant = np.sqrt(np.abs(jones_antenna[0])**2 + np.abs(jones_antenna[1])**2)
    norm_inc = np.sqrt(np.abs(jones_incident[0])**2 + np.abs(jones_incident[1])**2)

    if norm_ant < 1e-15 or norm_inc < 1e-15:
        return np.array(0.0)

    jones_ant_norm = jones_antenna / norm_ant
    jones_inc_norm = jones_incident / norm_inc

    # PLF = |antenna . incident*|^2
    plf = np.abs(np.sum(jones_ant_norm * np.conj(jones_inc_norm)))**2

    return plf


def ludwig3_decomposition(
    theta: np.ndarray,
    phi: np.ndarray,
    E_theta: np.ndarray,
    E_phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose field into Ludwig-3 co-polar and cross-polar components.

    Ludwig-3 is the most common definition for co/cross-pol in antenna
    measurements. It's based on aligning the reference polarization
    with the principal planes.

    Parameters
    ----------
    theta : ndarray
        Theta angles in radians
    phi : ndarray
        Phi angles in radians
    E_theta : ndarray
        Theta component of electric field (complex)
    E_phi : ndarray
        Phi component of electric field (complex)

    Returns
    -------
    E_co : ndarray
        Co-polar component (Ludwig-3)
    E_cross : ndarray
        Cross-polar component (Ludwig-3)

    Notes
    -----
    Ludwig-3 definition (for reference polarization in phi=0 plane):
        E_co = E_theta * cos(phi) - E_phi * sin(phi)
        E_cross = E_theta * sin(phi) + E_phi * cos(phi)

    Examples
    --------
    At phi = 0, E_theta is co-pol:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> theta = np.array([np.pi/4])
    >>> phi = np.array([0.0])
    >>> E_theta = np.array([1.0 + 0j])
    >>> E_phi = np.array([0.0 + 0j])
    >>> E_co, E_cross = pa.ludwig3_decomposition(theta, phi, E_theta, E_phi)
    >>> np.isclose(np.abs(E_co[0]), 1.0)
    True
    >>> np.isclose(np.abs(E_cross[0]), 0.0)
    True
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    E_theta = np.asarray(E_theta, dtype=complex)
    E_phi = np.asarray(E_phi, dtype=complex)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    E_co = E_theta * cos_phi - E_phi * sin_phi
    E_cross = E_theta * sin_phi + E_phi * cos_phi

    return E_co, E_cross


def co_pol_pattern(
    theta: np.ndarray,
    phi: np.ndarray,
    E_theta: np.ndarray,
    E_phi: np.ndarray,
    reference_pol: str = 'ludwig3'
) -> np.ndarray:
    """
    Extract co-polar component of the radiation pattern.

    Parameters
    ----------
    theta : ndarray
        Theta angles in radians
    phi : ndarray
        Phi angles in radians
    E_theta : ndarray
        Theta component of electric field
    E_phi : ndarray
        Phi component of electric field
    reference_pol : str
        'ludwig3' - Ludwig-3 definition (default)
        'theta' - E_theta is co-pol
        'phi' - E_phi is co-pol

    Returns
    -------
    E_co : ndarray
        Co-polar component (complex)

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> theta = np.linspace(0, np.pi/2, 91)
    >>> phi = np.zeros_like(theta)
    >>> E_theta = np.cos(theta)  # Simple pattern
    >>> E_phi = np.zeros_like(theta)
    >>> E_co = pa.co_pol_pattern(theta, phi, E_theta, E_phi)
    >>> E_co.shape
    (91,)
    """
    if reference_pol == 'ludwig3':
        E_co, _ = ludwig3_decomposition(theta, phi, E_theta, E_phi)
        return E_co
    elif reference_pol == 'theta':
        return np.asarray(E_theta, dtype=complex)
    elif reference_pol == 'phi':
        return np.asarray(E_phi, dtype=complex)
    else:
        raise ValueError(f"Unknown reference polarization: {reference_pol}")


def cross_pol_pattern(
    theta: np.ndarray,
    phi: np.ndarray,
    E_theta: np.ndarray,
    E_phi: np.ndarray,
    reference_pol: str = 'ludwig3'
) -> np.ndarray:
    """
    Extract cross-polar component of the radiation pattern.

    Parameters
    ----------
    theta : ndarray
        Theta angles in radians
    phi : ndarray
        Phi angles in radians
    E_theta : ndarray
        Theta component of electric field
    E_phi : ndarray
        Phi component of electric field
    reference_pol : str
        'ludwig3' - Ludwig-3 definition (default)
        'theta' - E_phi is cross-pol
        'phi' - E_theta is cross-pol

    Returns
    -------
    E_cross : ndarray
        Cross-polar component (complex)

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> theta = np.linspace(0, np.pi/2, 91)
    >>> phi = np.zeros_like(theta)
    >>> E_theta = np.cos(theta)
    >>> E_phi = 0.1 * np.sin(theta)  # Small cross-pol
    >>> E_cross = pa.cross_pol_pattern(theta, phi, E_theta, E_phi)
    >>> E_cross.shape
    (91,)
    """
    if reference_pol == 'ludwig3':
        _, E_cross = ludwig3_decomposition(theta, phi, E_theta, E_phi)
        return E_cross
    elif reference_pol == 'theta':
        return np.asarray(E_phi, dtype=complex)
    elif reference_pol == 'phi':
        return np.asarray(E_theta, dtype=complex)
    else:
        raise ValueError(f"Unknown reference polarization: {reference_pol}")
