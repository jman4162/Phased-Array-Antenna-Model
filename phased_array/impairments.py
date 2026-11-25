"""
Realistic impairment models for phased arrays.

Includes mutual coupling, phase quantization, element failures, and scan blindness.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .geometry import ArrayGeometry

# ============== Mutual Coupling ==============

def mutual_coupling_matrix_theoretical(
    geometry: ArrayGeometry,
    k: float,
    coupling_model: str = 'sinc',
    coupling_coeff: float = 0.3
) -> np.ndarray:
    """
    Compute theoretical mutual coupling matrix.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber (2*pi/wavelength)
    coupling_model : str
        'sinc' - heuristic ``coupling_coeff * sin(kr)/kr * exp(-jkr)``
        decay. It is isotropic and is not an electromagnetic dipole model
        (it is exactly zero at r = lambda/2).
        'exponential' - heuristic exponential decay model
    coupling_coeff : float
        Coupling coefficient (typical: 0.1-0.5)

    Returns
    -------
    C : ndarray
        N x N complex mutual coupling matrix
    """
    if coupling_model not in ('sinc', 'exponential'):
        raise ValueError(
            "coupling_model must be 'sinc' or 'exponential', got "
            f"{coupling_model!r}"
        )

    n = geometry.n_elements
    C = np.eye(n, dtype=complex)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Distance between elements
                dx = geometry.x[i] - geometry.x[j]
                dy = geometry.y[i] - geometry.y[j]
                dz = 0
                if geometry.z is not None:
                    dz = geometry.z[i] - geometry.z[j]

                r = np.sqrt(dx**2 + dy**2 + dz**2)
                kr = k * r

                if coupling_model == 'sinc':
                    # Sinc model: approximates dipole coupling
                    if kr > 1e-10:
                        coupling = coupling_coeff * np.sin(kr) / kr
                    else:
                        coupling = coupling_coeff
                else:
                    # Exponential decay
                    coupling = coupling_coeff * np.exp(-kr / 2)

                # Add phase based on distance
                C[i, j] = coupling * np.exp(-1j * kr)

    return C


def mutual_coupling_matrix_measured(
    s_parameters: np.ndarray,
    formulation: str = 'impedance'
) -> np.ndarray:
    """
    Convert measured S-parameters to coupling matrix.

    The coupling matrix relates actual element currents to excitation
    voltages: I = C^(-1) @ V

    Parameters
    ----------
    s_parameters : ndarray
        N x N S-parameter matrix (complex)
    formulation : str
        'impedance' (default): C = (I + S)(I - S)^(-1), the normalized
        impedance matrix. 'voltage': C = I + S, the total-voltage
        coupling used before v1.3.2.

    Returns
    -------
    C : ndarray
        Mutual coupling matrix

    Notes
    -----
    Before v1.3.2 this function returned I + S while documenting the
    impedance formulation; pass ``formulation='voltage'`` to reproduce
    the old behavior.
    """
    n = s_parameters.shape[0]
    identity = np.eye(n, dtype=complex)

    if formulation == 'impedance':
        try:
            C = np.linalg.solve((identity - s_parameters).T, (identity + s_parameters).T).T
        except np.linalg.LinAlgError:
            C = (identity + s_parameters) @ np.linalg.pinv(identity - s_parameters)
    elif formulation == 'voltage':
        C = identity + s_parameters
    else:
        raise ValueError(
            f"formulation must be 'impedance' or 'voltage', got {formulation!r}"
        )
    return C


def apply_mutual_coupling(
    weights: np.ndarray,
    C: np.ndarray,
    mode: str = 'transmit'
) -> np.ndarray:
    """
    Apply mutual coupling to element weights.

    Parameters
    ----------
    weights : ndarray
        Ideal element weights (N,)
    C : ndarray
        Mutual coupling matrix (N x N)
    mode : str
        'transmit' - coupling affects radiated field
        'receive' - coupling affects received signal
        'compensate' - pre-distort to compensate coupling

    Returns
    -------
    effective_weights : ndarray
        Weights after coupling effects
    """
    if mode == 'transmit':
        # Actual element excitations given desired weights
        # Radiated field is C @ weights
        return C @ weights
    elif mode == 'receive':
        # Coupled signal at element ports
        return C.T @ weights
    elif mode == 'compensate':
        # Pre-distort to achieve desired radiation
        # Want C @ w_comp = w_desired, so w_comp = C^(-1) @ w_desired
        try:
            return np.linalg.solve(C, weights)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(C, weights, rcond=None)[0]
    else:
        raise ValueError(f"Unknown coupling mode: {mode}")


def active_element_pattern(
    theta: np.ndarray,
    phi: np.ndarray,
    geometry: ArrayGeometry,
    element_idx: int,
    C: np.ndarray,
    k: float,
    isolated_element_pattern: Optional[callable] = None
) -> np.ndarray:
    """
    Compute active element pattern including mutual coupling.

    The active element pattern is the pattern of a single element
    when all other elements are terminated in matched loads.

    Parameters
    ----------
    theta : ndarray
        Observation theta angles
    phi : ndarray
        Observation phi angles
    geometry : ArrayGeometry
        Array geometry
    element_idx : int
        Index of the element to compute pattern for
    C : ndarray
        Mutual coupling matrix
    k : float
        Wavenumber
    isolated_element_pattern : callable, optional
        Pattern function for isolated element

    Returns
    -------
    pattern : ndarray
        Active element pattern (complex)
    """
    from .core import array_factor_vectorized

    n = geometry.n_elements

    # Excite only the element of interest
    excitation = np.zeros(n, dtype=complex)
    excitation[element_idx] = 1.0

    # Apply coupling
    effective_excitation = C @ excitation

    # Compute pattern (AF with coupled excitations)
    AF = array_factor_vectorized(
        theta, phi,
        geometry.x, geometry.y,
        effective_excitation, k,
        geometry.z
    )

    # Apply isolated element pattern if provided
    if isolated_element_pattern is not None:
        EP = isolated_element_pattern(theta, phi)
        return AF * EP

    return AF


# ============== Phase Quantization ==============

def quantize_phase(
    weights: np.ndarray,
    n_bits: int
) -> np.ndarray:
    """
    Quantize phase shifter settings to discrete levels.

    Parameters
    ----------
    weights : ndarray
        Complex weights (phase will be quantized)
    n_bits : int
        Number of bits for phase quantization (e.g., 3 bits = 8 levels)

    Returns
    -------
    quantized_weights : ndarray
        Weights with quantized phases

    Examples
    --------
    Quantize to 3-bit phase shifters (8 levels, 45 deg steps):

    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=15, phi0_deg=0)
    >>> weights_q = pa.quantize_phase(weights, n_bits=3)
    >>> weights_q.shape
    (64,)

    Check phase quantization levels:

    >>> phases_deg = np.rad2deg(np.angle(weights_q))
    >>> np.unique(np.round(phases_deg / 45) * 45).size <= 8
    True

    Compare effect of different bit depths:

    >>> rms_3bit = pa.quantization_rms_error(3)  # ~13 degrees
    >>> rms_6bit = pa.quantization_rms_error(6)  # ~1.6 degrees
    >>> rms_3bit > rms_6bit
    True
    """
    n_levels = 2 ** n_bits
    phase_step = 2 * np.pi / n_levels

    # Extract amplitude and phase
    amplitude = np.abs(weights)
    phase = np.angle(weights)

    # Quantize phase to nearest level
    quantized_phase = np.round(phase / phase_step) * phase_step

    return amplitude * np.exp(1j * quantized_phase)


def quantization_rms_error(n_bits: int) -> float:
    """
    Compute theoretical RMS phase error for quantization.

    Parameters
    ----------
    n_bits : int
        Number of phase quantization bits

    Returns
    -------
    rms_error_deg : float
        RMS phase error in degrees
    """
    # Uniform quantization: RMS error = step / sqrt(12)
    n_levels = 2 ** n_bits
    step_deg = 360.0 / n_levels
    return step_deg / np.sqrt(12)


def quantization_sidelobe_increase(n_bits: int) -> float:
    """
    Deprecated: returns ``20*log10(2*sigma)``, which is not a sidelobe level.

    .. deprecated:: 1.6.0
        The returned value does not depend on the array size or taper and
        disagrees with both textbook models by 10-25 dB. Use
        :func:`quantization_sidelobe_level` for the average sidelobe floor
        from uncorrelated quantization error, or :func:`quantization_lobes`
        for the quantization lobes that form at small scan angles.

    Parameters
    ----------
    n_bits : int
        Number of phase quantization bits

    Returns
    -------
    value_dB : float
        ``20*log10(2*sigma)`` with sigma the RMS phase error in radians
    """
    warnings.warn(
        "quantization_sidelobe_increase is deprecated and does not estimate a "
        "sidelobe level; use quantization_sidelobe_level (uncorrelated error) "
        "or quantization_lobes (correlated error at small scan angles).",
        DeprecationWarning,
        stacklevel=2,
    )
    rms_error_rad = np.deg2rad(quantization_rms_error(n_bits))
    return 20 * np.log10(2 * rms_error_rad + 1e-10)


def quantization_sidelobe_level(
    n_bits: int,
    n_elements: int,
    taper_efficiency: float = 1.0
) -> float:
    """
    Average sidelobe level from uncorrelated phase quantization error.

    When the quantization error varies randomly from element to element
    (large scan angles, or phase dithering), it acts like a random phase
    error with standard deviation ``sigma = 2*pi / (2**n_bits * sqrt(12))``.
    The average sidelobe level relative to the main-beam peak is
    (Haupt, *Antenna Arrays*, eqs. 2.119 and 2.129)

        SLL = (1 - exp(-sigma**2)) / (N * eta * exp(-sigma**2))

    Parameters
    ----------
    n_bits : int
        Number of phase-shifter bits
    n_elements : int
        Number of elements N
    taper_efficiency : float
        Aperture taper efficiency eta (1 for uniform), see
        :func:`compute_taper_efficiency`

    Returns
    -------
    sll_dB : float
        Average sidelobe level in dB relative to the beam peak

    Examples
    --------
    >>> import phased_array as pa
    >>> round(pa.quantization_sidelobe_level(4, 100), 1)
    -38.9
    """
    sigma = np.deg2rad(quantization_rms_error(n_bits))
    sll = np.expm1(sigma**2) / (n_elements * taper_efficiency)
    return float(10 * np.log10(sll))


def quantization_lobes(
    n_bits: int,
    u_scan: float,
    max_order: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positions and levels of phase-quantization lobes at small scan.

    At small scan angles neighboring elements share the same quantized
    phase, so the phase error is a periodic sawtooth across the aperture
    and produces quantization lobes (Haupt, *Antenna Arrays*, eqs.
    2.130-2.133). Lobe ``m`` lies at

        u_m = u_scan * (1 + m * 2**n_bits)

    with a field level of ``1 / |1 + m * 2**n_bits|`` relative to the main
    beam, from the Fourier series of the sawtooth error. The model assumes
    a large array and round-to-nearest quantization; lobes with
    ``|u_m| > 1`` fall in invisible space.

    Parameters
    ----------
    n_bits : int
        Number of phase-shifter bits
    u_scan : float
        Scan direction cosine, ``sin(theta0)`` in the scan plane
    max_order : int
        Highest lobe order; orders ``-max_order..-1, 1..max_order`` are
        returned

    Returns
    -------
    u_positions : ndarray
        Lobe positions in direction-cosine space, shape (2*max_order,)
    levels_dB : ndarray
        Lobe peak levels in dB relative to the main beam

    Examples
    --------
    >>> import phased_array as pa
    >>> u, lvl = pa.quantization_lobes(3, 0.05)
    >>> [round(float(x), 2) for x in u], [round(float(x), 1) for x in lvl]
    ([-0.35, 0.45], [-16.9, -19.1])
    """
    orders = np.r_[np.arange(-max_order, 0), np.arange(1, max_order + 1)]
    factor = 1 + orders * 2.0**n_bits
    u_positions = u_scan * factor
    levels_dB = -20 * np.log10(np.abs(factor))
    return u_positions, levels_dB


def analyze_quantization_effect(
    weights: np.ndarray,
    geometry: ArrayGeometry,
    k: float,
    n_bits: int,
    theta_range: Tuple[float, float] = (0, np.pi/2),
    n_points: int = 361
) -> Dict[str, np.ndarray]:
    """
    Analyze effect of phase quantization on the pattern.

    Parameters
    ----------
    weights : ndarray
        Ideal complex weights
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    n_bits : int
        Quantization bits
    theta_range : tuple
        Range for pattern computation
    n_points : int
        Number of angle points

    Returns
    -------
    results : dict
        'theta_deg': angle array
        'pattern_ideal_dB': ideal pattern
        'pattern_quantized_dB': quantized pattern
        'difference_dB': pattern difference
    """
    from .core import array_factor_vectorized
    from .utils import linear_to_db

    # Quantize weights
    weights_q = quantize_phase(weights, n_bits)

    # Compute patterns
    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    phi = np.zeros_like(theta)

    theta_grid = theta.reshape(-1, 1)
    phi_grid = phi.reshape(-1, 1)

    AF_ideal = array_factor_vectorized(
        theta_grid, phi_grid,
        geometry.x, geometry.y, weights, k, geometry.z
    ).ravel()

    AF_quant = array_factor_vectorized(
        theta_grid, phi_grid,
        geometry.x, geometry.y, weights_q, k, geometry.z
    ).ravel()

    # Convert to dB
    pattern_ideal = linear_to_db(np.abs(AF_ideal)**2)
    pattern_quant = linear_to_db(np.abs(AF_quant)**2)

    # Normalize
    pattern_ideal -= np.max(pattern_ideal)
    pattern_quant -= np.max(pattern_quant)

    return {
        'theta_deg': np.rad2deg(theta),
        'pattern_ideal_dB': pattern_ideal,
        'pattern_quantized_dB': pattern_quant,
        'difference_dB': pattern_quant - pattern_ideal,
        'rms_error_deg': quantization_rms_error(n_bits)
    }


# ============== Element Failures ==============

def simulate_element_failures(
    weights: np.ndarray,
    failure_rate: float,
    mode: str = 'off',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate random element failures.

    Parameters
    ----------
    weights : ndarray
        Nominal element weights
    failure_rate : float
        Probability of failure per element (0 to 1)
    mode : str
        'off' - failed elements have zero output
        'stuck' - failed elements stuck at nominal magnitude, random phase
        'full' - failed elements at full power, random phase
    seed : int, optional
        Random seed

    Returns
    -------
    degraded_weights : ndarray
        Weights with failures applied
    failure_mask : ndarray
        Boolean array, True for failed elements

    Examples
    --------
    Simulate 5% element failure rate:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=0, phi0_deg=0)
    >>> degraded, mask = pa.simulate_element_failures(
    ...     weights, failure_rate=0.05, mode='off', seed=42
    ... )
    >>> n_failed = np.sum(mask)
    >>> degraded.shape
    (256,)

    Compare failure modes:

    >>> # 'off' mode: failed elements produce no output
    >>> w_off, m_off = pa.simulate_element_failures(weights, 0.1, mode='off', seed=1)
    >>> np.all(w_off[m_off] == 0)
    True

    Analyze graceful degradation:

    >>> results = pa.analyze_graceful_degradation(
    ...     weights, geom, k, failure_rates=[0.0, 0.05, 0.1], n_trials=5
    ... )
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(weights)
    failure_mask = np.random.random(n) < failure_rate
    degraded_weights = weights.copy()

    if mode == 'off':
        degraded_weights[failure_mask] = 0
    elif mode == 'stuck':
        # Random phase, same magnitude
        random_phase = np.random.uniform(0, 2*np.pi, np.sum(failure_mask))
        degraded_weights[failure_mask] = (
            np.abs(degraded_weights[failure_mask]) * np.exp(1j * random_phase)
        )
    elif mode == 'full':
        # Full power, random phase
        random_phase = np.random.uniform(0, 2*np.pi, np.sum(failure_mask))
        degraded_weights[failure_mask] = np.exp(1j * random_phase)
    else:
        raise ValueError(f"Unknown failure mode: {mode}")

    return degraded_weights, failure_mask


def analyze_graceful_degradation(
    weights: np.ndarray,
    geometry: ArrayGeometry,
    k: float,
    failure_rates: List[float],
    n_trials: int = 100,
    mode: str = 'off'
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo analysis of graceful degradation vs failure rate.

    Parameters
    ----------
    weights : ndarray
        Nominal weights
    geometry : ArrayGeometry
        Array geometry
    k : float
        Wavenumber
    failure_rates : list
        Failure rates to test
    n_trials : int
        Number of Monte Carlo trials per rate
    mode : str
        Failure mode

    Returns
    -------
    results : dict
        'failure_rates': input rates
        'gain_loss_mean_dB': mean drop in beam peak power (positive = loss)
        'gain_loss_std_dB': std of gain loss
        'sidelobe_increase_mean_dB': mean change in peak sidelobe level
        'sidelobe_ref_dB': peak sidelobe level of the nominal weights

    Notes
    -----
    Patterns are evaluated on a cut through the beam peak (the
    ``phi`` plane of the beam, theta from -90 to 90 degrees), and the
    sidelobe level is measured outside the first nulls with
    :func:`~phased_array.core.compute_peak_sidelobe_level`. Trials use
    NumPy's global random state; call ``np.random.seed`` first for
    repeatable results.
    """
    from .core import array_factor_vectorized, compute_peak_sidelobe_level
    from .utils import linear_to_db

    x, y, z = geometry.x, geometry.y, geometry.z

    # Beam direction of the nominal weights, from a coarse hemisphere search
    th_c, ph_c = np.meshgrid(np.linspace(0, np.pi / 2, 91),
                             np.linspace(0, 2 * np.pi, 180, endpoint=False),
                             indexing='ij')
    af_c = np.abs(array_factor_vectorized(th_c, ph_c, x, y, weights, k, z))
    i_pk = np.unravel_index(np.argmax(af_c), af_c.shape)
    phi_beam = ph_c[i_pk]

    # Cut through the beam: the phi_beam plane, theta from -90 to 90 deg
    angles_deg = np.linspace(-90, 90, 721)
    t = np.deg2rad(angles_deg)
    theta_cut = np.abs(t)
    phi_cut = np.where(t >= 0, phi_beam, phi_beam + np.pi)

    def cut_power_dB(w):
        af = array_factor_vectorized(theta_cut, phi_cut, x, y, w, k, z)
        return linear_to_db(np.abs(af)**2, min_db=-300.0)

    ref_dB = cut_power_dB(weights)
    peak_ref_dB = float(np.max(ref_dB))
    sidelobe_ref = compute_peak_sidelobe_level(angles_deg, ref_dB)

    gain_loss_mean = []
    gain_loss_std = []
    sidelobe_increase_mean = []

    for rate in failure_rates:
        gain_losses = []
        sidelobe_increases = []

        for trial in range(n_trials):
            degraded, _ = simulate_element_failures(
                weights, rate, mode, seed=None
            )
            pattern_dB = cut_power_dB(degraded)

            # Gain loss: drop in peak power on the beam cut (positive = loss)
            gain_losses.append(peak_ref_dB - float(np.max(pattern_dB)))

            # Peak sidelobe level relative to the degraded beam peak
            sidelobe_degraded = compute_peak_sidelobe_level(angles_deg, pattern_dB)
            sidelobe_increases.append(sidelobe_degraded - sidelobe_ref)

        gain_loss_mean.append(np.mean(gain_losses))
        gain_loss_std.append(np.std(gain_losses))
        sidelobe_increase_mean.append(np.mean(sidelobe_increases))

    return {
        'failure_rates': np.array(failure_rates),
        'gain_loss_mean_dB': np.array(gain_loss_mean),
        'gain_loss_std_dB': np.array(gain_loss_std),
        'sidelobe_increase_mean_dB': np.array(sidelobe_increase_mean),
        'sidelobe_ref_dB': sidelobe_ref
    }


# ============== Scan Blindness ==============

def _tm0_surface_wave_beta(substrate_er: float, substrate_h: float) -> float:
    """
    Normalized TM0 surface-wave propagation constant of a grounded slab.

    Solves the TM dispersion relation of a grounded dielectric slab
    (normalized to k0, h in wavelengths)

        sqrt(er - b**2) * tan(k0 h sqrt(er - b**2)) = er * sqrt(b**2 - 1)

    for the TM0 root ``b = beta/k0`` in ``(1, sqrt(er))``. TM0 has no
    cutoff, so the root exists for any ``er > 1`` and ``h > 0``.
    """
    from scipy.optimize import brentq

    if substrate_er <= 1:
        raise ValueError(f"substrate_er must exceed 1, got {substrate_er}")
    if substrate_h <= 0:
        raise ValueError(f"substrate_h must be positive, got {substrate_h}")

    k0h = 2 * np.pi * substrate_h
    n_max = np.sqrt(substrate_er)

    def residual(b):
        kd = np.sqrt(substrate_er - b * b)
        return kd * np.tan(k0h * kd) - substrate_er * np.sqrt(b * b - 1)

    # Keep k0 h sqrt(er - b^2) below pi/2 so the bracket excludes the tan
    # pole; the TM0 root lies in that branch.
    lower = 1.0
    if k0h * np.sqrt(substrate_er - 1) >= np.pi / 2:
        lower = np.sqrt(substrate_er - (np.pi / (2 * k0h))**2)
    span = n_max - lower
    eps = 1e-13 * max(span, 1e-300)
    if residual(lower + eps) <= 0:
        # Extremely thin slab: root is within rounding of b = 1
        return 1.0 + 0.5 * (k0h * (substrate_er - 1) / substrate_er)**2
    return float(brentq(residual, lower + eps, n_max - eps, xtol=1e-15, rtol=1e-15))


def surface_wave_scan_angle(
    dx: float,
    dy: float,
    substrate_er: float = 4.0,
    substrate_h: float = 0.1
) -> Tuple[float, float]:
    """
    Estimate scan blindness angles due to surface wave excitation.

    Blindness occurs where a Floquet (grating-lobe) mode of the scanned
    array matches the TM0 surface wave of the grounded substrate:
    ``|sin(theta) - lambda/d| = beta_sw / k0`` for scan in the plane of
    spacing ``d``, so ``sin(theta_blind) = |lambda/d - beta_sw/k0|``.
    ``beta_sw`` comes from the exact grounded-slab TM0 dispersion relation.

    Parameters
    ----------
    dx : float
        Element spacing in x (wavelengths)
    dy : float
        Element spacing in y (wavelengths)
    substrate_er : float
        Substrate relative permittivity (> 1)
    substrate_h : float
        Substrate height (wavelengths)

    Returns
    -------
    theta_blind_E : float
        Blind scan angle in the x (E-) plane in degrees; 90.0 when no
        blind angle falls in visible space
    theta_blind_H : float
        Blind scan angle in the y (H-) plane in degrees; 90.0 when none

    Notes
    -----
    This is the geometric matching condition only. How deep the blindness
    is depends on element type and polarization: printed dipoles and
    patches couple strongly to TM0 in the E-plane. The TE1 surface wave,
    present on thicker substrates, is not modeled.

    For er = 12.8, h = 0.06 lambda and d = 0.5 lambda this gives about
    45.6 degrees, consistent with the E-plane blindness Pozar and
    Schaubert report for printed dipoles (IEEE Trans. AP-32, 1984).

    Examples
    --------
    >>> import phased_array as pa
    >>> theta_e, theta_h = pa.surface_wave_scan_angle(0.5, 0.5, 12.8, 0.06)
    >>> round(theta_e, 1)
    45.6
    """
    beta = _tm0_surface_wave_beta(substrate_er, substrate_h)

    def blind_angle(d):
        s_blind = abs(1.0 / d - beta)
        if s_blind > 1.0:
            return 90.0
        return float(np.rad2deg(np.arcsin(s_blind)))

    return blind_angle(dx), blind_angle(dy)


def scan_blindness_model(
    theta: np.ndarray,
    phi: np.ndarray,
    theta_blind: float,
    phi_blind: Optional[float] = None,
    null_width_deg: float = 5.0,
    null_depth_dB: float = -30.0
) -> np.ndarray:
    """
    Model scan blindness as a Gaussian null at the blind angle.

    Parameters
    ----------
    theta : ndarray
        Observation theta angles (radians)
    phi : ndarray
        Observation phi angles (radians)
    theta_blind : float
        Blind angle theta (degrees)
    phi_blind : float, optional
        Blind angle phi (degrees). If None, blindness is phi-independent
    null_width_deg : float
        Width of the null (degrees, 1-sigma)
    null_depth_dB : float
        Depth of null in dB (negative)

    Returns
    -------
    factor : ndarray
        Multiplicative factor (0 to 1)
    """
    theta_deg = np.rad2deg(theta)

    if phi_blind is None:
        # Phi-independent blindness (conical null)
        angular_distance = np.abs(theta_deg - theta_blind)
    else:
        # Point null at specific direction
        phi_deg = np.rad2deg(phi)
        phi_blind_rad = np.deg2rad(phi_blind)
        theta_blind_rad = np.deg2rad(theta_blind)

        # Angular distance on sphere
        cos_dist = (np.cos(theta) * np.cos(theta_blind_rad) +
                   np.sin(theta) * np.sin(theta_blind_rad) *
                   np.cos(phi - phi_blind_rad))
        angular_distance = np.rad2deg(np.arccos(np.clip(cos_dist, -1, 1)))

    # Gaussian null
    null_depth_linear = 10 ** (null_depth_dB / 10)
    factor = 1 - (1 - null_depth_linear) * np.exp(
        -0.5 * (angular_distance / null_width_deg) ** 2
    )

    return factor


def apply_scan_blindness(
    pattern: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    theta_blind_list: List[float],
    phi_blind_list: Optional[List[float]] = None,
    null_width_deg: float = 5.0,
    null_depth_dB: float = -30.0
) -> np.ndarray:
    """
    Apply scan blindness model to a computed pattern.

    Parameters
    ----------
    pattern : ndarray
        Complex or magnitude pattern
    theta : ndarray
        Theta angles (radians)
    phi : ndarray
        Phi angles (radians)
    theta_blind_list : list
        List of blind angles (degrees)
    phi_blind_list : list, optional
        List of blind phi angles
    null_width_deg : float
        Null width
    null_depth_dB : float
        Null depth

    Returns
    -------
    modified_pattern : ndarray
        Pattern with scan blindness applied
    """
    modified = pattern.copy()

    if phi_blind_list is None:
        phi_blind_list = [None] * len(theta_blind_list)

    for theta_blind, phi_blind in zip(theta_blind_list, phi_blind_list):
        factor = scan_blindness_model(
            theta, phi, theta_blind, phi_blind,
            null_width_deg, null_depth_dB
        )
        modified = modified * np.sqrt(factor)  # sqrt for voltage pattern

    return modified


def compute_scan_loss(
    geometry: ArrayGeometry,
    weights: np.ndarray,
    k: float,
    theta_scan_deg: float,
    phi_scan_deg: float,
    element_pattern_func: Optional[callable] = None
) -> float:
    """
    Compute scan loss (reduction in peak gain at scan angle).

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    weights : ndarray
        Element weights
    k : float
        Wavenumber
    theta_scan_deg : float
        Scan angle theta
    phi_scan_deg : float
        Scan angle phi
    element_pattern_func : callable, optional
        Element pattern function

    Returns
    -------
    scan_loss_dB : float
        Reduction in gain relative to broadside (negative or zero)

    Notes
    -----
    The reference is the same amplitude distribution, ``|weights|``,
    steered to broadside, so an amplitude taper is not counted as scan
    loss. The ratio is of the pattern power in the two directions; with
    ``element_pattern_func=None`` a planar array shows no loss, because
    the array-factor peak does not change with scan.
    """
    from .core import steering_vector, total_pattern

    # Reference: the same amplitude distribution steered to broadside
    theta_bs = np.array([[0.0]])
    phi_bs = np.array([[0.0]])
    weights_bs = np.abs(weights) * steering_vector(
        k, geometry.x, geometry.y, 0.0, 0.0, geometry.z
    )

    pattern_bs = total_pattern(
        theta_bs, phi_bs, geometry.x, geometry.y,
        weights_bs, k, element_pattern_func, geometry.z
    )
    gain_bs = np.abs(pattern_bs.item()) ** 2

    # Compute gain at scan angle
    theta_scan = np.array([[np.deg2rad(theta_scan_deg)]])
    phi_scan = np.array([[np.deg2rad(phi_scan_deg)]])

    pattern_scan = total_pattern(
        theta_scan, phi_scan, geometry.x, geometry.y,
        weights, k, element_pattern_func, geometry.z
    )
    gain_scan = np.abs(pattern_scan.item()) ** 2

    if gain_bs > 0 and gain_scan > 0:
        return 10 * np.log10(gain_scan / gain_bs)
    else:
        return -100.0


# ============== Active Impedance and VSWR ==============

def _coupling_to_s(C: np.ndarray, formulation: str) -> np.ndarray:
    """Scattering matrix from a coupling matrix in the given formulation."""
    C = np.asarray(C, dtype=complex)
    identity = np.eye(C.shape[0], dtype=complex)
    if formulation == 'impedance':
        # C = (I + S)(I - S)^-1  =>  S = (C + I)^-1 (C - I)
        return np.linalg.solve(C + identity, C - identity)
    if formulation == 'voltage':
        return C - identity
    raise ValueError(
        f"formulation must be 'impedance' or 'voltage', got {formulation!r}"
    )


def _active_gammas(
    C: np.ndarray,
    weights: np.ndarray,
    formulation: str
) -> np.ndarray:
    """Active reflection coefficients of all elements, (S w)_n / w_n."""
    S = _coupling_to_s(C, formulation)
    weights = np.asarray(weights, dtype=complex)
    reflected = S @ weights
    gamma = np.zeros_like(reflected)
    nonzero = weights != 0
    gamma[nonzero] = reflected[nonzero] / weights[nonzero]
    return gamma


def active_reflection_coefficient(
    C: np.ndarray,
    weights: np.ndarray,
    element_idx: int,
    formulation: str = 'impedance'
) -> complex:
    """
    Compute active reflection coefficient for an element.

    The active reflection coefficient accounts for mutual coupling
    from all other elements when the array is excited with given weights.

    Parameters
    ----------
    C : ndarray
        Mutual coupling matrix (N x N)
    weights : ndarray
        Complex excitation weights (N,), the incident waves at the ports
    element_idx : int
        Index of the element to compute reflection for
    formulation : str
        How ``C`` relates to the scattering matrix S. It must match how C
        was built:

        - ``'impedance'`` (default): ``C = (I + S)(I - S)^-1``, the
          normalized impedance matrix returned by
          :func:`mutual_coupling_matrix_measured` by default. The
          unit-diagonal heuristic of
          :func:`mutual_coupling_matrix_theoretical` is also treated this
          way.
        - ``'voltage'``: ``C = I + S`` (the pre-1.6 behavior of this
          function, and ``mutual_coupling_matrix_measured(...,
          formulation='voltage')``).

    Returns
    -------
    gamma : complex
        Active reflection coefficient

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)
    >>> weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)
    >>> gamma = pa.active_reflection_coefficient(C, weights, element_idx=0)
    >>> np.abs(gamma) < 1  # Should be reasonable
    True

    Notes
    -----
    The active reflection coefficient is

        gamma_n = sum_m(S_nm * w_m) / w_n

    with S obtained from C according to ``formulation``. It equals
    ``(z_n - 1) / (z_n + 1)`` for the normalized active impedance ``z_n``.
    It differs from the isolated reflection coefficient because power
    couples from neighboring elements. Elements with zero weight return 0.
    """
    weights = np.asarray(weights, dtype=complex)
    S = _coupling_to_s(C, formulation)
    if weights[element_idx] == 0:
        return complex(0, 0)
    return complex(S[element_idx, :] @ weights / weights[element_idx])


def active_impedance(
    C: np.ndarray,
    weights: np.ndarray,
    element_idx: int,
    Z0: float = 50.0,
    formulation: str = 'impedance'
) -> complex:
    """
    Compute active impedance of an element.

    Parameters
    ----------
    C : ndarray
        Mutual coupling matrix (N x N)
    weights : ndarray
        Complex excitation weights (N,)
    element_idx : int
        Index of the element
    Z0 : float
        Reference impedance in ohms (default 50)
    formulation : str
        Form of ``C``: ``'impedance'`` (default) or ``'voltage'``; see
        :func:`active_reflection_coefficient`

    Returns
    -------
    Z_active : complex
        Active impedance in ohms

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)
    >>> weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)
    >>> Z = pa.active_impedance(C, weights, element_idx=0)
    >>> np.real(Z) > 0  # Should have positive resistance
    True

    Notes
    -----
    Active impedance is computed from the active reflection coefficient:
        Z_active = Z0 * (1 + gamma) / (1 - gamma)

    This is the impedance seen looking into the element port when
    all elements are excited with the given weights.
    """
    gamma = active_reflection_coefficient(C, weights, element_idx, formulation)

    # Avoid division by zero
    if np.abs(1 - gamma) < 1e-10:
        return complex(np.inf, 0)

    Z_active = Z0 * (1 + gamma) / (1 - gamma)

    return Z_active


def vswr_vs_scan(
    geometry: ArrayGeometry,
    C: np.ndarray,
    k: float,
    theta_range: Tuple[float, float] = (0, 60),
    n_angles: int = 61,
    phi_deg: float = 0.0,
    formulation: str = 'impedance'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute VSWR for all elements versus scan angle.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    C : ndarray
        Mutual coupling matrix
    k : float
        Wavenumber
    theta_range : tuple
        (min, max) scan angles in degrees
    n_angles : int
        Number of scan angles to compute
    phi_deg : float
        Phi scan plane in degrees
    formulation : str
        Form of ``C``: ``'impedance'`` (default) or ``'voltage'``; see
        :func:`active_reflection_coefficient`

    Returns
    -------
    theta_deg : ndarray
        Scan angles in degrees (n_angles,)
    vswr_per_element : ndarray
        VSWR for each element at each angle (n_angles x n_elements)
    vswr_max : ndarray
        Maximum VSWR across all elements at each angle (n_angles,)

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)
    >>> theta_deg, vswr_all, vswr_max = pa.vswr_vs_scan(
    ...     geom, C, k, theta_range=(0, 45), n_angles=10
    ... )
    >>> len(vswr_max) == 10
    True
    >>> np.all(vswr_max >= 1.0)  # VSWR always >= 1
    True

    Notes
    -----
    VSWR is computed from reflection coefficient::

        VSWR = (1 + |gamma|) / (1 - |gamma|)

    High VSWR at certain scan angles indicates potential scan blindness
    or poor matching conditions.
    """
    from .core import steering_vector

    theta_deg = np.linspace(theta_range[0], theta_range[1], n_angles)
    n_elements = geometry.n_elements

    vswr_per_element = np.zeros((n_angles, n_elements))

    for i, theta in enumerate(theta_deg):
        # Compute steering weights
        weights = steering_vector(
            k, geometry.x, geometry.y,
            theta, phi_deg, geometry.z
        )

        # VSWR for each element; |gamma| >= 1 maps to inf
        gamma_mag = np.abs(_active_gammas(C, weights, formulation))
        with np.errstate(divide='ignore'):
            vswr_per_element[i] = np.where(
                gamma_mag >= 1.0, np.inf, (1 + gamma_mag) / (1 - gamma_mag)
            )

    vswr_max = np.max(vswr_per_element, axis=1)

    return theta_deg, vswr_per_element, vswr_max


def mismatch_loss(gamma: np.ndarray) -> np.ndarray:
    """
    Compute mismatch loss from reflection coefficient.

    Parameters
    ----------
    gamma : ndarray
        Complex reflection coefficient(s)

    Returns
    -------
    loss_dB : ndarray
        Mismatch loss in dB (negative)

    Examples
    --------
    Perfect match (gamma=0) has zero loss:

    >>> import numpy as np
    >>> import phased_array as pa
    >>> loss = pa.mismatch_loss(0.0)
    >>> np.isclose(loss, 0.0)
    True

    Typical 2:1 VSWR (gamma=0.333):

    >>> loss = pa.mismatch_loss(0.333)
    >>> -0.6 < loss < -0.4  # About 0.5 dB
    True

    Notes
    -----
    Mismatch loss is::

        Loss_dB = 10 * log10(1 - |gamma|^2)

    This represents the power reflected back due to mismatch.
    """
    gamma = np.asarray(gamma)
    gamma_mag_sq = np.abs(gamma)**2

    # Handle |gamma| >= 1 case
    with np.errstate(divide='ignore', invalid='ignore'):
        transmission = 1 - gamma_mag_sq
        loss_dB = np.where(transmission > 0,
                          10 * np.log10(transmission),
                          -100.0)

    return loss_dB


def active_scan_impedance_matrix(
    geometry: ArrayGeometry,
    C: np.ndarray,
    k: float,
    theta_deg: float,
    phi_deg: float,
    Z0: float = 50.0,
    formulation: str = 'impedance'
) -> np.ndarray:
    """
    Compute active impedance for all elements at a given scan angle.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    C : ndarray
        Mutual coupling matrix
    k : float
        Wavenumber
    theta_deg : float
        Scan angle theta in degrees
    phi_deg : float
        Scan angle phi in degrees
    Z0 : float
        Reference impedance
    formulation : str
        Form of ``C``: ``'impedance'`` (default) or ``'voltage'``; see
        :func:`active_reflection_coefficient`

    Returns
    -------
    Z_active : ndarray
        Active impedance for each element (complex, n_elements)

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)
    >>> Z = pa.active_scan_impedance_matrix(geom, C, k, theta_deg=30, phi_deg=0)
    >>> Z.shape
    (16,)
    >>> np.all(np.real(Z) > 0)  # All should have positive resistance
    True

    Notes
    -----
    This function computes the active impedance seen by each element
    when the array is steered to the specified direction. Elements
    at different positions in the array may have different active
    impedances due to edge effects and the scan angle.
    """
    from .core import steering_vector

    weights = steering_vector(
        k, geometry.x, geometry.y,
        theta_deg, phi_deg, geometry.z
    )

    gamma = _active_gammas(C, weights, formulation)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_active = np.where(
            np.abs(1 - gamma) < 1e-10,
            complex(np.inf, 0),
            Z0 * (1 + gamma) / (1 - gamma),
        )

    return Z_active
