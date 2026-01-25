"""
Realistic impairment models for phased arrays.

Includes mutual coupling, phase quantization, element failures, and scan blindness.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Union
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
        'sinc' - sinc function model (good for dipoles)
        'exponential' - exponential decay model
    coupling_coeff : float
        Coupling coefficient (typical: 0.1-0.5)

    Returns
    -------
    C : ndarray
        N x N complex mutual coupling matrix
    """
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
                elif coupling_model == 'exponential':
                    # Exponential decay
                    coupling = coupling_coeff * np.exp(-kr / 2)
                else:
                    coupling = 0

                # Add phase based on distance
                C[i, j] = coupling * np.exp(-1j * kr)

    return C


def mutual_coupling_matrix_measured(
    s_parameters: np.ndarray
) -> np.ndarray:
    """
    Convert measured S-parameters to coupling matrix.

    The coupling matrix relates actual element currents to excitation
    voltages: I = C^(-1) @ V

    Parameters
    ----------
    s_parameters : ndarray
        N x N S-parameter matrix (complex)

    Returns
    -------
    C : ndarray
        Mutual coupling matrix
    """
    n = s_parameters.shape[0]
    # C = (I + S)(I - S)^(-1) for impedance normalization
    # Or simply use S directly for voltage coupling
    C = np.eye(n, dtype=complex) + s_parameters
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
    Estimate sidelobe level increase due to phase quantization.

    Parameters
    ----------
    n_bits : int
        Number of phase quantization bits

    Returns
    -------
    increase_dB : float
        Expected sidelobe increase in dB
    """
    # Approximate formula: sidelobe ratio ~ -6*n_bits dB
    # for uniformly distributed phase errors
    rms_error_rad = np.deg2rad(quantization_rms_error(n_bits))
    # Peak sidelobe from quantization noise ~ 2*sigma
    return 20 * np.log10(2 * rms_error_rad + 1e-10)


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
        geometry.x, geometry.y, weights, k
    ).ravel()

    AF_quant = array_factor_vectorized(
        theta_grid, phi_grid,
        geometry.x, geometry.y, weights_q, k
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
        'gain_loss_mean_dB': mean gain loss
        'gain_loss_std_dB': std of gain loss
        'sidelobe_increase_mean_dB': mean sidelobe increase
    """
    from .core import array_factor_vectorized, compute_half_power_beamwidth
    from .utils import linear_to_db

    # Reference pattern (no failures)
    theta = np.linspace(0, np.pi/2, 181)
    phi = np.zeros_like(theta)

    AF_ref = array_factor_vectorized(
        theta.reshape(-1, 1), phi.reshape(-1, 1),
        geometry.x, geometry.y, weights, k
    ).ravel()
    pattern_ref_dB = linear_to_db(np.abs(AF_ref)**2)
    pattern_ref_dB -= np.max(pattern_ref_dB)

    peak_ref = 0  # Normalized
    # Find first sidelobe
    main_beam_end = np.argmax(pattern_ref_dB < -3)
    sidelobe_ref = np.max(pattern_ref_dB[main_beam_end:]) if main_beam_end > 0 else -20

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

            AF = array_factor_vectorized(
                theta.reshape(-1, 1), phi.reshape(-1, 1),
                geometry.x, geometry.y, degraded, k
            ).ravel()

            pattern_dB = linear_to_db(np.abs(AF)**2)
            peak_degraded = np.max(pattern_dB)
            pattern_dB -= peak_degraded

            # Gain loss
            gain_loss = peak_ref - (peak_degraded - np.max(linear_to_db(np.abs(AF_ref)**2)))
            gain_losses.append(gain_loss)

            # Sidelobe increase
            sidelobe_degraded = np.max(pattern_dB[main_beam_end:]) if main_beam_end > 0 else -20
            sidelobe_increases.append(sidelobe_degraded - sidelobe_ref)

        gain_loss_mean.append(np.mean(gain_losses))
        gain_loss_std.append(np.std(gain_losses))
        sidelobe_increase_mean.append(np.mean(sidelobe_increases))

    return {
        'failure_rates': np.array(failure_rates),
        'gain_loss_mean_dB': np.array(gain_loss_mean),
        'gain_loss_std_dB': np.array(gain_loss_std),
        'sidelobe_increase_mean_dB': np.array(sidelobe_increase_mean)
    }


# ============== Scan Blindness ==============

def surface_wave_scan_angle(
    dx: float,
    dy: float,
    substrate_er: float = 4.0,
    substrate_h: float = 0.1
) -> Tuple[float, float]:
    """
    Estimate scan blindness angles due to surface wave excitation.

    Parameters
    ----------
    dx : float
        Element spacing in x (wavelengths)
    dy : float
        Element spacing in y (wavelengths)
    substrate_er : float
        Substrate relative permittivity
    substrate_h : float
        Substrate height (wavelengths)

    Returns
    -------
    theta_blind_E : float
        Blind angle in E-plane (degrees)
    theta_blind_H : float
        Blind angle in H-plane (degrees)
    """
    # Surface wave propagation constant (approximate)
    # For thin substrates: beta_sw ~ k0 * sqrt(er) * (1 + some correction)
    # Simplified model
    n_eff = np.sqrt(substrate_er) * (1 + 0.5 * substrate_h * np.sqrt(substrate_er - 1))
    n_eff = min(n_eff, np.sqrt(substrate_er))

    # Blind angle occurs when grating lobe enters surface wave
    # sin(theta_blind) = n_eff - 1/d
    sin_theta_E = n_eff - 1 / dx
    sin_theta_H = n_eff - 1 / dy

    # Clamp to valid range
    sin_theta_E = np.clip(sin_theta_E, -1, 1)
    sin_theta_H = np.clip(sin_theta_H, -1, 1)

    theta_blind_E = np.rad2deg(np.arcsin(sin_theta_E)) if abs(sin_theta_E) <= 1 else 90
    theta_blind_H = np.rad2deg(np.arcsin(sin_theta_H)) if abs(sin_theta_H) <= 1 else 90

    return abs(theta_blind_E), abs(theta_blind_H)


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
    """
    from .core import total_pattern

    # Compute gain at broadside
    theta_bs = np.array([[0.0]])
    phi_bs = np.array([[0.0]])
    weights_bs = np.ones_like(weights)

    pattern_bs = total_pattern(
        theta_bs, phi_bs, geometry.x, geometry.y,
        weights_bs, k, element_pattern_func
    )
    gain_bs = np.abs(pattern_bs.item()) ** 2

    # Compute gain at scan angle
    theta_scan = np.array([[np.deg2rad(theta_scan_deg)]])
    phi_scan = np.array([[np.deg2rad(phi_scan_deg)]])

    pattern_scan = total_pattern(
        theta_scan, phi_scan, geometry.x, geometry.y,
        weights, k, element_pattern_func
    )
    gain_scan = np.abs(pattern_scan.item()) ** 2

    if gain_bs > 0 and gain_scan > 0:
        return 10 * np.log10(gain_scan / gain_bs)
    else:
        return -100.0
