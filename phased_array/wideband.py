"""
Wideband beamforming functions for phased array antennas.

Includes true-time delay (TTD), hybrid phase/TTD steering,
beam squint analysis, and wideband pattern computation.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .geometry import ArrayGeometry, SubarrayArchitecture

ArrayLike = Union[np.ndarray, float]

# Speed of light
C = 299792458.0  # m/s


def steering_vector_ttd(
    x: np.ndarray,
    y: np.ndarray,
    theta0_deg: float,
    phi0_deg: float,
    frequency: float,
    c: float = C
) -> np.ndarray:
    """
    Compute true-time delay (TTD) steering vector.

    TTD provides frequency-independent beam pointing by applying
    actual time delays rather than phase shifts.

    Parameters
    ----------
    x : ndarray
        Element x-positions in meters
    y : ndarray
        Element y-positions in meters
    theta0_deg : float
        Steering elevation angle in degrees (from broadside)
    phi0_deg : float
        Steering azimuth angle in degrees
    frequency : float
        Operating frequency in Hz
    c : float, optional
        Speed of light in m/s (default: 299792458)

    Returns
    -------
    weights : ndarray
        Complex steering weights with TTD phases

    Notes
    -----
    The time delay for each element is:
        tau_n = (x_n * sin(theta) * cos(phi) + y_n * sin(theta) * sin(phi)) / c

    The phase is: phi_n = -2 * pi * f * tau_n

    This is equivalent to phase steering at the given frequency, but
    the key difference is that the TIME DELAY is what's physically
    implemented, so the beam points correctly at all frequencies.
    """
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    # Direction cosines
    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    # Time delay for each element (negative for steering towards direction)
    tau = -(x * u0 + y * v0) / c

    # Phase at the given frequency
    phase = 2 * np.pi * frequency * tau

    return np.exp(1j * phase)


def steering_delays_ttd(
    x: np.ndarray,
    y: np.ndarray,
    theta0_deg: float,
    phi0_deg: float,
    c: float = C
) -> np.ndarray:
    """
    Compute true-time delays for steering (frequency-independent).

    Parameters
    ----------
    x : ndarray
        Element x-positions in meters
    y : ndarray
        Element y-positions in meters
    theta0_deg : float
        Steering elevation angle in degrees
    phi0_deg : float
        Steering azimuth angle in degrees
    c : float, optional
        Speed of light in m/s

    Returns
    -------
    delays : ndarray
        Time delays in seconds for each element
    """
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    # Time delay (negative = delay for positive steering)
    delays = -(x * u0 + y * v0) / c

    # Normalize so minimum delay is zero
    delays = delays - np.min(delays)

    return delays


def steering_vector_hybrid(
    geometry: ArrayGeometry,
    architecture: SubarrayArchitecture,
    theta0_deg: float,
    phi0_deg: float,
    frequency: float,
    c: float = C
) -> np.ndarray:
    """
    Compute hybrid TTD + phase steering vector.

    True-time delay is applied at the subarray level (coarse steering),
    and phase shifters provide fine adjustment within each subarray.
    This is the most common architecture for wideband phased arrays.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry with element positions
    architecture : SubarrayArchitecture
        Subarray partitioning
    theta0_deg : float
        Steering elevation angle in degrees
    phi0_deg : float
        Steering azimuth angle in degrees
    frequency : float
        Operating frequency in Hz
    c : float, optional
        Speed of light in m/s

    Returns
    -------
    weights : ndarray
        Complex steering weights

    Notes
    -----
    For each element:
    1. TTD is computed based on subarray center position
    2. Phase shift compensates for element offset from subarray center

    This reduces beam squint compared to phase-only steering,
    with squint now determined by subarray size rather than array size.
    """
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    k = 2 * np.pi * frequency / c

    weights = np.zeros(geometry.n_elements, dtype=complex)

    for sub_idx in range(architecture.n_subarrays):
        # Get elements in this subarray
        mask = architecture.subarray_assignments == sub_idx
        elem_indices = np.where(mask)[0]

        # Subarray center position
        center_x = architecture.subarray_centers[sub_idx, 0]
        center_y = architecture.subarray_centers[sub_idx, 1]

        # TTD phase based on subarray center (frequency-independent pointing)
        tau_subarray = -(center_x * u0 + center_y * v0) / c
        ttd_phase = 2 * np.pi * frequency * tau_subarray

        # Phase shifter compensation for element offset from center
        for idx in elem_indices:
            dx = geometry.x[idx] - center_x
            dy = geometry.y[idx] - center_y

            # Phase shift for offset (this causes residual squint)
            phase_offset = -k * (dx * u0 + dy * v0)

            weights[idx] = np.exp(1j * (ttd_phase + phase_offset))

    return weights


def compute_subarray_delays_ttd(
    architecture: SubarrayArchitecture,
    theta0_deg: float,
    phi0_deg: float,
    c: float = C
) -> np.ndarray:
    """
    Compute TTD values for each subarray.

    Parameters
    ----------
    architecture : SubarrayArchitecture
        Subarray partitioning with center positions
    theta0_deg : float
        Steering elevation angle in degrees
    phi0_deg : float
        Steering azimuth angle in degrees
    c : float, optional
        Speed of light in m/s

    Returns
    -------
    delays : ndarray
        Time delay for each subarray in seconds (shape: n_subarrays,)
    """
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    delays = np.zeros(architecture.n_subarrays)

    for sub_idx in range(architecture.n_subarrays):
        center_x = architecture.subarray_centers[sub_idx, 0]
        center_y = architecture.subarray_centers[sub_idx, 1]
        delays[sub_idx] = -(center_x * u0 + center_y * v0) / c

    # Normalize so minimum delay is zero
    delays = delays - np.min(delays)

    return delays


def compute_beam_squint(
    x: np.ndarray,
    y: np.ndarray,
    theta0_deg: float,
    phi0_deg: float,
    center_frequency: float,
    frequencies: np.ndarray,
    steering_mode: str = 'phase',
    architecture: Optional[SubarrayArchitecture] = None,
    n_points: int = 361
) -> Dict[str, np.ndarray]:
    """
    Compute beam squint (pointing error) vs frequency.

    Parameters
    ----------
    x : ndarray
        Element x-positions in meters
    y : ndarray
        Element y-positions in meters
    theta0_deg : float
        Intended steering angle in degrees
    phi0_deg : float
        Intended azimuth angle in degrees
    center_frequency : float
        Center frequency in Hz (where weights are computed)
    frequencies : ndarray
        Frequencies to evaluate in Hz
    steering_mode : str
        'phase' - phase-only steering (maximum squint)
        'ttd' - true-time delay (no squint)
        'hybrid' - TTD at subarray + phase at element
    architecture : SubarrayArchitecture, optional
        Required for 'hybrid' mode
    n_points : int
        Number of angle points for pattern computation

    Returns
    -------
    results : dict
        'frequencies' : Frequency values (Hz)
        'beam_angles' : Actual beam pointing angle at each frequency (deg)
        'squint' : Pointing error (deg), positive = beam moved towards broadside
        'relative_gain' : Gain relative to center frequency (dB)
    """
    from .core import array_factor_vectorized

    c = C
    k_center = 2 * np.pi * center_frequency / c

    # Compute steering weights at center frequency
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)
    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    if steering_mode == 'phase':
        # Phase-only steering - compute at center frequency
        weights = np.exp(-1j * k_center * (x * u0 + y * v0))
    elif steering_mode == 'ttd':
        # TTD - delays are frequency-independent
        weights = steering_vector_ttd(x, y, theta0_deg, phi0_deg, center_frequency)
    elif steering_mode == 'hybrid':
        if architecture is None:
            raise ValueError("SubarrayArchitecture required for hybrid mode")
        geom = ArrayGeometry(x=x, y=y)
        weights = steering_vector_hybrid(geom, architecture, theta0_deg, phi0_deg, center_frequency)
    else:
        raise ValueError(f"Unknown steering_mode: {steering_mode}")

    # Scan angles around the intended steering direction
    scan_range = 30  # degrees
    angles = np.linspace(theta0_deg - scan_range, theta0_deg + scan_range, n_points)
    angles = np.clip(angles, -90, 90)
    theta_rad = np.deg2rad(angles)

    beam_angles = []
    relative_gains = []

    for freq in frequencies:
        k = 2 * np.pi * freq / c

        if steering_mode == 'ttd':
            # For TTD, recompute weights at each frequency
            # (the TIME delays are fixed, but phase = 2*pi*f*tau)
            weights_freq = steering_vector_ttd(x, y, theta0_deg, phi0_deg, freq)
        elif steering_mode == 'hybrid' and architecture is not None:
            geom = ArrayGeometry(x=x, y=y)
            weights_freq = steering_vector_hybrid(geom, architecture, theta0_deg, phi0_deg, freq)
        else:
            # Phase-only: weights are fixed, but k changes
            weights_freq = weights

        # Compute pattern along the scan cut (phi = phi0)
        pattern = np.zeros(len(angles), dtype=complex)
        for i, th in enumerate(theta_rad):
            u = np.sin(th) * np.cos(phi0)
            v = np.sin(th) * np.sin(phi0)
            pattern[i] = np.sum(weights_freq * np.exp(1j * k * (x * u + y * v)))

        pattern_mag = np.abs(pattern)
        peak_idx = np.argmax(pattern_mag)
        beam_angles.append(angles[peak_idx])

        # Relative gain (compared to peak at center frequency)
        relative_gains.append(20 * np.log10(pattern_mag[peak_idx] / len(x)))

    beam_angles = np.array(beam_angles)
    squint = beam_angles - theta0_deg

    # Normalize relative gain to 0 dB at center frequency
    center_idx = np.argmin(np.abs(frequencies - center_frequency))
    relative_gains = np.array(relative_gains) - relative_gains[center_idx]

    return {
        'frequencies': frequencies,
        'beam_angles': beam_angles,
        'squint': squint,
        'relative_gain': relative_gains
    }


def analyze_instantaneous_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    theta0_deg: float,
    phi0_deg: float,
    center_frequency: float,
    squint_tolerance_deg: float = 0.5,
    steering_mode: str = 'phase',
    architecture: Optional[SubarrayArchitecture] = None
) -> Dict[str, float]:
    """
    Analyze instantaneous bandwidth for given squint tolerance.

    Parameters
    ----------
    x : ndarray
        Element x-positions in meters
    y : ndarray
        Element y-positions in meters
    theta0_deg : float
        Steering angle in degrees
    phi0_deg : float
        Azimuth angle in degrees
    center_frequency : float
        Center frequency in Hz
    squint_tolerance_deg : float
        Maximum acceptable beam squint in degrees
    steering_mode : str
        'phase', 'ttd', or 'hybrid'
    architecture : SubarrayArchitecture, optional
        Required for 'hybrid' mode

    Returns
    -------
    results : dict
        'ibw_hz' : Instantaneous bandwidth in Hz
        'ibw_percent' : Bandwidth as percentage of center frequency
        'ibw_ratio' : Bandwidth ratio (f_high / f_low)
    """
    # For TTD, bandwidth is theoretically infinite (limited by other factors)
    if steering_mode == 'ttd':
        return {
            'ibw_hz': np.inf,
            'ibw_percent': np.inf,
            'ibw_ratio': np.inf,
            'note': 'TTD provides theoretically unlimited instantaneous bandwidth'
        }

    # Search for bandwidth limits
    # Start with a wide range and narrow down
    max_bw_percent = 100  # Start with +/- 50%

    for bw_percent in np.linspace(1, max_bw_percent, 100):
        bw_hz = center_frequency * bw_percent / 100
        frequencies = np.array([
            center_frequency - bw_hz / 2,
            center_frequency,
            center_frequency + bw_hz / 2
        ])

        results = compute_beam_squint(
            x, y, theta0_deg, phi0_deg, center_frequency, frequencies,
            steering_mode=steering_mode, architecture=architecture
        )

        max_squint = np.max(np.abs(results['squint']))

        if max_squint > squint_tolerance_deg:
            # Found the limit
            ibw_hz = bw_hz * squint_tolerance_deg / max_squint
            ibw_percent = ibw_hz / center_frequency * 100
            f_low = center_frequency - ibw_hz / 2
            f_high = center_frequency + ibw_hz / 2

            return {
                'ibw_hz': ibw_hz,
                'ibw_percent': ibw_percent,
                'ibw_ratio': f_high / f_low,
                'f_low': f_low,
                'f_high': f_high
            }

    # If we get here, bandwidth exceeds our search range
    return {
        'ibw_hz': center_frequency * max_bw_percent / 100,
        'ibw_percent': max_bw_percent,
        'ibw_ratio': 1.5,
        'note': f'Exceeds {max_bw_percent}% bandwidth'
    }


def compute_pattern_vs_frequency(
    x: np.ndarray,
    y: np.ndarray,
    theta0_deg: float,
    phi0_deg: float,
    center_frequency: float,
    frequencies: np.ndarray,
    steering_mode: str = 'phase',
    architecture: Optional[SubarrayArchitecture] = None,
    n_points: int = 181,
    phi_cut_deg: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Compute radiation pattern at multiple frequencies.

    Parameters
    ----------
    x : ndarray
        Element x-positions in meters
    y : ndarray
        Element y-positions in meters
    theta0_deg : float
        Steering angle in degrees
    phi0_deg : float
        Azimuth angle in degrees
    center_frequency : float
        Center frequency in Hz
    frequencies : ndarray
        Frequencies to compute patterns at
    steering_mode : str
        'phase', 'ttd', or 'hybrid'
    architecture : SubarrayArchitecture, optional
        Required for 'hybrid' mode
    n_points : int
        Number of angle points
    phi_cut_deg : float, optional
        Phi angle for cut (default: phi0_deg)

    Returns
    -------
    results : dict
        'angles' : Theta angles in degrees
        'frequencies' : Frequency values
        'patterns' : 2D array of patterns (n_freq x n_angles) in dB
    """
    c = C
    k_center = 2 * np.pi * center_frequency / c

    if phi_cut_deg is None:
        phi_cut_deg = phi0_deg
    phi_cut = np.deg2rad(phi_cut_deg)

    # Steering direction
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)
    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    # Compute base weights at center frequency
    if steering_mode == 'phase':
        weights_base = np.exp(-1j * k_center * (x * u0 + y * v0))
    elif steering_mode == 'ttd':
        weights_base = None  # Will compute per-frequency
    elif steering_mode == 'hybrid':
        if architecture is None:
            raise ValueError("SubarrayArchitecture required for hybrid mode")
        weights_base = None
    else:
        raise ValueError(f"Unknown steering_mode: {steering_mode}")

    angles = np.linspace(-90, 90, n_points)
    theta_rad = np.deg2rad(angles)

    patterns = np.zeros((len(frequencies), n_points))

    for f_idx, freq in enumerate(frequencies):
        k = 2 * np.pi * freq / c

        if steering_mode == 'ttd':
            weights = steering_vector_ttd(x, y, theta0_deg, phi0_deg, freq)
        elif steering_mode == 'hybrid':
            geom = ArrayGeometry(x=x, y=y)
            weights = steering_vector_hybrid(geom, architecture, theta0_deg, phi0_deg, freq)
        else:
            weights = weights_base

        # Compute pattern
        for i, th in enumerate(theta_rad):
            u = np.sin(th) * np.cos(phi_cut)
            v = np.sin(th) * np.sin(phi_cut)
            af = np.abs(np.sum(weights * np.exp(1j * k * (x * u + y * v))))
            patterns[f_idx, i] = af

        # Normalize to peak and convert to dB
        peak = np.max(patterns[f_idx, :])
        if peak > 0:
            patterns[f_idx, :] = 20 * np.log10(patterns[f_idx, :] / peak + 1e-10)

    return {
        'angles': angles,
        'frequencies': frequencies,
        'patterns': patterns
    }


def compute_subarray_weights_hybrid(
    geometry: ArrayGeometry,
    architecture: SubarrayArchitecture,
    theta0_deg: float,
    phi0_deg: float,
    frequency: float,
    amplitude_taper: Optional[np.ndarray] = None,
    ttd_quantization_bits: Optional[int] = None,
    phase_quantization_bits: Optional[int] = None,
    c: float = C
) -> Dict[str, np.ndarray]:
    """
    Compute hybrid TTD + phase weights with optional quantization.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    architecture : SubarrayArchitecture
        Subarray partitioning
    theta0_deg : float
        Steering elevation angle in degrees
    phi0_deg : float
        Steering azimuth angle in degrees
    frequency : float
        Operating frequency in Hz
    amplitude_taper : ndarray, optional
        Amplitude weights for each element
    ttd_quantization_bits : int, optional
        Bits for TTD quantization (e.g., 6 bits = 64 delay steps)
    phase_quantization_bits : int, optional
        Bits for phase shifter quantization
    c : float, optional
        Speed of light

    Returns
    -------
    results : dict
        'weights' : Complex element weights
        'subarray_delays' : TTD value for each subarray (seconds)
        'element_phases' : Phase shift for each element (radians)
    """
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    k = 2 * np.pi * frequency / c
    wavelength = c / frequency

    weights = np.ones(geometry.n_elements, dtype=complex)
    subarray_delays = np.zeros(architecture.n_subarrays)
    element_phases = np.zeros(geometry.n_elements)

    # Compute maximum delay for quantization
    if ttd_quantization_bits is not None:
        max_delay = np.max(np.abs(architecture.subarray_centers[:, 0] * u0 +
                                   architecture.subarray_centers[:, 1] * v0)) / c
        delay_step = 2 * max_delay / (2**ttd_quantization_bits - 1)

    for sub_idx in range(architecture.n_subarrays):
        mask = architecture.subarray_assignments == sub_idx
        elem_indices = np.where(mask)[0]

        center_x = architecture.subarray_centers[sub_idx, 0]
        center_y = architecture.subarray_centers[sub_idx, 1]

        # TTD for subarray center
        tau = -(center_x * u0 + center_y * v0) / c

        # Quantize TTD if specified
        if ttd_quantization_bits is not None and delay_step > 0:
            tau = np.round(tau / delay_step) * delay_step

        subarray_delays[sub_idx] = tau
        ttd_phase = 2 * np.pi * frequency * tau

        # Phase shifter for each element
        for idx in elem_indices:
            dx = geometry.x[idx] - center_x
            dy = geometry.y[idx] - center_y

            phase = -k * (dx * u0 + dy * v0)

            # Quantize phase if specified
            if phase_quantization_bits is not None:
                n_levels = 2**phase_quantization_bits
                phase_step = 2 * np.pi / n_levels
                phase = np.round(phase / phase_step) * phase_step

            element_phases[idx] = phase
            weights[idx] = np.exp(1j * (ttd_phase + phase))

    # Apply amplitude taper
    if amplitude_taper is not None:
        weights = weights * amplitude_taper

    return {
        'weights': weights,
        'subarray_delays': subarray_delays,
        'element_phases': element_phases
    }


def compare_steering_modes(
    geometry: ArrayGeometry,
    architecture: SubarrayArchitecture,
    theta0_deg: float,
    phi0_deg: float,
    center_frequency: float,
    bandwidth_percent: float = 20.0,
    n_freq_points: int = 11
) -> Dict[str, Dict]:
    """
    Compare beam squint for different steering modes.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    architecture : SubarrayArchitecture
        Subarray partitioning
    theta0_deg : float
        Steering angle in degrees
    phi0_deg : float
        Azimuth angle in degrees
    center_frequency : float
        Center frequency in Hz
    bandwidth_percent : float
        Total bandwidth as percentage of center frequency
    n_freq_points : int
        Number of frequency points to evaluate

    Returns
    -------
    results : dict
        Results for each steering mode ('phase', 'hybrid', 'ttd')
    """
    bw_hz = center_frequency * bandwidth_percent / 100
    frequencies = np.linspace(
        center_frequency - bw_hz / 2,
        center_frequency + bw_hz / 2,
        n_freq_points
    )

    results = {}

    for mode in ['phase', 'hybrid', 'ttd']:
        arch = architecture if mode == 'hybrid' else None

        squint_results = compute_beam_squint(
            geometry.x, geometry.y,
            theta0_deg, phi0_deg,
            center_frequency, frequencies,
            steering_mode=mode,
            architecture=arch
        )

        results[mode] = {
            'frequencies': frequencies,
            'squint': squint_results['squint'],
            'max_squint': np.max(np.abs(squint_results['squint'])),
            'beam_angles': squint_results['beam_angles']
        }

    return results
