"""
Phased Array Antenna Modeling Package

A comprehensive Python library for computing and visualizing phased array
antenna radiation patterns, including:

- Vectorized array factor computation (50-100x faster than loops)
- FFT-based pattern computation for uniform rectangular arrays
- Various array geometries (rectangular, circular, conformal, sparse)
- Beamforming techniques (tapering, null steering, multi-beam)
- Realistic impairments (mutual coupling, quantization, failures, scan blindness)
- Interactive 3D visualization with Plotly
- UV-space pattern representation

Example
-------
>>> import phased_array as pa
>>> import numpy as np
>>>
>>> # Create a 10x10 rectangular array, half-wavelength spacing
>>> geom = pa.create_rectangular_array(10, 10, dx=0.5, dy=0.5)
>>>
>>> # Compute steering weights for 30 deg scan
>>> k = pa.wavelength_to_k(1.0)  # normalized wavelength
>>> weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
>>>
>>> # Apply Taylor taper for sidelobe control
>>> taper = pa.taylor_taper_2d(10, 10, sidelobe_dB=-30)
>>> weights = weights * taper
>>>
>>> # Compute and plot pattern
>>> theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights, k)
"""

__version__ = "1.1.0"

# Beamforming functions
from .beamforming import (  # Amplitude tapers; Null steering; Multiple beams
    apply_taper_to_geometry, chebyshev_taper_1d, chebyshev_taper_2d,
    compute_beam_isolation, compute_null_depth, compute_taper_directivity_loss,
    compute_taper_efficiency, cosine_on_pedestal_taper_1d,
    cosine_on_pedestal_taper_2d, cosine_taper_1d, cosine_taper_2d,
    gaussian_taper_1d, gaussian_taper_2d, hamming_taper_1d, hamming_taper_2d,
    hanning_taper_1d, hanning_taper_2d, monopulse_weights,
    multi_beam_weights_orthogonal, multi_beam_weights_superposition,
    null_steering_lcmv, null_steering_projection, taylor_taper_1d,
    taylor_taper_2d)
# Core computation functions
from .core import (array_factor_fft, array_factor_uv, array_factor_vectorized,
                   compute_directivity, compute_full_pattern,
                   compute_half_power_beamwidth, compute_pattern_cuts,
                   element_pattern, element_pattern_cosine_tapered,
                   steering_vector, total_pattern)
# Export functions
from .export import (export_array_config_json, export_coupling_matrix_csv,
                     export_geometry_csv, export_pattern_2d_csv,
                     export_pattern_csv, export_pattern_npz,
                     export_summary_report, export_uv_pattern_csv,
                     export_weights_csv, load_pattern_npz)
# Geometry classes and functions
from .geometry import (ArrayGeometry, SubarrayArchitecture,
                       array_factor_conformal, compute_subarray_weights,
                       create_circular_array, create_concentric_rings_array,
                       create_cylindrical_array, create_elliptical_array,
                       create_rectangular_array, create_rectangular_subarrays,
                       create_spherical_array, create_triangular_array,
                       thin_array_density_tapered,
                       thin_array_genetic_algorithm, thin_array_random)
# Impairment models
from .impairments import (  # Mutual coupling; Phase quantization; Element failures; Scan blindness
    active_element_pattern, analyze_graceful_degradation,
    analyze_quantization_effect, apply_mutual_coupling, apply_scan_blindness,
    compute_scan_loss, mutual_coupling_matrix_measured,
    mutual_coupling_matrix_theoretical, quantization_rms_error,
    quantization_sidelobe_increase, quantize_phase, scan_blindness_model,
    simulate_element_failures, surface_wave_scan_angle)
# Utility functions
from .utils import (azel_to_thetaphi, create_theta_phi_grid, create_uv_grid,
                    db_to_linear, deg2rad, frequency_to_k,
                    frequency_to_wavelength, is_visible_region, linear_to_db,
                    normalize_pattern, rad2deg, theta_phi_to_uv,
                    thetaphi_to_azel, uv_to_theta_phi, wavelength_to_k)
# Visualization functions
from .visualization import (  # 2D matplotlib plots; UV-space; 3D Plotly plots; Utilities
    compute_pattern_uv_space, create_pattern_animation_plotly,
    plot_array_geometry, plot_array_geometry_3d_plotly,
    plot_comparison_patterns, plot_pattern_2d,
    plot_pattern_3d_cartesian_plotly, plot_pattern_3d_plotly,
    plot_pattern_contour, plot_pattern_polar, plot_pattern_uv_plotly,
    plot_pattern_uv_space)

__all__ = [
    # Version
    "__version__",
    # Core
    "steering_vector",
    "array_factor_vectorized",
    "array_factor_uv",
    "array_factor_fft",
    "element_pattern",
    "element_pattern_cosine_tapered",
    "total_pattern",
    "compute_pattern_cuts",
    "compute_full_pattern",
    "compute_directivity",
    "compute_half_power_beamwidth",
    # Utils
    "deg2rad",
    "rad2deg",
    "azel_to_thetaphi",
    "thetaphi_to_azel",
    "theta_phi_to_uv",
    "uv_to_theta_phi",
    "is_visible_region",
    "wavelength_to_k",
    "frequency_to_wavelength",
    "frequency_to_k",
    "db_to_linear",
    "linear_to_db",
    "normalize_pattern",
    "create_theta_phi_grid",
    "create_uv_grid",
    # Geometry
    "ArrayGeometry",
    "SubarrayArchitecture",
    "create_rectangular_array",
    "create_triangular_array",
    "create_elliptical_array",
    "create_circular_array",
    "create_concentric_rings_array",
    "create_cylindrical_array",
    "create_spherical_array",
    "thin_array_random",
    "thin_array_density_tapered",
    "thin_array_genetic_algorithm",
    "create_rectangular_subarrays",
    "compute_subarray_weights",
    "array_factor_conformal",
    # Beamforming
    "taylor_taper_1d",
    "taylor_taper_2d",
    "chebyshev_taper_1d",
    "chebyshev_taper_2d",
    "hamming_taper_1d",
    "hamming_taper_2d",
    "hanning_taper_1d",
    "hanning_taper_2d",
    "cosine_taper_1d",
    "cosine_taper_2d",
    "cosine_on_pedestal_taper_1d",
    "cosine_on_pedestal_taper_2d",
    "gaussian_taper_1d",
    "gaussian_taper_2d",
    "compute_taper_efficiency",
    "compute_taper_directivity_loss",
    "apply_taper_to_geometry",
    "null_steering_projection",
    "null_steering_lcmv",
    "compute_null_depth",
    "multi_beam_weights_superposition",
    "multi_beam_weights_orthogonal",
    "compute_beam_isolation",
    "monopulse_weights",
    # Impairments
    "mutual_coupling_matrix_theoretical",
    "mutual_coupling_matrix_measured",
    "apply_mutual_coupling",
    "active_element_pattern",
    "quantize_phase",
    "quantization_rms_error",
    "quantization_sidelobe_increase",
    "analyze_quantization_effect",
    "simulate_element_failures",
    "analyze_graceful_degradation",
    "surface_wave_scan_angle",
    "scan_blindness_model",
    "apply_scan_blindness",
    "compute_scan_loss",
    # Visualization
    "plot_pattern_2d",
    "plot_pattern_polar",
    "plot_pattern_contour",
    "plot_array_geometry",
    "compute_pattern_uv_space",
    "plot_pattern_uv_space",
    "plot_pattern_3d_plotly",
    "plot_pattern_3d_cartesian_plotly",
    "plot_array_geometry_3d_plotly",
    "plot_pattern_uv_plotly",
    "plot_comparison_patterns",
    "create_pattern_animation_plotly",
    # Export
    "export_pattern_csv",
    "export_pattern_2d_csv",
    "export_uv_pattern_csv",
    "export_weights_csv",
    "export_geometry_csv",
    "export_array_config_json",
    "export_pattern_npz",
    "load_pattern_npz",
    "export_coupling_matrix_csv",
    "export_summary_report",
]
