# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Python package and Jupyter notebooks for computing and visualizing phased array antenna radiation patterns. The project includes:

1. **`phased_array/` package** - Comprehensive library for array modeling with:
   - Vectorized array factor computation (50-100x faster than loops)
   - Multiple array geometries (rectangular, circular, conformal, sparse)
   - Beamforming techniques (tapering, null steering, multi-beam)
   - Realistic impairments (mutual coupling, quantization, failures, scan blindness)
   - Interactive 3D visualization with Plotly

2. **Notebooks** (designed for Google Colab):
   - `Phased_Array_Demo.ipynb` - Comprehensive demo of all package features
   - `Phased_Array_Antennas_Computing_Radiation_Patterns_Using_Python.ipynb` - Original tutorial
   - `Generating_Rectangular_and_Offset_Triangular_Grid_of_Phased_Array_Elements_within_an_Elliptical_Boundary.ipynb` - Grid generation tutorial

## Package Structure

```
phased_array/
├── __init__.py      # Public API exports
├── core.py          # Vectorized AF, FFT, steering, element patterns
├── geometry.py      # ArrayGeometry, conformal, sparse, subarrays
├── beamforming.py   # Tapers, null steering, multi-beam
├── impairments.py   # Coupling, quantization, failures, scan blindness
├── visualization.py # 2D, 3D Plotly, UV-space plotting
└── utils.py         # Coordinate transforms, helpers
```

## Dependencies

```bash
pip install numpy matplotlib scipy plotly seaborn requests
```

Or use: `pip install -r requirements.txt`

## Quick Start

```python
import phased_array as pa
import numpy as np

# Create a 16x16 rectangular array, half-wavelength spacing
geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)

# Wavenumber for normalized wavelength
k = pa.wavelength_to_k(1.0)

# Steering weights for 30 deg scan with Taylor taper
weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

# Compute and plot pattern
theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights, k)
pa.plot_pattern_contour(np.rad2deg(theta), np.rad2deg(phi), pattern_dB)
```

## Key Concepts and Code Patterns

### Coordinate Systems
- Element positions use Cartesian coordinates (x, y, z) in meters
- Observation angles use theta/phi (elevation/azimuth) in radians
- UV-space: direction cosines u = sin(θ)cos(φ), v = sin(θ)sin(φ)
- Conversion functions in `utils.py`: `theta_phi_to_uv()`, `uv_to_theta_phi()`, etc.

### Core Computations
- **Steering vector**: `steering_vector(k, x, y, theta0_deg, phi0_deg)`
- **Array factor**: `array_factor_vectorized()` (fast) or `array_factor_fft()` (uniform arrays)
- **Element pattern**: `element_pattern()` with raised cosine model
- **Total pattern**: `total_pattern()` = element × array factor

### Array Geometries (`geometry.py`)
- `ArrayGeometry` dataclass holds positions and optional element normals
- `create_rectangular_array()`, `create_triangular_array()`, `create_elliptical_array()`
- `create_circular_array()`, `create_cylindrical_array()`, `create_spherical_array()`
- `thin_array_random()`, `thin_array_density_tapered()`, `thin_array_genetic_algorithm()`
- `SubarrayArchitecture` for subarray-level beamforming

### Beamforming (`beamforming.py`)
- **Amplitude tapers**: `taylor_taper_2d()`, `chebyshev_taper_2d()`, `hamming_taper_2d()`, etc.
- **Null steering**: `null_steering_projection()`, `null_steering_lcmv()`
- **Multi-beam**: `multi_beam_weights_superposition()`, `multi_beam_weights_orthogonal()`

### Impairments (`impairments.py`)
- **Mutual coupling**: `mutual_coupling_matrix_theoretical()`, `apply_mutual_coupling()`
- **Quantization**: `quantize_phase()`, `analyze_quantization_effect()`
- **Failures**: `simulate_element_failures()`, `analyze_graceful_degradation()`
- **Scan blindness**: `surface_wave_scan_angle()`, `apply_scan_blindness()`

### Visualization (`visualization.py`)
- **2D matplotlib**: `plot_pattern_2d()`, `plot_pattern_contour()`, `plot_array_geometry()`
- **UV-space**: `compute_pattern_uv_space()`, `plot_pattern_uv_space()`
- **3D Plotly**: `plot_pattern_3d_plotly()`, `plot_array_geometry_3d_plotly()`

### Antenna Parameters
- Grid spacing typically λ/2 to avoid grating lobes
- `Nx`, `Ny` define array dimensions
- `dx`, `dy` spacing in wavelengths
- `theta0`, `phi0` beam steering angles in degrees

## Running the Notebooks

Open in Google Colab using the badge links at the top of each notebook, or run locally:
```bash
jupyter notebook
```

## Testing

Verify package installation and basic functionality:
```python
import phased_array as pa
print(f"Version: {pa.__version__}")

# Quick test
geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
k = pa.wavelength_to_k(1.0)
weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)
print(f"Array has {geom.n_elements} elements")
```
