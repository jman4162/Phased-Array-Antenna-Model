# Phased Array Antenna Modeling

[![CI](https://github.com/jman4162/Phased-Array-Antenna-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/jman4162/Phased-Array-Antenna-Model/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/phased-array-modeling.svg)](https://pypi.org/project/phased-array-modeling/)
[![PyPI downloads](https://img.shields.io/pypi/dm/phased-array-modeling.svg)](https://pypi.org/project/phased-array-modeling/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/Phased-Array-Antenna-Model/blob/main/Phased_Array_Demo.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phased-array-antenna-model.streamlit.app/)

A comprehensive Python library for computing and visualizing phased array antenna radiation patterns. Features **125x faster** vectorized computations, multiple array geometries, advanced beamforming, and interactive 3D visualization.

## Features

- **High Performance**: Vectorized array factor computation (125x faster than naive loops)
- **Multiple Geometries**: Rectangular, triangular, circular, cylindrical, spherical, sparse/thinned arrays
- **Beamforming**: Amplitude tapering (Taylor, Chebyshev, etc.), null steering, multi-beam
- **Realistic Impairments**: Mutual coupling, phase quantization, element failures, scan blindness
- **Visualization**: 2D matplotlib, interactive 3D Plotly, UV-space representation
- **Subarray Support**: Subarray-level beamforming with quantized phase shifters
- **Data Export**: CSV, JSON, and NumPy formats for patterns, weights, and geometry

## Try it Online

**[Launch Interactive Web App](https://phased-array-antenna-model.streamlit.app/)** - No installation required!

The Streamlit app provides an interactive interface for:
- Designing array geometries (rectangular, triangular, circular, concentric rings, elliptical)
- Beam steering with real-time pattern visualization
- Amplitude tapering with sidelobe comparison
- Impairment simulation (phase quantization, element failures, mutual coupling)
- UV-space pattern analysis
- Export data to CSV for further analysis

## Installation

### From GitHub (recommended)

```bash
pip install phased-array-modeling
```

### With optional dependencies

```bash
# Include Plotly for 3D visualization
pip install "phased-array-modeling[plotting]"

# Include all optional dependencies
pip install "phased-array-modeling[full]"
```

### For development

```bash
git clone https://github.com/jman4162/Phased-Array-Antenna-Model.git
cd Phased-Array-Antenna-Model
pip install -e ".[dev]"
```

## Quick Start

```python
import phased_array as pa
import numpy as np

# Create a 16x16 rectangular array with half-wavelength spacing
geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)

# Wavenumber for normalized wavelength
k = pa.wavelength_to_k(1.0)

# Steering weights for 30 degree scan with Taylor taper
weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

# Compute full 2D pattern
theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights, k)

# Plot
pa.plot_pattern_contour(np.rad2deg(theta), np.rad2deg(phi), pattern_dB,
                        title="16x16 Array - 30deg Scan with Taylor Taper")
```

## Examples

### Beam Steering

```python
# Steer beam to theta=25 deg, phi=45 deg
weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=25, phi0_deg=45)

# Compute E-plane and H-plane cuts
angles, E_plane, H_plane = pa.compute_pattern_cuts(
    geom.x, geom.y, weights, k,
    theta0_deg=25, phi0_deg=45
)
```

### Null Steering

```python
# Place nulls at specific directions to reject interference
null_directions = [(20, 0), (-20, 0)]  # (theta, phi) in degrees

weights = pa.null_steering_projection(
    geom, k,
    theta_main_deg=0, phi_main_deg=0,
    null_directions=null_directions
)
```

### Multiple Simultaneous Beams

```python
# Create 3 simultaneous beams
beam_directions = [(0, 0), (25, 0), (25, 180)]

weights = pa.multi_beam_weights_superposition(
    geom, k, beam_directions,
    amplitudes=[1.0, 0.7, 0.7]
)
```

### Circular/Conformal Arrays

```python
# Cylindrical array
geom_cyl = pa.create_cylindrical_array(
    n_azimuth=16, n_vertical=8,
    radius=2.0, height=4.0
)

# Compute pattern accounting for element orientations
AF = pa.array_factor_conformal(theta, phi, geom_cyl, weights, k)
```

### Phase Quantization Analysis

```python
# Simulate 4-bit phase shifters
weights_quantized = pa.quantize_phase(weights, n_bits=4)

# Analyze effect on pattern
results = pa.analyze_quantization_effect(weights, geom, k, n_bits=4)
print(f"RMS phase error: {results['rms_error_deg']:.1f} degrees")
```

### 3D Interactive Visualization

```python
# Create interactive 3D pattern plot
fig = pa.plot_pattern_3d_plotly(theta, phi, pattern_dB,
                                 title="3D Radiation Pattern")
fig.show()
```

## Documentation

For comprehensive documentation, see the [demo notebook](Phased_Array_Demo.ipynb) which covers:

1. Basic array factor computation
2. Beam steering
3. Performance benchmarking
4. Amplitude tapering comparison
5. Array geometries (rectangular, triangular, circular, etc.)
6. Null steering
7. Multiple simultaneous beams
8. Phase quantization effects
9. Element failure analysis
10. UV-space visualization
11. 3D Plotly visualization
12. Conformal array patterns
13. Mutual coupling effects
14. Subarray beamforming

## Package Structure

```
phased_array/
├── core.py          # Vectorized AF, FFT, steering, element patterns
├── geometry.py      # Array geometries and subarray architectures
├── beamforming.py   # Tapering, null steering, multi-beam
├── impairments.py   # Coupling, quantization, failures, scan blindness
├── visualization.py # 2D, 3D Plotly, UV-space plotting
└── utils.py         # Coordinate transforms, helpers
```

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0
- Plotly >= 5.0.0 (optional, for 3D visualization)

## Running Tests

```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{phased_array,
  author = {Hodge, John},
  title = {Phased Array Antenna Modeling},
  url = {https://github.com/jman4162/Phased-Array-Antenna-Model},
  year = {2024}
}
```

## Contact

John Hodge - jah70@vt.edu

Project Link: [https://github.com/jman4162/Phased-Array-Antenna-Model](https://github.com/jman4162/Phased-Array-Antenna-Model)
