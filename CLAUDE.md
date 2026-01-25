# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Jupyter notebooks for computing and visualizing phased array antenna radiation patterns using Python. The notebooks are designed to run in Google Colab and cover:

1. **Radiation Pattern Computation** (`Phased_Array_Antennas_Computing_Radiation_Patterns_Using_Python.ipynb`) - Main tutorial covering beam steering, array factor calculation, and pattern visualization
2. **Element Grid Generation** (`Generating_Rectangular_and_Offset_Triangular_Grid_of_Phased_Array_Elements_within_an_Elliptical_Boundary.ipynb`) - Generates rectangular and offset triangular (hexagonal) grid layouts within elliptical boundaries

## Dependencies

```bash
pip install numpy matplotlib scipy seaborn requests
```

## Running the Notebooks

Open in Google Colab using the badge links at the top of each notebook, or run locally:
```bash
jupyter notebook
```

## Key Concepts and Code Patterns

### Coordinate Systems
- Element positions use Cartesian coordinates (x, y) in meters
- Observation angles use theta/phi (elevation/azimuth) in radians
- Conversion functions `azel_to_thetaphi()` and `thetaphi_to_azel()` handle coordinate transforms

### Core Computations
- **Steering vector**: Phase shifts for beam steering calculated via `steering_vector(k, xv, yv, theta_deg, phi_deg)`
- **Array factor**: Summed contribution from all elements via `AF(theta, phi, x, y, w, k)`
- **Element pattern**: Raised cosine model via `antenna_element_pattern(theta, phi, cos_factor_theta, cos_factor_phi, max_gain_dBi)`
- **Total pattern**: Element pattern × Array factor

### Antenna Parameters
- Grid spacing typically λ/2 to avoid grating lobes
- `Nx`, `Ny` define array dimensions
- `dx`, `dy` spacing in wavelengths
- `theta0`, `phi0` beam steering angles in degrees

### Grid Types
- Rectangular: uniform spacing in x and y
- Offset triangular (hexagonal): rows offset by half-step with √3/2 vertical spacing
- Both constrained to elliptical boundaries using `(x/a)² + (y/b)² ≤ 1`
