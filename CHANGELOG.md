# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `compute_directivity` no longer calls `np.trapz`, which NumPy removed
  in 2.4 (deprecated in 2.0). The function raised `AttributeError` on
  any NumPy 2.4 or newer install. It now uses
  `scipy.integrate.trapezoid`, which is available across the supported
  NumPy 1.x and 2.x range; SciPy is already a required dependency
- `compute_directivity` no longer discards the power radiated at the
  poles. Weighting samples by `sin(theta)` gives theta = 0 and theta =
  pi zero weight, biasing directivity high — most visibly for
  high-gain end-fire patterns. Each theta row is now weighted by the
  exact solid angle of its spherical cell, so an isotropic pattern
  integrates to 4*pi within floating-point error. Method contributed
  by @bhicks-leolabs (#8)

### Changed

- `compute_directivity` validates its grid and raises `ValueError` with
  an actionable message for inputs it cannot integrate: non-2D or
  mismatched arrays, non-finite values, a single-sample axis, an
  `indexing='xy'` or transposed mesh, non-uniform or descending
  spacing, or theta outside [0, pi]. Previously such inputs returned a
  plausible-looking number
- `compute_directivity` docstring now states the grid contract, the
  piecewise-constant power approximation, and that exact isotropic
  integration is an invariant rather than a general accuracy
  guarantee: for smooth patterns that vanish at the poles the previous
  `sin(theta)` rule can still be the more accurate of the two

### Added

- `tests/test_directivity.py`: acceptance tests against the public API
  on `create_theta_phi_grid` output, and regression tests covering
  analytic directivities (isotropic, short dipole, `cos(theta)**10`,
  an azimuth-dependent pattern), convergence at 10, 5, 2.5 and 1.25
  degree theta spacing, partial-sphere and partial-azimuth grids,
  amplitude and phase invariance, and grid validation.
  `compute_directivity` previously had no tests
- CI `deps` job pinning three dependency sets that the Python-version
  matrix never reached: Python 3.9 with the oldest declared minimums
  (NumPy 1.20.0, SciPy 1.7.0, Matplotlib 3.5.0), Python 3.11 with
  NumPy 1.26.4, and Python 3.12 with NumPy 2.4 or newer

## [1.4.0] - 2026-07-20

### Added

#### Vector (Polarized) Pattern Engine
- New `phased_array/vector_patterns.py` module wiring the v1.3.0
  polarization math into the pattern engine: polarized element models
  produce complex (E_theta, E_phi) components that multiply the array
  factor
- `VectorPattern` dataclass with `power`, `power_dB()`, `co_cross()`
  (Ludwig-3 x/y reference), `axial_ratio_map()`, and `xpd_map()`
- Polarized element factories: `dipole_element()` (x/y/z Hertzian
  dipole), `ideal_patch_element()` (cos^n co-pol, zero cross-pol),
  `cos_q_polarized_element()` (arbitrary Jones state; successor to the
  deprecated `cos_exp_phi`), `crossed_dipole_element()` (CP turnstile)
- `GriddedElementPattern` for measured/simulated element patterns
  (interpolated, usable anywhere an element function is accepted),
  with `from_scalar()` constructor
- `vector_total_pattern()`, `compute_full_vector_pattern()`,
  `compute_co_cross_pattern_cuts()`, `dual_pol_weights()`
- Conformal support: `element_rotation_matrices()`,
  `global_to_local_angles()`, and `vector_array_factor_conformal()`
  evaluate each element in its local frame and rotate fields back to
  the global spherical basis; `ArrayGeometry` gains optional element
  tangent fields (`tx`, `ty`, `tz`) to set the local polarization
  reference
- 36 new tests (`tests/test_vector_patterns.py`)

### Changed
- `array_factor_conformal()` now passes local (theta, phi) — both
  derived from the element's rotation matrix — to the element pattern
  function instead of local theta with global phi. Results are
  unchanged for the bundled phi-independent element models; custom
  phi-dependent callables now receive correct local angles
- Dropped Python 3.8 support (EOL October 2024); `requires-python >= 3.9`

## [1.3.2] - 2026-07-20

### Fixed
- `overlapped_subarray_weights()` raised `ImportError` for non-overlapped
  architectures (it imported `compute_subarray_weights` from the wrong
  module)
- `mutual_coupling_matrix_measured()` now implements the documented
  impedance formulation C = (I + S)(I - S)^-1. Pass
  `formulation='voltage'` to reproduce the previous C = I + S behavior
- `azel_to_thetaphi()` and `thetaphi_to_azel()` used inconsistent az/el
  conventions and were not mutual inverses. Both now use the documented
  convention (boresight +z, azimuth toward +x, elevation toward +y) and
  round-trip exactly
- `create_concentric_rings_array()` produced in-plane element normals
  (zero for the center element); all normals are now +z, consistent
  with the other planar array factories
- `compute_half_power_beamwidth()` now interpolates the -3 dB crossings
  instead of snapping to grid samples
- Docstring examples for `compute_null_depth`,
  `analyze_graceful_degradation`, `compute_beam_squint`, and
  `stokes_parameters` called functions with wrong arguments or asserted
  wrong expected values; all module docstrings now run as doctests in CI
- Importing the package no longer emits a `SyntaxWarning` from an
  invalid escape sequence in a docstring

### Deprecated
- The unused `cos_exp_phi` parameter of `element_pattern()` now emits a
  `DeprecationWarning` when set; the basic element model is
  phi-symmetric

### Added
- Test coverage for `utils.py` and `visualization.py` (previously
  untested): 48 new tests plus a doctest suite over all modules

## [1.3.1] - 2026-02-01

### Fixed
- Fixed import ordering in `__init__.py` and `beamforming.py` to satisfy isort linting

## [1.3.0] - 2026-02-01

### Added

#### Polarization Module (Issue #2)
- New `phased_array/polarization.py` module for polarization analysis
- `jones_vector()` - Create Jones vectors for polarization states
- `stokes_parameters()` - Compute Stokes parameters (S0, S1, S2, S3)
- `axial_ratio()` - Calculate polarization ellipse axial ratio
- `tilt_angle()` - Calculate polarization ellipse tilt angle
- `cross_pol_discrimination()` - Compute XPD between polarizations
- `polarization_loss_factor()` - Calculate PLF for polarization mismatch
- `co_pol_pattern()` - Extract co-polar component of radiation pattern
- `cross_pol_pattern()` - Extract cross-polar component of radiation pattern
- `ludwig3_decomposition()` - Ludwig-3 co/cross-pol decomposition

#### Coordinate Transforms (Issue #3)
- New `phased_array/coordinates.py` module for coordinate system conversions
- `antenna_to_radar()` - Convert antenna (theta/phi) to radar (az/el) coordinates
- `radar_to_antenna()` - Convert radar (az/el) to antenna (theta/phi) coordinates
- `antenna_to_cone()` - Convert antenna to cone/clock coordinates
- `cone_to_antenna()` - Convert cone/clock to antenna coordinates
- `rotation_matrix_roll()` - 3x3 rotation matrix for roll (x-axis)
- `rotation_matrix_pitch()` - 3x3 rotation matrix for pitch (y-axis)
- `rotation_matrix_yaw()` - 3x3 rotation matrix for yaw (z-axis)
- `rotate_pattern()` - Rotate radiation pattern by Euler angles with interpolation

#### Beam Spoiling (Issue #4)
- `quadratic_phase_spoil()` - Apply quadratic phase distribution for beam broadening
- `compute_spoil_factor()` - Calculate spoil factor for desired beamwidth
- `spoiled_beam_gain()` - Estimate gain of spoiled beam
- `spoiled_beamwidth()` - Estimate beamwidth of spoiled beam

#### Overlapped Subarrays (Issue #5)
- Extended `SubarrayArchitecture` dataclass with overlapped subarray support
  - New fields: `overlapped`, `subarray_elements`, `overlap_weights`
  - New method: `get_element_subarrays()` - Get subarrays containing an element
- `create_overlapped_subarrays()` - Create overlapped subarray architecture
- `overlapped_subarray_weights()` - Compute element weights for overlapped subarrays
- `compute_overlapped_pattern()` - Compute radiation pattern for overlapped architecture

#### Adaptive Beamforming SMI/GSC (Issue #6)
- `adaptive_weights_smi()` - Sample Matrix Inversion (MVDR) adaptive weights
- `adaptive_weights_gsc()` - Generalized Sidelobe Canceller adaptive weights
- `compute_sinr_improvement()` - Calculate SINR improvement from adaptation
- `plot_adapted_pattern()` - Visualize quiescent vs adapted patterns

#### Active Impedance/VSWR (Issue #7)
- `active_reflection_coefficient()` - Compute active reflection coefficient with coupling
- `active_impedance()` - Compute active impedance from reflection coefficient
- `vswr_vs_scan()` - Calculate VSWR for all elements versus scan angle
- `mismatch_loss()` - Compute mismatch loss from reflection coefficient
- `active_scan_impedance_matrix()` - Compute active impedance for all elements at scan angle

### Changed
- Updated `__init__.py` to export all 33 new functions
- Version bumped to 1.3.0

### Tests
- Added `tests/test_polarization.py` with 26 tests
- Added `tests/test_coordinates.py` with 15 tests
- Added `TestBeamSpoiling` class to `tests/test_beamforming.py` (7 tests)
- Added `TestAdaptiveBeamforming` class to `tests/test_beamforming.py` (4 tests)
- Added `TestOverlappedSubarrays` class to `tests/test_geometry.py` (8 tests)
- Added `TestActiveImpedance` class to `tests/test_impairments.py` (12 tests)

## [1.2.0] - Previous Release

### Added
- Wideband/TTD (True Time Delay) support
- Subarray configuration in Array Design
- Comprehensive Sphinx documentation

## [1.1.0] - Earlier Release

### Added
- Export module: patterns, weights, geometry, and coupling matrices to
  CSV, JSON, and NPZ formats
- Summary report generation
- Export buttons in the Streamlit app pages

## [1.0.0] - Initial Release

### Added
- Core array factor computation (vectorized and FFT-based)
- Rectangular, triangular, circular, elliptical array geometries
- Conformal array support (cylindrical, spherical)
- Sparse/thinned array generation
- Beamforming with amplitude tapers (Taylor, Chebyshev, etc.)
- Null steering (projection and LCMV methods)
- Multi-beam pattern generation
- Impairment models (mutual coupling, quantization, failures, scan blindness)
- 2D and polar pattern visualization
- Interactive 3D visualization with Plotly
- UV-space pattern representation
