# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Export functionality for patterns and configurations
- Comprehensive Sphinx documentation

## [1.1.0] - Earlier Release

### Added
- Interactive 3D visualization with Plotly
- UV-space pattern representation
- Conformal array support (cylindrical, spherical)
- Sparse/thinned array generation

## [1.0.0] - Initial Release

### Added
- Core array factor computation (vectorized and FFT-based)
- Rectangular, triangular, circular, elliptical array geometries
- Beamforming with amplitude tapers (Taylor, Chebyshev, etc.)
- Null steering (projection and LCMV methods)
- Multi-beam pattern generation
- Impairment models (mutual coupling, quantization, failures, scan blindness)
- 2D and polar pattern visualization
