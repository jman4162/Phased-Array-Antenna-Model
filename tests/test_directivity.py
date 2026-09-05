"""
Tests for compute_directivity.

Acceptance tests exercise the public API on grids built by
create_theta_phi_grid. Regression tests pin the behavior corrected in this
change: the rectangular-grid failure, convergence toward analytic answers,
partial-sphere support, and grid validation.

Analytic directivities used throughout (amplitude patterns, since the
function squares its input):

    amplitude 1              -> D = 1
    amplitude sin(theta)     -> D = 3/2      (short dipole)
    amplitude cos(theta)**10 -> D = 21       (cos(theta)**20 power)
    amplitude sqrt(1 + 0.5 sin^2(theta) cos(2 phi)) -> D = 3/2
"""

import ast
import inspect

import numpy as np
import pytest
from scipy.integrate import trapezoid

import phased_array as pa


def _grid(d_theta_deg, theta_range=(0, np.pi), phi_range=(0, 2 * np.pi)):
    """Build an ij-indexed grid with the given angular step in degrees."""
    theta_span = np.rad2deg(theta_range[1] - theta_range[0])
    phi_span = np.rad2deg(phi_range[1] - phi_range[0])
    n_theta = int(round(theta_span / d_theta_deg)) + 1
    n_phi = int(round(phi_span / d_theta_deg)) + 1
    return pa.create_theta_phi_grid(
        theta_range=theta_range,
        phi_range=phi_range,
        n_theta=n_theta,
        n_phi=n_phi,
    )


def _isotropic(theta):
    return np.ones_like(theta)


def _dipole(theta):
    return np.sin(theta)


def _pole_peaked(theta):
    return np.cos(theta) ** 10


def _azimuth_dependent(theta, phi):
    return np.sqrt(1 + 0.5 * np.sin(theta) ** 2 * np.cos(2 * phi))


class TestDirectivityAcceptance:
    """Public-API behavior on the grids create_theta_phi_grid produces."""

    def test_isotropic_on_default_grid(self):
        _, _, theta, phi = pa.create_theta_phi_grid()
        assert theta.shape == (181, 361)

        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))

        assert np.isfinite(directivity)
        assert abs(directivity - 1.0) < 1e-12

    def test_short_dipole_on_default_grid(self):
        _, _, theta, phi = pa.create_theta_phi_grid()

        directivity = pa.compute_directivity(theta, phi, _dipole(theta))

        assert abs(directivity - 1.5) < 1e-3

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("theta_max, expected", [(np.pi, 1.0), (np.pi / 2, 2.0)])
    def test_isotropic_grid_storage_precision(self, dtype, theta_max, expected):
        _, _, theta, phi = pa.create_theta_phi_grid(theta_range=(0, theta_max))
        theta, phi = theta.astype(dtype), phi.astype(dtype)

        directivity = pa.compute_directivity(theta, phi, np.ones_like(theta))

        assert np.isclose(directivity, expected, rtol=1e-6)

    @pytest.mark.parametrize("steering_angle", [0, 90])
    def test_half_wavelength_linear_array(self, steering_angle):
        n = 64
        x = y = np.zeros(n)
        z = np.arange(n) * 0.5
        k = 2 * np.pi
        weights = pa.steering_vector(k, x, y, steering_angle, 0, z=z)
        _, _, theta, phi = pa.create_theta_phi_grid(n_theta=361, n_phi=3)
        pattern = pa.array_factor_vectorized(theta, phi, x, y, weights, k, z=z)

        directivity = pa.compute_directivity(theta, phi, pattern)

        # Half-wavelength-separated cross terms integrate to zero, so the
        # total power is 4*pi*n and exact directivity is n at either steering.
        assert abs(10 * np.log10(directivity / n)) < 0.002

    def test_pole_peaked_on_default_grid(self):
        _, _, theta, phi = pa.create_theta_phi_grid()

        directivity = pa.compute_directivity(theta, phi, _pole_peaked(theta))

        assert abs(directivity - 21.0) < 0.01

    def test_azimuth_dependent_pattern(self):
        _, _, theta, phi = pa.create_theta_phi_grid()

        pattern = _azimuth_dependent(theta, phi)
        directivity = pa.compute_directivity(theta, phi, pattern)

        # Peak power 1.5 sits at theta = pi/2, phi in {0, pi, 2 pi}, all of
        # which are grid points, and the mean power over the sphere is 1.
        assert abs(directivity - 1.5) < 1e-12

    def test_array_pattern_exceeds_isotropic(self):
        geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
        k = pa.wavelength_to_k(1.0)
        weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)
        _, _, theta, phi = pa.create_theta_phi_grid(n_theta=91, n_phi=181)
        pattern = pa.array_factor_vectorized(
            theta, phi, geom.x, geom.y, weights, k
        )

        directivity = pa.compute_directivity(theta, phi, pattern)

        assert np.isfinite(directivity)
        assert directivity > 1.0

    def test_unsupported_grid_raises_instead_of_returning_a_number(self):
        theta_1d, phi_1d, _, _ = pa.create_theta_phi_grid()

        with pytest.raises(ValueError, match="2D grid"):
            pa.compute_directivity(theta_1d, phi_1d, np.ones_like(theta_1d))


class TestDirectivityRegression:
    """Behavior corrected in this change."""

    def test_rectangular_grid_with_unequal_axis_counts(self):
        # Theta weights must be a 1D vector of length n_theta. Building them
        # from the 2D theta grid instead broadcasts against n_phi, which is
        # only silent when n_theta == n_phi.
        _, _, theta, phi = pa.create_theta_phi_grid(n_theta=37, n_phi=181)
        assert theta.shape[0] != theta.shape[1]

        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))

        assert abs(directivity - 1.0) < 1e-12

    def test_grid_dtype_does_not_change_acceptance(self):
        # The uniformity tolerance must key off the axis span, not the array
        # dtype: upcasting a valid float32 grid changes no coordinate value
        # and must not turn an accepted grid into a ValueError.
        _, _, theta, phi = pa.create_theta_phi_grid()
        theta32, phi32 = theta.astype(np.float32), phi.astype(np.float32)
        pattern = np.ones(theta.shape)

        as_float32 = pa.compute_directivity(theta32, phi32, pattern)
        upcast = pa.compute_directivity(
            theta32.astype(np.float64), phi32.astype(np.float64), pattern
        )

        assert np.isclose(as_float32, 1.0, rtol=1e-6)
        assert np.isclose(upcast, 1.0, rtol=1e-6)

    def test_array_like_input_is_accepted(self):
        theta_1d = np.linspace(0, np.pi, 19)
        phi_1d = np.linspace(0, 2 * np.pi, 37)
        theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")

        directivity = pa.compute_directivity(
            theta.tolist(), phi.tolist(), np.ones_like(theta).tolist()
        )

        assert abs(directivity - 1.0) < 1e-12

    def test_integration_avoids_numpy_only_trapezoid_names(self):
        # np.trapz was removed in NumPy 2.4 and np.trapezoid does not exist on
        # NumPy 1.x, so neither name can appear in the integration path. The
        # original PR also called np.trapezoidz, which never existed.
        tree = ast.parse(inspect.getsource(pa.core.compute_directivity))
        numpy_attrs = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "np"
        }

        assert numpy_attrs.isdisjoint({"trapz", "trapezoid", "trapezoidz"})

        _, _, theta, phi = _grid(5.0)
        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))
        assert np.isfinite(directivity)

    def test_isotropic_exact_at_every_spacing(self):
        for d_theta_deg in (10.0, 5.0, 2.5, 1.25):
            _, _, theta, phi = _grid(d_theta_deg)
            directivity = pa.compute_directivity(theta, phi, _isotropic(theta))
            assert abs(directivity - 1.0) < 1e-12, d_theta_deg

    def test_dipole_converges(self):
        errors = []
        for d_theta_deg in (10.0, 5.0, 2.5, 1.25):
            _, _, theta, phi = _grid(d_theta_deg)
            directivity = pa.compute_directivity(theta, phi, _dipole(theta))
            errors.append(abs(directivity - 1.5))

        assert all(b < a for a, b in zip(errors, errors[1:])), errors
        assert errors[-1] < 5e-5, errors[-1]

    def test_pole_peaked_converges(self):
        errors = []
        for d_theta_deg in (10.0, 5.0, 2.5, 1.25):
            _, _, theta, phi = _grid(d_theta_deg)
            pattern = _pole_peaked(theta)
            directivity = pa.compute_directivity(theta, phi, pattern)
            errors.append(abs(directivity - 21.0))

        assert all(b < a for a, b in zip(errors, errors[1:])), errors
        assert errors[-1] < 0.01, errors[-1]

    def test_pole_peaked_quadrature_improves_at_coarse_resolution(self):
        # This particular coarse-grid pattern benefits from cell weighting;
        # the improvement is not a universal property of the quadrature.
        _, _, theta, phi = _grid(10.0)
        pattern = _pole_peaked(theta)

        power = np.abs(pattern) ** 2
        legacy_total = trapezoid(
            trapezoid(power * np.sin(theta), x=phi[0, :], axis=1),
            x=theta[:, 0],
        )
        legacy = 4 * np.pi * power.max() / legacy_total

        directivity = pa.compute_directivity(theta, phi, pattern)

        assert legacy > 21.0 + 1.0
        assert abs(directivity - 21.0) < abs(legacy - 21.0)

    def test_hemisphere_grid_is_supported(self):
        # compute_full_pattern defaults to theta_range=(0, pi/2); a hemisphere
        # grid must integrate the sampled solid angle, not raise.
        _, _, theta, phi = _grid(1.0, theta_range=(0, np.pi / 2))

        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))

        # 4 pi / 2 pi: isotropic power confined to the sampled hemisphere.
        assert abs(directivity - 2.0) < 1e-12

    def test_partial_theta_band_is_supported(self):
        theta_range = (np.pi / 6, np.pi / 3)
        _, _, theta, phi = _grid(0.5, theta_range=theta_range)

        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))

        cap = np.cos(theta_range[0]) - np.cos(theta_range[1])
        solid_angle = 2 * np.pi * cap
        assert abs(directivity - 4 * np.pi / solid_angle) < 1e-9

    def test_narrow_polar_cap_avoids_cancellation(self):
        theta_max = 1e-8
        _, _, theta, phi = pa.create_theta_phi_grid(
            theta_range=(0, theta_max), n_theta=19, n_phi=37
        )
        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))
        expected = 1 / np.sin(theta_max / 2) ** 2
        assert np.isclose(directivity, expected, rtol=1e-12)

    def test_partial_azimuth_sector_is_supported(self):
        _, _, theta, phi = _grid(1.0, phi_range=(0, np.pi))

        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))

        assert abs(directivity - 2.0) < 1e-12

    def test_invariant_under_amplitude_scaling(self):
        _, _, theta, phi = _grid(2.5)
        pattern = _pole_peaked(theta)

        reference = pa.compute_directivity(theta, phi, pattern)

        for scale in (1e-6, 0.5, 3.0, 1e6):
            scaled = pa.compute_directivity(theta, phi, scale * pattern)
            assert np.isclose(scaled, reference, rtol=1e-12)

    def test_invariant_under_complex_phase(self):
        _, _, theta, phi = _grid(2.5)
        pattern = _pole_peaked(theta)

        reference = pa.compute_directivity(theta, phi, pattern)
        phased = pattern * np.exp(1j * (2 * theta + 3 * phi))

        assert np.isclose(
            pa.compute_directivity(theta, phi, phased), reference, rtol=1e-12
        )

    def test_zero_pattern_falls_back_to_unity(self):
        _, _, theta, phi = _grid(5.0)

        assert pa.compute_directivity(theta, phi, np.zeros_like(theta)) == 1.0

    def test_returns_builtin_float(self):
        _, _, theta, phi = _grid(5.0)

        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))
        assert type(directivity) is float


class TestDirectivityValidation:
    """Unsupported grids raise ValueError with an actionable message."""

    def test_one_dimensional_input(self):
        theta = np.linspace(0, np.pi, 19)
        phi = np.linspace(0, 2 * np.pi, 37)

        with pytest.raises(ValueError, match="2D grid"):
            pa.compute_directivity(theta, phi, np.ones(19))

    def test_mismatched_shapes(self):
        _, _, theta, phi = _grid(5.0)

        with pytest.raises(ValueError, match="same shape"):
            pa.compute_directivity(theta, phi, np.ones((3, 4)))

    def test_single_sample_axis(self):
        theta, phi = np.meshgrid(
            np.array([0.5]), np.linspace(0, 2 * np.pi, 37), indexing="ij"
        )

        with pytest.raises(ValueError, match="at least two samples"):
            pa.compute_directivity(theta, phi, np.ones_like(theta))

    def test_nonuniform_theta_spacing(self):
        theta_1d = np.concatenate(
            [
                np.linspace(0, np.pi / 2, 10),
                np.linspace(np.pi / 2, np.pi, 40)[1:],
            ]
        )
        theta, phi = np.meshgrid(
            theta_1d, np.linspace(0, 2 * np.pi, 37), indexing="ij"
        )

        with pytest.raises(ValueError, match="uniformly spaced"):
            pa.compute_directivity(theta, phi, np.ones_like(theta))

    @pytest.mark.parametrize("axis", ["theta", "phi"])
    def test_float32_nonuniform_grid_is_still_rejected(self, axis):
        _, _, theta, phi = pa.create_theta_phi_grid()
        theta, phi = theta.astype(np.float32), phi.astype(np.float32)
        if axis == "theta":
            theta[90, :] += 1e-3
        else:
            phi[:, 180] += 1e-3

        with pytest.raises(ValueError, match="uniformly spaced"):
            pa.compute_directivity(theta, phi, np.ones_like(theta))

    def test_ragged_nested_sequence(self):
        with pytest.raises(ValueError):
            pa.compute_directivity([[0.0, 1.0], [2.0]], [[0.0]], [[1.0]])

    def test_degree_valued_phi_grid(self):
        # A grid built in degrees spans 360 "radians" and would otherwise
        # return D = 0.017, which is not physically reachable.
        _, _, theta, phi = pa.create_theta_phi_grid(
            phi_range=(0, 360.0), n_phi=361
        )

        with pytest.raises(ValueError, match="span at most"):
            pa.compute_directivity(theta, phi, np.ones_like(theta))

    def test_phi_spanning_more_than_one_turn(self):
        _, _, theta, phi = pa.create_theta_phi_grid(
            phi_range=(0, 4 * np.pi), n_phi=721
        )

        with pytest.raises(ValueError, match="span at most"):
            pa.compute_directivity(theta, phi, np.ones_like(theta))

    def test_full_turn_phi_is_accepted(self):
        # The boundary case must not be caught by the span check.
        _, _, theta, phi = pa.create_theta_phi_grid()

        directivity = pa.compute_directivity(theta, phi, _isotropic(theta))

        assert abs(directivity - 1.0) < 1e-12

    def test_descending_theta(self):
        theta, phi = np.meshgrid(
            np.linspace(np.pi, 0, 19),
            np.linspace(0, 2 * np.pi, 37),
            indexing="ij",
        )

        with pytest.raises(ValueError, match="strictly increasing"):
            pa.compute_directivity(theta, phi, np.ones_like(theta))

    def test_xy_indexed_mesh_is_rejected(self):
        theta, phi = np.meshgrid(
            np.linspace(0, np.pi, 19),
            np.linspace(0, 2 * np.pi, 37),
            indexing="xy",
        )

        with pytest.raises(ValueError, match="constant along axis"):
            pa.compute_directivity(theta, phi, np.ones_like(theta))

    def test_theta_outside_zero_to_pi(self):
        theta, phi = np.meshgrid(
            np.linspace(0, 1.5 * np.pi, 19),
            np.linspace(0, 2 * np.pi, 37),
            indexing="ij",
        )

        with pytest.raises(ValueError, match=r"within \[0, pi\]"):
            pa.compute_directivity(theta, phi, np.ones_like(theta))

    def test_non_finite_pattern(self):
        _, _, theta, phi = _grid(5.0)
        pattern = _isotropic(theta)
        pattern[0, 0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            pa.compute_directivity(theta, phi, pattern)

    def test_non_finite_grid(self):
        _, _, theta, phi = _grid(5.0)
        theta = theta.copy()
        theta[0, 0] = np.inf

        with pytest.raises(ValueError, match="non-finite"):
            pa.compute_directivity(theta, phi, _isotropic(theta))
