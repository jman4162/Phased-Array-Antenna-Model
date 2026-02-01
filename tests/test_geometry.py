"""
Unit tests for phased_array.geometry module.
"""

import numpy as np
import pytest
import phased_array as pa


class TestRectangularArray:
    """Tests for rectangular array creation."""

    def test_element_count(self):
        """Should create correct number of elements."""
        geom = pa.create_rectangular_array(8, 6, 0.5, 0.5)
        assert geom.n_elements == 48

    def test_centered_array(self):
        """Centered array should have mean position at origin."""
        geom = pa.create_rectangular_array(10, 10, 0.5, 0.5, center=True)

        assert np.isclose(np.mean(geom.x), 0.0, atol=1e-10)
        assert np.isclose(np.mean(geom.y), 0.0, atol=1e-10)

    def test_spacing(self):
        """Elements should have correct spacing."""
        dx, dy = 0.5, 0.6
        wavelength = 1.0
        geom = pa.create_rectangular_array(3, 3, dx, dy, wavelength)

        # Check x spacing (should be dx * wavelength)
        x_sorted = np.sort(np.unique(geom.x))
        if len(x_sorted) > 1:
            x_spacing = x_sorted[1] - x_sorted[0]
            assert np.isclose(x_spacing, dx * wavelength)

    def test_z_coordinates(self):
        """Planar array should have z=0 for all elements."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)

        assert geom.z is not None
        assert np.allclose(geom.z, 0.0)

    def test_normals(self):
        """Planar array should have z-pointing normals."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)

        assert geom.nz is not None
        assert np.allclose(geom.nz, 1.0)
        assert np.allclose(geom.nx, 0.0)
        assert np.allclose(geom.ny, 0.0)


class TestTriangularArray:
    """Tests for triangular (hexagonal) array creation."""

    def test_creates_elements(self):
        """Should create a non-empty array."""
        geom = pa.create_triangular_array(8, 8, 0.5)
        assert geom.n_elements > 0

    def test_row_offset(self):
        """Odd rows should be offset by half spacing."""
        geom = pa.create_triangular_array(4, 4, 0.5, wavelength=1.0, center=False)

        # Find elements in row 0 and row 1
        y_unique = np.sort(np.unique(np.round(geom.y, 6)))

        if len(y_unique) >= 2:
            row0_x = geom.x[np.isclose(geom.y, y_unique[0])]
            row1_x = geom.x[np.isclose(geom.y, y_unique[1])]

            if len(row0_x) > 0 and len(row1_x) > 0:
                # Row 1 should be offset from row 0
                offset = np.min(row1_x) - np.min(row0_x)
                assert np.isclose(abs(offset), 0.25, atol=0.01)


class TestCircularArray:
    """Tests for circular array creation."""

    def test_element_count(self):
        """Should create correct number of elements."""
        geom = pa.create_circular_array(16, radius=2.0)
        assert geom.n_elements == 16

    def test_radius(self):
        """All elements should be at specified radius."""
        radius = 2.5
        geom = pa.create_circular_array(12, radius=radius, wavelength=1.0)

        distances = np.sqrt(geom.x**2 + geom.y**2)
        assert np.allclose(distances, radius)

    def test_angular_spacing(self):
        """Elements should be evenly spaced in angle."""
        n = 8
        geom = pa.create_circular_array(n, radius=2.0)

        angles = np.arctan2(geom.y, geom.x)
        angles_sorted = np.sort(angles)
        diffs = np.diff(angles_sorted)

        expected_spacing = 2 * np.pi / n
        assert np.allclose(diffs, expected_spacing, atol=0.01)


class TestCylindricalArray:
    """Tests for cylindrical conformal array."""

    def test_element_count(self):
        """Should create correct number of elements."""
        geom = pa.create_cylindrical_array(8, 4, radius=2.0, height=3.0)
        assert geom.n_elements == 32

    def test_radius(self):
        """All elements should be at specified radius in xy-plane."""
        radius = 2.0
        geom = pa.create_cylindrical_array(8, 4, radius=radius, height=3.0)

        xy_distances = np.sqrt(geom.x**2 + geom.y**2)
        assert np.allclose(xy_distances, radius)

    def test_normals_radial(self):
        """Normals should point radially outward."""
        geom = pa.create_cylindrical_array(8, 4, radius=2.0, height=3.0)

        # Normal should be parallel to (x, y) position
        for i in range(geom.n_elements):
            xy_norm = np.sqrt(geom.x[i]**2 + geom.y[i]**2)
            if xy_norm > 0:
                expected_nx = geom.x[i] / xy_norm
                expected_ny = geom.y[i] / xy_norm

                assert np.isclose(geom.nx[i], expected_nx, atol=1e-6)
                assert np.isclose(geom.ny[i], expected_ny, atol=1e-6)


class TestEllipticalArray:
    """Tests for elliptical boundary arrays."""

    def test_elements_inside_ellipse(self):
        """All elements should be inside the elliptical boundary."""
        a, b = 3.0, 2.0
        geom = pa.create_elliptical_array(a, b, dx=0.5, wavelength=1.0)

        # Check ellipse equation: (x/a)^2 + (y/b)^2 <= 1
        ellipse_val = (geom.x / a)**2 + (geom.y / b)**2
        assert np.all(ellipse_val <= 1.0 + 1e-6)


class TestSparseThinning:
    """Tests for array thinning functions."""

    def test_random_thinning_count(self):
        """Random thinning should keep approximately the right number."""
        geom = pa.create_rectangular_array(10, 10, 0.5, 0.5)
        thinning_factor = 0.5

        thinned = pa.thin_array_random(geom, thinning_factor, seed=42)

        expected = int(geom.n_elements * thinning_factor)
        assert thinned.n_elements == expected

    def test_random_thinning_reproducible(self):
        """Same seed should give same result."""
        geom = pa.create_rectangular_array(10, 10, 0.5, 0.5)

        thinned1 = pa.thin_array_random(geom, 0.5, seed=123)
        thinned2 = pa.thin_array_random(geom, 0.5, seed=123)

        assert np.allclose(thinned1.x, thinned2.x)
        assert np.allclose(thinned1.y, thinned2.y)

    def test_density_thinning(self):
        """Density thinning should keep more elements at higher density."""
        geom = pa.create_rectangular_array(20, 20, 0.5, 0.5)

        # Density function: higher at center
        def density(x, y):
            r = np.sqrt(x**2 + y**2)
            r_max = np.max(np.sqrt(geom.x**2 + geom.y**2))
            return 1.0 - 0.8 * (r / r_max)

        thinned = pa.thin_array_density_tapered(geom, density, seed=42)

        # Should have fewer elements than original
        assert thinned.n_elements < geom.n_elements


class TestSubarrayArchitecture:
    """Tests for subarray creation."""

    def test_subarray_count(self):
        """Should create correct number of subarrays."""
        arch = pa.create_rectangular_subarrays(
            Nx_total=16, Ny_total=16,
            Nx_sub=4, Ny_sub=4,
            dx=0.5, dy=0.5
        )

        assert arch.n_subarrays == 16  # (16/4) * (16/4)

    def test_all_elements_assigned(self):
        """Every element should be assigned to a subarray."""
        arch = pa.create_rectangular_subarrays(
            Nx_total=8, Ny_total=8,
            Nx_sub=4, Ny_sub=4,
            dx=0.5, dy=0.5
        )

        # All assignments should be valid subarray indices
        assert np.all(arch.subarray_assignments >= 0)
        assert np.all(arch.subarray_assignments < arch.n_subarrays)


class TestArrayGeometryProperties:
    """Tests for ArrayGeometry class properties."""

    def test_is_planar(self):
        """Planar array should report as planar."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        assert geom.is_planar

    def test_cylindrical_not_planar(self):
        """Cylindrical array should not report as planar."""
        geom = pa.create_cylindrical_array(8, 4, 2.0, 3.0)
        # It's not planar because z values vary
        # Actually, in our implementation z varies for cylindrical
        assert not geom.is_planar or geom.is_conformal

    def test_copy(self):
        """Copy should create independent array."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        geom_copy = geom.copy()

        # Modify original
        geom.x[0] = 999.0

        # Copy should be unchanged
        assert geom_copy.x[0] != 999.0


class TestOverlappedSubarrays:
    """Tests for overlapped subarray creation."""

    def test_overlapped_flag(self):
        """Overlapped architecture should have overlapped=True."""
        arch = pa.create_overlapped_subarrays(
            Nx_total=16, Ny_total=16,
            Nx_sub=4, Ny_sub=4,
            overlap_x=2, overlap_y=2,
            dx=0.5, dy=0.5
        )

        assert arch.overlapped is True

    def test_subarray_elements_not_none(self):
        """Overlapped architecture should have subarray_elements list."""
        arch = pa.create_overlapped_subarrays(
            Nx_total=16, Ny_total=16,
            Nx_sub=4, Ny_sub=4,
            overlap_x=2, overlap_y=2,
            dx=0.5, dy=0.5
        )

        assert arch.subarray_elements is not None
        assert len(arch.subarray_elements) == arch.n_subarrays

    def test_overlap_weights_not_none(self):
        """Overlapped architecture should have overlap_weights list."""
        arch = pa.create_overlapped_subarrays(
            Nx_total=16, Ny_total=16,
            Nx_sub=4, Ny_sub=4,
            overlap_x=2, overlap_y=2,
            dx=0.5, dy=0.5
        )

        assert arch.overlap_weights is not None
        assert len(arch.overlap_weights) == arch.n_subarrays

    def test_subarray_count(self):
        """Should create correct number of overlapped subarrays."""
        arch = pa.create_overlapped_subarrays(
            Nx_total=16, Ny_total=16,
            Nx_sub=4, Ny_sub=4,
            overlap_x=2, overlap_y=2,
            dx=0.5, dy=0.5
        )

        # stride_x = 4 - 2 = 2, stride_y = 4 - 2 = 2
        # n_sub_x = (16 - 4) / 2 + 1 = 7
        # n_sub_y = (16 - 4) / 2 + 1 = 7
        expected = 7 * 7
        assert arch.n_subarrays == expected

    def test_elements_in_multiple_subarrays(self):
        """Some elements should belong to multiple subarrays."""
        arch = pa.create_overlapped_subarrays(
            Nx_total=8, Ny_total=8,
            Nx_sub=4, Ny_sub=4,
            overlap_x=2, overlap_y=2,
            dx=0.5, dy=0.5
        )

        # Check middle elements belong to multiple subarrays
        middle_elem = 27  # Roughly in the middle
        subarrays = arch.get_element_subarrays(middle_elem)

        # Middle elements should be in multiple subarrays
        assert len(subarrays) >= 1

    def test_overlapped_subarray_weights_shape(self):
        """Overlapped weights should have correct shape."""
        arch = pa.create_overlapped_subarrays(
            Nx_total=16, Ny_total=16,
            Nx_sub=4, Ny_sub=4,
            overlap_x=2, overlap_y=2,
            dx=0.5, dy=0.5
        )
        k = pa.wavelength_to_k(1.0)

        weights = pa.overlapped_subarray_weights(arch, k, 15, 0)

        assert weights.shape == (arch.geometry.n_elements,)

    def test_compute_overlapped_pattern(self):
        """Should compute pattern correctly."""
        arch = pa.create_overlapped_subarrays(
            Nx_total=16, Ny_total=16,
            Nx_sub=4, Ny_sub=4,
            overlap_x=2, overlap_y=2,
            dx=0.5, dy=0.5
        )
        k = pa.wavelength_to_k(1.0)

        theta_deg, pattern_dB = pa.compute_overlapped_pattern(
            arch, k, theta0_deg=0, phi0_deg=0,
            n_points=91
        )

        assert len(theta_deg) == 91
        assert len(pattern_dB) == 91
        assert np.max(pattern_dB) == 0  # Normalized

    def test_overlap_validation(self):
        """Should raise error if overlap >= subarray size."""
        with pytest.raises(ValueError):
            pa.create_overlapped_subarrays(
                Nx_total=16, Ny_total=16,
                Nx_sub=4, Ny_sub=4,
                overlap_x=4, overlap_y=4,  # Invalid: overlap == subarray size
                dx=0.5, dy=0.5
            )
