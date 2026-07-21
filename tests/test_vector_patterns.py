"""Tests for phased_array.vector_patterns."""

import numpy as np
import pytest

import phased_array as pa
from phased_array.polarization import ludwig3_decomposition
from phased_array.vector_patterns import (
    GriddedElementPattern,
    VectorPattern,
    compute_co_cross_pattern_cuts,
    compute_full_vector_pattern,
    cos_q_polarized_element,
    crossed_dipole_element,
    dipole_element,
    dual_pol_weights,
    element_rotation_matrices,
    global_to_local_angles,
    ideal_patch_element,
    vector_array_factor_conformal,
    vector_total_pattern,
)


@pytest.fixture(scope="module")
def small_array():
    geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    k = pa.wavelength_to_k(1.0)
    weights = np.ones(16, dtype=complex)
    return geom, k, weights


class TestElementFactories:
    def test_dipole_x_boresight_copol(self):
        f = dipole_element("x")
        E_theta, E_phi = f(np.array([0.0]), np.array([0.0]))
        assert np.isclose(abs(E_theta[0]), 1.0)
        assert np.isclose(abs(E_phi[0]), 0.0)

    def test_dipole_x_zero_crosspol_principal_planes(self):
        f = dipole_element("x")
        theta = np.deg2rad(np.array([0.0, 30.0, 60.0]))
        for phi_val in (0.0, np.pi / 2):
            phi = np.full_like(theta, phi_val)
            E_theta, E_phi = f(theta, phi)
            _, E_cross = ludwig3_decomposition(theta, phi, E_theta, E_phi)
            assert np.allclose(np.abs(E_cross), 0.0, atol=1e-12)

    def test_dipole_z_pattern(self):
        f = dipole_element("z")
        theta = np.array([0.0, np.pi / 2])
        E_theta, E_phi = f(theta, np.zeros(2))
        assert np.isclose(abs(E_theta[0]), 0.0)  # null along axis
        assert np.isclose(abs(E_theta[1]), 1.0)  # peak broadside to axis
        assert np.allclose(np.abs(E_phi), 0.0)

    def test_dipole_invalid_orientation(self):
        with pytest.raises(ValueError):
            dipole_element("w")

    def test_ideal_patch_zero_crosspol_everywhere(self):
        f = ideal_patch_element("x", cos_exp=1.5)
        theta = np.deg2rad(np.linspace(0, 89, 10))
        phi = np.deg2rad(np.linspace(0, 350, 10))
        E_theta, E_phi = f(theta, phi)
        _, E_cross = ludwig3_decomposition(theta, phi, E_theta, E_phi)
        assert np.allclose(np.abs(E_cross), 0.0, atol=1e-12)

    def test_ideal_patch_backlobe_zero(self):
        f = ideal_patch_element("x")
        E_theta, E_phi = f(np.array([2.0]), np.array([0.0]))  # theta > 90 deg
        assert abs(E_theta[0]) == 0.0
        assert abs(E_phi[0]) == 0.0

    def test_ideal_patch_y_copol(self):
        f = ideal_patch_element("y")
        theta = np.array([0.3])
        phi = np.array([1.1])
        E_theta, E_phi = f(theta, phi)
        E_co, E_cross = ludwig3_decomposition(theta, phi - np.pi / 2, E_theta, E_phi)
        # y-pol element: cross term of the x-referenced basis carries it
        _, E_cross_x = ludwig3_decomposition(theta, phi, E_theta, E_phi)
        assert abs(E_cross_x[0]) > 0.9 * np.cos(theta[0])

    def test_crossed_dipole_boresight_circular(self):
        f = crossed_dipole_element(-np.pi / 2)
        E_theta, E_phi = f(np.array([0.0]), np.array([0.0]))
        jones = np.stack([E_theta, E_phi])
        ar = pa.axial_ratio(jones)
        assert np.isclose(ar[0], 1.0, atol=1e-10)

    def test_cos_q_matches_scalar_element_pattern(self):
        q = 2.0
        f = cos_q_polarized_element(q_theta=q)
        theta = np.deg2rad(np.linspace(0, 89, 20))
        phi = np.deg2rad(np.linspace(0, 340, 20))
        E_theta, E_phi = f(theta, phi)
        power = np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2
        scalar = pa.element_pattern(theta, phi, cos_exp_theta=2 * q)
        assert np.allclose(power, scalar, atol=1e-12)

    def test_cos_q_gain_scale(self):
        f = cos_q_polarized_element(max_gain_dBi=6.0)
        E_theta, E_phi = f(np.array([0.0]), np.array([0.0]))
        power_dB = 10 * np.log10(abs(E_theta[0]) ** 2 + abs(E_phi[0]) ** 2)
        assert np.isclose(power_dB, 6.0)

    def test_cos_q_zero_jones_raises(self):
        with pytest.raises(ValueError):
            cos_q_polarized_element(jones=np.array([0.0, 0.0]))


class TestVectorTotalPattern:
    def test_boresight_peak(self, small_array):
        geom, k, weights = small_array
        f = ideal_patch_element("x")
        E_theta, E_phi = vector_total_pattern(
            np.array([0.0]), np.array([0.0]), geom.x, geom.y, weights, k, f
        )
        assert np.isclose(abs(E_theta[0]), 16.0)

    def test_matches_scalar_path(self, small_array):
        geom, k, weights = small_array
        q = 1.5
        f = ideal_patch_element("x", cos_exp=q)
        _, _, THETA, PHI = pa.create_theta_phi_grid(
            theta_range=(0, np.pi / 2), n_theta=19, n_phi=37
        )
        E_theta, E_phi = vector_total_pattern(
            THETA, PHI, geom.x, geom.y, weights, k, f
        )
        vec_power = np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2
        AF = pa.array_factor_vectorized(THETA, PHI, geom.x, geom.y, weights, k)
        expected = np.abs(AF) ** 2 * np.where(
            np.cos(THETA) > 0, np.cos(THETA) ** (2 * q), 0.0
        )
        assert np.allclose(vec_power, expected, atol=1e-9)

    def test_shapes_1d_2d(self, small_array):
        geom, k, weights = small_array
        f = dipole_element("x")
        E_theta, _ = vector_total_pattern(
            np.zeros(5), np.zeros(5), geom.x, geom.y, weights, k, f
        )
        assert E_theta.shape == (5,)
        E_theta, _ = vector_total_pattern(
            np.zeros((3, 4)), np.zeros((3, 4)), geom.x, geom.y, weights, k, f
        )
        assert E_theta.shape == (3, 4)


class TestVectorPatternContainer:
    def test_full_pattern_and_power(self, small_array):
        geom, k, weights = small_array
        vp = compute_full_vector_pattern(
            geom.x, geom.y, weights, k, n_theta=19, n_phi=37
        )
        assert isinstance(vp, VectorPattern)
        assert vp.E_theta.shape == (19, 37)
        assert np.isclose(np.max(vp.power_dB()), 0.0)

    def test_xpd_ideal_patch_interior(self, small_array):
        geom, k, weights = small_array
        vp = compute_full_vector_pattern(
            geom.x, geom.y, weights, k, n_theta=19, n_phi=37,
            theta_range=(0, np.pi / 3),
        )
        xpd = vp.xpd_map()
        # Ideal patch has zero cross-pol: XPD is enormous wherever the
        # field is significant (at pattern nulls both components are
        # floating-point noise, so exclude them)
        significant = vp.power > 1e-3 * np.max(vp.power)
        assert np.min(xpd[significant]) > 100.0

    def test_axial_ratio_crossed_dipole(self, small_array):
        geom, k, weights = small_array
        vp = compute_full_vector_pattern(
            geom.x, geom.y, weights, k,
            element_func=crossed_dipole_element(), n_theta=19, n_phi=37,
        )
        ar = vp.axial_ratio_map()
        assert np.isclose(ar[0, 0], 1.0, atol=1e-6)

    def test_co_cross_reference_y(self, small_array):
        geom, k, weights = small_array
        vp = compute_full_vector_pattern(
            geom.x, geom.y, weights, k,
            element_func=ideal_patch_element("y"), n_theta=19, n_phi=37,
        )
        E_co_y, E_cross_y = vp.co_cross(reference_pol="ludwig3-y")
        # y-pol element is pure co-pol in the y-referenced basis
        assert np.max(np.abs(E_cross_y)) < 1e-9 * np.max(np.abs(E_co_y))

    def test_co_cross_invalid_reference(self, small_array):
        geom, k, weights = small_array
        vp = compute_full_vector_pattern(
            geom.x, geom.y, weights, k, n_theta=5, n_phi=9
        )
        with pytest.raises(ValueError):
            vp.co_cross("bogus")


class TestCoCrossCuts:
    def test_copol_peak_normalized(self, small_array):
        geom, k, weights = small_array
        theta_deg, co_dB, cross_dB = compute_co_cross_pattern_cuts(
            geom.x, geom.y, weights, k, dipole_element("x"), phi_cut_deg=45.0
        )
        assert theta_deg.shape == co_dB.shape == cross_dB.shape
        assert np.isclose(np.max(co_dB), 0.0)
        # Diagonal-plane dipole cross-pol stays below co-pol peak
        assert np.max(cross_dB) < 0.0

    def test_principal_plane_low_crosspol(self, small_array):
        geom, k, weights = small_array
        _, _, cross_dB = compute_co_cross_pattern_cuts(
            geom.x, geom.y, weights, k, dipole_element("x"), phi_cut_deg=0.0
        )
        assert np.max(cross_dB) < -90.0


class TestDualPolWeights:
    def test_circular_split(self):
        w = np.ones(8, dtype=complex)
        jones = pa.jones_vector(1.0, 1.0, phase_diff=-np.pi / 2)
        wx, wy = dual_pol_weights(w, w, jones)
        assert np.allclose(np.abs(wx), 1 / np.sqrt(2))
        assert np.allclose(np.abs(wy), 1 / np.sqrt(2))
        assert np.allclose(np.angle(wy / wx), -np.pi / 2)

    def test_zero_goal_raises(self):
        with pytest.raises(ValueError):
            dual_pol_weights(np.ones(4), np.ones(4), np.zeros(2))


class TestRotationHelpers:
    def test_planar_identity(self):
        geom = pa.create_rectangular_array(3, 3, dx=0.5, dy=0.5)
        R = element_rotation_matrices(geom)
        assert R.shape == (9, 3, 3)
        assert np.allclose(R, np.eye(3))

    def test_identity_angles_roundtrip(self):
        theta = np.deg2rad(np.array([10.0, 45.0, 80.0]))
        phi = np.deg2rad(np.array([0.0, 120.0, 300.0]))
        lt, lp = global_to_local_angles(np.eye(3), theta, phi)
        assert np.allclose(lt, theta)
        # phi may wrap by 2*pi
        assert np.allclose(np.mod(lp, 2 * np.pi), np.mod(phi, 2 * np.pi))

    def test_tilted_element_known_value(self):
        # Element normal along +x: global +x direction is local boresight
        geom = pa.ArrayGeometry(
            x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
            nx=np.ones(1), ny=np.zeros(1), nz=np.zeros(1),
        )
        R = element_rotation_matrices(geom)
        lt, lp = global_to_local_angles(
            R[0], np.array([np.pi / 2]), np.array([0.0])
        )
        assert np.isclose(lt[0], 0.0, atol=1e-12)

    def test_tangent_override(self):
        geom = pa.ArrayGeometry(
            x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
            nx=np.zeros(1), ny=np.zeros(1), nz=np.ones(1),
            tx=np.zeros(1), ty=np.ones(1), tz=np.zeros(1),
        )
        R = element_rotation_matrices(geom)
        # local x-axis is global +y
        assert np.allclose(R[0, 0], [0.0, 1.0, 0.0])

    def test_geometry_copy_preserves_tangents(self):
        geom = pa.ArrayGeometry(
            x=np.zeros(2), y=np.zeros(2), z=np.zeros(2),
            nx=np.zeros(2), ny=np.zeros(2), nz=np.ones(2),
            tx=np.ones(2), ty=np.zeros(2), tz=np.zeros(2),
        )
        copied = geom.copy()
        assert np.allclose(copied.tx, geom.tx)
        assert copied.tx is not geom.tx


class TestConformalVector:
    def test_planar_equals_planar_path(self, small_array):
        geom, k, weights = small_array
        f = ideal_patch_element("x", cos_exp=1.5)
        theta = np.deg2rad(np.linspace(0, 80, 9))
        phi = np.full_like(theta, np.deg2rad(20))
        Et_c, Ep_c = vector_array_factor_conformal(
            theta, phi, geom, weights, k, element_func=f
        )
        Et_p, Ep_p = vector_total_pattern(
            theta, phi, geom.x, geom.y, weights, k, f
        )
        assert np.allclose(Et_c, Et_p, atol=1e-9)
        assert np.allclose(Ep_c, Ep_p, atol=1e-9)

    def test_cylindrical_smoke(self):
        cyl = pa.create_cylindrical_array(8, 4, radius=1.0, height=2.0)
        k = pa.wavelength_to_k(1.0)
        w = np.ones(cyl.n_elements, dtype=complex)
        Et, Ep = vector_array_factor_conformal(
            np.array([np.pi / 3]), np.array([0.2]), cyl, w, k
        )
        assert np.isfinite(Et).all() and np.isfinite(Ep).all()
        assert abs(Et[0]) + abs(Ep[0]) > 0

    def test_per_element_funcs(self, small_array):
        geom, k, weights = small_array
        funcs = [ideal_patch_element("x")] * geom.n_elements
        Et_a, Ep_a = vector_array_factor_conformal(
            np.array([0.2]), np.array([0.4]), geom, weights, k,
            per_element_funcs=funcs,
        )
        Et_b, Ep_b = vector_array_factor_conformal(
            np.array([0.2]), np.array([0.4]), geom, weights, k,
            element_func=ideal_patch_element("x"),
        )
        assert np.allclose(Et_a, Et_b)
        assert np.allclose(Ep_a, Ep_b)

    def test_per_element_funcs_wrong_length(self, small_array):
        geom, k, weights = small_array
        with pytest.raises(ValueError):
            vector_array_factor_conformal(
                np.array([0.2]), np.array([0.4]), geom, weights, k,
                per_element_funcs=[ideal_patch_element("x")],
            )

    def test_scalar_conformal_regression(self):
        # array_factor_conformal default cos-pattern and cos^1 element
        # func agree after the local-frame fix
        cyl = pa.create_cylindrical_array(8, 4, radius=1.0, height=2.0)
        k = pa.wavelength_to_k(1.0)
        w = np.ones(cyl.n_elements, dtype=complex)
        af_default = pa.array_factor_conformal(
            np.array([0.3]), np.array([0.1]), cyl, w, k
        )
        af_cos = pa.array_factor_conformal(
            np.array([0.3]), np.array([0.1]), cyl, w, k,
            element_pattern_func=pa.element_pattern, cos_exp_theta=1.0,
        )
        assert np.allclose(af_default, af_cos, atol=1e-9)


class TestGriddedElementPattern:
    def test_reproduces_analytic(self):
        f = ideal_patch_element("x", cos_exp=1.5)
        theta_grid = np.linspace(0, np.pi / 2, 91)
        phi_grid = np.linspace(0, 2 * np.pi, 181)
        THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing="ij")
        E_theta, E_phi = f(THETA, PHI)
        gridded = GriddedElementPattern(theta_grid, phi_grid, E_theta, E_phi)
        theta_test = np.deg2rad(np.array([13.0, 47.0]))
        phi_test = np.deg2rad(np.array([33.0, 250.0]))
        Et_i, Ep_i = gridded(theta_test, phi_test)
        Et_a, Ep_a = f(theta_test, phi_test)
        assert np.allclose(Et_i, Et_a, atol=2e-3)
        assert np.allclose(Ep_i, Ep_a, atol=2e-3)

    def test_from_scalar(self):
        theta_grid = np.linspace(0, np.pi / 2, 46)
        phi_grid = np.linspace(0, 2 * np.pi, 91)
        THETA, _ = np.meshgrid(theta_grid, phi_grid, indexing="ij")
        gridded = GriddedElementPattern.from_scalar(
            theta_grid, phi_grid, np.cos(THETA)
        )
        E_theta, E_phi = gridded(np.array([0.0]), np.array([0.0]))
        assert np.isclose(abs(E_theta[0]) ** 2 + abs(E_phi[0]) ** 2, 1.0, atol=1e-6)

    def test_usable_in_array_pattern(self, small_array):
        geom, k, weights = small_array
        theta_grid = np.linspace(0, np.pi / 2, 46)
        phi_grid = np.linspace(0, 2 * np.pi, 91)
        THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing="ij")
        E_theta, E_phi = ideal_patch_element("x")(THETA, PHI)
        gridded = GriddedElementPattern(theta_grid, phi_grid, E_theta, E_phi)
        Et, Ep = vector_total_pattern(
            np.array([0.1]), np.array([0.2]), geom.x, geom.y, weights, k, gridded
        )
        assert np.isfinite(Et).all()
