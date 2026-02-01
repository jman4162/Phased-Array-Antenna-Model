"""
Unit tests for phased_array.coordinates module.
"""

import numpy as np
import pytest
import phased_array as pa


class TestAntennaRadarConversion:
    """Tests for antenna to/from radar coordinate conversion."""

    def test_boresight_conversion(self):
        """Boresight should convert correctly."""
        az, el = pa.antenna_to_radar(0.0, 0.0)
        # At theta=0 (boresight), el should be 90 degrees
        assert np.isclose(el, np.pi/2, atol=1e-10)

    def test_round_trip_conversion(self):
        """Round-trip conversion should recover original angles."""
        theta_orig = np.pi/4
        phi_orig = np.pi/3

        az, el = pa.antenna_to_radar(theta_orig, phi_orig)
        theta, phi = pa.radar_to_antenna(az, el)

        assert np.isclose(theta, theta_orig, atol=1e-10)
        assert np.isclose(np.cos(phi - phi_orig), 1.0, atol=1e-10)

    def test_array_input(self):
        """Should handle array inputs."""
        theta = np.array([0, np.pi/6, np.pi/4])
        phi = np.array([0, np.pi/4, np.pi/2])

        az, el = pa.antenna_to_radar(theta, phi)

        assert az.shape == theta.shape
        assert el.shape == theta.shape


class TestConeClockConversion:
    """Tests for cone/clock coordinate conversion."""

    def test_identity_conversion(self):
        """Cone/clock should be same as theta/phi for standard definition."""
        theta = np.pi/6
        phi = np.pi/4

        cone, clock = pa.antenna_to_cone(theta, phi)

        assert np.isclose(cone, theta)
        assert np.isclose(clock, phi)

    def test_round_trip(self):
        """Round-trip conversion should recover original."""
        cone_orig = np.pi/6
        clock_orig = np.pi/4

        theta, phi = pa.cone_to_antenna(cone_orig, clock_orig)
        cone, clock = pa.antenna_to_cone(theta, phi)

        assert np.isclose(cone, cone_orig)
        assert np.isclose(clock, clock_orig)


class TestRotationMatrices:
    """Tests for rotation matrices."""

    def test_roll_identity(self):
        """Zero roll should give identity matrix."""
        R = pa.rotation_matrix_roll(0.0)
        assert np.allclose(R, np.eye(3))

    def test_pitch_identity(self):
        """Zero pitch should give identity matrix."""
        R = pa.rotation_matrix_pitch(0.0)
        assert np.allclose(R, np.eye(3))

    def test_yaw_identity(self):
        """Zero yaw should give identity matrix."""
        R = pa.rotation_matrix_yaw(0.0)
        assert np.allclose(R, np.eye(3))

    def test_roll_90(self):
        """90-degree roll should rotate y to z."""
        R = pa.rotation_matrix_roll(np.pi/2)
        v = np.array([0, 1, 0])
        v_rot = R @ v
        assert np.allclose(v_rot, [0, 0, 1], atol=1e-10)

    def test_pitch_90(self):
        """90-degree pitch should rotate z to x."""
        R = pa.rotation_matrix_pitch(np.pi/2)
        v = np.array([0, 0, 1])
        v_rot = R @ v
        assert np.allclose(v_rot, [1, 0, 0], atol=1e-10)

    def test_yaw_90(self):
        """90-degree yaw should rotate x to y."""
        R = pa.rotation_matrix_yaw(np.pi/2)
        v = np.array([1, 0, 0])
        v_rot = R @ v
        assert np.allclose(v_rot, [0, 1, 0], atol=1e-10)

    def test_orthogonality(self):
        """Rotation matrices should be orthogonal."""
        for angle in [np.pi/6, np.pi/4, np.pi/3]:
            for R in [pa.rotation_matrix_roll(angle),
                      pa.rotation_matrix_pitch(angle),
                      pa.rotation_matrix_yaw(angle)]:
                assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
                assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestRotatePattern:
    """Tests for pattern rotation."""

    def test_identity_rotation(self):
        """Zero rotation should preserve pattern."""
        theta = np.linspace(0, np.pi/2, 10)
        phi = np.linspace(0, 2*np.pi, 10)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        pattern = np.cos(theta_grid)

        theta_r, phi_r, pattern_r = pa.rotate_pattern(
            theta_grid, phi_grid, pattern,
            roll_deg=0, pitch_deg=0, yaw_deg=0
        )

        assert theta_r.shape == theta_grid.shape
        # Pattern should be approximately preserved (may differ slightly due to interpolation)
        assert np.allclose(pattern_r, pattern, atol=0.1)

    def test_output_shape(self):
        """Rotated output should have same shape as input."""
        theta = np.linspace(0, np.pi/2, 46)
        phi = np.linspace(0, 2*np.pi, 73)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        pattern = np.cos(theta_grid) + 0j

        theta_r, phi_r, pattern_r = pa.rotate_pattern(
            theta_grid, phi_grid, pattern,
            roll_deg=10, pitch_deg=20, yaw_deg=30
        )

        assert theta_r.shape == theta_grid.shape
        assert phi_r.shape == phi_grid.shape
        assert pattern_r.shape == pattern.shape

    def test_rotation_changes_pattern(self):
        """Non-zero rotation should change pattern values."""
        theta = np.linspace(0, np.pi/2, 10)
        phi = np.linspace(0, 2*np.pi, 10)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        pattern = np.cos(theta_grid)

        _, _, pattern_r = pa.rotate_pattern(
            theta_grid, phi_grid, pattern,
            roll_deg=45, pitch_deg=0, yaw_deg=0
        )

        # Pattern should be different due to rotation
        assert not np.allclose(pattern_r, pattern)
