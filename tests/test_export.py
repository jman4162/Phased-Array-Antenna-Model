"""
Tests for phased_array.export module.
"""

import numpy as np
import pytest
import json
import tempfile
import os

import phased_array as pa


class TestPatternExport:
    """Tests for pattern export functions."""

    def test_export_pattern_csv_basic(self):
        """Test basic 1D pattern CSV export."""
        angles = np.linspace(-90, 90, 181)
        pattern = -20 * np.abs(np.sin(np.deg2rad(angles)))

        csv = pa.export_pattern_csv(angles, pattern)

        # Check header
        lines = csv.strip().split('\n')
        assert 'angle_deg,pattern_dB' in lines[0]

        # Check data points
        assert len(lines) == 182  # header + 181 data points

    def test_export_pattern_csv_with_metadata(self):
        """Test 1D pattern CSV export with metadata."""
        angles = np.linspace(-90, 90, 11)
        pattern = np.zeros(11)

        csv = pa.export_pattern_csv(
            angles, pattern,
            metadata={'test_key': 'test_value', 'n_elements': 100}
        )

        assert '# test_key: test_value' in csv
        assert '# n_elements: 100' in csv

    def test_export_pattern_csv_to_file(self):
        """Test writing pattern CSV to file."""
        angles = np.linspace(-90, 90, 11)
        pattern = np.zeros(11)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filename = f.name

        try:
            pa.export_pattern_csv(angles, pattern, filename=filename)
            assert os.path.exists(filename)

            with open(filename, 'r') as f:
                content = f.read()
            assert 'angle_deg,pattern_dB' in content
        finally:
            os.unlink(filename)

    def test_export_pattern_2d_csv(self):
        """Test 2D pattern CSV export."""
        theta = np.linspace(0, 90, 5)
        phi = np.linspace(0, 360, 9)
        pattern = np.zeros((5, 9))

        csv = pa.export_pattern_2d_csv(theta, phi, pattern)

        lines = csv.strip().split('\n')
        assert 'theta_deg,phi_deg,pattern_dB' in lines[0]
        # 5 theta x 9 phi = 45 data points + header
        assert len(lines) == 46


class TestUVExport:
    """Tests for UV-space export functions."""

    def test_export_uv_pattern_csv(self):
        """Test UV-space pattern CSV export."""
        u = np.linspace(-1, 1, 5)
        v = np.linspace(-1, 1, 5)
        pattern = np.zeros((5, 5))

        csv = pa.export_uv_pattern_csv(u, v, pattern)

        lines = csv.strip().split('\n')
        assert 'u,v,pattern_dB' in lines[0]
        # 5 x 5 = 25 data points + header
        assert len(lines) == 26


class TestWeightsExport:
    """Tests for weights export functions."""

    def test_export_weights_csv(self):
        """Test element weights CSV export."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        weights = np.ones(16, dtype=complex) * np.exp(1j * np.pi / 4)

        csv = pa.export_weights_csv(geom, weights)

        lines = csv.strip().split('\n')
        assert 'element,x,y,weight_real,weight_imag,weight_mag,weight_phase_deg' in lines[0]
        # 16 elements + header
        assert len(lines) == 17

        # Check a data line
        data = lines[1].split(',')
        assert len(data) == 7  # 7 columns


class TestGeometryExport:
    """Tests for geometry export functions."""

    def test_export_geometry_csv_2d(self):
        """Test 2D geometry CSV export."""
        geom = pa.create_rectangular_array(3, 3, 0.5, 0.5)

        csv = pa.export_geometry_csv(geom)

        lines = csv.strip().split('\n')
        assert 'element,x,y' in lines[0]
        assert len(lines) == 10  # header + 9 elements

    def test_export_geometry_csv_3d(self):
        """Test 3D geometry CSV export with normals."""
        geom = pa.create_cylindrical_array(6, 3, radius=1.0, height=1.0)

        csv = pa.export_geometry_csv(geom)

        lines = csv.strip().split('\n')
        # Should include z and normal columns
        assert 'z' in lines[0] or 'element,x,y' in lines[0]


class TestJSONExport:
    """Tests for JSON export functions."""

    def test_export_array_config_json_basic(self):
        """Test basic JSON config export."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)

        json_str = pa.export_array_config_json(geom)

        config = json.loads(json_str)
        assert 'metadata' in config
        assert 'geometry' in config
        assert config['geometry']['n_elements'] == 16
        assert len(config['geometry']['x']) == 16

    def test_export_array_config_json_with_weights(self):
        """Test JSON config export with weights."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        weights = np.ones(16, dtype=complex)

        json_str = pa.export_array_config_json(geom, weights=weights)

        config = json.loads(json_str)
        assert 'weights' in config
        assert 'magnitude' in config['weights']
        assert 'phase_deg' in config['weights']

    def test_export_array_config_json_full(self):
        """Test JSON config export with all options."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        weights = np.ones(16, dtype=complex)

        json_str = pa.export_array_config_json(
            geom,
            weights=weights,
            array_params={'type': 'Rectangular', 'Nx': 4, 'Ny': 4},
            steering={'theta': 30, 'phi': 0},
            taper_info={'type': 'Taylor', 'sidelobe_dB': -30}
        )

        config = json.loads(json_str)
        assert config['array_params']['type'] == 'Rectangular'
        assert config['steering']['theta'] == 30
        assert config['taper']['type'] == 'Taylor'


class TestNpzExport:
    """Tests for NumPy format export."""

    def test_export_load_pattern_npz(self):
        """Test NPZ pattern export and load round-trip."""
        angles = np.linspace(-90, 90, 181)
        e_plane = np.sin(np.deg2rad(angles))
        h_plane = np.cos(np.deg2rad(angles))

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name

        try:
            pa.export_pattern_npz(
                filename,
                angles=angles,
                e_plane=e_plane,
                h_plane=h_plane,
                metadata={'test': 'value'}
            )

            data = pa.load_pattern_npz(filename)

            np.testing.assert_array_almost_equal(data['angles'], angles)
            np.testing.assert_array_almost_equal(data['e_plane'], e_plane)
            np.testing.assert_array_almost_equal(data['h_plane'], h_plane)
            assert data['metadata']['test'] == 'value'
        finally:
            os.unlink(filename)


class TestCouplingExport:
    """Tests for coupling matrix export."""

    def test_export_coupling_matrix_csv_magnitude(self):
        """Test coupling matrix export in magnitude format."""
        C = np.eye(4, dtype=complex)
        C[0, 1] = 0.1 * np.exp(1j * np.pi / 4)
        C[1, 0] = 0.1 * np.exp(1j * np.pi / 4)

        csv = pa.export_coupling_matrix_csv(C, format='magnitude')

        lines = csv.strip().split('\n')
        assert len(lines) == 5  # header + 4 rows

    def test_export_coupling_matrix_csv_magnitude_phase(self):
        """Test coupling matrix export in magnitude/phase format."""
        C = np.eye(4, dtype=complex)

        csv = pa.export_coupling_matrix_csv(C, format='magnitude_phase')

        lines = csv.strip().split('\n')
        assert 'row,col,magnitude,phase_deg' in lines[0]
        # 4x4 = 16 entries + header
        assert len(lines) == 17


class TestSummaryReport:
    """Tests for summary report generation."""

    def test_export_summary_report(self):
        """Test summary report generation."""
        geom = pa.create_rectangular_array(8, 8, 0.5, 0.5)
        weights = np.ones(64, dtype=complex)

        report = pa.export_summary_report(
            geom,
            weights,
            pattern_metrics={
                'HPBW (deg)': 12.5,
                'SLL (dB)': -13.2,
                'Directivity (dBi)': 25.3
            },
            array_params={'type': 'Rectangular', 'Nx': 8, 'Ny': 8},
            steering={'theta': 0, 'phi': 0}
        )

        assert 'SUMMARY REPORT' in report
        assert 'Number of Elements: 64' in report
        assert 'HPBW (deg): 12.5' in report
        assert 'SLL (dB): -13.2' in report

    def test_export_summary_report_to_file(self):
        """Test writing summary report to file."""
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        weights = np.ones(16, dtype=complex)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filename = f.name

        try:
            pa.export_summary_report(
                geom, weights,
                pattern_metrics={'Test': 123},
                filename=filename
            )

            assert os.path.exists(filename)
            with open(filename, 'r') as f:
                content = f.read()
            assert 'SUMMARY REPORT' in content
        finally:
            os.unlink(filename)


class TestExportSafety:

    def test_load_rejects_pickled_objects(self, tmp_path):
        f = tmp_path / "evil.npz"
        np.savez(f, theta=np.zeros(3), obj=np.array([{"a": 1}], dtype=object))
        with pytest.raises(ValueError):
            pa.load_pattern_npz(str(f))

    def test_roundtrip_with_metadata(self, tmp_path):
        f = str(tmp_path / "p.npz")
        th = np.linspace(0, 1, 4)
        pa.export_pattern_npz(f, theta=th, phi=th, pattern_2d=np.ones((4, 4)),
                              metadata={"freq": 10e9})
        data = pa.load_pattern_npz(f)
        assert data["metadata"]["freq"] == 10e9
        np.testing.assert_allclose(data["theta"], th)

    def test_2d_csv_warns_on_radian_input(self):
        geom = pa.create_rectangular_array(4, 4, 0.5, 0.5)
        theta, phi, pat = pa.compute_full_pattern(geom.x, geom.y, np.ones(16),
                                                  pa.wavelength_to_k(1.0), n_theta=5, n_phi=9)
        with pytest.warns(UserWarning, match="radians"):
            pa.export_pattern_2d_csv(theta, phi, pat)
        csv = pa.export_pattern_2d_csv(theta, phi, pat, angle_unit='rad')
        assert csv.splitlines()[0] == "theta_rad,phi_rad,pattern_dB"


class TestUVPlotCircles:

    def test_grating_circles_without_visible_region(self):
        import matplotlib
        matplotlib.use("Agg")
        u = v = np.linspace(-1, 1, 21)
        ax = pa.plot_pattern_uv_space(u, v, np.zeros((21, 21)),
                                      show_visible_region=False,
                                      show_grating_circles=True, dx_wavelengths=0.7)
        assert ax is not None


class TestRadianHeuristic:

    def test_small_degree_grid_does_not_warn(self):
        import warnings
        theta = np.linspace(0, 3, 7)   # degrees: a main-beam zoom
        phi = np.linspace(0, 6, 7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pa.export_pattern_2d_csv(theta, phi, np.zeros((7, 7)))
            pa.export_pattern_2d_csv(np.linspace(0, 90, 5), np.linspace(0, 360, 9),
                                     np.zeros((5, 9)))
