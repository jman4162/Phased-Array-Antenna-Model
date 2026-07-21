"""Smoke tests for phased_array.visualization plotting functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phased_array as pa
from phased_array.visualization import (
    compute_pattern_uv_space,
    plot_array_geometry,
    plot_beam_squint,
    plot_comparison_patterns,
    plot_pattern_2d,
    plot_pattern_contour,
    plot_pattern_polar,
    plot_pattern_uv_space,
    plot_pattern_vs_frequency,
    plot_subarray_delays,
)

plotly = pytest.importorskip("plotly", reason="plotly not installed")
import plotly.graph_objects as go  # noqa: E402

from phased_array.visualization import (  # noqa: E402
    create_pattern_animation_plotly,
    plot_array_geometry_3d_plotly,
    plot_pattern_3d_cartesian_plotly,
    plot_pattern_3d_plotly,
    plot_pattern_uv_plotly,
    plot_pattern_vs_frequency_plotly,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def small_array():
    geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    k = pa.wavelength_to_k(1.0)
    weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)
    return geom, k, weights


@pytest.fixture(scope="module")
def pattern_cut(small_array):
    geom, k, weights = small_array
    angles = np.linspace(-90, 90, 181)
    theta = np.deg2rad(np.abs(angles))
    phi = np.where(angles >= 0, 0.0, np.pi)
    af = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, weights, k)
    pattern_dB = 20 * np.log10(np.abs(af) / np.abs(af).max() + 1e-12)
    return angles, pattern_dB


@pytest.fixture(scope="module")
def full_pattern(small_array):
    geom, k, weights = small_array
    theta, phi, pattern_dB = pa.compute_full_pattern(
        geom.x, geom.y, weights, k, n_theta=19, n_phi=37
    )
    return theta, phi, pattern_dB


class TestMatplotlibPlots:
    def test_plot_pattern_2d(self, pattern_cut):
        ax = plot_pattern_2d(*pattern_cut)
        assert ax is not None

    def test_plot_pattern_2d_existing_ax(self, pattern_cut):
        fig, ax_in = plt.subplots()
        ax = plot_pattern_2d(*pattern_cut, ax=ax_in)
        assert ax is ax_in

    def test_plot_pattern_polar(self, pattern_cut):
        ax = plot_pattern_polar(*pattern_cut)
        assert ax is not None

    def test_plot_pattern_contour(self, full_pattern):
        theta, phi, pattern_dB = full_pattern
        ax = plot_pattern_contour(np.rad2deg(theta), np.rad2deg(phi), pattern_dB)
        assert ax is not None

    def test_plot_array_geometry(self, small_array):
        geom, _, weights = small_array
        ax = plot_array_geometry(geom, weights=weights)
        assert ax is not None

    def test_plot_comparison_patterns(self, pattern_cut):
        angles, pattern_dB = pattern_cut
        ax = plot_comparison_patterns(
            angles, {"a": pattern_dB, "b": pattern_dB - 3.0}
        )
        assert ax is not None

    def test_plot_beam_squint(self):
        freqs = np.linspace(9e9, 11e9, 5)
        squint = {"phase": np.linspace(-2, 2, 5), "ttd": np.zeros(5)}
        ax = plot_beam_squint(freqs, squint, center_frequency=10e9)
        assert ax is not None

    def test_plot_pattern_vs_frequency(self, pattern_cut):
        angles, pattern_dB = pattern_cut
        freqs = np.linspace(9e9, 11e9, 4)
        patterns = np.tile(pattern_dB, (4, 1))
        ax = plot_pattern_vs_frequency(angles, freqs, patterns, center_frequency=10e9)
        assert ax is not None

    def test_plot_subarray_delays(self):
        arch = pa.create_rectangular_subarrays(8, 8, 4, 4, dx=0.5, dy=0.5)
        delays = pa.compute_subarray_delays_ttd(arch, theta0_deg=20, phi0_deg=0)
        ax = plot_subarray_delays(arch, delays)
        assert ax is not None


class TestUVSpace:
    def test_compute_pattern_uv_space(self, small_array):
        geom, k, weights = small_array
        u, v, pattern_dB = compute_pattern_uv_space(geom, weights, k, n_u=41, n_v=41)
        assert pattern_dB.shape == (41, 41)
        # Invisible region must be masked or finite-valued; peak is 0 dB
        assert np.isclose(np.nanmax(pattern_dB), 0.0, atol=1e-6)

    def test_plot_pattern_uv_space(self, small_array):
        geom, k, weights = small_array
        u, v, pattern_dB = compute_pattern_uv_space(geom, weights, k, n_u=41, n_v=41)
        ax = plot_pattern_uv_space(
            u, v, pattern_dB, show_grating_circles=True,
            dx_wavelengths=0.5, dy_wavelengths=0.5,
        )
        assert ax is not None


class TestPlotlyPlots:
    def test_plot_pattern_3d_plotly(self, full_pattern):
        fig = plot_pattern_3d_plotly(*full_pattern)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_plot_pattern_3d_cartesian_plotly(self, full_pattern):
        theta, phi, pattern_dB = full_pattern
        fig = plot_pattern_3d_cartesian_plotly(
            np.rad2deg(theta), np.rad2deg(phi), pattern_dB
        )
        assert isinstance(fig, go.Figure)

    def test_plot_array_geometry_3d_plotly(self, small_array):
        geom, _, weights = small_array
        fig = plot_array_geometry_3d_plotly(geom, weights=weights)
        assert isinstance(fig, go.Figure)

    def test_plot_pattern_uv_plotly(self, small_array):
        geom, k, weights = small_array
        u, v, pattern_dB = compute_pattern_uv_space(geom, weights, k, n_u=41, n_v=41)
        fig = plot_pattern_uv_plotly(u, v, pattern_dB)
        assert isinstance(fig, go.Figure)

    def test_create_pattern_animation_plotly(self, full_pattern):
        theta, phi, pattern_dB = full_pattern
        fig = create_pattern_animation_plotly(
            theta, phi, [pattern_dB, pattern_dB - 3.0], ["f0", "f1"]
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 2

    def test_plot_pattern_vs_frequency_plotly(self, pattern_cut):
        angles, pattern_dB = pattern_cut
        freqs = np.linspace(9e9, 11e9, 4)
        patterns = np.tile(pattern_dB, (4, 1))
        fig = plot_pattern_vs_frequency_plotly(
            angles, freqs, patterns, center_frequency=10e9
        )
        assert isinstance(fig, go.Figure)
