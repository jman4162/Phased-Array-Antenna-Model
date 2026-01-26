"""
Visualization functions for phased array patterns.

Includes 3D Plotly plots, UV-space representation, and array geometry plots.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .geometry import ArrayGeometry
from .utils import (is_visible_region, linear_to_db, theta_phi_to_uv,
                    uv_to_theta_phi)

# ============== Matplotlib Plots (2D) ==============

def plot_pattern_2d(
    angles_deg: np.ndarray,
    pattern_dB: np.ndarray,
    title: str = "Radiation Pattern",
    xlabel: str = "Angle (degrees)",
    ylabel: str = "Normalized Gain (dB)",
    min_dB: float = -50.0,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[Any] = None,
    label: Optional[str] = None,
    **plot_kwargs
) -> Any:
    """
    Plot a 1D pattern cut.

    Parameters
    ----------
    angles_deg : ndarray
        Angle values in degrees
    pattern_dB : ndarray
        Pattern values in dB
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    min_dB : float
        Minimum dB level to display
    figsize : tuple
        Figure size (width, height)
    ax : matplotlib axis, optional
        Existing axis to plot on
    label : str, optional
        Legend label
    **plot_kwargs
        Additional arguments for plt.plot()

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    pattern_clipped = np.clip(pattern_dB, min_dB, 0)
    ax.plot(angles_deg, pattern_clipped, label=label, **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim([min_dB, 5])
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend()

    return ax


def plot_pattern_polar(
    angles_deg: np.ndarray,
    pattern_dB: np.ndarray,
    title: str = "Radiation Pattern",
    min_dB: float = -40.0,
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[Any] = None,
    **plot_kwargs
) -> Any:
    """
    Plot a 1D pattern cut in polar coordinates.

    Parameters
    ----------
    angles_deg : ndarray
        Angle values in degrees
    pattern_dB : ndarray
        Pattern values in dB (normalized)
    title : str
        Plot title
    min_dB : float
        Minimum dB level (becomes r=0)
    figsize : tuple
        Figure size
    ax : matplotlib polar axis, optional
        Existing axis

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # Convert dB to radius (shift so min_dB = 0)
    r = pattern_dB - min_dB
    r = np.clip(r, 0, None)

    ax.plot(np.deg2rad(angles_deg), r, **plot_kwargs)
    ax.set_title(title)
    ax.set_theta_zero_location('N')  # 0 degrees at top
    ax.set_theta_direction(-1)  # Clockwise

    return ax


def plot_pattern_contour(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    pattern_dB: np.ndarray,
    title: str = "Radiation Pattern",
    min_dB: float = -40.0,
    levels: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'jet',
    ax: Optional[Any] = None
) -> Any:
    """
    Plot 2D pattern as contour plot.

    Parameters
    ----------
    theta_deg : ndarray
        Theta values (1D or 2D grid)
    phi_deg : ndarray
        Phi values (1D or 2D grid)
    pattern_dB : ndarray
        Pattern in dB (2D)
    title : str
        Plot title
    min_dB : float
        Minimum dB level
    levels : int
        Number of contour levels
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    ax : matplotlib axis, optional
        Existing axis

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create meshgrid if 1D inputs
    if theta_deg.ndim == 1 and phi_deg.ndim == 1:
        theta_grid, phi_grid = np.meshgrid(theta_deg, phi_deg, indexing='ij')
    else:
        theta_grid, phi_grid = theta_deg, phi_deg

    pattern_clipped = np.clip(pattern_dB, min_dB, 0)

    cf = ax.contourf(phi_grid, theta_grid, pattern_clipped, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax, label='Gain (dB)')

    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')
    ax.set_title(title)

    return ax


def plot_array_geometry(
    geometry: ArrayGeometry,
    weights: Optional[np.ndarray] = None,
    title: str = "Array Geometry",
    show_indices: bool = False,
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[Any] = None
) -> Any:
    """
    Plot array element positions (2D view).

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    weights : ndarray, optional
        Element weights (color by magnitude)
    title : str
        Plot title
    show_indices : bool
        Show element index numbers
    figsize : tuple
        Figure size
    ax : matplotlib axis, optional
        Existing axis

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if weights is not None:
        colors = np.abs(weights)
        scatter = ax.scatter(geometry.x, geometry.y, c=colors, cmap='viridis', s=50)
        plt.colorbar(scatter, ax=ax, label='|Weight|')
    else:
        ax.scatter(geometry.x, geometry.y, s=50)

    if show_indices:
        for i, (x, y) in enumerate(zip(geometry.x, geometry.y)):
            ax.annotate(str(i), (x, y), fontsize=8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


# ============== UV-Space Visualization ==============

def compute_pattern_uv_space(
    geometry: ArrayGeometry,
    weights: np.ndarray,
    k: float,
    n_u: int = 201,
    n_v: int = 201,
    u_range: Tuple[float, float] = (-1, 1),
    v_range: Tuple[float, float] = (-1, 1)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pattern directly in UV-space.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    weights : ndarray
        Element weights
    k : float
        Wavenumber
    n_u, n_v : int
        Number of points in u and v
    u_range, v_range : tuple
        Range for u and v

    Returns
    -------
    u : ndarray
        U values (1D)
    v : ndarray
        V values (1D)
    pattern_dB : ndarray
        Pattern in dB (2D: n_u x n_v)
    """
    from .core import array_factor_uv

    u = np.linspace(u_range[0], u_range[1], n_u)
    v = np.linspace(v_range[0], v_range[1], n_v)
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')

    AF = array_factor_uv(u_grid, v_grid, geometry.x, geometry.y, weights, k)

    pattern_dB = linear_to_db(np.abs(AF)**2)
    pattern_dB -= np.max(pattern_dB)

    return u, v, pattern_dB


def plot_pattern_uv_space(
    u: np.ndarray,
    v: np.ndarray,
    pattern_dB: np.ndarray,
    title: str = "UV-Space Pattern",
    min_dB: float = -40.0,
    show_visible_region: bool = True,
    show_grating_circles: bool = False,
    dx_wavelengths: Optional[float] = None,
    dy_wavelengths: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'jet',
    ax: Optional[Any] = None
) -> Any:
    """
    Plot pattern in UV-space with optional visible region and grating lobe circles.

    Parameters
    ----------
    u, v : ndarray
        Direction cosine values (1D)
    pattern_dB : ndarray
        Pattern in dB (2D)
    title : str
        Plot title
    min_dB : float
        Minimum dB level
    show_visible_region : bool
        Show unit circle (visible space boundary)
    show_grating_circles : bool
        Show grating lobe circles
    dx_wavelengths, dy_wavelengths : float, optional
        Element spacing for grating lobe calculation
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    ax : matplotlib axis, optional
        Existing axis

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    pattern_clipped = np.clip(pattern_dB, min_dB, 0)

    cf = ax.contourf(u_grid, v_grid, pattern_clipped, levels=20, cmap=cmap)
    plt.colorbar(cf, ax=ax, label='Gain (dB)')

    if show_visible_region:
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'w--', linewidth=2,
                label='Visible region')

    if show_grating_circles and dx_wavelengths is not None:
        # Grating lobe positions: u = u0 + m/dx, v = v0 + n/dy
        for m in [-1, 1]:
            u_grating = m / dx_wavelengths
            ax.plot(np.cos(theta_circle) + u_grating, np.sin(theta_circle),
                    'r--', linewidth=1, alpha=0.7)
        if dy_wavelengths is not None:
            for n in [-1, 1]:
                v_grating = n / dy_wavelengths
                ax.plot(np.cos(theta_circle), np.sin(theta_circle) + v_grating,
                        'r--', linewidth=1, alpha=0.7)

    ax.set_xlabel('u = sin(θ)cos(φ)')
    ax.set_ylabel('v = sin(θ)sin(φ)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim([u.min(), u.max()])
    ax.set_ylim([v.min(), v.max()])

    return ax


# ============== 3D Plotly Plots ==============

def plot_pattern_3d_plotly(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern_dB: np.ndarray,
    title: str = "3D Radiation Pattern",
    min_dB: float = -40.0,
    colorscale: str = 'Jet',
    surface_type: str = 'spherical'
) -> Any:
    """
    Create interactive 3D pattern plot using Plotly.

    Parameters
    ----------
    theta : ndarray
        Theta values in radians (1D)
    phi : ndarray
        Phi values in radians (1D)
    pattern_dB : ndarray
        Pattern in dB (2D: n_theta x n_phi)
    title : str
        Plot title
    min_dB : float
        Minimum dB level
    colorscale : str
        Plotly colorscale name
    surface_type : str
        'spherical' - radius proportional to gain
        'cartesian' - theta/phi/gain surface

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for this function. Install with: pip install plotly")

    # Create meshgrid
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    pattern_clipped = np.clip(pattern_dB, min_dB, 0)

    if surface_type == 'spherical':
        # Map gain to radius (linear scale for better visualization)
        r = (pattern_clipped - min_dB) / (-min_dB)  # 0 to 1
        r = np.clip(r, 0.1, 1)  # Minimum radius for visibility

        # Spherical to Cartesian
        x = r * np.sin(theta_grid) * np.cos(phi_grid)
        y = r * np.sin(theta_grid) * np.sin(phi_grid)
        z = r * np.cos(theta_grid)

        fig = go.Figure(data=[go.Surface(
            x=x, y=y, z=z,
            surfacecolor=pattern_clipped,
            colorscale=colorscale,
            colorbar=dict(title='Gain (dB)'),
            showscale=True
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            )
        )

    else:  # cartesian
        fig = go.Figure(data=[go.Surface(
            x=np.rad2deg(theta_grid),
            y=np.rad2deg(phi_grid),
            z=pattern_clipped,
            colorscale=colorscale,
            colorbar=dict(title='Gain (dB)')
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Theta (deg)',
                yaxis_title='Phi (deg)',
                zaxis_title='Gain (dB)',
            )
        )

    return fig


def plot_pattern_3d_cartesian_plotly(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    pattern_dB: np.ndarray,
    title: str = "3D Radiation Pattern",
    min_dB: float = -40.0,
    colorscale: str = 'Jet'
) -> Any:
    """
    Create 3D surface plot with theta/phi/gain axes.

    Parameters
    ----------
    theta_deg : ndarray
        Theta values in degrees (1D)
    phi_deg : ndarray
        Phi values in degrees (1D)
    pattern_dB : ndarray
        Pattern in dB (2D)
    title : str
        Plot title
    min_dB : float
        Minimum dB
    colorscale : str
        Colorscale name

    Returns
    -------
    fig : plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for this function")

    theta_grid, phi_grid = np.meshgrid(theta_deg, phi_deg, indexing='ij')
    pattern_clipped = np.clip(pattern_dB, min_dB, 0)

    fig = go.Figure(data=[go.Surface(
        x=theta_grid,
        y=phi_grid,
        z=pattern_clipped,
        colorscale=colorscale,
        colorbar=dict(title='Gain (dB)')
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Theta (deg)',
            yaxis_title='Phi (deg)',
            zaxis_title='Gain (dB)',
        )
    )

    return fig


def plot_array_geometry_3d_plotly(
    geometry: ArrayGeometry,
    weights: Optional[np.ndarray] = None,
    title: str = "Array Geometry",
    show_normals: bool = True,
    normal_scale: float = 0.1
) -> Any:
    """
    Create interactive 3D array geometry plot using Plotly.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry with positions and optional normals
    weights : ndarray, optional
        Element weights for coloring
    title : str
        Plot title
    show_normals : bool
        Show element normal vectors
    normal_scale : float
        Scale factor for normal vectors

    Returns
    -------
    fig : plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for this function")

    z = geometry.z if geometry.z is not None else np.zeros_like(geometry.x)

    # Element colors
    if weights is not None:
        colors = np.abs(weights)
        color_label = '|Weight|'
    else:
        colors = np.arange(len(geometry.x))
        color_label = 'Element Index'

    # Element positions
    scatter = go.Scatter3d(
        x=geometry.x, y=geometry.y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale='Viridis',
            colorbar=dict(title=color_label),
            showscale=True
        ),
        name='Elements'
    )

    data = [scatter]

    # Element normals
    if show_normals and geometry.nx is not None and geometry.ny is not None:
        nz = geometry.nz if geometry.nz is not None else np.zeros_like(geometry.nx)

        for i in range(len(geometry.x)):
            data.append(go.Scatter3d(
                x=[geometry.x[i], geometry.x[i] + normal_scale * geometry.nx[i]],
                y=[geometry.y[i], geometry.y[i] + normal_scale * geometry.ny[i]],
                z=[z[i], z[i] + normal_scale * nz[i]],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False
            ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        )
    )

    return fig


def plot_pattern_uv_plotly(
    u: np.ndarray,
    v: np.ndarray,
    pattern_dB: np.ndarray,
    title: str = "UV-Space Pattern",
    min_dB: float = -40.0,
    colorscale: str = 'Jet',
    show_visible_circle: bool = True
) -> Any:
    """
    Interactive UV-space pattern plot using Plotly.

    Parameters
    ----------
    u, v : ndarray
        Direction cosines (1D)
    pattern_dB : ndarray
        Pattern in dB (2D)
    title : str
        Plot title
    min_dB : float
        Minimum dB
    colorscale : str
        Colorscale
    show_visible_circle : bool
        Show visible region boundary

    Returns
    -------
    fig : plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for this function")

    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    pattern_clipped = np.clip(pattern_dB, min_dB, 0)

    fig = go.Figure()

    # Heatmap of pattern
    fig.add_trace(go.Heatmap(
        x=u, y=v, z=pattern_clipped.T,
        colorscale=colorscale,
        colorbar=dict(title='Gain (dB)'),
        zmin=min_dB, zmax=0
    ))

    # Visible region circle
    if show_visible_circle:
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines',
            line=dict(color='white', width=2, dash='dash'),
            name='Visible Region'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='u = sin(θ)cos(φ)',
        yaxis_title='v = sin(θ)sin(φ)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        yaxis=dict(constrain='domain')
    )

    return fig


def plot_comparison_patterns(
    angles_deg: np.ndarray,
    patterns_dB: Dict[str, np.ndarray],
    title: str = "Pattern Comparison",
    min_dB: float = -50.0,
    figsize: Tuple[int, int] = (12, 6)
) -> Any:
    """
    Plot multiple patterns for comparison.

    Parameters
    ----------
    angles_deg : ndarray
        Angle values
    patterns_dB : dict
        Dictionary of {label: pattern_dB}
    title : str
        Plot title
    min_dB : float
        Minimum dB
    figsize : tuple
        Figure size

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    fig, ax = plt.subplots(figsize=figsize)

    for label, pattern in patterns_dB.items():
        pattern_clipped = np.clip(pattern, min_dB, 0)
        ax.plot(angles_deg, pattern_clipped, label=label, linewidth=1.5)

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Gain (dB)')
    ax.set_title(title)
    ax.set_ylim([min_dB, 5])
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def create_pattern_animation_plotly(
    theta: np.ndarray,
    phi: np.ndarray,
    patterns_dB: List[np.ndarray],
    frame_labels: List[str],
    title: str = "Pattern Animation",
    min_dB: float = -40.0,
    colorscale: str = 'Jet'
) -> Any:
    """
    Create animated pattern plot (e.g., beam scanning).

    Parameters
    ----------
    theta : ndarray
        Theta values (1D)
    phi : ndarray
        Phi values (1D)
    patterns_dB : list of ndarray
        List of 2D patterns for each frame
    frame_labels : list of str
        Label for each frame
    title : str
        Plot title
    min_dB : float
        Minimum dB
    colorscale : str
        Colorscale

    Returns
    -------
    fig : plotly Figure with animation
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for this function")

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # First frame
    fig = go.Figure(
        data=[go.Surface(
            x=np.rad2deg(theta_grid),
            y=np.rad2deg(phi_grid),
            z=np.clip(patterns_dB[0], min_dB, 0),
            colorscale=colorscale,
            cmin=min_dB,
            cmax=0
        )],
        layout=go.Layout(
            title=title,
            scene=dict(
                xaxis_title='Theta (deg)',
                yaxis_title='Phi (deg)',
                zaxis_title='Gain (dB)',
                zaxis=dict(range=[min_dB, 0])
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, {'frame': {'duration': 500, 'redraw': True},
                                     'fromcurrent': True}]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate'}])
                ]
            )]
        ),
        frames=[go.Frame(
            data=[go.Surface(
                x=np.rad2deg(theta_grid),
                y=np.rad2deg(phi_grid),
                z=np.clip(p, min_dB, 0),
                colorscale=colorscale,
                cmin=min_dB,
                cmax=0
            )],
            name=label
        ) for p, label in zip(patterns_dB, frame_labels)]
    )

    return fig


# ============== Wideband / Beam Squint Plots ==============

def plot_beam_squint(
    frequencies: np.ndarray,
    squint_data: Dict[str, np.ndarray],
    center_frequency: float,
    title: str = "Beam Squint vs Frequency",
    figsize: Tuple[int, int] = (10, 6)
) -> Any:
    """
    Plot beam squint comparison for different steering modes.

    Parameters
    ----------
    frequencies : ndarray
        Frequency values in Hz
    squint_data : dict
        Dictionary of {mode_name: squint_array} in degrees
    center_frequency : float
        Center frequency in Hz (for normalization)
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize frequency to percentage of center
    freq_percent = (frequencies - center_frequency) / center_frequency * 100

    colors = {'phase': 'red', 'hybrid': 'blue', 'ttd': 'green'}
    labels = {'phase': 'Phase-only', 'hybrid': 'Hybrid (TTD + Phase)', 'ttd': 'True-Time Delay'}

    for mode, squint in squint_data.items():
        color = colors.get(mode, None)
        label = labels.get(mode, mode)
        ax.plot(freq_percent, squint, 'o-', label=label, color=color, linewidth=2, markersize=6)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Frequency Offset (%)')
    ax.set_ylabel('Beam Squint (degrees)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_pattern_vs_frequency(
    angles: np.ndarray,
    frequencies: np.ndarray,
    patterns: np.ndarray,
    center_frequency: float,
    title: str = "Pattern vs Frequency",
    min_dB: float = -40.0,
    figsize: Tuple[int, int] = (12, 8)
) -> Any:
    """
    Plot radiation patterns at multiple frequencies as a waterfall/heatmap.

    Parameters
    ----------
    angles : ndarray
        Angle values in degrees
    frequencies : ndarray
        Frequency values in Hz
    patterns : ndarray
        2D array (n_freq x n_angles) of patterns in dB
    center_frequency : float
        Center frequency for labeling
    title : str
        Plot title
    min_dB : float
        Minimum dB for colormap
    figsize : tuple
        Figure size

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize frequency to percentage
    freq_percent = (frequencies - center_frequency) / center_frequency * 100

    patterns_clipped = np.clip(patterns, min_dB, 0)

    im = ax.pcolormesh(angles, freq_percent, patterns_clipped, cmap='jet', shading='auto')
    plt.colorbar(im, ax=ax, label='Gain (dB)')

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Frequency Offset (%)')
    ax.set_title(title)

    return ax


def plot_pattern_vs_frequency_plotly(
    angles: np.ndarray,
    frequencies: np.ndarray,
    patterns: np.ndarray,
    center_frequency: float,
    title: str = "Pattern vs Frequency",
    min_dB: float = -40.0
) -> Any:
    """
    Interactive Plotly plot of patterns vs frequency.

    Parameters
    ----------
    angles : ndarray
        Angle values in degrees
    frequencies : ndarray
        Frequency values in Hz
    patterns : ndarray
        2D array (n_freq x n_angles) in dB
    center_frequency : float
        Center frequency
    title : str
        Plot title
    min_dB : float
        Minimum dB

    Returns
    -------
    fig : plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for this function")

    freq_percent = (frequencies - center_frequency) / center_frequency * 100
    patterns_clipped = np.clip(patterns, min_dB, 0)

    fig = go.Figure(data=go.Heatmap(
        x=angles,
        y=freq_percent,
        z=patterns_clipped,
        colorscale='Jet',
        zmin=min_dB,
        zmax=0,
        colorbar=dict(title='Gain (dB)')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Angle (degrees)',
        yaxis_title='Frequency Offset (%)'
    )

    return fig


def plot_subarray_delays(
    architecture,
    delays: np.ndarray,
    title: str = "Subarray Time Delays",
    figsize: Tuple[int, int] = (10, 8)
) -> Any:
    """
    Visualize TTD values across subarrays.

    Parameters
    ----------
    architecture : SubarrayArchitecture
        Subarray architecture with centers
    delays : ndarray
        Time delay for each subarray in seconds
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this function")

    fig, ax = plt.subplots(figsize=figsize)

    centers = architecture.subarray_centers
    delays_ns = delays * 1e9  # Convert to nanoseconds

    scatter = ax.scatter(centers[:, 0], centers[:, 1], c=delays_ns,
                         cmap='viridis', s=200, edgecolors='black')
    plt.colorbar(scatter, ax=ax, label='Delay (ns)')

    # Add labels
    for i, (x, y) in enumerate(centers):
        ax.annotate(f'SA{i}', (x, y), ha='center', va='center', fontsize=8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax
