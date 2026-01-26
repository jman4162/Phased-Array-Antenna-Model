"""
Export functions for phased array antenna data.

Supports CSV, JSON, NumPy, and MATLAB formats.
"""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .geometry import ArrayGeometry


def export_pattern_csv(
    angles: np.ndarray,
    pattern_dB: np.ndarray,
    filename: Optional[str] = None,
    angle_label: str = "angle_deg",
    pattern_label: str = "pattern_dB",
    include_header: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export 1D pattern data to CSV format.

    Parameters
    ----------
    angles : ndarray
        Angle values (typically in degrees)
    pattern_dB : ndarray
        Pattern values in dB
    filename : str, optional
        If provided, write to file. Otherwise return string.
    angle_label : str
        Column label for angles
    pattern_label : str
        Column label for pattern
    include_header : bool
        Whether to include column headers
    metadata : dict, optional
        Additional metadata to include as comments

    Returns
    -------
    csv_string : str
        CSV formatted string (also written to file if filename provided)
    """
    buffer = io.StringIO()

    # Write metadata as comments
    if metadata:
        buffer.write(f"# Phased Array Pattern Export\n")
        buffer.write(f"# Generated: {datetime.now().isoformat()}\n")
        for key, value in metadata.items():
            buffer.write(f"# {key}: {value}\n")
        buffer.write("#\n")

    # Write header
    if include_header:
        buffer.write(f"{angle_label},{pattern_label}\n")

    # Write data
    for angle, pattern in zip(angles, pattern_dB):
        buffer.write(f"{angle:.6f},{pattern:.4f}\n")

    csv_string = buffer.getvalue()

    if filename:
        with open(filename, 'w') as f:
            f.write(csv_string)

    return csv_string


def export_pattern_2d_csv(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern_dB: np.ndarray,
    filename: Optional[str] = None,
    include_header: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export 2D pattern data to CSV format.

    Parameters
    ----------
    theta : ndarray
        Theta values (1D or meshgrid)
    phi : ndarray
        Phi values (1D or meshgrid)
    pattern_dB : ndarray
        2D pattern in dB
    filename : str, optional
        If provided, write to file
    include_header : bool
        Whether to include column headers
    metadata : dict, optional
        Additional metadata as comments

    Returns
    -------
    csv_string : str
        CSV formatted string
    """
    buffer = io.StringIO()

    # Write metadata
    if metadata:
        buffer.write(f"# Phased Array 2D Pattern Export\n")
        buffer.write(f"# Generated: {datetime.now().isoformat()}\n")
        for key, value in metadata.items():
            buffer.write(f"# {key}: {value}\n")
        buffer.write("#\n")

    # Write header
    if include_header:
        buffer.write("theta_deg,phi_deg,pattern_dB\n")

    # Handle meshgrid or 1D arrays
    if theta.ndim == 1 and phi.ndim == 1:
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    else:
        theta_grid, phi_grid = theta, phi

    # Flatten and write
    for i in range(theta_grid.shape[0]):
        for j in range(theta_grid.shape[1]):
            buffer.write(f"{theta_grid[i,j]:.4f},{phi_grid[i,j]:.4f},{pattern_dB[i,j]:.4f}\n")

    csv_string = buffer.getvalue()

    if filename:
        with open(filename, 'w') as f:
            f.write(csv_string)

    return csv_string


def export_uv_pattern_csv(
    u: np.ndarray,
    v: np.ndarray,
    pattern_dB: np.ndarray,
    filename: Optional[str] = None,
    include_header: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export UV-space pattern data to CSV format.

    Parameters
    ----------
    u : ndarray
        U direction cosine values
    v : ndarray
        V direction cosine values
    pattern_dB : ndarray
        2D pattern in dB
    filename : str, optional
        If provided, write to file
    include_header : bool
        Whether to include column headers
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    csv_string : str
        CSV formatted string
    """
    buffer = io.StringIO()

    if metadata:
        buffer.write(f"# Phased Array UV-Space Pattern Export\n")
        buffer.write(f"# Generated: {datetime.now().isoformat()}\n")
        for key, value in metadata.items():
            buffer.write(f"# {key}: {value}\n")
        buffer.write("#\n")

    if include_header:
        buffer.write("u,v,pattern_dB\n")

    # Handle meshgrid or 1D arrays
    if u.ndim == 1 and v.ndim == 1:
        for i in range(len(u)):
            for j in range(len(v)):
                buffer.write(f"{u[i]:.6f},{v[j]:.6f},{pattern_dB[i,j]:.4f}\n")
    else:
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                buffer.write(f"{u[i,j]:.6f},{v[i,j]:.6f},{pattern_dB[i,j]:.4f}\n")

    csv_string = buffer.getvalue()

    if filename:
        with open(filename, 'w') as f:
            f.write(csv_string)

    return csv_string


def export_weights_csv(
    geometry: ArrayGeometry,
    weights: np.ndarray,
    filename: Optional[str] = None,
    include_header: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export element weights to CSV format.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry with element positions
    weights : ndarray
        Complex element weights
    filename : str, optional
        If provided, write to file
    include_header : bool
        Whether to include column headers
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    csv_string : str
        CSV formatted string with columns:
        element, x, y, weight_real, weight_imag, weight_mag, weight_phase_deg
    """
    buffer = io.StringIO()

    if metadata:
        buffer.write(f"# Phased Array Weights Export\n")
        buffer.write(f"# Generated: {datetime.now().isoformat()}\n")
        buffer.write(f"# N_elements: {geometry.n_elements}\n")
        for key, value in metadata.items():
            buffer.write(f"# {key}: {value}\n")
        buffer.write("#\n")

    if include_header:
        buffer.write("element,x,y,weight_real,weight_imag,weight_mag,weight_phase_deg\n")

    for i in range(geometry.n_elements):
        w = weights[i]
        buffer.write(f"{i},{geometry.x[i]:.6f},{geometry.y[i]:.6f},"
                    f"{w.real:.8f},{w.imag:.8f},"
                    f"{np.abs(w):.8f},{np.rad2deg(np.angle(w)):.4f}\n")

    csv_string = buffer.getvalue()

    if filename:
        with open(filename, 'w') as f:
            f.write(csv_string)

    return csv_string


def export_geometry_csv(
    geometry: ArrayGeometry,
    filename: Optional[str] = None,
    include_header: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export array geometry to CSV format.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry with element positions
    filename : str, optional
        If provided, write to file
    include_header : bool
        Whether to include column headers
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    csv_string : str
        CSV formatted string
    """
    buffer = io.StringIO()

    if metadata:
        buffer.write(f"# Phased Array Geometry Export\n")
        buffer.write(f"# Generated: {datetime.now().isoformat()}\n")
        buffer.write(f"# N_elements: {geometry.n_elements}\n")
        for key, value in metadata.items():
            buffer.write(f"# {key}: {value}\n")
        buffer.write("#\n")

    # Check if we have 3D geometry and normal vectors
    has_z = geometry.z is not None
    has_normals = hasattr(geometry, 'nx') and geometry.nx is not None

    if include_header:
        header_parts = ["element", "x", "y"]
        if has_z:
            header_parts.append("z")
        if has_normals:
            header_parts.extend(["nx", "ny", "nz"])
        buffer.write(",".join(header_parts) + "\n")

    for i in range(geometry.n_elements):
        row = [str(i), f"{geometry.x[i]:.6f}", f"{geometry.y[i]:.6f}"]
        if has_z:
            row.append(f"{geometry.z[i]:.6f}")
        if has_normals:
            row.extend([f"{geometry.nx[i]:.6f}",
                       f"{geometry.ny[i]:.6f}",
                       f"{geometry.nz[i]:.6f}"])
        buffer.write(",".join(row) + "\n")

    csv_string = buffer.getvalue()

    if filename:
        with open(filename, 'w') as f:
            f.write(csv_string)

    return csv_string


def export_array_config_json(
    geometry: ArrayGeometry,
    weights: Optional[np.ndarray] = None,
    array_params: Optional[Dict[str, Any]] = None,
    steering: Optional[Dict[str, float]] = None,
    taper_info: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None
) -> str:
    """
    Export full array configuration to JSON format.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    weights : ndarray, optional
        Complex weights (exported as mag/phase)
    array_params : dict, optional
        Array parameters (type, spacing, etc.)
    steering : dict, optional
        Steering direction (theta, phi)
    taper_info : dict, optional
        Taper information
    filename : str, optional
        If provided, write to file

    Returns
    -------
    json_string : str
        JSON formatted string
    """
    config = {
        "metadata": {
            "export_time": datetime.now().isoformat(),
            "format_version": "1.0",
            "package": "phased-array-modeling"
        },
        "geometry": {
            "n_elements": geometry.n_elements,
            "x": geometry.x.tolist(),
            "y": geometry.y.tolist(),
        }
    }

    if geometry.z is not None:
        config["geometry"]["z"] = geometry.z.tolist()

    if hasattr(geometry, 'nx') and geometry.nx is not None:
        config["geometry"]["nx"] = geometry.nx.tolist()
        config["geometry"]["ny"] = geometry.ny.tolist()
        config["geometry"]["nz"] = geometry.nz.tolist()

    if weights is not None:
        config["weights"] = {
            "magnitude": np.abs(weights).tolist(),
            "phase_deg": np.rad2deg(np.angle(weights)).tolist()
        }

    if array_params:
        config["array_params"] = array_params

    if steering:
        config["steering"] = steering

    if taper_info:
        config["taper"] = taper_info

    json_string = json.dumps(config, indent=2)

    if filename:
        with open(filename, 'w') as f:
            f.write(json_string)

    return json_string


def export_pattern_npz(
    filename: str,
    angles: Optional[np.ndarray] = None,
    e_plane: Optional[np.ndarray] = None,
    h_plane: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
    phi: Optional[np.ndarray] = None,
    pattern_2d: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    pattern_uv: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    geometry_x: Optional[np.ndarray] = None,
    geometry_y: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export pattern data to compressed NumPy format.

    Parameters
    ----------
    filename : str
        Output filename (should end in .npz)
    angles : ndarray, optional
        1D cut angles
    e_plane : ndarray, optional
        E-plane pattern
    h_plane : ndarray, optional
        H-plane pattern
    theta, phi : ndarray, optional
        2D grid coordinates
    pattern_2d : ndarray, optional
        Full 2D pattern
    u, v : ndarray, optional
        UV-space coordinates
    pattern_uv : ndarray, optional
        UV-space pattern
    weights : ndarray, optional
        Complex element weights
    geometry_x, geometry_y : ndarray, optional
        Element positions
    metadata : dict, optional
        Additional metadata (converted to string for storage)
    """
    data = {}

    if angles is not None:
        data['angles'] = angles
    if e_plane is not None:
        data['e_plane'] = e_plane
    if h_plane is not None:
        data['h_plane'] = h_plane
    if theta is not None:
        data['theta'] = theta
    if phi is not None:
        data['phi'] = phi
    if pattern_2d is not None:
        data['pattern_2d'] = pattern_2d
    if u is not None:
        data['u'] = u
    if v is not None:
        data['v'] = v
    if pattern_uv is not None:
        data['pattern_uv'] = pattern_uv
    if weights is not None:
        data['weights'] = weights
    if geometry_x is not None:
        data['geometry_x'] = geometry_x
    if geometry_y is not None:
        data['geometry_y'] = geometry_y
    if metadata is not None:
        data['metadata'] = np.array([json.dumps(metadata)])

    np.savez_compressed(filename, **data)


def load_pattern_npz(filename: str) -> Dict[str, Any]:
    """
    Load pattern data from NumPy format.

    Parameters
    ----------
    filename : str
        Input filename

    Returns
    -------
    data : dict
        Dictionary with loaded arrays and metadata
    """
    loaded = np.load(filename, allow_pickle=True)
    data = {key: loaded[key] for key in loaded.files}

    # Parse metadata if present
    if 'metadata' in data:
        try:
            data['metadata'] = json.loads(str(data['metadata'][0]))
        except:
            pass

    return data


def export_coupling_matrix_csv(
    coupling_matrix: np.ndarray,
    filename: Optional[str] = None,
    format: str = 'magnitude_phase',
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export mutual coupling matrix to CSV format.

    Parameters
    ----------
    coupling_matrix : ndarray
        Complex N x N coupling matrix
    filename : str, optional
        If provided, write to file
    format : str
        'magnitude_phase' - two columns per element (mag, phase_deg)
        'real_imag' - two columns per element (real, imag)
        'magnitude' - magnitude only
        'dB' - magnitude in dB only
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    csv_string : str
        CSV formatted string
    """
    buffer = io.StringIO()
    n = coupling_matrix.shape[0]

    if metadata:
        buffer.write(f"# Mutual Coupling Matrix Export\n")
        buffer.write(f"# Generated: {datetime.now().isoformat()}\n")
        buffer.write(f"# Format: {format}\n")
        buffer.write(f"# Size: {n} x {n}\n")
        for key, value in metadata.items():
            buffer.write(f"# {key}: {value}\n")
        buffer.write("#\n")

    if format == 'magnitude':
        buffer.write(",".join([f"col_{j}" for j in range(n)]) + "\n")
        for i in range(n):
            row = [f"{np.abs(coupling_matrix[i,j]):.8f}" for j in range(n)]
            buffer.write(",".join(row) + "\n")

    elif format == 'dB':
        buffer.write(",".join([f"col_{j}" for j in range(n)]) + "\n")
        for i in range(n):
            row = [f"{20*np.log10(max(np.abs(coupling_matrix[i,j]), 1e-10)):.4f}" for j in range(n)]
            buffer.write(",".join(row) + "\n")

    elif format == 'magnitude_phase':
        buffer.write("row,col,magnitude,phase_deg\n")
        for i in range(n):
            for j in range(n):
                c = coupling_matrix[i, j]
                buffer.write(f"{i},{j},{np.abs(c):.8f},{np.rad2deg(np.angle(c)):.4f}\n")

    elif format == 'real_imag':
        buffer.write("row,col,real,imag\n")
        for i in range(n):
            for j in range(n):
                c = coupling_matrix[i, j]
                buffer.write(f"{i},{j},{c.real:.8f},{c.imag:.8f}\n")

    csv_string = buffer.getvalue()

    if filename:
        with open(filename, 'w') as f:
            f.write(csv_string)

    return csv_string


def export_summary_report(
    geometry: ArrayGeometry,
    weights: np.ndarray,
    pattern_metrics: Dict[str, float],
    array_params: Optional[Dict[str, Any]] = None,
    steering: Optional[Dict[str, float]] = None,
    filename: Optional[str] = None
) -> str:
    """
    Generate a human-readable summary report.

    Parameters
    ----------
    geometry : ArrayGeometry
        Array geometry
    weights : ndarray
        Complex weights
    pattern_metrics : dict
        Dictionary with metrics like HPBW, SLL, directivity, etc.
    array_params : dict, optional
        Array configuration parameters
    steering : dict, optional
        Steering direction
    filename : str, optional
        If provided, write to file

    Returns
    -------
    report : str
        Formatted text report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PHASED ARRAY ANTENNA - SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Array Configuration
    lines.append("-" * 40)
    lines.append("ARRAY CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Number of Elements: {geometry.n_elements}")

    aperture_x = geometry.x.max() - geometry.x.min()
    aperture_y = geometry.y.max() - geometry.y.min()
    lines.append(f"Aperture Size: {aperture_x:.3f} x {aperture_y:.3f} m")

    if array_params:
        for key, value in array_params.items():
            if key != 'wavelength':
                lines.append(f"{key}: {value}")
        if 'wavelength' in array_params:
            lines.append(f"Wavelength: {array_params['wavelength']} m")
    lines.append("")

    # Steering
    if steering:
        lines.append("-" * 40)
        lines.append("BEAM STEERING")
        lines.append("-" * 40)
        lines.append(f"Theta (elevation): {steering.get('theta', 0)}°")
        lines.append(f"Phi (azimuth): {steering.get('phi', 0)}°")
        lines.append("")

    # Pattern Metrics
    lines.append("-" * 40)
    lines.append("PATTERN METRICS")
    lines.append("-" * 40)
    for key, value in pattern_metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.2f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")

    # Weight Statistics
    lines.append("-" * 40)
    lines.append("WEIGHT STATISTICS")
    lines.append("-" * 40)
    mag = np.abs(weights)
    phase = np.rad2deg(np.angle(weights))
    lines.append(f"Magnitude Range: {mag.min():.4f} - {mag.max():.4f}")
    lines.append(f"Phase Range: {phase.min():.1f}° - {phase.max():.1f}°")
    lines.append(f"Total Power: {np.sum(mag**2):.4f}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    report = "\n".join(lines)

    if filename:
        with open(filename, 'w') as f:
            f.write(report)

    return report
