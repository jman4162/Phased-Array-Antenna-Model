"""
Pattern summary metrics for an array excitation.

:func:`pattern_summary` collects the figures usually quoted for a beam:
direction, directivity, half-power and first-null beamwidths and peak
sidelobe level in two orthogonal cuts, the peak sidelobe over the whole
visible hemisphere, and directions where a grating lobe reaches the
main-beam level.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from .core import (compute_directivity, compute_first_null_beamwidth,
                   compute_half_power_beamwidth, compute_pattern_cuts,
                   compute_peak_sidelobe_level, total_pattern)
from .impairments import _beam_direction
from .utils import create_theta_phi_grid


def pattern_summary(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    k: float,
    theta0_deg: Optional[float] = None,
    phi0_deg: Optional[float] = None,
    z: Optional[np.ndarray] = None,
    element_pattern_func: Optional[Callable] = None,
    hemisphere: bool = True,
    grating_lobe_threshold_dB: float = -3.0,
    n_theta: int = 181,
    n_phi: int = 361,
    **element_kwargs: Any
) -> Dict[str, Any]:
    """
    Summarize the beam of an array excitation.

    Parameters
    ----------
    x, y : ndarray
        Element positions in meters
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber in rad/m
    theta0_deg, phi0_deg : float, optional
        Beam direction in degrees. If omitted, the array-factor peak in the
        front hemisphere is located numerically.
    z : ndarray, optional
        Element z-positions in meters
    element_pattern_func : callable, optional
        Element field pattern ``f(theta, phi, **element_kwargs)``
    hemisphere : bool
        Integrate directivity over the front hemisphere (ground-plane
        backed array, the default) instead of the full sphere
    grating_lobe_threshold_dB : float
        Level relative to the beam peak above which a lobe outside the
        main beam is reported as a grating lobe
    n_theta, n_phi : int
        Grid used for directivity and the hemisphere sidelobe search
    **element_kwargs
        Passed to ``element_pattern_func``

    Returns
    -------
    summary : dict
        - ``'theta0_deg'``, ``'phi0_deg'``: beam direction used
        - ``'directivity_dBi'``: peak directivity (hemisphere or sphere)
        - ``'hpbw_deg'``: (cut 1, cut 2) half-power beamwidths
        - ``'fnbw_deg'``: (cut 1, cut 2) first-null beamwidths
        - ``'sll_dB'``: (cut 1, cut 2) peak sidelobe levels
        - ``'peak_sidelobe_dB'``: highest level outside the main beam
          over the whole grid
        - ``'grating_lobes'``: list of ``(theta_deg, phi_deg, level_dB)``
          for lobes outside the main beam at or above the threshold

        Cut 1 is the ``phi = phi0`` plane, cut 2 the orthogonal great
        circle through the beam (see :func:`compute_pattern_cuts`).

    Notes
    -----
    On the grid, the main lobe is every direction reachable from the
    pattern peak without climbing, so it ends at the surrounding null
    valleys whatever its shape. Sidelobes and grating lobes are searched
    outside it. The grid must resolve those nulls: for arrays more than
    about 60 wavelengths across, raise ``n_theta`` and ``n_phi``.

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, 0.5, 0.5)
    >>> k = pa.wavelength_to_k(1.0)
    >>> s = pa.pattern_summary(geom.x, geom.y, np.ones(256), k)
    >>> [round(v, 2) for v in s['hpbw_deg']], [round(v, 2) for v in s['sll_dB']]
    ([6.36, 6.36], [-13.15, -13.15])
    >>> s['grating_lobes']
    []
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.broadcast_to(np.asarray(weights), x.shape)

    if theta0_deg is None or phi0_deg is None:
        theta_b, phi_b, _ = _beam_direction(x, y, z, weights, k)
        theta0_deg = float(np.rad2deg(theta_b)) if theta0_deg is None else theta0_deg
        phi0_deg = float(np.rad2deg(phi_b)) if phi0_deg is None else phi0_deg

    # Two orthogonal cuts through the beam, sampled finely enough for the
    # narrowest beams likely here (0.01 deg steps)
    angles, cut1, cut2 = compute_pattern_cuts(
        x, y, weights, k, theta0_deg=theta0_deg, phi0_deg=phi0_deg,
        n_points=18001, element_pattern_func=element_pattern_func, z=z,
        **element_kwargs
    )
    hpbw = (compute_half_power_beamwidth(angles, cut1),
            compute_half_power_beamwidth(angles, cut2))
    fnbw = (compute_first_null_beamwidth(angles, cut1),
            compute_first_null_beamwidth(angles, cut2))
    sll = (compute_peak_sidelobe_level(angles, cut1),
           compute_peak_sidelobe_level(angles, cut2))

    # Directivity and the hemisphere-wide sidelobe search on one grid
    theta_range = (0, np.pi / 2) if hemisphere else (0, np.pi)
    _, _, theta, phi = create_theta_phi_grid(theta_range, (0, 2 * np.pi),
                                             n_theta, n_phi)
    field = total_pattern(theta, phi, x, y, weights, k, element_pattern_func,
                          z, **element_kwargs)
    directivity = compute_directivity(theta, phi, field)

    power = np.abs(field) ** 2
    peak = power.max()
    level_dB = 10 * np.log10(np.maximum(power / peak, 1e-30))

    # Main lobe on the grid: every cell reachable from the peak by moving
    # downhill (or level), which stops at the null valleys around it
    main_lobe = _downhill_region(level_dB, np.unravel_index(np.argmax(power), power.shape))
    outside = ~main_lobe

    peak_sidelobe = float(level_dB[outside].max()) if np.any(outside) else -np.inf

    # Grating lobes: local maxima outside the main lobe above the threshold
    grating = []
    candidates = outside & (level_dB >= grating_lobe_threshold_dB)
    if np.any(candidates):
        is_max = _local_maxima(level_dB)
        found = [(float(np.rad2deg(theta[i, j])), float(np.rad2deg(phi[i, j])),
                  float(level_dB[i, j]))
                 for i, j in zip(*np.nonzero(candidates & is_max))]
        # One entry per lobe: merge duplicates at the pole and the phi seam
        for g in sorted(found, key=lambda t: -t[2]):
            if all(_angular_distance_deg(g, u) > 2.0 for u in grating):
                grating.append(g)

    return {
        'theta0_deg': float(theta0_deg),
        'phi0_deg': float(phi0_deg),
        'directivity_dBi': float(10 * np.log10(directivity)),
        'hpbw_deg': hpbw,
        'fnbw_deg': fnbw,
        'sll_dB': sll,
        'peak_sidelobe_dB': peak_sidelobe,
        'grating_lobes': grating,
    }


def _downhill_region(level: np.ndarray, start) -> np.ndarray:
    """Cells reachable from ``start`` through non-increasing levels.

    The grid is periodic in phi (axis 1); the theta = 0 row is a single
    point, so all of its cells are neighbors of each other.
    """
    n_theta, n_phi = level.shape
    mask = np.zeros(level.shape, dtype=bool)
    stack = [tuple(int(v) for v in start)]
    mask[stack[0]] = True
    tol = 1e-9
    while stack:
        i, j = stack.pop()
        current = level[i, j]
        neighbours = [(i + di, (j + dj) % n_phi)
                      for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        if i == 0:
            neighbours += [(0, jj) for jj in range(n_phi)]
        for a, b in neighbours:
            if 0 <= a < n_theta and not mask[a, b] and level[a, b] <= current + tol:
                mask[a, b] = True
                stack.append((a, b))
    return mask


def _local_maxima(level: np.ndarray) -> np.ndarray:
    """Cells at least as high as their 8 neighbors (phi periodic)."""
    padded = np.pad(level, ((1, 1), (0, 0)), mode='edge')
    padded = np.concatenate([padded[:, -1:], padded, padded[:, :1]], axis=1)
    centre = padded[1:-1, 1:-1]
    is_max = np.ones(level.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbour = padded[1 + di:padded.shape[0] - 1 + di,
                               1 + dj:padded.shape[1] - 1 + dj]
            is_max &= centre >= neighbour
    return is_max


def _angular_distance_deg(a, b) -> float:
    """Great-circle angle between (theta, phi) directions given in degrees."""
    ta, pa_, tb, pb = np.deg2rad([a[0], a[1], b[0], b[1]])
    cos_d = (np.sin(ta) * np.sin(tb) * np.cos(pa_ - pb) + np.cos(ta) * np.cos(tb))
    return float(np.rad2deg(np.arccos(np.clip(cos_d, -1.0, 1.0))))

