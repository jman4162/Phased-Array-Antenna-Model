"""
Pattern summary metrics for an array excitation.

:func:`pattern_summary` collects the figures usually quoted for a beam:
direction, directivity, half-power and first-null beamwidths and peak
sidelobe level in two orthogonal cuts, the peak sidelobe over the whole
visible hemisphere, and directions where a grating lobe reaches the
main-beam level.
"""

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    Sidelobes and grating lobes are searched over the front hemisphere
    (theta <= 90 deg) whatever ``hemisphere`` says, so the mirror image of
    the beam behind a planar array is not reported. On that grid the main
    lobe is grown from the beam direction through cells that rise no more
    than 1 dB above the lowest level on the path, so it follows the null
    valleys around a pencil beam and the whole ridge of a fan beam. Each
    group of connected cells above the threshold outside it is one
    grating lobe. Levels and directivity are referred to the pattern at
    ``(theta0_deg, phi0_deg)`` when that exceeds the grid maximum. The grid
    must resolve the nulls: for arrays more than about 60 wavelengths
    across, raise ``n_theta`` and ``n_phi``.

    Without a beam direction, the array-factor peak is located and, when
    an element pattern is given, refined on the total pattern.

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
        theta_b, phi_b, beamwidth = _beam_direction(x, y, z, weights, k)
        if element_pattern_func is not None:
            # The element pattern pulls the total-pattern peak off the
            # array-factor peak; refine on the total pattern
            theta_b, phi_b = _refine_total_peak(
                x, y, z, weights, k, element_pattern_func, element_kwargs,
                theta_b, phi_b, beamwidth)
        theta0_deg = float(np.rad2deg(theta_b)) if theta0_deg is None else theta0_deg
        phi0_deg = float(np.rad2deg(phi_b)) if phi0_deg is None else phi0_deg

    # Two orthogonal cuts through the beam, sampled finely enough for the
    # narrowest beams likely here (0.01 deg steps)
    angles, cut1, cut2 = compute_pattern_cuts(
        x, y, weights, k, theta0_deg=theta0_deg, phi0_deg=phi0_deg,
        n_points=18001, element_pattern_func=element_pattern_func, z=z,
        **element_kwargs
    )
    hpbw = (float(compute_half_power_beamwidth(angles, cut1)),
            float(compute_half_power_beamwidth(angles, cut2)))
    fnbw = (float(compute_first_null_beamwidth(angles, cut1)),
            float(compute_first_null_beamwidth(angles, cut2)))
    sll = (float(compute_peak_sidelobe_level(angles, cut1)),
           float(compute_peak_sidelobe_level(angles, cut2)))

    # Reference peak: the beam direction itself, which a coarse grid can
    # miss by a fraction of a beamwidth
    th0, ph0 = np.deg2rad(theta0_deg), np.deg2rad(phi0_deg)
    beam_field = total_pattern(np.array([th0]), np.array([ph0]), x, y, weights,
                               k, element_pattern_func, z, **element_kwargs)
    beam_power = float(np.abs(beam_field[0]) ** 2)

    # Directivity on the hemisphere or full-sphere grid
    theta_range = (0, np.pi / 2) if hemisphere else (0, np.pi)
    _, _, theta, phi = create_theta_phi_grid(theta_range, (0, 2 * np.pi),
                                             n_theta, n_phi)
    field = total_pattern(theta, phi, x, y, weights, k, element_pattern_func,
                          z, **element_kwargs)
    power = np.abs(field) ** 2
    grid_peak = float(power.max())
    reference = max(grid_peak, beam_power)
    directivity = compute_directivity(theta, phi, field) * reference / grid_peak

    # Sidelobe and grating-lobe search over the front hemisphere, without
    # the phi = 2*pi column that repeats phi = 0
    front = theta[:, 0] <= np.pi / 2 + 1e-12
    s_theta, s_phi = theta[front, :-1], phi[front, :-1]
    level_dB = 10 * np.log10(np.maximum(power[front, :-1] / reference, 1e-30))

    start = _climb(level_dB, _nearest_cell(s_theta, s_phi, th0, ph0))
    main_lobe = _main_lobe_region(level_dB, start)
    outside = ~main_lobe

    peak_sidelobe = float(level_dB[outside].max()) if np.any(outside) else -np.inf

    # Grating lobes: one entry per connected group of cells outside the
    # main lobe at or above the threshold, at its highest cell
    grating = []
    candidates = outside & (level_dB >= grating_lobe_threshold_dB)
    for cells in _components(candidates):
        i, j = max(cells, key=lambda c: level_dB[c])
        grating.append((float(np.rad2deg(s_theta[i, j])),
                        float(np.rad2deg(s_phi[i, j])),
                        float(level_dB[i, j])))
    grating.sort(key=lambda g: -g[2])

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


def _neighbours(i: int, j: int, n_theta: int, n_phi: int) -> List[Tuple[int, int]]:
    """8-neighbors on a theta/phi grid, periodic in phi; row 0 is the pole."""
    if i == 0:
        cells = [(0, jj) for jj in range(n_phi)]
        cells += [(1, jj) for jj in range(n_phi)] if n_theta > 1 else []
        return cells
    cells = [(i + di, (j + dj) % n_phi)
             for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    return [(a, b) for a, b in cells if 0 <= a < n_theta]


def _nearest_cell(theta: np.ndarray, phi: np.ndarray, th0: float, ph0: float):
    """Grid cell closest in angle to the direction (th0, ph0)."""
    cos_d = (np.sin(theta) * np.sin(th0) * np.cos(phi - ph0)
             + np.cos(theta) * np.cos(th0))
    return np.unravel_index(int(np.argmax(cos_d)), theta.shape)


def _climb(level: np.ndarray, start) -> Tuple[int, int]:
    """Steepest ascent from ``start`` to a local maximum."""
    n_theta, n_phi = level.shape
    i, j = (int(v) for v in start)
    while True:
        best = max(_neighbours(i, j, n_theta, n_phi), key=lambda c: level[c])
        if level[best] <= level[i, j]:
            return i, j
        i, j = best


def _main_lobe_region(level: np.ndarray, start, tolerance_dB: float = 1.0) -> np.ndarray:
    """Cells reachable from ``start`` without rising more than
    ``tolerance_dB`` above the lowest level met on the path.

    Processing cells in order of decreasing path minimum gives every cell
    its most permissive path; grid ripple along a ridge is crossed, a null
    valley is not.
    """
    n_theta, n_phi = level.shape
    region = np.zeros(level.shape, dtype=bool)
    best = np.full(level.shape, -np.inf)
    i0, j0 = (int(v) for v in start)
    best[i0, j0] = level[i0, j0]
    heap = [(-level[i0, j0], i0, j0)]
    while heap:
        neg_min, i, j = heapq.heappop(heap)
        if region[i, j]:
            continue
        region[i, j] = True
        path_min = -neg_min
        for a, b in _neighbours(i, j, n_theta, n_phi):
            if region[a, b] or level[a, b] > path_min + tolerance_dB:
                continue
            new_min = min(path_min, level[a, b])
            if new_min > best[a, b]:
                best[a, b] = new_min
                heapq.heappush(heap, (-new_min, a, b))
    return region


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Connected groups of True cells (8-connected, phi periodic, pole)."""
    n_theta, n_phi = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    for i, j in zip(*np.nonzero(mask)):
        if seen[i, j]:
            continue
        stack, cells = [(int(i), int(j))], []
        seen[i, j] = True
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for c in _neighbours(a, b, n_theta, n_phi):
                if mask[c] and not seen[c]:
                    seen[c] = True
                    stack.append(c)
        groups.append(cells)
    return groups


def _refine_total_peak(x, y, z, weights, k, element_pattern_func,
                       element_kwargs, theta_b, phi_b, beamwidth):
    """Nelder-Mead on the total pattern, starting at the array-factor peak."""
    from scipy.optimize import minimize

    def neg_power(p):
        return -float(np.abs(total_pattern(
            np.array([abs(p[0])]),
            np.array([p[1] + (np.pi if p[0] < 0 else 0.0)]),
            x, y, weights, k, element_pattern_func, z, **element_kwargs)[0]) ** 2)

    step = beamwidth / 4
    x0 = np.array([theta_b, phi_b])
    res = minimize(neg_power, x0, method='Nelder-Mead',
                   options={'initial_simplex': [x0, x0 + [step, 0], x0 + [0, step]],
                            'xatol': 1e-7, 'fatol': 1e-12, 'maxiter': 400})
    theta_r, phi_r = res.x
    if theta_r < 0:
        theta_r, phi_r = -theta_r, phi_r + np.pi
    return float(theta_r), float(np.mod(phi_r, 2 * np.pi))
