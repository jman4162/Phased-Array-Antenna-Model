"""
Timing and memory benchmark for the array-factor engine.

Compares ``array_factor_vectorized`` with two loop baselines on the same
grid and weights. Rectangular, triangular and thinned grids take the
lattice-factorized path; the random-position case shows the direct sum.

Baselines:

- element loop: a NumPy loop over elements, each step vectorized over all
  angles (the baseline used in ``Phased_Array_Demo.ipynb``)
- pure Python: nested loops over angles and elements (run on a reduced
  grid and scaled per evaluated point)

Run from the repository root::

    python benchmarks/bench_array_factor.py

Results depend on the machine and the NumPy/BLAS build, so the docs quote
ratios measured with this script rather than fixed speedups.
"""

import cmath
import math
import platform
import timeit
import tracemalloc

import numpy as np

import phased_array as pa


def element_loop(theta, phi, x, y, weights, k):
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    af = np.zeros(theta.shape, dtype=complex)
    for xn, yn, wn in zip(x, y, weights):
        af += wn * np.exp(1j * k * (xn * u + yn * v))
    return af


def pure_python(theta, phi, x, y, weights, k):
    out = []
    for th, ph in zip(theta.ravel(), phi.ravel()):
        u = math.sin(th) * math.cos(ph)
        v = math.sin(th) * math.sin(ph)
        acc = 0j
        for xn, yn, wn in zip(x, y, weights):
            acc += wn * cmath.exp(1j * k * (xn * u + yn * v))
        out.append(acc)
    return np.array(out)


def best_time(fn, repeat=5):
    return min(timeit.repeat(fn, number=1, repeat=repeat))


def main():
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, "
          f"{platform.machine()}")
    k = pa.wavelength_to_k(1.0)

    print("\nTiming (best of 5), 181 x 361 theta/phi grid:")
    print(f"{'array':>8} {'vectorized':>12} {'element loop':>14} {'pure Python*':>14}"
          f" {'vs loop':>9} {'vs Python':>10}")
    _, _, theta, phi = pa.create_theta_phi_grid(n_theta=181, n_phi=361)
    for n in (8, 16, 32):
        geom = pa.create_rectangular_array(n, n, 0.5, 0.5)
        w = pa.steering_vector(k, geom.x, geom.y, 30, 0)
        t_vec = best_time(lambda: pa.array_factor_vectorized(theta, phi, geom.x, geom.y, w, k))
        t_loop = best_time(lambda: element_loop(theta, phi, geom.x, geom.y, w, k), repeat=3)
        # Pure Python on 1/50 of the points, scaled to the full grid
        th_s, ph_s = theta.ravel()[::50], phi.ravel()[::50]
        t_py = best_time(lambda: pure_python(th_s, ph_s, geom.x, geom.y, w, k),
                         repeat=1) * theta.size / th_s.size
        print(f"{n:>4}x{n:<3} {t_vec * 1e3:>10.1f}ms {t_loop * 1e3:>12.1f}ms "
              f"{t_py:>13.1f}s {t_loop / t_vec:>8.1f}x {t_py / t_vec:>9.0f}x")
    # Irregular positions (no lattice): the direct chunked sum is used
    rng = np.random.default_rng(0)
    n_irr = 1024
    x_irr, y_irr = rng.uniform(-8, 8, (2, n_irr))
    w_irr = np.ones(n_irr, dtype=complex)
    t_vec = best_time(lambda: pa.array_factor_vectorized(theta, phi, x_irr, y_irr, w_irr, k))
    t_loop = best_time(lambda: element_loop(theta, phi, x_irr, y_irr, w_irr, k), repeat=3)
    print(f"{'1024 irr':>8} {t_vec * 1e3:>10.1f}ms {t_loop * 1e3:>12.1f}ms "
          f"{'':>14} {t_loop / t_vec:>8.1f}x")
    print("* pure Python timed on 1/50 of the grid and scaled; "
          "'irr' = random positions (no lattice)")

    print("\nPeak memory of one call, 64 x 64 array (4096 elements):")
    geom = pa.create_rectangular_array(64, 64, 0.5, 0.5)
    w = pa.steering_vector(k, geom.x, geom.y, 30, 0)
    tracemalloc.start()
    pa.array_factor_vectorized(theta, phi, geom.x, geom.y, w, k)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    unchunked = theta.size * geom.n_elements * 16
    print(f"  chunked: {peak / 2**20:.0f} MiB "
          f"(a single (angles x elements) complex matrix would be "
          f"{unchunked / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
