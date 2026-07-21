"""
Subarrays and Spoiling Page - Overlapped subarray architectures and beam spoiling.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Subarrays & Spoiling", page_icon="🧩", layout="wide")

st.title("🧩 Subarrays & Beam Spoiling")
st.markdown("""
Two aperture-level techniques:
- **Overlapped subarrays**: sharing tapered elements between adjacent subarrays smooths
  the effective subarray pattern and suppresses the grating (quantization) lobes caused
  by subarray-level-only phase steering.
- **Beam spoiling**: a quadratic phase profile defocuses the beam, trading gain for
  beamwidth (search modes).
""")

# Sidebar controls
st.sidebar.header("Array Configuration")

N = st.sidebar.select_slider("Array Size (N×N)", options=[8, 16, 32], value=16)
spacing = st.sidebar.number_input("Spacing (λ)", min_value=0.25, max_value=1.0, value=0.5, step=0.05)

k = pa.wavelength_to_k(1.0)

tab1, tab2 = st.tabs(["Overlapped Subarrays", "Beam Spoiling"])

with tab1:
    st.subheader("Overlapped vs Contiguous Subarrays")

    col1, col2 = st.columns([1, 2])

    with col1:
        sub_options = [i for i in [2, 4, 8] if N % i == 0 and i < N]
        N_sub = st.selectbox("Subarray Size", sub_options, index=min(1, len(sub_options) - 1))
        overlap = st.selectbox(
            "Overlap (elements)",
            [i for i in [1, 2, 4] if i < N_sub],
            index=min(1, len([i for i in [1, 2, 4] if i < N_sub]) - 1)
        )
        scan_deg = st.slider("Scan Angle (deg)", 0, 30, 10)

        arch_contig = pa.create_rectangular_subarrays(N, N, N_sub, N_sub, spacing, spacing)
        arch_over = pa.create_overlapped_subarrays(
            N, N, Nx_sub=N_sub, Ny_sub=N_sub,
            overlap_x=overlap, overlap_y=overlap,
            dx=spacing, dy=spacing
        )

        st.metric("Contiguous Subarrays", arch_contig.n_subarrays)
        st.metric("Overlapped Subarrays", arch_over.n_subarrays)

        # Predicted grating lobe location for subarray-level steering:
        # u_gl = sin(theta0) - lambda / d_sub
        d_sub = N_sub * spacing  # in wavelengths
        u_gl = np.sin(np.deg2rad(scan_deg)) + 1.0 / d_sub
        if abs(u_gl) <= 1:
            gl_deg = np.rad2deg(np.arcsin(u_gl))
            st.write(f"**Predicted grating lobe:** {gl_deg:.1f}°")
        else:
            gl_deg = None
            st.write("**Predicted grating lobe:** outside visible space")

    with col2:
        # Pattern cuts: subarray-center phase steering only.
        # compute_overlapped_pattern falls back to contiguous weights
        # for a non-overlapped architecture.
        th_sub, pat_contig = pa.compute_overlapped_pattern(arch_contig, k, scan_deg, 0)
        _, pat_over = pa.compute_overlapped_pattern(arch_over, k, scan_deg, 0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(th_sub, pat_contig, 'r-', linewidth=1.5, label='Contiguous')
        ax.plot(th_sub, pat_over, 'b-', linewidth=1.5, label='Overlapped')
        ax.axvline(scan_deg, color='gray', linestyle=':', alpha=0.7, label=f'Scan ({scan_deg}°)')
        if gl_deg is not None:
            ax.axvline(gl_deg, color='black', linestyle='--', alpha=0.5, label='Predicted GL')
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel('Normalized Gain (dB)')
        ax.set_ylim([-40, 2])
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Subarray-level steering to {scan_deg}° ({N_sub}×{N_sub} subarrays)')
        st.pyplot(fig)
        plt.close()

        if gl_deg is not None:
            i_gl = np.argmin(np.abs(th_sub - gl_deg))
            suppression = pat_contig[i_gl] - pat_over[i_gl]
            st.metric(
                "Grating Lobe Suppression",
                f"{suppression:.1f} dB",
                help=f"Contiguous {pat_contig[i_gl]:.1f} dB vs overlapped {pat_over[i_gl]:.1f} dB at {gl_deg:.1f}°"
            )

with tab2:
    st.subheader("Quadratic Phase Beam Spoiling")

    col1, col2 = st.columns([1, 2])

    geom = pa.create_rectangular_array(N, N, spacing, spacing)

    with col1:
        spoil_factor = st.slider("Spoil Factor", 0.5, 5.0, 3.0, 0.5)
        spoil_axis = st.selectbox("Spoil Axis", ["both", "x", "y"])

        st.markdown(f"""
        Predicted broadening: `sqrt(1 + {spoil_factor}²)` =
        **{np.sqrt(1 + spoil_factor**2):.2f}×**
        """)

    with col2:
        w_pencil = pa.steering_vector(k, geom.x, geom.y, 0, 0)
        w_spoiled = pa.quadratic_phase_spoil(
            geom, k, theta0_deg=0, phi0_deg=0,
            spoil_factor=spoil_factor, axis=spoil_axis
        )

        angles, E_pencil, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_pencil, k)
        _, E_spoiled, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_spoiled, k)

        def beamwidth_3dB(ang, E_dB):
            """Contiguous -3 dB width around the pattern peak."""
            i = np.argmax(E_dB)
            lo, hi = i, i
            while lo > 0 and E_dB[lo - 1] >= -3.0:
                lo -= 1
            while hi < len(E_dB) - 1 and E_dB[hi + 1] >= -3.0:
                hi += 1
            return ang[hi] - ang[lo]

        bw_pencil = beamwidth_3dB(angles, E_pencil)
        bw_spoiled = beamwidth_3dB(angles, E_spoiled)
        bw_predicted = pa.spoiled_beamwidth(bw_pencil, spoil_factor)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(angles, E_pencil, 'b-', linewidth=1.5, label=f'Pencil ({bw_pencil:.1f}°)')
        ax.plot(angles, E_spoiled, 'r-', linewidth=1.5, label=f'Spoiled ({bw_spoiled:.1f}°)')
        ax.axhline(-3, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel('Normalized Gain (dB)')
        ax.set_ylim([-50, 2])
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Beam spoiling, factor {spoil_factor} ({spoil_axis} axis)')
        st.pyplot(fig)
        plt.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pencil Beamwidth", f"{bw_pencil:.1f}°")
    with col2:
        st.metric("Spoiled Beamwidth", f"{bw_spoiled:.1f}°")
    with col3:
        st.metric(
            "Measured Broadening",
            f"{bw_spoiled / bw_pencil:.2f}×",
            help=f"Model prediction: {bw_predicted / bw_pencil:.2f}×"
        )

st.success("✅ Subarray and spoiling analysis complete!")
