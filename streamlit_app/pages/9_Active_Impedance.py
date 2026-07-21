"""
Active Impedance Page - Scan-dependent VSWR and active reflection coefficient.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Active Impedance", page_icon="🔌", layout="wide")

st.title("🔌 Active Impedance & VSWR vs Scan")
st.markdown("""
When the array scans, each element sees coupled power from its neighbors, so its
reflection coefficient depends on the excitation:
`Γ_active,n = Σ_m C_nm w_m / w_n`. Active VSWR rises with scan angle and can spike at
scan-blindness conditions; front-ends are therefore specified against active (scan)
impedance, not isolated-element match.
""")

# Sidebar controls
st.sidebar.header("Array Configuration")

N = st.sidebar.select_slider("Array Size (N×N)", options=[4, 8, 12, 16], value=8)
spacing = st.sidebar.number_input("Spacing (λ)", min_value=0.25, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Coupling Model")

coupling_model = st.sidebar.selectbox("Model", ["sinc", "exponential"])
coupling_coeff = st.sidebar.slider("Coupling Strength", 0.05, 0.5, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("Scan Sweep")

max_scan = st.sidebar.slider("Max Scan Angle (deg)", 30, 80, 60)
phi_plane = st.sidebar.selectbox("Scan Plane φ (deg)", [0, 45, 90])
scan_readout = st.sidebar.slider("Readout Scan Angle (deg)", 0, 80, 45)

# Build array and coupling matrix
geom = pa.create_rectangular_array(N, N, spacing, spacing)
k = pa.wavelength_to_k(1.0)

C = pa.mutual_coupling_matrix_theoretical(
    geom, k, coupling_model=coupling_model, coupling_coeff=coupling_coeff
)

# VSWR vs scan sweep
theta_scan, vswr_all, vswr_max = pa.vswr_vs_scan(
    geom, C, k, theta_range=(0, max_scan), n_angles=41, phi_deg=float(phi_plane)
)

# Active reflection coefficients at the readout angle
w_scan = pa.steering_vector(k, geom.x, geom.y, scan_readout, float(phi_plane))
gammas = np.array([
    pa.active_reflection_coefficient(C, w_scan, i) for i in range(geom.n_elements)
])
gamma_mag = np.abs(gammas)
worst_idx = int(np.argmax(gamma_mag))
worst_gamma = gamma_mag[worst_idx]
worst_vswr = (1 + worst_gamma) / (1 - worst_gamma) if worst_gamma < 1 else np.inf

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Elements", geom.n_elements)
with col2:
    st.metric("Worst |Γ| at Readout", f"{worst_gamma:.3f}",
              help=f"Element {worst_idx} at θ={scan_readout}°, φ={phi_plane}°")
with col3:
    st.metric("Worst VSWR at Readout", f"{worst_vswr:.2f}" if np.isfinite(worst_vswr) else "∞")
with col4:
    st.metric("Return Loss", f"{-20 * np.log10(worst_gamma):.1f} dB" if worst_gamma > 0 else "∞")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("VSWR vs Scan Angle")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(theta_scan, vswr_max, 'r-', linewidth=2, label='Worst element')
    ax.plot(theta_scan, np.median(vswr_all, axis=1), 'b--', linewidth=1.5, label='Median element')
    ax.fill_between(theta_scan, np.min(vswr_all, axis=1), vswr_max,
                    alpha=0.15, color='red', label='Element spread')
    ax.axhline(2.0, color='gray', linestyle=':', label='VSWR = 2 spec line')
    ax.axvline(scan_readout, color='green', linestyle='--', alpha=0.7,
               label=f'Readout ({scan_readout}°)')
    ax.set_xlabel('Scan Angle (degrees)')
    ax.set_ylabel('Active VSWR')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Active VSWR vs scan ({coupling_model} coupling, coeff={coupling_coeff})')
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Element |Γ| Map")
    st.markdown(f"Active reflection magnitude at θ={scan_readout}°, φ={phi_plane}°.")

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(geom.x, geom.y, c=gamma_mag, cmap='hot', s=80,
                         edgecolors='black', linewidths=0.5)
    ax.scatter(geom.x[worst_idx], geom.y[worst_idx], facecolors='none',
               edgecolors='lime', s=250, linewidths=2, label=f'Worst (el. {worst_idx})')
    plt.colorbar(scatter, ax=ax, label='|Γ|')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Active Reflection per Element')
    st.pyplot(fig)
    plt.close()

st.markdown("""
**Reading the plots:** edge and corner elements typically show different active match
than interior elements because they see asymmetric coupling. Raising the coupling
strength or scanning further from broadside pushes the worst-element VSWR up.
""")

# Export section
st.markdown("---")
st.subheader("Export VSWR Data")

if st.button("Export VSWR vs Scan CSV"):
    import io
    buffer = io.StringIO()
    buffer.write("scan_deg,vswr_max,vswr_median\n")
    med = np.median(vswr_all, axis=1)
    for i in range(len(theta_scan)):
        buffer.write(f"{theta_scan[i]:.2f},{vswr_max[i]:.4f},{med[i]:.4f}\n")

    st.download_button(
        label="Download VSWR CSV",
        data=buffer.getvalue(),
        file_name=f"vswr_vs_scan_{coupling_model}_c{coupling_coeff}.csv",
        mime="text/csv"
    )

st.success("✅ Active impedance analysis complete!")
