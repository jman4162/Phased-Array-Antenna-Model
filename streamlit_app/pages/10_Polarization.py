"""
Polarization Page - Polarized element patterns, co/cross cuts, axial ratio.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Polarization", page_icon="🌀", layout="wide")

st.title("🌀 Polarized Array Patterns")
st.markdown("""
The v1.4 vector pattern engine carries full `(E_θ, E_φ)` fields through the array
computation. Pick a polarized element model and inspect co/cross-polarization cuts
(Ludwig-3 for linear elements, RHCP/LHCP decomposition for circular) and the axial
ratio at boresight.
""")

# Sidebar controls
st.sidebar.header("Element Model")

element_type = st.sidebar.selectbox(
    "Element Type",
    ["X Dipole", "Ideal Patch (x-pol)", "Crossed Dipole (RHCP)", "Crossed Dipole (LHCP)"]
)
is_circular = element_type.startswith("Crossed")

st.sidebar.markdown("---")
st.sidebar.header("Array Configuration")

N = st.sidebar.select_slider("Array Size (N×N)", options=[4, 8, 16], value=16)
spacing = st.sidebar.number_input("Spacing (λ)", min_value=0.25, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Scan & Cut")

theta_scan = st.sidebar.slider("Scan θ (deg)", 0, 60, 0)
phi_cut = st.sidebar.selectbox("φ Cut Plane (deg)", [0, 45, 90], index=1)
st.sidebar.caption("The beam is steered within the selected cut plane so the peak stays visible.")

# Element factory
if element_type == "X Dipole":
    element_func = pa.dipole_element('x')
elif element_type == "Ideal Patch (x-pol)":
    element_func = pa.ideal_patch_element('x')
elif element_type == "Crossed Dipole (RHCP)":
    element_func = pa.crossed_dipole_element(-np.pi / 2)
else:  # LHCP
    element_func = pa.crossed_dipole_element(np.pi / 2)

# Array and weights (steer in the cut plane)
geom = pa.create_rectangular_array(int(N), int(N), spacing, spacing)
k = pa.wavelength_to_k(1.0)
weights = pa.steering_vector(k, geom.x, geom.y, theta_scan, phi_cut)

if is_circular:
    # Vector pattern on a hemisphere grid, then RHCP/LHCP decomposition.
    # In this package's convention: E_RHCP = (E_co + j*E_cx)/sqrt(2) from the
    # Ludwig-3 pair, E_LHCP the conjugate combination.
    n_theta, n_phi = 91, 361  # 1-degree grids
    vp = pa.compute_full_vector_pattern(
        geom.x, geom.y, weights, k,
        element_func=element_func, n_theta=n_theta, n_phi=n_phi
    )
    E_co_l3, E_cx_l3 = vp.co_cross()
    E_rhcp = (E_co_l3 + 1j * E_cx_l3) / np.sqrt(2)
    E_lhcp = (E_co_l3 - 1j * E_cx_l3) / np.sqrt(2)
    if "RHCP" in element_type:
        co_grid, cx_grid = E_rhcp, E_lhcp
        co_label, cx_label = "Co-pol (RHCP)", "Cross-pol (LHCP)"
    else:
        co_grid, cx_grid = E_lhcp, E_rhcp
        co_label, cx_label = "Co-pol (LHCP)", "Cross-pol (RHCP)"

    # Stitch a -90..+90 cut from phi_cut and phi_cut + 180 columns
    ip = int(round(phi_cut))            # 1-degree phi grid
    ip2 = (ip + 180) % 360
    theta_1d = np.linspace(0, 90, n_theta)
    theta_cut_deg = np.concatenate([-theta_1d[::-1], theta_1d[1:]])
    co_cut = np.concatenate([np.abs(co_grid[::-1, ip2]), np.abs(co_grid[1:, ip])])
    cx_cut = np.concatenate([np.abs(cx_grid[::-1, ip2]), np.abs(cx_grid[1:, ip])])

    ref = np.max(co_cut)
    co_dB = 20 * np.log10(np.maximum(co_cut / ref, 1e-6))
    cx_dB = 20 * np.log10(np.maximum(cx_cut / ref, 1e-6))

    ar_map_dB = 20 * np.log10(vp.axial_ratio_map())
    boresight_ar = ar_map_dB[0, 0]
    i_peak_grid = np.argmin(np.abs(theta_1d - theta_scan))
    peak_ar = ar_map_dB[i_peak_grid, ip]
else:
    # Linear elements: Ludwig-3 co/cross cuts (normalized to co-pol peak)
    theta_cut_deg, co_dB, cx_dB = pa.compute_co_cross_pattern_cuts(
        geom.x, geom.y, weights, k,
        element_func=element_func, phi_cut_deg=float(phi_cut)
    )
    co_label, cx_label = "Co-pol (Ludwig-3)", "Cross-pol (Ludwig-3)"
    boresight_ar = None
    peak_ar = None

# Metric row
i_peak = int(np.nanargmax(co_dB))
max_cross = float(np.nanmax(cx_dB))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Elements", geom.n_elements)
with col2:
    st.metric("Co-pol Peak Direction", f"{theta_cut_deg[i_peak]:.1f}°",
              help=f"Commanded scan: {theta_scan}° in the φ={phi_cut}° plane")
with col3:
    st.metric("Max Cross-pol", f"{max_cross:.1f} dB",
              help="Relative to the co-pol peak, over the displayed cut")
with col4:
    if is_circular:
        st.metric("Boresight Axial Ratio", f"{boresight_ar:.2f} dB",
                  help=f"AR at the {theta_scan}° scanned peak: {peak_ar:.2f} dB")
    else:
        st.metric("Boresight Axial Ratio", "∞ (linear)")

# Pattern plot
st.subheader(f"Co/Cross-Polarization Cut (φ = {phi_cut}°)")

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(theta_cut_deg, co_dB, 'b-', linewidth=1.8, label=co_label)
ax.plot(theta_cut_deg, cx_dB, 'r--', linewidth=1.5, label=cx_label)
ax.axvline(theta_scan, color='gray', linestyle=':', alpha=0.7, label=f'Scan ({theta_scan}°)')
ax.set_xlabel('Theta (degrees)')
ax.set_ylabel('Normalized Gain (dB)')
ax.set_ylim([-80, 5])
ax.set_xlim([-90, 90])
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(f'{element_type} elements, {N}×{N} array')
st.pyplot(fig)
plt.close()

if is_circular:
    st.subheader("Axial Ratio Map")
    phi_1d = np.linspace(0, 360, 361)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.pcolormesh(phi_1d, np.linspace(0, 90, 91),
                       np.clip(ar_map_dB, 0, 20), cmap='viridis_r', shading='auto')
    fig.colorbar(im, ax=ax, label='Axial Ratio (dB)')
    ax.axvline(phi_cut, color='white', linestyle='--', alpha=0.7)
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')
    ax.set_title('Axial ratio over the forward hemisphere (0 dB = circular)')
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Reading the plots:** the crossed-dipole element is perfectly circular at
    boresight; axial ratio degrades off-axis and with scan because the two dipole
    projections become unequal.
    """)
else:
    st.markdown("""
    **Reading the plot:** Ludwig-3 cross-pol for an x-polarized element peaks in the
    diagonal plane (φ = 45°) and vanishes in the principal planes. The ideal patch
    model has zero cross-pol by construction, so its curve sits at the numerical floor.
    """)

# Export section
st.markdown("---")
st.subheader("Export Pattern Cut")

if st.button("Export Co/Cross CSV"):
    import io
    buffer = io.StringIO()
    buffer.write("theta_deg,co_dB,cross_dB\n")
    for i in range(len(theta_cut_deg)):
        buffer.write(f"{theta_cut_deg[i]:.2f},{co_dB[i]:.4f},{cx_dB[i]:.4f}\n")

    st.download_button(
        label="Download Cut CSV",
        data=buffer.getvalue(),
        file_name=f"co_cross_{element_type.replace(' ', '_')}_phi{phi_cut}.csv",
        mime="text/csv"
    )

st.success("✅ Polarization analysis complete!")
