"""
Beam Steering Page - Interactive beam direction control.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Beam Steering", page_icon="🎯", layout="wide")

st.title("🎯 Beam Steering")
st.markdown("Control beam direction and visualize radiation patterns.")

# Check for geometry
if 'geometry' not in st.session_state or st.session_state.geometry is None:
    st.warning("⚠️ No array defined. Please go to **Array Design** first.")
    st.stop()

geom = st.session_state.geometry
params = st.session_state.array_params
wavelength = params.get('wavelength', 1.0)
k = pa.wavelength_to_k(wavelength)

# Sidebar controls
st.sidebar.header("Beam Direction")

theta0 = st.sidebar.slider(
    "Theta (θ) - Elevation",
    min_value=0,
    max_value=80,
    value=0,
    step=1,
    help="Scan angle from broadside (degrees)"
)

phi0 = st.sidebar.slider(
    "Phi (φ) - Azimuth",
    min_value=0,
    max_value=360,
    value=0,
    step=5,
    help="Azimuth angle (degrees)"
)

st.sidebar.markdown("---")
st.sidebar.header("Display Options")

show_3d = st.sidebar.checkbox("Show 3D Pattern", value=False)
n_points = st.sidebar.select_slider("Resolution", options=[91, 181, 361], value=181)

# Compute weights
weights = pa.steering_vector(k, geom.x, geom.y, theta0, phi0)

# Store weights in session state for other pages
st.session_state.weights = weights
st.session_state.steering = {'theta': theta0, 'phi': phi0}

# Display beam info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Theta (θ)", f"{theta0}°")
with col2:
    st.metric("Phi (φ)", f"{phi0}°")
with col3:
    st.metric("Elements", geom.n_elements)
with col4:
    # Scan loss estimate (cos(theta) for element factor)
    scan_loss = 20 * np.log10(max(np.cos(np.deg2rad(theta0)), 0.01))
    st.metric("Scan Loss", f"{scan_loss:.1f} dB")

# Compute pattern cuts
angles, E_plane, H_plane = pa.compute_pattern_cuts(
    geom.x, geom.y, weights, k,
    theta0_deg=theta0, phi0_deg=phi0,
    n_points=n_points
)

# Compute beamwidths
hpbw_e = pa.compute_half_power_beamwidth(angles, E_plane)
hpbw_h = pa.compute_half_power_beamwidth(angles, H_plane)

# Plots
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"E-Plane Cut (φ = {phi0}°)")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles, E_plane, 'b-', linewidth=1.5)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3 dB')
    ax.axvline(theta0, color='g', linestyle=':', alpha=0.5, label=f'Scan: {theta0}°')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Gain (dB)')
    ax.set_ylim([-50, 5])
    ax.set_xlim([-90, 90])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"HPBW: {hpbw_e:.1f}°")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader(f"H-Plane Cut (φ = {phi0 + 90}°)")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles, H_plane, 'b-', linewidth=1.5)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3 dB')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Gain (dB)')
    ax.set_ylim([-50, 5])
    ax.set_xlim([-90, 90])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"HPBW: {hpbw_h:.1f}°")
    st.pyplot(fig)
    plt.close()

# 3D plot (optional)
if show_3d:
    st.subheader("3D Radiation Pattern")

    try:
        theta_3d, phi_3d, pattern_3d = pa.compute_full_pattern(
            geom.x, geom.y, weights, k,
            n_theta=61, n_phi=121
        )

        fig = pa.plot_pattern_3d_plotly(
            theta_3d, phi_3d, pattern_3d,
            title=f"3D Pattern - Scan θ={theta0}°, φ={phi0}°",
            surface_type='spherical'
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Plotly not installed. Run: `pip install plotly`")

# Polar plot
st.subheader("Polar Pattern")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    r = E_plane - E_plane.min()  # Shift to positive
    ax.plot(np.deg2rad(angles), r)
    ax.set_title(f"E-Plane (φ = {phi0}°)")
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    r = H_plane - H_plane.min()
    ax.plot(np.deg2rad(angles), r)
    ax.set_title(f"H-Plane (φ = {phi0 + 90}°)")
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    st.pyplot(fig)
    plt.close()

# Export section
st.markdown("---")
st.subheader("Export Pattern Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export E-Plane CSV"):
        csv_data = pa.export_pattern_csv(
            angles, E_plane,
            angle_label="angle_deg",
            pattern_label="E_plane_dB",
            metadata={
                'cut': 'E-plane',
                'theta': theta0,
                'phi': phi0,
                'n_elements': geom.n_elements
            }
        )
        st.download_button(
            label="Download E-Plane CSV",
            data=csv_data,
            file_name=f"e_plane_theta{theta0}_phi{phi0}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export H-Plane CSV"):
        csv_data = pa.export_pattern_csv(
            angles, H_plane,
            angle_label="angle_deg",
            pattern_label="H_plane_dB",
            metadata={
                'cut': 'H-plane',
                'theta': theta0,
                'phi': phi0 + 90,
                'n_elements': geom.n_elements
            }
        )
        st.download_button(
            label="Download H-Plane CSV",
            data=csv_data,
            file_name=f"h_plane_theta{theta0}_phi{phi0}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("Export Weights CSV"):
        csv_data = pa.export_weights_csv(
            geom, weights,
            metadata={
                'steering_theta': theta0,
                'steering_phi': phi0,
                'taper': 'uniform'
            }
        )
        st.download_button(
            label="Download Weights CSV",
            data=csv_data,
            file_name=f"weights_theta{theta0}_phi{phi0}.csv",
            mime="text/csv"
        )

st.success("✅ Pattern computed! Try **Tapering** to reduce sidelobes.")
