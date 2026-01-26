"""
UV-Space Page - Visualize pattern in direction cosine space.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

import phased_array as pa

st.set_page_config(page_title="UV-Space", page_icon="🌐", layout="wide")

st.title("🌐 UV-Space Visualization")
st.markdown("""
Visualize the radiation pattern in UV-space (direction cosines).
This representation clearly shows the visible region and grating lobes.
""")

# Check for geometry
if 'geometry' not in st.session_state or st.session_state.geometry is None:
    st.warning("⚠️ No array defined. Please go to **Array Design** first.")
    st.stop()

geom = st.session_state.geometry
params = st.session_state.array_params
wavelength = params.get('wavelength', 1.0)
k = pa.wavelength_to_k(wavelength)

# Get weights
if 'weights' not in st.session_state:
    st.session_state.weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)

weights = st.session_state.weights
steering = st.session_state.get('steering', {'theta': 0, 'phi': 0})

# Sidebar controls
st.sidebar.header("UV-Space Options")

n_points = st.sidebar.select_slider("Resolution", options=[101, 151, 201, 301], value=201)
min_dB = st.sidebar.slider("Min dB Level", -60, -20, -40, 5)
show_visible = st.sidebar.checkbox("Show Visible Region", value=True)
show_grating = st.sidebar.checkbox("Show Grating Lobe Circles", value=True)

# Compute pattern in UV-space
u, v, pattern_uv = pa.compute_pattern_uv_space(
    geom, weights, k,
    n_u=n_points, n_v=n_points
)

# Main plot
st.subheader("UV-Space Pattern")

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 10))

    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    pattern_clipped = np.clip(pattern_uv, min_dB, 0)

    cf = ax.contourf(u_grid, v_grid, pattern_clipped, levels=20, cmap='jet')
    plt.colorbar(cf, ax=ax, label='Gain (dB)')

    # Visible region circle
    if show_visible:
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'w--', linewidth=2,
                label='Visible Region')

    # Grating lobe circles
    if show_grating:
        dx = params.get('dx', 0.5)
        dy = params.get('dy', dx)

        # Grating lobes at u = u0 + m/dx, v = v0 + n/dy
        u0 = np.sin(np.deg2rad(steering['theta'])) * np.cos(np.deg2rad(steering['phi']))
        v0 = np.sin(np.deg2rad(steering['theta'])) * np.sin(np.deg2rad(steering['phi']))

        for m in [-1, 1]:
            u_grating = u0 + m / dx
            if abs(u_grating) < 2:
                ax.plot(np.cos(theta_circle) + u_grating, np.sin(theta_circle),
                        'r--', linewidth=1.5, alpha=0.7)

        for n in [-1, 1]:
            v_grating = v0 + n / dy
            if abs(v_grating) < 2:
                ax.plot(np.cos(theta_circle), np.sin(theta_circle) + v_grating,
                        'r--', linewidth=1.5, alpha=0.7)

    # Mark beam position
    u_beam = np.sin(np.deg2rad(steering['theta'])) * np.cos(np.deg2rad(steering['phi']))
    v_beam = np.sin(np.deg2rad(steering['theta'])) * np.sin(np.deg2rad(steering['phi']))
    ax.plot(u_beam, v_beam, 'w+', markersize=15, markeredgewidth=3, label='Beam')

    ax.set_xlabel('u = sin(θ)cos(φ)', fontsize=12)
    ax.set_ylabel('v = sin(θ)sin(φ)', fontsize=12)
    ax.set_title('UV-Space Radiation Pattern')
    ax.set_aspect('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("UV-Space Info")

    st.markdown("""
    **Direction Cosines:**
    - u = sin(θ)cos(φ)
    - v = sin(θ)sin(φ)

    **Visible Region:**
    - u² + v² ≤ 1
    - Points outside represent evanescent waves

    **Grating Lobes:**
    - Appear when element spacing > λ/2
    - Located at u = u₀ + m/dx
    """)

    st.markdown("---")

    st.subheader("Current Beam")
    st.write(f"**θ:** {steering['theta']}°")
    st.write(f"**φ:** {steering['phi']}°")
    st.write(f"**u:** {u_beam:.3f}")
    st.write(f"**v:** {v_beam:.3f}")

    st.markdown("---")

    st.subheader("Array Spacing")
    dx = params.get('dx', 0.5)
    dy = params.get('dy', dx)
    st.write(f"**dx:** {dx}λ")
    st.write(f"**dy:** {dy}λ")

    # Grating lobe onset
    if dx > 0:
        theta_grating = np.rad2deg(np.arcsin(min(1/dx - 1, 1))) if 1/dx > 1 else 90
        st.write(f"**Grating onset:** {theta_grating:.0f}°")

# Interactive Plotly version
st.subheader("Interactive 3D View")

try:
    fig = pa.plot_pattern_uv_plotly(
        u, v, pattern_uv,
        title="UV-Space Pattern (Interactive)",
        min_dB=min_dB,
        show_visible_circle=show_visible
    )
    st.plotly_chart(fig, use_container_width=True)
except ImportError:
    st.info("Install Plotly for interactive 3D plots: `pip install plotly`")
except Exception as e:
    st.warning(f"Could not create Plotly chart: {e}")

# U and V cuts
st.subheader("Principal Cuts")

col1, col2 = st.columns(2)

with col1:
    # U-cut (v=0)
    v_zero_idx = len(v) // 2
    u_cut = pattern_uv[:, v_zero_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(u, u_cut, 'b-', linewidth=1.5)
    ax.axvline(-1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-3, color='r', linestyle=':', alpha=0.5, label='-3 dB')
    ax.set_xlabel('u')
    ax.set_ylabel('Gain (dB)')
    ax.set_title('U-Cut (v = 0)')
    ax.set_ylim([min_dB, 5])
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close()

with col2:
    # V-cut (u=0)
    u_zero_idx = len(u) // 2
    v_cut = pattern_uv[u_zero_idx, :]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(v, v_cut, 'b-', linewidth=1.5)
    ax.axvline(-1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-3, color='r', linestyle=':', alpha=0.5, label='-3 dB')
    ax.set_xlabel('v')
    ax.set_ylabel('Gain (dB)')
    ax.set_title('V-Cut (u = 0)')
    ax.set_ylim([min_dB, 5])
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close()

# Export data
st.subheader("Export Data")

col1, col2 = st.columns(2)

with col1:
    # Export pattern as CSV
    if st.button("Generate CSV Data"):
        import io
        buffer = io.StringIO()
        buffer.write("u,v,pattern_dB\n")
        for i in range(len(u)):
            for j in range(len(v)):
                buffer.write(f"{u[i]:.4f},{v[j]:.4f},{pattern_uv[i,j]:.2f}\n")

        st.download_button(
            label="Download Pattern CSV",
            data=buffer.getvalue(),
            file_name="uv_pattern.csv",
            mime="text/csv"
        )

with col2:
    # Export weights
    if st.button("Generate Weights CSV"):
        import io
        buffer = io.StringIO()
        buffer.write("element,x,y,weight_real,weight_imag,weight_mag,weight_phase_deg\n")
        for i in range(geom.n_elements):
            w = weights[i]
            buffer.write(f"{i},{geom.x[i]:.4f},{geom.y[i]:.4f},"
                        f"{w.real:.6f},{w.imag:.6f},"
                        f"{np.abs(w):.6f},{np.rad2deg(np.angle(w)):.2f}\n")

        st.download_button(
            label="Download Weights CSV",
            data=buffer.getvalue(),
            file_name="array_weights.csv",
            mime="text/csv"
        )

st.success("✅ UV-Space analysis complete!")
