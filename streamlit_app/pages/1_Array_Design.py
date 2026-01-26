"""
Array Design Page - Create and configure array geometries.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Array Design", page_icon="🔲", layout="wide")

st.title("🔲 Array Design")
st.markdown("Create and configure your phased array geometry.")

# Store geometry in session state
if 'geometry' not in st.session_state:
    st.session_state.geometry = None
if 'array_params' not in st.session_state:
    st.session_state.array_params = {}

# Sidebar controls
st.sidebar.header("Array Configuration")

array_type = st.sidebar.selectbox(
    "Array Type",
    ["Rectangular", "Triangular", "Circular Ring", "Concentric Rings", "Elliptical"]
)

wavelength = st.sidebar.number_input(
    "Wavelength (m)",
    min_value=0.001,
    max_value=10.0,
    value=1.0,
    help="Operating wavelength in meters"
)

st.sidebar.markdown("---")

# Array-specific parameters
if array_type == "Rectangular":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        Nx = st.number_input("Elements X", min_value=2, max_value=64, value=16)
    with col2:
        Ny = st.number_input("Elements Y", min_value=2, max_value=64, value=16)

    col3, col4 = st.sidebar.columns(2)
    with col3:
        dx = st.number_input("Spacing X (λ)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    with col4:
        dy = st.number_input("Spacing Y (λ)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    if st.sidebar.button("Create Array", type="primary"):
        st.session_state.geometry = pa.create_rectangular_array(Nx, Ny, dx, dy, wavelength)
        st.session_state.array_params = {
            'type': 'Rectangular', 'Nx': Nx, 'Ny': Ny, 'dx': dx, 'dy': dy,
            'wavelength': wavelength
        }

elif array_type == "Triangular":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        Nx = st.number_input("Elements X", min_value=2, max_value=64, value=16)
    with col2:
        Ny = st.number_input("Rows Y", min_value=2, max_value=64, value=16)

    dx = st.sidebar.number_input("Spacing (λ)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    if st.sidebar.button("Create Array", type="primary"):
        st.session_state.geometry = pa.create_triangular_array(Nx, Ny, dx, wavelength=wavelength)
        st.session_state.array_params = {
            'type': 'Triangular', 'Nx': Nx, 'Ny': Ny, 'dx': dx,
            'wavelength': wavelength
        }

elif array_type == "Circular Ring":
    n_elements = st.sidebar.number_input("Number of Elements", min_value=4, max_value=64, value=16)
    radius = st.sidebar.number_input("Radius (λ)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)

    if st.sidebar.button("Create Array", type="primary"):
        st.session_state.geometry = pa.create_circular_array(n_elements, radius, wavelength)
        st.session_state.array_params = {
            'type': 'Circular Ring', 'n_elements': n_elements, 'radius': radius,
            'wavelength': wavelength
        }

elif array_type == "Concentric Rings":
    n_rings = st.sidebar.number_input("Number of Rings", min_value=1, max_value=10, value=3)
    elements_per_ring = st.sidebar.number_input("Elements per Ring", min_value=4, max_value=32, value=12)
    ring_spacing = st.sidebar.number_input("Ring Spacing (λ)", min_value=0.3, max_value=2.0, value=0.6, step=0.1)
    include_center = st.sidebar.checkbox("Include Center Element", value=True)

    if st.sidebar.button("Create Array", type="primary"):
        st.session_state.geometry = pa.create_concentric_rings_array(
            n_rings, elements_per_ring, ring_spacing, wavelength, include_center
        )
        st.session_state.array_params = {
            'type': 'Concentric Rings', 'n_rings': n_rings,
            'elements_per_ring': elements_per_ring, 'ring_spacing': ring_spacing,
            'wavelength': wavelength
        }

elif array_type == "Elliptical":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("Semi-axis X (λ)", min_value=1.0, max_value=20.0, value=4.0)
    with col2:
        b = st.number_input("Semi-axis Y (λ)", min_value=1.0, max_value=20.0, value=3.0)

    dx = st.sidebar.number_input("Spacing (λ)", min_value=0.3, max_value=1.0, value=0.5, step=0.1)
    grid_type = st.sidebar.selectbox("Grid Type", ["rectangular", "triangular"])

    if st.sidebar.button("Create Array", type="primary"):
        st.session_state.geometry = pa.create_elliptical_array(a, b, dx, grid_type=grid_type, wavelength=wavelength)
        st.session_state.array_params = {
            'type': 'Elliptical', 'a': a, 'b': b, 'dx': dx, 'grid_type': grid_type,
            'wavelength': wavelength
        }

# Display results
if st.session_state.geometry is not None:
    geom = st.session_state.geometry
    params = st.session_state.array_params

    # Info columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Elements", geom.n_elements)
    with col2:
        aperture_x = geom.x.max() - geom.x.min()
        st.metric("Aperture X", f"{aperture_x:.2f} m")
    with col3:
        aperture_y = geom.y.max() - geom.y.min()
        st.metric("Aperture Y", f"{aperture_y:.2f} m")

    # Plot
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Array Layout")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(geom.x, geom.y, s=50, c='blue', alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f"{params['type']} Array - {geom.n_elements} Elements")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Array Parameters")
        st.json(params)

        st.subheader("Expected Performance")
        k = pa.wavelength_to_k(params['wavelength'])

        # Estimate beamwidth (approximate)
        bw_x = None
        bw_y = None
        if aperture_x > 0:
            bw_x = np.rad2deg(0.886 * params['wavelength'] / aperture_x)
            st.write(f"**Est. Beamwidth X:** {bw_x:.1f}°")
        if aperture_y > 0:
            bw_y = np.rad2deg(0.886 * params['wavelength'] / aperture_y)
            st.write(f"**Est. Beamwidth Y:** {bw_y:.1f}°")

        # Directivity estimate
        D_est = 4 * np.pi * aperture_x * aperture_y / (params['wavelength']**2)
        D_dB = 10 * np.log10(max(D_est, 1))
        st.write(f"**Est. Directivity:** {D_dB:.1f} dBi")

    # Export section
    st.markdown("---")
    st.subheader("Export Array Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export Geometry CSV"):
            csv_data = pa.export_geometry_csv(
                geom,
                metadata={'array_type': params['type'], 'wavelength': params['wavelength']}
            )
            st.download_button(
                label="Download Geometry CSV",
                data=csv_data,
                file_name="array_geometry.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Export Config JSON"):
            json_data = pa.export_array_config_json(
                geom,
                array_params=params
            )
            st.download_button(
                label="Download Config JSON",
                data=json_data,
                file_name="array_config.json",
                mime="application/json"
            )

    with col3:
        if st.button("Generate Summary Report"):
            report = pa.export_summary_report(
                geom,
                weights=np.ones(geom.n_elements, dtype=complex),
                pattern_metrics={
                    'Est. Directivity (dBi)': D_dB,
                    'Est. Beamwidth X (deg)': bw_x if bw_x is not None else 'N/A',
                    'Est. Beamwidth Y (deg)': bw_y if bw_y is not None else 'N/A',
                },
                array_params=params
            )
            st.download_button(
                label="Download Summary Report",
                data=report,
                file_name="array_report.txt",
                mime="text/plain"
            )

    st.success("✅ Array created! Go to **Beam Steering** to compute patterns.")
else:
    st.info("👈 Configure array parameters in the sidebar and click **Create Array**")
