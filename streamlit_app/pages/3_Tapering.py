"""
Tapering Page - Apply amplitude tapers for sidelobe control.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Tapering", page_icon="📊", layout="wide")

st.title("📊 Amplitude Tapering")
st.markdown("Apply window functions to reduce sidelobes at the cost of beamwidth and efficiency.")

# Check for geometry
if 'geometry' not in st.session_state or st.session_state.geometry is None:
    st.warning("⚠️ No array defined. Please go to **Array Design** first.")
    st.stop()

geom = st.session_state.geometry
params = st.session_state.array_params
wavelength = params.get('wavelength', 1.0)
k = pa.wavelength_to_k(wavelength)

# Get steering from session state
steering = st.session_state.get('steering', {'theta': 0, 'phi': 0})
theta0 = steering['theta']
phi0 = steering['phi']

# Sidebar controls
st.sidebar.header("Taper Selection")

taper_type = st.sidebar.selectbox(
    "Window Type",
    ["Uniform", "Taylor", "Chebyshev", "Hamming", "Hanning", "Gaussian", "Cosine-on-Pedestal"]
)

# Taper-specific parameters
if taper_type == "Taylor":
    sidelobe_dB = st.sidebar.slider("Sidelobe Level (dB)", -50, -20, -30, 1)
    nbar = st.sidebar.slider("nbar", 2, 8, 4)
elif taper_type == "Chebyshev":
    sidelobe_dB = st.sidebar.slider("Sidelobe Level (dB)", -60, -30, -40, 1)
elif taper_type == "Gaussian":
    sigma = st.sidebar.slider("Sigma", 0.2, 0.6, 0.4, 0.05)
elif taper_type == "Cosine-on-Pedestal":
    pedestal = st.sidebar.slider("Pedestal", 0.0, 0.5, 0.1, 0.05)

st.sidebar.markdown("---")
compare_uniform = st.sidebar.checkbox("Compare with Uniform", value=True)

# Determine array dimensions for 2D tapers
Nx = params.get('Nx', int(np.sqrt(geom.n_elements)))
Ny = params.get('Ny', int(np.sqrt(geom.n_elements)))

# Generate taper
if taper_type == "Uniform":
    taper = np.ones(geom.n_elements)
elif taper_type == "Taylor":
    try:
        taper = pa.taylor_taper_2d(Nx, Ny, sidelobe_dB, nbar)
    except:
        taper = pa.taylor_taper_1d(geom.n_elements, sidelobe_dB, nbar)
elif taper_type == "Chebyshev":
    try:
        taper = pa.chebyshev_taper_2d(Nx, Ny, sidelobe_dB)
    except:
        taper = pa.chebyshev_taper_1d(geom.n_elements, sidelobe_dB)
elif taper_type == "Hamming":
    try:
        taper = pa.hamming_taper_2d(Nx, Ny)
    except:
        taper = pa.hamming_taper_1d(geom.n_elements)
elif taper_type == "Hanning":
    try:
        taper = pa.hanning_taper_2d(Nx, Ny)
    except:
        taper = pa.hanning_taper_1d(geom.n_elements)
elif taper_type == "Gaussian":
    try:
        taper = pa.gaussian_taper_2d(Nx, Ny, sigma)
    except:
        taper = pa.gaussian_taper_1d(geom.n_elements, sigma)
elif taper_type == "Cosine-on-Pedestal":
    try:
        taper = pa.cosine_on_pedestal_taper_2d(Nx, Ny, pedestal)
    except:
        taper = pa.cosine_on_pedestal_taper_1d(geom.n_elements, pedestal)

# Ensure taper matches array size
if len(taper) != geom.n_elements:
    taper = np.ones(geom.n_elements)
    st.warning("Taper size mismatch - using uniform. Try Rectangular array for best taper support.")

# Compute efficiency
efficiency = pa.compute_taper_efficiency(taper)
loss_dB = pa.compute_taper_directivity_loss(taper)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Taper Type", taper_type)
with col2:
    st.metric("Efficiency", f"{efficiency*100:.1f}%")
with col3:
    st.metric("Directivity Loss", f"{loss_dB:.2f} dB")
with col4:
    st.metric("Taper Range", f"{taper.min():.2f} - {taper.max():.2f}")

# Compute patterns
steering_weights = pa.steering_vector(k, geom.x, geom.y, theta0, phi0)
weights_tapered = steering_weights * taper

# Store tapered weights
st.session_state.weights = weights_tapered
st.session_state.taper = taper

# Compute pattern cuts
angles, E_tapered, _ = pa.compute_pattern_cuts(
    geom.x, geom.y, weights_tapered, k,
    theta0_deg=theta0, phi0_deg=phi0
)

if compare_uniform:
    _, E_uniform, _ = pa.compute_pattern_cuts(
        geom.x, geom.y, steering_weights, k,
        theta0_deg=theta0, phi0_deg=phi0
    )

# Plot pattern comparison
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Pattern Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))

    if compare_uniform:
        ax.plot(angles, E_uniform, 'b--', linewidth=1, alpha=0.7, label='Uniform')

    ax.plot(angles, E_tapered, 'r-', linewidth=1.5, label=taper_type)
    ax.axhline(-3, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Gain (dB)')
    ax.set_ylim([-60, 5])
    ax.set_xlim([-90, 90])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"E-Plane Cut - Scan θ={theta0}°")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Taper Weights")

    # Visualize taper on array
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(geom.x, geom.y, c=taper, cmap='viridis', s=50)
    plt.colorbar(scatter, ax=ax, label='Weight')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.set_title('Amplitude Distribution')
    st.pyplot(fig)
    plt.close()

# Beamwidth comparison
st.subheader("Performance Metrics")

hpbw_tapered = pa.compute_half_power_beamwidth(angles, E_tapered)
if compare_uniform:
    hpbw_uniform = pa.compute_half_power_beamwidth(angles, E_uniform)

    # Find first sidelobe level
    peak_idx = np.argmax(E_tapered)
    main_lobe_end = peak_idx + np.argmax(E_tapered[peak_idx:] < -3)
    if main_lobe_end < len(E_tapered) - 10:
        sll_tapered = np.max(E_tapered[main_lobe_end + 5:])
        sll_uniform = np.max(E_uniform[main_lobe_end + 5:])
    else:
        sll_tapered = -30
        sll_uniform = -13

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("HPBW (Uniform)", f"{hpbw_uniform:.1f}°")
    with col2:
        st.metric("HPBW (Tapered)", f"{hpbw_tapered:.1f}°",
                  delta=f"{hpbw_tapered - hpbw_uniform:+.1f}°")
    with col3:
        st.metric("SLL (Uniform)", f"{sll_uniform:.1f} dB")
    with col4:
        st.metric("SLL (Tapered)", f"{sll_tapered:.1f} dB",
                  delta=f"{sll_tapered - sll_uniform:+.1f} dB")
else:
    st.metric("Half-Power Beamwidth", f"{hpbw_tapered:.1f}°")

# Taper comparison chart
st.subheader("Taper Comparison")

tapers_to_compare = {
    'Uniform': np.ones(geom.n_elements),
}

try:
    tapers_to_compare['Taylor -25dB'] = pa.taylor_taper_2d(Nx, Ny, -25)
    tapers_to_compare['Taylor -35dB'] = pa.taylor_taper_2d(Nx, Ny, -35)
    tapers_to_compare['Chebyshev -40dB'] = pa.chebyshev_taper_2d(Nx, Ny, -40)
    tapers_to_compare['Hamming'] = pa.hamming_taper_2d(Nx, Ny)
except:
    pass

if len(tapers_to_compare) > 1:
    comparison_data = []
    for name, t in tapers_to_compare.items():
        if len(t) == geom.n_elements:
            eff = pa.compute_taper_efficiency(t)
            loss = pa.compute_taper_directivity_loss(t)
            comparison_data.append({
                'Taper': name,
                'Efficiency (%)': f"{eff*100:.1f}",
                'Loss (dB)': f"{loss:.2f}"
            })

    st.table(comparison_data)

# Export section
st.markdown("---")
st.subheader("Export Tapered Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Tapered Pattern CSV"):
        csv_data = pa.export_pattern_csv(
            angles, E_tapered,
            angle_label="angle_deg",
            pattern_label="pattern_dB",
            metadata={
                'taper': taper_type,
                'efficiency': f"{efficiency*100:.1f}%",
                'theta': theta0,
                'phi': phi0
            }
        )
        st.download_button(
            label="Download Pattern CSV",
            data=csv_data,
            file_name=f"pattern_{taper_type.lower()}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export Tapered Weights CSV"):
        csv_data = pa.export_weights_csv(
            geom, weights_tapered,
            metadata={
                'taper': taper_type,
                'steering_theta': theta0,
                'steering_phi': phi0
            }
        )
        st.download_button(
            label="Download Weights CSV",
            data=csv_data,
            file_name=f"weights_{taper_type.lower()}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("Export Taper Comparison"):
        import io
        buffer = io.StringIO()
        buffer.write("angle_deg")
        for name in patterns_quant if 'patterns_quant' in dir() else {}:
            buffer.write(f",{name}")
        buffer.write(",uniform\n")
        for i, angle in enumerate(angles):
            buffer.write(f"{angle:.2f}")
            for name in patterns_quant if 'patterns_quant' in dir() else {}:
                buffer.write(f",{patterns_quant[name][i]:.2f}")
            if compare_uniform:
                buffer.write(f",{E_uniform[i]:.2f}")
            buffer.write("\n")

        st.download_button(
            label="Download Comparison CSV",
            data=buffer.getvalue(),
            file_name="taper_comparison.csv",
            mime="text/csv"
        )

st.success("✅ Taper applied! Check **Impairments** for realistic effects.")
