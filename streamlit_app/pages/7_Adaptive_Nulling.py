"""
Adaptive Nulling Page - SMI/MVDR adaptive beamforming against interferers.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Adaptive Nulling", page_icon="🎛️", layout="wide")

st.title("🎛️ Adaptive Nulling (SMI)")
st.markdown("""
Sample Matrix Inversion (SMI) computes MVDR weights `w = R⁻¹s / (sᴴR⁻¹s)` from
interference-plus-noise snapshots: unit gain is held on the desired direction while a
null forms on the interferer. Compare the quiescent and adapted pattern cuts and the
resulting SINR improvement.
""")

# Sidebar controls
st.sidebar.header("Array Configuration")

col1, col2 = st.sidebar.columns(2)
with col1:
    Nx = st.number_input("Elements X", min_value=4, max_value=32, value=8)
with col2:
    Ny = st.number_input("Elements Y", min_value=4, max_value=32, value=8)

spacing = st.sidebar.number_input("Spacing (λ)", min_value=0.25, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Signal Scenario")

theta_sig = st.sidebar.slider("Desired Signal θ (deg)", -60, 60, 0)
theta_int = st.sidebar.slider("Interferer θ (deg)", -60, 60, 20)
inr_dB = st.sidebar.slider("Interferer Power INR (dB)", 0, 50, 30)

st.sidebar.markdown("---")
st.sidebar.header("SMI Parameters")

n_snapshots = st.sidebar.slider("Snapshots", 32, 512, 200, step=8)
diag_loading = st.sidebar.select_slider(
    "Diagonal Loading",
    options=[0.0, 1e-3, 1e-2, 1e-1, 1.0],
    value=1e-2
)
seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)

if abs(theta_int - theta_sig) < 3:
    st.warning("⚠️ Interferer is within ~3° of the desired signal. The adaptive null "
               "will eat into the main beam and SINR improvement collapses.")

# Build array
geom = pa.create_rectangular_array(int(Nx), int(Ny), spacing, spacing)
k = pa.wavelength_to_k(1.0)

# Simulate received snapshots: interferer plane wave + unit-power noise.
# Received-array manifold is the conjugate of the transmit steering vector.
rng = np.random.default_rng(int(seed))
s_int = pa.steering_vector(k, geom.x, geom.y, theta_int, 0).conj()
amp = 10 ** (inr_dB / 20)
jam = amp * (rng.standard_normal((n_snapshots, 1))
             + 1j * rng.standard_normal((n_snapshots, 1))) / np.sqrt(2)
noise = (rng.standard_normal((n_snapshots, geom.n_elements))
         + 1j * rng.standard_normal((n_snapshots, geom.n_elements))) / np.sqrt(2)
snapshots = jam * s_int[None, :] + noise

# Quiescent and adapted weights
w_quiescent = pa.steering_vector(k, geom.x, geom.y, theta_sig, 0)
w_adapted = pa.adaptive_weights_smi(
    geom, k,
    theta_desired_deg=theta_sig, phi_desired_deg=0,
    interference_data=snapshots,
    diagonal_loading=diag_loading
)

# SINR improvement
sinr_before, sinr_after, improvement = pa.compute_sinr_improvement(
    w_quiescent, w_adapted, geom, k,
    signal_direction=(theta_sig, 0),
    interference_directions=[(theta_int, 0)],
    signal_power=1.0,
    interference_powers=[10 ** (inr_dB / 10)],
    noise_power=1.0
)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Elements", geom.n_elements)
with col2:
    st.metric("SINR (Quiescent)", f"{sinr_before:.1f} dB")
with col3:
    st.metric("SINR (Adapted)", f"{sinr_after:.1f} dB")
with col4:
    st.metric("SINR Improvement", f"{improvement:.1f} dB")

# Pattern comparison
col1, col2 = st.columns([2, 1])

angles, E_quiescent, _ = pa.compute_pattern_cuts(
    geom.x, geom.y, w_quiescent, k, theta0_deg=theta_sig, phi0_deg=0
)
_, E_adapted, _ = pa.compute_pattern_cuts(
    geom.x, geom.y, w_adapted, k, theta0_deg=theta_sig, phi0_deg=0
)

with col1:
    st.subheader("Quiescent vs Adapted Pattern")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(angles, E_quiescent, 'b-', linewidth=1.5, label='Quiescent')
    ax.plot(angles, E_adapted, 'r-', linewidth=1.5, label='SMI Adapted')
    ax.axvline(theta_sig, color='green', linestyle='--', alpha=0.7, label=f'Signal ({theta_sig}°)')
    ax.axvline(theta_int, color='black', linestyle='--', alpha=0.7, label=f'Interferer ({theta_int}°)')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Gain (dB)')
    ax.set_ylim([-70, 5])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Adaptive Nulling: {inr_dB} dB interferer at {theta_int}°')
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Null Depth")

    idx_int = np.argmin(np.abs(angles - theta_int))
    st.write(f"**Pattern at interferer angle:**")
    st.write(f"- Quiescent: **{E_quiescent[idx_int]:.1f} dB**")
    st.write(f"- Adapted: **{E_adapted[idx_int]:.1f} dB**")
    st.metric("Null Deepening", f"{E_quiescent[idx_int] - E_adapted[idx_int]:.1f} dB")

    st.markdown("---")
    st.markdown("""
    **Reading the plot:** the adapted pattern keeps its main beam on the signal
    direction while pulling a deep null onto the interferer. With few snapshots or
    no diagonal loading, sidelobes away from the null degrade.
    """)

# Export section
st.markdown("---")
st.subheader("Export Adapted Pattern")

if st.button("Export Pattern CSV"):
    import io
    buffer = io.StringIO()
    buffer.write("angle_deg,quiescent_dB,adapted_dB\n")
    for i in range(len(angles)):
        buffer.write(f"{angles[i]:.2f},{E_quiescent[i]:.4f},{E_adapted[i]:.4f}\n")

    st.download_button(
        label="Download Pattern CSV",
        data=buffer.getvalue(),
        file_name=f"adaptive_nulling_int{theta_int}deg_inr{inr_dB}dB.csv",
        mime="text/csv"
    )

st.success("✅ Adaptive nulling analysis complete!")
