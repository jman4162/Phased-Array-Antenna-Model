"""
Impairments Page - Simulate real-world effects.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Impairments", page_icon="⚠️", layout="wide")

st.title("⚠️ Impairment Simulation")
st.markdown("Simulate real-world effects: phase quantization, element failures, and mutual coupling.")

# Check for geometry
if 'geometry' not in st.session_state or st.session_state.geometry is None:
    st.warning("⚠️ No array defined. Please go to **Array Design** first.")
    st.stop()

geom = st.session_state.geometry
params = st.session_state.array_params
wavelength = params.get('wavelength', 1.0)
k = pa.wavelength_to_k(wavelength)

# Get weights from session state
if 'weights' not in st.session_state:
    steering = st.session_state.get('steering', {'theta': 0, 'phi': 0})
    st.session_state.weights = pa.steering_vector(k, geom.x, geom.y, steering['theta'], steering['phi'])

weights_ideal = st.session_state.weights
steering = st.session_state.get('steering', {'theta': 0, 'phi': 0})

# Tabs for different impairments
tab1, tab2, tab3 = st.tabs(["Phase Quantization", "Element Failures", "Mutual Coupling"])

with tab1:
    st.subheader("Phase Quantization")
    st.markdown("""
    Phase shifters in real systems have limited resolution. This simulates the effect
    of discrete phase levels on the radiation pattern.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        n_bits = st.slider("Phase Shifter Bits", 2, 8, 4)
        n_levels = 2 ** n_bits
        rms_error = pa.quantization_rms_error(n_bits)

        st.metric("Phase Levels", n_levels)
        st.metric("Phase Step", f"{360/n_levels:.1f}°")
        st.metric("RMS Error", f"{rms_error:.1f}°")

    with col2:
        # Compute patterns
        weights_quantized = pa.quantize_phase(weights_ideal, n_bits)

        angles, E_ideal, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, weights_ideal, k,
            theta0_deg=steering['theta'], phi0_deg=steering['phi']
        )
        _, E_quant, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, weights_quantized, k,
            theta0_deg=steering['theta'], phi0_deg=steering['phi']
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(angles, E_ideal, 'b-', linewidth=1.5, label='Ideal')
        ax.plot(angles, E_quant, 'r--', linewidth=1.5, label=f'{n_bits}-bit Quantized')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Normalized Gain (dB)')
        ax.set_ylim([-50, 5])
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Phase Quantization Effect ({n_bits} bits = {n_levels} levels)')
        st.pyplot(fig)
        plt.close()

    # Comparison of multiple bit levels
    st.subheader("Quantization Comparison")

    patterns_quant = {}
    for bits in [2, 3, 4, 6, 8]:
        w_q = pa.quantize_phase(weights_ideal, bits)
        _, E_q, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, w_q, k,
            theta0_deg=steering['theta'], phi0_deg=steering['phi']
        )
        patterns_quant[f'{bits}-bit'] = E_q

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(patterns_quant)))
    for (name, E), color in zip(patterns_quant.items(), colors):
        ax.plot(angles, E, linewidth=1.5, label=name, color=color)
    ax.plot(angles, E_ideal, 'k--', linewidth=2, label='Ideal', alpha=0.7)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Gain (dB)')
    ax.set_ylim([-50, 5])
    ax.set_xlim([-60, 60])
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    st.pyplot(fig)
    plt.close()

with tab2:
    st.subheader("Element Failures")
    st.markdown("""
    Random element failures degrade array performance. This simulates the effect
    of failed elements on the radiation pattern.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        failure_rate = st.slider("Failure Rate (%)", 0, 30, 10) / 100
        failure_mode = st.selectbox("Failure Mode", ["off", "stuck", "full"])
        seed = st.number_input("Random Seed", 0, 1000, 42)

        n_failed = int(geom.n_elements * failure_rate)
        st.metric("Failed Elements", f"{n_failed} / {geom.n_elements}")

    with col2:
        if failure_rate > 0:
            weights_failed, failure_mask = pa.simulate_element_failures(
                weights_ideal, failure_rate, failure_mode, seed=seed
            )
        else:
            weights_failed = weights_ideal
            failure_mask = np.zeros(geom.n_elements, dtype=bool)

        _, E_failed, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, weights_failed, k,
            theta0_deg=steering['theta'], phi0_deg=steering['phi']
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(angles, E_ideal, 'b-', linewidth=1.5, label='Ideal')
        ax.plot(angles, E_failed, 'r--', linewidth=1.5, label=f'{failure_rate*100:.0f}% Failed')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Normalized Gain (dB)')
        ax.set_ylim([-50, 5])
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Element Failure Effect ({n_failed} elements {failure_mode})')
        st.pyplot(fig)
        plt.close()

    # Show failed elements on array
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Array Status")
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['red' if f else 'blue' for f in failure_mask]
        ax.scatter(geom.x, geom.y, c=colors, s=50, alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.set_title(f'Red = Failed ({np.sum(failure_mask)} elements)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Graceful Degradation")

        failure_rates_test = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
        gain_losses = []

        for rate in failure_rates_test:
            losses_trial = []
            for trial in range(10):
                if rate > 0:
                    w_f, _ = pa.simulate_element_failures(weights_ideal, rate, 'off', seed=trial)
                else:
                    w_f = weights_ideal
                # Estimate gain loss from sum of weights
                gain_loss = 20 * np.log10(np.abs(np.sum(w_f)) / np.abs(np.sum(weights_ideal)))
                losses_trial.append(gain_loss)
            gain_losses.append(np.mean(losses_trial))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r*100 for r in failure_rates_test], gain_losses, 'bo-', linewidth=2)
        ax.set_xlabel('Failure Rate (%)')
        ax.set_ylabel('Gain Loss (dB)')
        ax.set_title('Graceful Degradation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("Mutual Coupling")
    st.markdown("""
    Electromagnetic coupling between elements affects the element patterns and
    steering accuracy.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        coupling_model = st.selectbox("Coupling Model", ["sinc", "exponential"])
        coupling_coeff = st.slider("Coupling Coefficient", 0.0, 0.5, 0.2, 0.05)
        apply_compensation = st.checkbox("Apply Compensation", value=False)

    with col2:
        # Compute coupling matrix
        C = pa.mutual_coupling_matrix_theoretical(
            geom, k,
            coupling_model=coupling_model,
            coupling_coeff=coupling_coeff
        )

        # Apply coupling
        if apply_compensation:
            weights_coupled = pa.apply_mutual_coupling(weights_ideal, C, mode='compensate')
            weights_coupled = pa.apply_mutual_coupling(weights_coupled, C, mode='transmit')
            label = 'Compensated'
        else:
            weights_coupled = pa.apply_mutual_coupling(weights_ideal, C, mode='transmit')
            label = 'With Coupling'

        _, E_coupled, _ = pa.compute_pattern_cuts(
            geom.x, geom.y, weights_coupled, k,
            theta0_deg=steering['theta'], phi0_deg=steering['phi']
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(angles, E_ideal, 'b-', linewidth=1.5, label='Ideal (No Coupling)')
        ax.plot(angles, E_coupled, 'r--', linewidth=1.5, label=label)
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Normalized Gain (dB)')
        ax.set_ylim([-50, 5])
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Mutual Coupling Effect ({coupling_model}, coeff={coupling_coeff})')
        st.pyplot(fig)
        plt.close()

    # Coupling matrix visualization
    st.subheader("Coupling Matrix")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.abs(C), cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax, label='|Coupling|')
    ax.set_xlabel('Element j')
    ax.set_ylabel('Element i')
    ax.set_title('Coupling Matrix Magnitude')
    st.pyplot(fig)
    plt.close()

st.success("✅ Impairment analysis complete! View **UV-Space** for advanced visualization.")
