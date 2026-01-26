"""
Wideband / TTD Page - True-time delay and hybrid beamforming analysis.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import phased_array as pa

st.set_page_config(page_title="Wideband/TTD", page_icon="📡", layout="wide")

st.title("📡 Wideband Beamforming & TTD")
st.markdown("""
Analyze beam squint and compare steering architectures:
- **Phase-only**: Maximum squint, simplest hardware
- **Hybrid**: TTD at subarray level, phase at element level
- **True-Time Delay (TTD)**: No squint, most complex hardware
""")

# Check for geometry
if 'geometry' not in st.session_state or st.session_state.geometry is None:
    st.warning("⚠️ No array defined. Please go to **Array Design** first.")
    st.stop()

geom = st.session_state.geometry
params = st.session_state.array_params
steering = st.session_state.get('steering', {'theta': 0, 'phi': 0})

# Sidebar controls
st.sidebar.header("Wideband Parameters")

center_freq_ghz = st.sidebar.number_input(
    "Center Frequency (GHz)",
    min_value=0.1, max_value=100.0, value=10.0, step=0.1
)
center_freq = center_freq_ghz * 1e9

bandwidth_pct = st.sidebar.slider(
    "Bandwidth (%)",
    min_value=5, max_value=50, value=20, step=5
)

n_freq_points = st.sidebar.slider(
    "Frequency Points",
    min_value=5, max_value=21, value=11, step=2
)

st.sidebar.markdown("---")
st.sidebar.header("Subarray Configuration")

# Determine subarray size based on array dimensions
Nx = params.get('Nx', int(np.sqrt(geom.n_elements)))
Ny = params.get('Ny', int(np.sqrt(geom.n_elements)))

# Subarray size options
sub_options_x = [i for i in [2, 4, 8] if Nx % i == 0 and i <= Nx]
sub_options_y = [i for i in [2, 4, 8] if Ny % i == 0 and i <= Ny]

if not sub_options_x:
    sub_options_x = [2]
if not sub_options_y:
    sub_options_y = [2]

Nx_sub = st.sidebar.selectbox("Subarray Size X", sub_options_x, index=0)
Ny_sub = st.sidebar.selectbox("Subarray Size Y", sub_options_y, index=0)

# Create subarray architecture
try:
    dx = params.get('dx', 0.5)
    dy = params.get('dy', dx)
    arch = pa.create_rectangular_subarrays(Nx, Ny, Nx_sub, Ny_sub, dx, dy)
    n_subarrays = arch.n_subarrays
except Exception as e:
    st.error(f"Could not create subarrays: {e}")
    st.info("Try a Rectangular array for best subarray support.")
    st.stop()

# Display configuration
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Center Frequency", f"{center_freq_ghz} GHz")
with col2:
    st.metric("Bandwidth", f"{bandwidth_pct}%")
with col3:
    st.metric("Subarrays", n_subarrays)
with col4:
    st.metric("Elements/Subarray", Nx_sub * Ny_sub)

# Compute frequencies
bw_hz = center_freq * bandwidth_pct / 100
frequencies = np.linspace(
    center_freq - bw_hz / 2,
    center_freq + bw_hz / 2,
    n_freq_points
)

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["Beam Squint", "Pattern vs Frequency", "Subarray TTD", "IBW Analysis"])

with tab1:
    st.subheader("Beam Squint Comparison")
    st.markdown("""
    Beam squint is the change in beam pointing direction with frequency.
    Phase-only steering causes the beam to move towards broadside at higher frequencies.
    """)

    # Compare steering modes
    with st.spinner("Computing beam squint..."):
        results = pa.compare_steering_modes(
            geom, arch,
            theta0_deg=steering['theta'],
            phi0_deg=steering['phi'],
            center_frequency=center_freq,
            bandwidth_percent=bandwidth_pct,
            n_freq_points=n_freq_points
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))

        freq_offset_pct = (results['phase']['frequencies'] - center_freq) / center_freq * 100

        ax.plot(freq_offset_pct, results['phase']['squint'], 'ro-',
                label=f"Phase-only (max: {results['phase']['max_squint']:.2f}°)", linewidth=2)
        ax.plot(freq_offset_pct, results['hybrid']['squint'], 'bs-',
                label=f"Hybrid (max: {results['hybrid']['max_squint']:.2f}°)", linewidth=2)
        ax.plot(freq_offset_pct, results['ttd']['squint'], 'g^-',
                label=f"TTD (max: {results['ttd']['max_squint']:.2f}°)", linewidth=2)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Frequency Offset (%)', fontsize=12)
        ax.set_ylabel('Beam Squint (degrees)', fontsize=12)
        ax.set_title(f'Beam Squint at θ={steering["theta"]}°, φ={steering["phi"]}°')
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Squint Summary")

        st.markdown("**Maximum Squint:**")
        st.write(f"- Phase-only: **{results['phase']['max_squint']:.3f}°**")
        st.write(f"- Hybrid: **{results['hybrid']['max_squint']:.3f}°**")
        st.write(f"- TTD: **{results['ttd']['max_squint']:.3f}°**")

        if results['phase']['max_squint'] > 0:
            improvement = (1 - results['hybrid']['max_squint'] / results['phase']['max_squint']) * 100
            st.markdown("---")
            st.metric("Hybrid Improvement", f"{improvement:.0f}%",
                     help="Reduction in max squint vs phase-only")

with tab2:
    st.subheader("Pattern vs Frequency")
    st.markdown("Visualize how the radiation pattern changes across the bandwidth.")

    steering_mode = st.selectbox(
        "Steering Mode",
        ["phase", "hybrid", "ttd"],
        format_func=lambda x: {"phase": "Phase-only", "hybrid": "Hybrid (TTD + Phase)", "ttd": "True-Time Delay"}[x]
    )

    with st.spinner("Computing wideband patterns..."):
        arch_to_use = arch if steering_mode == 'hybrid' else None
        pattern_results = pa.compute_pattern_vs_frequency(
            geom.x, geom.y,
            theta0_deg=steering['theta'],
            phi0_deg=steering['phi'],
            center_frequency=center_freq,
            frequencies=frequencies,
            steering_mode=steering_mode,
            architecture=arch_to_use,
            n_points=181
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(12, 8))

        freq_offset_pct = (pattern_results['frequencies'] - center_freq) / center_freq * 100
        patterns_clipped = np.clip(pattern_results['patterns'], -40, 0)

        im = ax.pcolormesh(
            pattern_results['angles'],
            freq_offset_pct,
            patterns_clipped,
            cmap='jet',
            shading='auto'
        )
        plt.colorbar(im, ax=ax, label='Gain (dB)')

        # Mark steering angle
        ax.axvline(steering['theta'], color='white', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Frequency Offset (%)', fontsize=12)
        ax.set_title(f'Pattern vs Frequency ({steering_mode.upper()})')
        ax.set_xlim([-60, 60])

        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Pattern Cuts")

        # Plot patterns at band edges
        fig, ax = plt.subplots(figsize=(6, 5))

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(frequencies)))
        for i, (freq, color) in enumerate(zip(frequencies, colors)):
            freq_label = f"{(freq - center_freq) / center_freq * 100:+.0f}%"
            ax.plot(pattern_results['angles'], pattern_results['patterns'][i],
                   color=color, alpha=0.7, label=freq_label if i % 3 == 0 else None)

        ax.axvline(steering['theta'], color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Gain (dB)')
        ax.set_ylim([-40, 5])
        ax.set_xlim([-60, 60])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("Subarray Time Delays")
    st.markdown("Visualize the TTD values assigned to each subarray.")

    # Compute subarray delays
    subarray_delays = pa.compute_subarray_delays_ttd(
        arch,
        theta0_deg=steering['theta'],
        phi0_deg=steering['phi']
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 8))

        centers = arch.subarray_centers
        delays_ns = subarray_delays * 1e9  # Convert to nanoseconds

        scatter = ax.scatter(
            centers[:, 0], centers[:, 1],
            c=delays_ns, cmap='viridis', s=300, edgecolors='black', linewidths=2
        )
        plt.colorbar(scatter, ax=ax, label='Delay (ns)')

        # Add labels
        for i, (x, y) in enumerate(centers):
            ax.annotate(f'SA{i}\n{delays_ns[i]:.2f}ns',
                       (x, y), ha='center', va='center', fontsize=8,
                       color='white' if delays_ns[i] > np.mean(delays_ns) else 'black')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Subarray TTD Values (θ={steering["theta"]}°, φ={steering["phi"]}°)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("TTD Statistics")

        st.write(f"**Number of Subarrays:** {n_subarrays}")
        st.write(f"**Max Delay:** {np.max(delays_ns):.3f} ns")
        st.write(f"**Min Delay:** {np.min(delays_ns):.3f} ns")
        st.write(f"**Delay Range:** {np.max(delays_ns) - np.min(delays_ns):.3f} ns")

        st.markdown("---")
        st.subheader("Hardware Implications")

        max_delay_ps = np.max(subarray_delays) * 1e12
        st.write(f"**Max TTD Required:** {max_delay_ps:.1f} ps")

        # Estimate bits needed
        delay_resolution_ps = 10  # Typical TTD resolution
        bits_needed = int(np.ceil(np.log2(max_delay_ps / delay_resolution_ps + 1)))
        st.write(f"**Est. TTD Bits:** {bits_needed} bits")
        st.write(f"*(at {delay_resolution_ps} ps resolution)*")

with tab4:
    st.subheader("Instantaneous Bandwidth Analysis")
    st.markdown("Calculate the maximum bandwidth for a given beam squint tolerance.")

    squint_tolerance = st.slider(
        "Squint Tolerance (degrees)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Phase-only Steering")

        ibw_phase = pa.analyze_instantaneous_bandwidth(
            geom.x, geom.y,
            theta0_deg=steering['theta'],
            phi0_deg=steering['phi'],
            center_frequency=center_freq,
            squint_tolerance_deg=squint_tolerance,
            steering_mode='phase'
        )

        if ibw_phase['ibw_hz'] < np.inf:
            st.metric("IBW", f"{ibw_phase['ibw_hz']/1e6:.1f} MHz")
            st.metric("IBW (%)", f"{ibw_phase['ibw_percent']:.1f}%")
            st.write(f"**Frequency Range:** {ibw_phase.get('f_low', 0)/1e9:.3f} - {ibw_phase.get('f_high', 0)/1e9:.3f} GHz")
        else:
            st.metric("IBW", "Exceeds limit")

    with col2:
        st.markdown("### Hybrid Steering")

        ibw_hybrid = pa.analyze_instantaneous_bandwidth(
            geom.x, geom.y,
            theta0_deg=steering['theta'],
            phi0_deg=steering['phi'],
            center_frequency=center_freq,
            squint_tolerance_deg=squint_tolerance,
            steering_mode='hybrid',
            architecture=arch
        )

        if ibw_hybrid['ibw_hz'] < np.inf:
            st.metric("IBW", f"{ibw_hybrid['ibw_hz']/1e6:.1f} MHz")
            st.metric("IBW (%)", f"{ibw_hybrid['ibw_percent']:.1f}%")
            st.write(f"**Frequency Range:** {ibw_hybrid.get('f_low', 0)/1e9:.3f} - {ibw_hybrid.get('f_high', 0)/1e9:.3f} GHz")

            if ibw_phase['ibw_hz'] < np.inf and ibw_phase['ibw_hz'] > 0:
                improvement = ibw_hybrid['ibw_hz'] / ibw_phase['ibw_hz']
                st.metric("IBW Improvement", f"{improvement:.1f}x",
                         help="Bandwidth improvement vs phase-only")
        else:
            st.metric("IBW", "Exceeds limit")

    st.markdown("---")
    st.info("**Note:** True-Time Delay (TTD) has theoretically unlimited instantaneous bandwidth.")

# Export section
st.markdown("---")
st.subheader("Export Wideband Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Squint Data CSV"):
        import io
        buffer = io.StringIO()
        buffer.write("freq_offset_pct,phase_squint_deg,hybrid_squint_deg,ttd_squint_deg\n")
        freq_pct = (results['phase']['frequencies'] - center_freq) / center_freq * 100
        for i in range(len(freq_pct)):
            buffer.write(f"{freq_pct[i]:.2f},{results['phase']['squint'][i]:.4f},"
                        f"{results['hybrid']['squint'][i]:.4f},{results['ttd']['squint'][i]:.4f}\n")

        st.download_button(
            label="Download Squint CSV",
            data=buffer.getvalue(),
            file_name="beam_squint_comparison.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export TTD Values CSV"):
        import io
        buffer = io.StringIO()
        buffer.write("subarray,center_x,center_y,delay_ns\n")
        for i in range(n_subarrays):
            buffer.write(f"{i},{arch.subarray_centers[i,0]:.4f},"
                        f"{arch.subarray_centers[i,1]:.4f},{subarray_delays[i]*1e9:.6f}\n")

        st.download_button(
            label="Download TTD CSV",
            data=buffer.getvalue(),
            file_name="subarray_ttd_values.csv",
            mime="text/csv"
        )

st.success("✅ Wideband analysis complete!")
