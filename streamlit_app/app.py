"""
Phased Array Antenna Designer - Interactive Streamlit App

Main landing page with overview and navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="Phased Array Designer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 Phased Array Antenna Designer")

st.markdown("""
Welcome to the **Phased Array Antenna Designer** - an interactive tool for designing,
analyzing, and visualizing phased array antenna systems.

### Features

Use the sidebar to navigate between different tools:

1. **🔲 Array Design** - Create and configure array geometries
   - Rectangular, triangular, circular arrays
   - Adjust element count and spacing
   - Visualize element positions

2. **🎯 Beam Steering** - Control beam direction
   - Interactive theta/phi sliders
   - Real-time pattern updates
   - E-plane and H-plane cuts

3. **📊 Tapering** - Apply amplitude tapers for sidelobe control
   - Taylor, Chebyshev, Hamming windows
   - Compare taper efficiency
   - See sidelobe reduction

4. **⚠️ Impairments** - Simulate real-world effects
   - Phase quantization (2-8 bits)
   - Random element failures
   - Mutual coupling

5. **🌐 UV-Space** - Visualize in direction cosine space
   - See visible region
   - Identify grating lobes
   - Interactive Plotly plots

6. **📡 Wideband/TTD** - Analyze wideband beamforming
   - Compare phase-only, hybrid, and true-time delay steering
   - Visualize beam squint vs frequency
   - Configure subarray architectures
   - Compute instantaneous bandwidth (IBW)

### Quick Start

1. Go to **Array Design** to create your array (enable subarrays for wideband analysis)
2. Use **Beam Steering** to point the beam
3. Apply **Tapering** for sidelobe control
4. Check **Impairments** for realistic effects
5. View **UV-Space** for advanced analysis
6. Use **Wideband/TTD** for true-time delay and hybrid beamforming

---

### About

Built with [Streamlit](https://streamlit.io) and the
[phased_array](https://github.com/jman4162/Phased-Array-Antenna-Model) Python package.

**Author:** John Hodge
**License:** MIT
""")

# Sidebar info
st.sidebar.success("Select a page above to get started!")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Resources
- [GitHub Repository](https://github.com/jman4162/Phased-Array-Antenna-Model)
- [Demo Notebook](https://colab.research.google.com/github/jman4162/Phased-Array-Antenna-Model/blob/main/Phased_Array_Demo.ipynb)
- [Documentation](https://github.com/jman4162/Phased-Array-Antenna-Model#readme)
""")
