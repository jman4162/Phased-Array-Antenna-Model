Installation
============

Requirements
------------

- Python 3.8 or later
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

Optional dependencies:

- Plotly >= 5.0.0 (for interactive 3D visualizations)
- Seaborn >= 0.11.0 (for enhanced statistical plots)

Installing from PyPI
--------------------

The recommended way to install the package is via pip:

.. code-block:: bash

   pip install phased-array-modeling

To include optional plotting dependencies:

.. code-block:: bash

   pip install phased-array-modeling[plotting]

For all optional dependencies:

.. code-block:: bash

   pip install phased-array-modeling[full]

Installing from Source
----------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/jman4162/Phased-Array-Antenna-Model.git
   cd Phased-Array-Antenna-Model
   pip install -e .

For development with all dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Verifying Installation
----------------------

Verify that the package is installed correctly:

.. code-block:: python

   import phased_array as pa
   print(f"Version: {pa.__version__}")

   # Quick test - create a small array
   geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
   print(f"Created array with {geom.n_elements} elements")

Expected output:

.. code-block:: text

   Version: 1.2.0
   Created array with 16 elements

Google Colab
------------

The library works seamlessly in Google Colab. Install it in a notebook cell:

.. code-block:: bash

   !pip install phased-array-modeling[plotting]

The Jupyter notebooks included in the repository are designed for Colab use.

Troubleshooting
---------------

**ImportError: No module named 'phased_array'**

Ensure you installed the package in your active Python environment:

.. code-block:: bash

   pip show phased-array-modeling

**Plotly plots not displaying**

For Jupyter notebooks, you may need to install additional extensions:

.. code-block:: bash

   pip install nbformat

For JupyterLab:

.. code-block:: bash

   jupyter labextension install jupyterlab-plotly
