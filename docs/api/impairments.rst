Impairments Module
==================

.. module:: phased_array.impairments
   :synopsis: Realistic impairment models for phased arrays

The impairments module provides functions to model real-world effects that
degrade phased array performance, including mutual coupling, phase quantization,
element failures, and scan blindness.

Mutual Coupling
---------------

.. autofunction:: phased_array.mutual_coupling_matrix_theoretical

.. autofunction:: phased_array.mutual_coupling_matrix_measured

.. autofunction:: phased_array.apply_mutual_coupling

.. autofunction:: phased_array.active_element_pattern

Phase Quantization
------------------

.. autofunction:: phased_array.quantize_phase

.. autofunction:: phased_array.quantization_rms_error

.. autofunction:: phased_array.quantization_sidelobe_increase

.. autofunction:: phased_array.analyze_quantization_effect

Element Failures
----------------

.. autofunction:: phased_array.simulate_element_failures

.. autofunction:: phased_array.analyze_graceful_degradation

Scan Blindness
--------------

.. autofunction:: phased_array.surface_wave_scan_angle

.. autofunction:: phased_array.scan_blindness_model

.. autofunction:: phased_array.apply_scan_blindness

.. autofunction:: phased_array.compute_scan_loss
