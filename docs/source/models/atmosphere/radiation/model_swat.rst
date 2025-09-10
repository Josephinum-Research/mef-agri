SWAT - atmosphere.radiation
===========================

Used constants in this module

* ``EARTH_ROTATION_VELOCITY = 0.2618`` :math:`\frac{rad}{h}`
* ``SOLAR_CONSTANT = 4.921`` :math:`\frac{MJ}{m^2\cdot h}`
* ``BOLTZMANN_CONSTANT = 4.903`` :math:`\frac{MJ}{m^2\cdot ^\circ K^4\cdot day}`
* ``CC_A, CC_B = 1.2, -0.2`` - coefficients for cloud cover adjustment - [R1]_ (general values from table 1:1-3)
* ``EM_A, EM_B = 0.39, -0.158`` - coefficients for net emittence computation - [R1]_ (general values from table 1:1-3)

.. autofunction:: mef_agri.models.atmosphere.radiation.model_swat.eccentricity_correction

.. autofunction:: mef_agri.models.atmosphere.radiation.model_swat.extraterrestrial_radiation

.. autoclass:: mef_agri.models.atmosphere.radiation.model_swat.Radiation_V2009
    :members:
