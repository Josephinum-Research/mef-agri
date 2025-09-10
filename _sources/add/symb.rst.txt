Used Symbols
============

There are the following main models which consist of multiple submodels

* soil
* atmosphere
* crop
* management
* observing sensors and platforms

The symbols representing model quantities consist of a character indicating the 
main model. The index of these characters consists of two or three parts which 
are separated by a comma

* the first part equals an identifier of the model quantity
* the last part corresponds the time or epoch information (a number for a specific epoch and :math:`k` for arbitrary epochs)
* in the case of quantities which belong to soil layers, there is an additional index :math:`i` between the quantity id and the time index
* in the case of quantities which belong to the soil surface, there is an additional index :math:`s` between the quantity id and the time index

The index part with the quantity identifier is occasionally further separated by 
minus signs for better readability.
Capital letters in the beginning indicate a specific domain within the main 
models (e.g. :math:`s_{\textrm{W-...},...}` for soil water related quantities).
The whole identifier part does not exceed five characters (incl. minus signs).

Some examples

* :math:`s_{\textrm{clay},0}` ... clay content of the soil at epoch 0
* :math:`a_{\textrm{temp},k}` ... mean temperature at epoch :math:`k`
* :math:`s_{\textrm{T-t},i,k}` ... temperature of soil layer :math:`i`
* :math:`c_{\textrm{R-bm},k}` ... root biomass at epoch :math:`k`
* :math:`m_{\textrm{F-NO}_{3}^{-},k}` ... applied nitrate through mineral fertilization at epoch :math:`k`
