import numpy as np

from ...base import Model, Quantities as Q, Units as U
from .temperature.model_swat import Temperature_V2009
from .water.model_swat import Water_V2009
from .nutrients.model_swat import Nutrients_C_N_V2009


class Layer_V2009(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.ROUT, U.m)
    def thickness(self) -> float:
        r"""
        MQ - Random output

        :math:`s_{\textrm{lt},i,0}\ [m]`

        :return: layer thickness 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.m)
    def depth(self) -> float:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{ld},i,0}\ [m]`

        :return: depth of bottom layer border 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.m)
    def center_depth(self) -> float:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{lcd},i,0}\ [m]`

        :return: depth of the layer center 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.t_ha)
    def biomass(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{bm},i,k}\ [\frac{t}{ha}]`

        NOTE: this quantity is computed and set from outside of this model

        :return: (root) biomass in soil layer 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.t_haxday)
    def biomass_rate(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\Delta\textrm{bm},i,k}\ [\frac{t}{ha\cdot day}]`
        
        NOTE: this quantity is computed and set from outside of this model

        :return: increase of (root) biomass in soil layer 
        :rtype: numpy.ndarray
        """

    @Model.is_child_model(Temperature_V2009)
    def temperature(self) -> Temperature_V2009:
        """
        Child Model

        :return: soil temperature model
        :rtype: Temperature_V2009
        """

    @Model.is_child_model(Water_V2009)
    def water(self) -> Water_V2009:
        """
        Child Model

        :return: soil water model
        :rtype: Water_V2009
        """

    @Model.is_child_model(Nutrients_C_N_V2009)
    def nutrients(self) -> Nutrients_C_N_V2009:
        """
        Child Model

        :return: soil-layer nutrient model
        :rtype: Nutrients_C_N_V2009
        """

    def initialize(self, epoch):
        """
        Initialization of child models.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self.temperature.initialize(epoch)
        self.water.initialize(epoch)
        self.nutrients.initialize(epoch)

    def update(self, epoch):
        """
        Update of child models.

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        self.temperature.update(epoch)
        self.water.update(epoch)
        self.nutrients.update(epoch)
