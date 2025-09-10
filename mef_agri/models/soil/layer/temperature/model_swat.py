import numpy as np

from ....base import Model, Quantities as Q
from ....utils import Units as U
from ....requ import Requirement


class Temperature_V2009(Model):
    LAG_COEFF = 0.8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.STATE, U.degC)
    def temperature(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`s_{\textrm{T-t},i,k}\ [^\circ C]`

        :return: soil layer temperature 
        :rtype: numpy.ndarray
        """

    @Model.is_required('damping_depth', 'zone.soil', U.mm)
    def damping_depth(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{T-dd},k}\ [mm]`

        :return: current damping depth of the soil 
        :rtype: Requirement
        """

    @Model.is_required('center_depth', '.__parent__', U.mm)
    def layer_depth_center(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__'`` (parent soil layer model)

        :math:`s_{\textrm{lcd},i,0}\ [mm]`

        :return: depth of layer center
        :rtype: Requirement
        """

    @Model.is_required('temperature_mean_annual', 'zone.atmosphere.weather', U.degC)
    def temp_annual(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\bar{\textrm{t}}}\ [^\circ C]`

        :return: mean annual temperature
        :rtype: Requirement
        """

    @Model.is_required('temperature', 'zone.soil.surface.temperature', U.degC)
    def temp_surf(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.surface.temperature'``

        :math:`s_{\textrm{T-t},s,k}\ [^\circ C]`

        :return: soil surface temperature 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Just the super call happens in this method.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)

    def update(self, epoch):
        """
        The following computations are performed

        * computation of the depth factor - [R1]_ (equ. 1:1.3.4)
        * computation of the soil layer temperature - [R1]_ (equ. 1:1.3.3)

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        # depth factor to compute temperature [R1]_ equ. 1:1.3.4
        aux = self.layer_depth_center.value / self.damping_depth.value
        df = aux / (aux + np.exp(-0.867 - 2.078 * aux))
        # computation of soil layer temperature [R1]_ equ. 1:1.3.3
        td = self.temp_annual.value - self.temp_surf.value
        self.temperature = self.LAG_COEFF * self.temperature + (
            1.0 - self.LAG_COEFF) * (df  * td + self.temp_surf.value)
