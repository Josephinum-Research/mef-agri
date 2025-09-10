import numpy as np

from ....base import Model, Quantities as Q
from ....utils import Units as U
from ....requ import Requirement


class Temperature_V2009(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None

    @Model.is_quantity(Q.ROUT, U.degC)
    def temperature_bare(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{T-tb},s,k}\ [^\circ C]` - [R1]_ (equ. 1:1.3.9)

        :return: surface temperature of bare soil at current day 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.degC)
    def temperature(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{T-t},s,k}\ [^\circ C]` - [R1]_ (equ. 1:1.3.12)

        :return: soil surface temperature 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def radiation_term(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{T-rt},s,k}\ [\ ]` - [R1]_ (equ. 1:1.3.10)

        :return: radiation term 
        :rtype: numpy.ndarray
        """

    @Model.is_required('temperature_mean', 'zone.atmosphere.weather', U.degC)
    def temp_mean(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{temp},k}\ [^\circ C]`

        :return: daily mean temperature 
        :rtype: Requirement
        """

    @Model.is_required('temperature_min', 'zone.atmosphere.weather', U.degC)
    def temp_min(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{tmin},k}\ [^\circ C]`

        :return: daily min. temperature 
        :rtype: Requirement
        """

    @Model.is_required('temperature_max', 'zone.atmosphere.weather', U.degC)
    def temp_max(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{tmax},k}\ [^\circ C]`

        :return: daily max. temperature 
        :rtype: Requirement
        """

    @Model.is_required('radiation_sum', 'zone.atmosphere.weather', U.MJ_m2xday)
    def radiation(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{rad},k}\ [\frac{MJ}{m^2\cdot day}]`

        :return: daily radiation sum 
        :rtype: Requirement
        """

    @Model.is_required('albedo', 'zone.soil.surface', U.frac)
    def albedo(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.surface'``

        :math:`s_{\textrm{alb},s,k}\ [\ ]` - [R1]_ (equ. 1:1.2.15)

        :return: albedo at current day (considering crops) 
        :rtype: Requirement
        """

    @Model.is_required('biomass_aboveground', 'crop', U.kg_ha)
    def biomass(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop'``

        :math:`c_{\textrm{bma},k}\ [\frac{kg}{ha}]`

        :return: aboveground biomass of crop 
        :rtype: Requirement
        """

    @Model.is_required('temperature', 'zone.soil.layer01.temperature', U.degC)
    def temp_soil(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.layer01.temperature'``

        :math:`s_{\textrm{T-t},1,k}\ [^\circ C]`

        :return: temperatuer of first soil layer 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Just the super-call and initialization of generic zero array.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))

    def update(self, epoch):
        """
        The following computations are performed

        * compute :func:`radiation_term`
        * compute :func:`temperature_bare`
        * weighting factor (snow cover not considered) - [R1]_ (equ. 1:1.3.11)
        * compute :func:`temperature`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        cp = self.model_tree.get_model('zone').crop_rotation.crop_present

        self.radiation_term = (self.radiation.value * (
            1.0 - self.albedo.value) - 14.0) / 20.0
        self.temperature_bare = self.temp_mean.value + self.radiation_term * (
            0.5 * (self.temp_max.value - self.temp_min.value)
        )
        if cp:
            # weighting factor - [R1]_ equ. 1:1.3.11
            # TODO snow cover not considered yet
            wf = self.biomass.value / (self.biomass.value + np.exp(
                7.563 - 1.297 * 1e-4 * self.biomass.value
            ))
        else:
            wf = self._zv.copy()
        self.temperature = wf * self.temp_soil.value + \
            (1.0 - wf) * self.temperature_bare
