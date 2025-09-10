import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement
from .temperature.model_swat import Temperature_V2009
from .water.model_swat import Water_CNM_V2009
from .evapotranspiration.model_swat import Evapotranspiration_V2009


ALBEDO_SNOW = 0.8  # fraction of radiation, value from [1] equ. 1:1.2.13
ALBEDO_SOIL = 0.19  # mean value of bare fields, according to [2] table 1
ALBEDO_CROP = 0.23  # from [1] sec. 1:1.2.5.1


class Surface_V2009(Model):
    """
    Soil surface model following the outlines in [R1]_ .

    Snow cover is not considered yet.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ov:np.ndarray = None

    @Model.is_quantity(Q.ROUT, U.none)
    def soil_cover_index(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{sci},s,k}\ [\ ]` - [R1]_ (equ. 1:1.2.16)

        :return: soil cover index 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def albedo(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{alb},s,k}\ [\ ]` - [R1]_ (equ. 1:1.2.15)

        :return: albedo at current day (considering crops) 
        :rtype: numpy.ndarray
        """

    @Model.is_required('biomass_aboveground', 'crop', U.kg_ha)
    def biomass(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop'``

        :math:`c_{\textrm{bma},k}\ [\frac{kg}{ha}]`

        :return: aboveground biomass of crop 
        :rtype: Requirement
        """
    @Model.is_required('albedo', 'zone.soil', U.frac)
    def albedo_soil(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{alb},k}\ [\ ]`

        :return: albedo of bare soil considering moisture in first layer 
        :rtype: Requirement
        """

    @Model.is_child_model(Temperature_V2009)
    def temperature(self) -> Temperature_V2009:
        """
        Child Model

        :return: model for surface temperature computations
        :rtype: :class:`ssc_csm.models.soil.surface.temperature.model_swat.Temperature_V2009`
        """

    @Model.is_child_model(Evapotranspiration_V2009)
    def evapotranspiration(self) -> Evapotranspiration_V2009:
        """
        Child Model

        :return: model to compute potential evapotranspiration
        :rtype: Evapotranspiration_V2009
        """

    @Model.is_child_model(Water_CNM_V2009)
    def water(self) -> Water_CNM_V2009:
        """
        Child Model

        :return: surface water model
        :rtype: Water_CNM_V2009
        """

    def initialize(self, epoch):
        """
        Initialization of the child models.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._ov = np.ones((self.model_tree.n_particles,))
        self.albedo = self.albedo_soil.value.copy()
        self.temperature.initialize(epoch)
        self.evapotranspiration.initialize(epoch)
        self.water.initialize(epoch)

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`soil_cover_index`
        * :func:`albedo`
        * update of child models

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        cpres = self.model_tree.get_model('zone').crop_rotation.crop_present

        # TODO data source to determine snow cover
        if cpres:
            self.soil_cover_index = np.exp(-5e-5 * self.biomass.value)
            self.albedo = ALBEDO_CROP * (1.0 - self.soil_cover_index) + (
                self.albedo_soil.value * self.soil_cover_index)
        else:
            self.soil_cover_index = self._ov.copy()
            self.albedo = self.albedo_soil.value

        self.temperature.update(epoch)
        self.evapotranspiration.update(epoch)
        self.water.update(epoch)
