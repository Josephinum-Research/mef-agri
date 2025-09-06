from ..base import Model
from .daylength.model_swat import Daylength_V2009
from .radiation.model_swat import Radiation_V2009
from .weather.model_swat import Weather_V2009


class Atmosphere_V2009(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_child_model(Daylength_V2009)
    def daylength(self) -> Daylength_V2009:
        """
        :return: computation of daylength and related quantities
        :rtype: :class:`mef_agri.models.atmosphere.daylength.model_swat.Daylength_V2009`
        """

    @Model.is_child_model(Weather_V2009)
    def weather(self) -> Weather_V2009:
        """
        :return: weather module providing inca(geosphere) observations and other computed quantities
        :rtype: :class:`mef_agri.models.atmosphere.weather.model_swat.WeatherINCA`
        """

    @Model.is_child_model(Radiation_V2009)
    def radiation(self) -> Radiation_V2009:
        """
        :return: computation of quantities related to radiation
        :rtype: :class:`mef_agri.models.atmosphere.radiation.model_swat.Radiation_V2009`
        """

    def initialize(self, epoch=None):
        super().initialize(epoch)
        self.weather.initialize(epoch)
        self.daylength.initialize(epoch)
        self.radiation.initialize(epoch)

    def update(self, epoch=None):
        super().update(epoch)
        self.weather.update(epoch)
        self.daylength.update(epoch)
        self.radiation.update(epoch)
