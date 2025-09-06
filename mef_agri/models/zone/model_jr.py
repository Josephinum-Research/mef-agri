from ..base import Model
from .base import Zone as ZBase
from ..atmosphere.model_swat import Atmosphere_V2009
from ..soil.model_swat import Soil_V2009_EPIC
from ..management.model_jr import Management
from ..satellite.sentinel_2.model_inrae import Sentinel2_LAI10m


class ZoneJR_V1(ZBase):
    def __init__(self):
        """
        The "model computation chain" is done as follows

        * update of the atmosphere model components
        * update of the soil model components 
          * considering soil crop-cover (e.g. for evapotranspiration and temperature)
          * compute soil supply based on crop demand and soil state
        * update of the crop model components (including demands)
          * compute and set the actual crop uptake

        """
        super().__init__()

    @Model.is_child_model(Atmosphere_V2009)
    def atmosphere(self) -> Atmosphere_V2009:
        """
        :return: SWAT V2009 atmosphere model 
        :rtype: Atmosphere_V2009
        """

    @Model.is_child_model(Soil_V2009_EPIC)
    def soil(self) -> Soil_V2009_EPIC:
        """
        :return: SWAT V2009 soil model
        :rtype: Soil_V2009
        """

    @Model.is_child_model(Management)
    def management(self) -> Management:
        """
        :return: field management model
        :rtype: ManagementJR
        """

    @Model.is_child_model(Sentinel2_LAI10m)
    def sentinel2_lai(self) -> Sentinel2_LAI10m:
        """
        :return: model which provides lai observations from sentinel-2
        :rtype: Sentinel2_LAI
        """

    def initialize(self, epoch=None):
        super().initialize(epoch)
        self.management.initialize(epoch)
        self.atmosphere.initialize(epoch)
        self.soil.initialize(epoch)
        self.sentinel2_lai.initialize(epoch)

    def update(self, epoch):
        self.management.update(epoch)
        super().update(epoch)
        self.atmosphere.update(epoch)
        self.soil.update(epoch)
        self.sentinel2_lai.update(epoch)
        if self.crop_rotation.crop_present:
            # process the crop model from the day after sowing-date until 
            # it is not present anymore (i.e. harvesting or mulching)
            self.crop_rotation.current_crop.update(epoch)
