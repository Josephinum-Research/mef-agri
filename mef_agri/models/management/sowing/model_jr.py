import numpy as np
from datetime import timedelta

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement


class Sowing(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.OBS, U.kg_ha)
    def sowing_amount(self) -> np.ndarray:
        """
        :return: sowing amount in [kg/ha]
        :rtype: numpy.ndarray
        """

    @Model.is_required('biomass', 'crop', U.kg_ha)
    def bmt(self) -> Requirement:
        """
        RQ - ``'biomass'`` from model with id ``'crop'``

        :return: total biomass of crop
        :rtype: Requirement
        """

    @Model.is_required('biomass', 'crop.roots', U.kg_ha)
    def bmr(self) -> Requirement:
        """
        RQ - ``'biomass'`` from model with id ``'crop.roots'``

        :return: root biomass
        :rtype: Requirement
        """

    def update(self, epoch):
        super().update(epoch)
        if self.sowing_amount is None:
            return
        elif self.get_obs_epoch('sowing_amount') == (epoch - timedelta(days=1)):
            pass
            #self.bmt.value = self.sowing_amount
            #self.bmr.value = self.sowing_amount
        else:
            self.reset_quantities(self.observation_names, force=True)
