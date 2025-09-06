import numpy as np

from ...base import Model, Quantities as Q, Units as U
from ...utils_soil import distribute_nutrient_amount

class Mineral_N_Fertilization(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frs:np.ndarray = None

    @Model.is_quantity(Q.OBS, U.kg_ha)
    def NO3_applied(self) -> np.ndarray:
        r"""
        :return: applied amount of :math:`NO_{3}\ [\frac{kg}{ha}]`
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.kg_ha)
    def NH4_applied(self) -> np.ndarray:
        r"""
        :return: applied amount of :math:`NH_{4}\ [\frac{kg}{ha}]`
        :rtype: numpy.ndarray
        """

    def initialize(self, epoch=None):
        super().initialize(epoch)
        soil = self.model_tree.get_model('zone.soil')
        if soil is None:
            return
        self._frs = distribute_nutrient_amount(len(soil.layers), mode='linear')

    def update(self, epoch=None):
        super().update(epoch)
        soil = self.model_tree.get_model('zone.soil')
        if self._frs is None:
            self._frs = distribute_nutrient_amount(
                len(soil.layers), mode='linear'
            )

        add_no3 = False if self.NO3_applied is None else True
        add_nh4 = False if self.NH4_applied is None else True
        if not add_no3 and not add_nh4:
            return
        for layer, fr in zip(soil.layers, self._frs):
            lnmid = layer.model_id + '.nutrients.nitrogen'
            if add_no3:
                no3 = self.model_tree.get_quantity('NO3', lnmid, unit=U.kg_ha)
                no3 += fr * self.NO3_applied
                self.model_tree.set_quantity('NO3', lnmid, no3, unit=U.kg_ha)
            if add_nh4:
                nh4 = self.model_tree.get_quantity('NH4', lnmid, unit=U.kg_ha)
                nh4 += fr * self.NH4_applied
                self.model_tree.set_quantity('NH4', lnmid, nh4, unit=U.kg_ha)
                
        self.NO3_applied = None
        self.NH4_applied = None
