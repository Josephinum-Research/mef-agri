import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement


class Harvest(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None

    @Model.is_quantity(Q.OBS, U.kg_ha)
    def cyield(self) -> np.ndarray:
        """
        :return: yield of current zone in [kg/ha]
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.none)
    def cn_ratio_res(self) -> np.ndarray:
        """
        :return: C/N ratio of crop residues and roots
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.frac)
    def residues_removed(self) -> np.ndarray:
        """
        :return: fraction of aboveground biomass which is removed
        :rtype: numpy.ndarray
        """

    @Model.is_required('biomass_aboveground', 'crop', U.kg_ha)
    def bma(self) -> Requirement:
        """
        :return: crop aboveground biomass [kg/ha]
        :rtype: Requirement
        """

    @Model.is_required('biomass', 'crop.cyield', U.kg_ha)
    def bmy(self) -> Requirement:
        """
        :return: yield biomass [kg/ha]
        :rtype: Requirement
        """

    def initialize(self, epoch=None):
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))

    def update(self, epoch=None):
        super().update(epoch)

        if self.cn_ratio_res is None:
            return
        
        # crop residues from aboveground biomass
        if self.residues_removed is None:
            self.residues_removed = self._zv.copy()
        cr_abgr = (self.bma.value - self.bmy.value)
        cr_abgr *= (1. - self.residues_removed)
        
        # compute carbon amounts in soil layers from roots
        # crop residues from aboveground biomass are added to the first layer
        smdl = self.model_tree.get_model('zone.soil')
        for layer in smdl.layers:
            crbm = self.model_tree.get_quantity(
                'biomass', layer.model_id , U.kg_ha
            )
            if cr_abgr is not None:
                crbm += cr_abgr
                cr_abgr = None

            lnm = layer.model_id + '.nutrients'
            lcm = lnm + '.carbon'
            cres = self.model_tree.get_quantity('C_res', lcm, U.kg_ha)
            cnres = self.model_tree.get_quantity('CN_res', lnm, U.none)
            cadd = 0.58 * crbm  # assumption that 58% of biomass is carbon
            cnres = (cres * cnres + cadd * self.cn_ratio_res) / (cres + cadd)
            cres += cadd
            self.model_tree.set_quantity('C_res', lcm, cres, unit=U.kg_ha)
            self.model_tree.set_quantity('CN_res', lnm, cnres, unit=U.none)
            # set biomass in layers to zero
            self.model_tree.set_quantity(
                'biomass', layer.model_id, self._zv.copy(), unit=U.kg_ha
            )
            self.model_tree.set_quantity(
                'biomass_rate', layer.model_id, self._zv.copy(), 
                unit=U.kg_haxday
            )
        
        self.cn_ratio_res = None
        self.residues_removed = None
