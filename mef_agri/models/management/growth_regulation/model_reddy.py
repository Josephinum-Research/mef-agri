import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement


class GrowthRegulation(Model):
    """
    Model originally for mepiquat-chlorid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.OBS, U.kg_ha)
    def ag_applied(self) -> np.ndarray:
        """
        MQ - Observation

        :return: applied active ingredient
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.STATE, U.kg_t)
    def ag_concentration(self) -> np.ndarray:
        """
        MQ - State

        :return: concentration of active ingredient per pl
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.frac)
    def crop_interception(self) -> np.ndarray:
        """
        MQ - Parameter

        :return: fraction of active ingredient which is indeed intercepted by plant
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.frac)
    def decomposition_rate(self) -> np.ndarray:
        """
        MQ - Parameter

        :return: decomposition rate per day of active ingredient concentration in crop biomass
        :rtype: np.ndarray
        """

    @Model.is_required('biomass', 'crop', U.t_ha)
    def bm(self) -> Requirement:
        """
        RQ - 

        :return: _description_
        :rtype: Requirement
        """

    def update(self, epoch):
        super().update(epoch)
        
        self.ag_concentration *= (1. - self.decomposition_rate)
        if self.ag_applied is None:
            return

        agadd = self.crop_interception * self.ag_applied
        self.ag_concentration += agadd / self.bm.value

        # TODO how to generally integrate reduction factors in crop model?
        #rf_stem_elong = 1. - 18.619 * self.ag_concentration
        #rf_leaf_exp = 1. - 7.774 * self.ag_concentration
        #rf_photosynth = 1. - 25.167 * self.ag_concentration 
        #rf_photosynth += 590.286 * np.power(self.ag_concentration, 2.)
