from copy import deepcopy
import numpy as np
from datetime import date

from ...base import Model, Quantities as Q, Units as U
from ...requ import Requirement
from ...utils_soil import distribute_nutrient_amount


class Supply(Model):
    r"""
    Computation of water and nitrate supply from the soil to the crop according 
    to [R2]_

    kwargs :math:`\rightarrow` :class:`ssc_csm.models.base.Model`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None
        self._ov:np.ndarray = None

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def water(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-avl},k}\ [\frac{mm}{day}]\ \rightarrow` [R2]_ : equ. 23 + 24

        :return: amount of water which is available for the crop
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def nitrogen(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{N-avl},k}\ [\frac{kg}{ha\cdot day}]\ \rightarrow` [R2]_ : equ 27 - 29

        :return: amount of nitrogen which is available for the crop at current day
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def moisture_1m(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-m1m},k}\ [\ ]\ \rightarrow` [R2]_ : part of equ. 49

        :return: moisture of first 1.0 m of soil 
        :rtype: numpy.ndarray
        """

    @Model.is_required('water_use_distribution_factor', 'zone.soil', U.none)
    def wudf(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-udf},0}\ [\ ]`

        :return: water use distribution factor
        :rtype: Requirement
        """

    @Model.is_required('rooting_depth_max', 'zone.soil', U.m)
    def rdm(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{rdm},0}\ [m]`

        :return: maximum rootable depth of the soil
        :rtype: Requirement
        """

    @Model.is_required('field_capacity', 'zone.soil', U.frac)
    def fc(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-fc},0}\ [\ ]`

        :return: soil moisture at field capacity 
        :rtype: Requirement
        """

    @Model.is_required('wilting_point', 'zone.soil', U.frac)
    def wp(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-wp},0}\ [\ ]`

        :return: soil moisture at wilting point 
        :rtype: Requirement
        """

    @Model.is_required('water', 'crop.demand', U.mm_day)
    def water_demand(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop.demand'``

        :math:`c_{\textrm{W-dem},k} ... [\frac{mm}{day}]`

        :return: water demand of crop at current day 
        :rtype: Requirement
        """

    @Model.is_required('temperature_base', 'crop.development', U.degC)
    def tb(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop.development'``

        :math:`c_{\textrm{D-tb},0}\ [^\circ C]`

        :return: crop-specific base temperature 
        :rtype: Requirement
        """

    @Model.is_required('temperature_opt', 'crop.development', U.degC)
    def to(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop.development'``

        :math:`c_{\textrm{D-to},0}\ [^\circ C]`

        :return: crop-specific optimum temperature 
        :rtype: Requirement
        """

    @Model.is_required('biomass_rate', 'crop.roots', U.t_haxday)
    def drbm(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop.roots'``

        :math:`c_{\textrm{R-}\Delta\textrm{bm},k}\ [\frac{t}{ha\cdot day}]`

        :return: daily increase of root biomass 
        :rtype: Requirement
        """

    @Model.is_required('nitrogen', 'crop.demand', U.kg_haxday)
    def ndem(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop.demand'``

        :math:`c_{\textrm{N-dem},k}\ [\frac{kg}{ha}]`

        :return: nitrogen demand of the crop 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Initialization of generic zero- and one-arrays.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))
        self._ov = np.ones((self.model_tree.n_particles,))

    def update(self, epoch):
        """
        The following computations are performed

        * check if crop is present - if not, uptake rates are set to zero
        * layer: compute temperature stress affecting the water use - [R2]_ : equ. 46 (using soil layer temp. instead of surface temp.)
        * layer: compute potential water use - [R2]_ : equ. 22
        * layer: compute actual water use - [R2]_ : equ. 23
        * layer: adjust water amount considering water use
        * layer: nitrate uptake due to mass flow - [R2]_ : equ. 27
        * ensure, that the overall nitrate uptake does not exceed the crop demand
        * increase the overall nitrate uptake due to mass flow to account for diffusion uptake - [R2]_ : equ. 29 (NOTE: if mass flow uptake is very small, the adjusted uptake will be set to zero to avoid numerical instabilities)
        * layer: distribute root biomass to soil layers with respect ot water use - [R2]_ : equ. 13
        * layer: adjust nitrate uptake amounts to account for diffusion uptake
        * compute the moisture of the first meter of soil (also if no crop is present)

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        cp = self.model_tree.get_model('zone').crop_rotation.crop_present
        smdl = self.model_tree.get_model('zone.soil')

        sumw, wc_wsum = self._zv.copy(), self._zv.copy()
        if not cp:
            # CROP NOT PRESENT
            # set crop uptake rates to zero before returning
            for layer in smdl.layers:
                self.model_tree.set_quantity(
                    'uptake', layer.model_id + '.water', 
                    self._zv.copy(), unit=U.mm_day
                )
                self.model_tree.set_quantity(
                    'NO3_uptake', layer.model_id + '.nutrients.nitrogen',
                    self._zv.copy(), unit=U.kg_haxday
                )

                ld = self.model_tree.get_quantity(
                    'depth', layer.model_id, unit=U.m
                )
                lt = self.model_tree.get_quantity(
                    'thickness', layer.model_id, unit=U.m
                )
                wc = self.model_tree.get_quantity(
                    'moisture', layer.model_id + '.water', unit=U.frac
                )

                wi, exc1 = self._zv.copy(), ld - 1.0
                wi = np.where(exc1 < 0.0, lt, wi)
                wi = np.where(exc1 < lt, lt - exc1, wi)
                sumw += wi
                wc_wsum += wi * wc
            self.moisture_1m = wc_wsum / sumw
            return

        ldab, wusum = 0., self._zv.copy()
        aux1 = self.water_demand.value / (1. - np.exp(-self.wudf.value))
        nusum = self._zv.copy()
        for layer in smdl.layers:
            ##### get layer quantities #########################################
            lwmid = layer.model_id + '.water'
            # soil layer temperature
            slt = self.model_tree.get_quantity(
                'temperature', layer.model_id + '.temperature', unit=U.degC
            )
            # depth of bottom layer border
            ld = self.model_tree.get_quantity('depth', layer.model_id, unit=U.m)
            # thickness of layer
            lt = self.model_tree.get_quantity(
                'thickness', layer.model_id, unit=U.m
            )
            # current water amount
            wc = self.model_tree.get_quantity('moisture', lwmid, unit=U.frac)
            wa = wc * lt * 1e3
            
            ##### compute water use ############################################
            # temperature stress affecting water use [R2]_ equ. 46 (soil layer
            # temperature instead of soil surface temperature)
            ts = np.sin(0.5 * np.pi * (
                (slt - self.tb.value) / (self.to.value - self.tb.value)
            ))
            # potential water use
            e1 = np.exp(-self.wudf.value * (ld / self.rdm.value))
            e2 = np.exp(-self.wudf.value * (ldab / self.rdm.value))
            lwup = aux1 * (1.0 - e1 - ((1.0 - ts) * (1.0 - e2))) - ts * wusum  # [mm/day]
            # actual water use
            c2 = wc < ((self.fc.value - self.wp.value) / 4.0) + self.wp.value
            aux2 = (wc - self.wp.value) / (self.fc.value - self.wp.value)
            lwu2 = lwup * np.exp(5.0 * (4.0 * aux2 - 1.0))  # [mm/day]
            lwu = np.where(c2, lwu2, lwup)  # [mm/day]
            # adjust layer water amount
            # TODO maybe additional constraint, such that `wc` cannot get much 
            # TODO lower than `wp` (or even negative?)
            wctemp = wc - lwu / (lt * 1e3)  # []
            watemp = wa - lwu  # [mm]
            self.model_tree.set_quantity('moisture', lwmid, wctemp, unit=U.frac)
            self.model_tree.set_quantity('amount', lwmid, watemp, unit=U.mm)
            self.model_tree.set_quantity('uptake', lwmid, lwu, unit=U.mm_day)
            ##### weighted sum for saturation factor ###########################
            wi, exc1 = self._zv.copy(), ld - 1.0
            wi = np.where(exc1 < 0.0, lt, wi)
            wi = np.where(exc1 < lt, lt - exc1, wi)
            sumw += wi
            wc_wsum += wi * wctemp

            ##### updating values of "above" layers ############################
            ldab = deepcopy(ld)
            wusum += lwu

            ##### nutrient stuff ###############################################
            # no3 uptake because of mass flow (i.e. water uptake)
            lna = self.model_tree.get_quantity(
                'NO3', layer.model_id + '.nutrients.nitrogen', unit=U.kg_ha
            )
            lnu = (lwu / wa) * lna
            nusum += lnu


        ##### another iteration for root biomass and nutrients #################
        # ensure that nitrogen uptake due to mass flow does not exceed the 
        # nitrogen demand of the crop
        nusum = np.where(nusum > self.ndem.value, self.ndem.value, nusum)
        # if mass flow uptake is very small, the adjusted uptake will be set 
        # to zero to avoid numerical instabilities
        # otherwise it will be increased to account for diffusion uptake and not
        # exceeding the crop demand
        with np.errstate(divide='ignore', invalid='ignore'):
            fc = np.where(nusum >= 1e-6, self.ndem.value / nusum, 0.0)
        
        nusum_a = self._zv.copy()
        for layer in smdl.layers:
            ##### distribute root biomass to soil layers #######################
            lwu = self.model_tree.get_quantity(
                'uptake', layer.model_id + '.water', unit=U.mm_day
            )
            wc = self.model_tree.get_quantity('moisture', lwmid, unit=U.frac)
            wa = wc * lt * 1e3

            if np.min(wusum) < 1e-8:
                aux3 = self._zv.copy()
            else:
                aux3 = lwu / wusum
            ldrbm = self.drbm.value * aux3
            lrbm = self.model_tree.get_quantity(
                'biomass', layer.model_id, unit=U.t_ha
            )
            self.model_tree.set_quantity(
                'biomass_rate', layer.model_id, ldrbm, unit=U.t_haxday
            )
            self.model_tree.set_quantity(
                'biomass', layer.model_id, lrbm + ldrbm, unit=U.t_ha
            )

            ##### nitrogen supply computation ##################################
            lnmid = layer.model_id + '.nutrients.nitrogen'
            lna = self.model_tree.get_quantity('NO3', lnmid, unit=U.kg_ha)
            lnu_a = fc * (lwu / (wa + lwu)) * lna
            lnu_a = np.where(lnu_a <= lna, lnu_a, lna)
            # additional check to avoid cases where supplied nitrate exceeds 
            # the crop demand
            ctrl = self.ndem.value - nusum_a - lnu_a
            lnu_a = np.where(ctrl < 0.0, lnu_a + ctrl, lnu_a)

            nusum_a += lnu_a
            self.model_tree.set_quantity(
                'NO3_uptake', lnmid, lnu_a, unit=U.kg_haxday
            )
            self.model_tree.set_quantity(
                'NO3', lnmid, lna - lnu_a, unit=U.kg_ha
            )
        
        ##### compute model outputs ############################################
        self.water = wusum.copy()
        self.moisture_1m = wc_wsum / sumw
        self.nitrogen = nusum_a.copy()
