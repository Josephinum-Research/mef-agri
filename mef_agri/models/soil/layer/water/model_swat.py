import numpy as np

from ....base import Model, Quantities as Q
from ....utils import Units as U
from ....requ import Requirement


class Water_V2009(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None

    @Model.is_quantity(Q.STATE, U.frac)
    def moisture(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`s_{\textrm{W-m},i,k}\ [\ ]`

        :return: soil moisture content 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm)
    def amount(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-a},i,k}\ [mm]`

        :return: water amount 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def infiltrated(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-inf},i,k}\ [\frac{mm}{day}]`

        :return: water that infiltrated into the layer at current day 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def percolation(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-per},i,k}\ [\frac{mm}{day}]`

        :return: water that percolated into deeper layer at current day 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def evaporation(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-evp},i,k}\ [\frac{mm}{day}]`

        :return: water that evaporated at current day 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def uptake(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{W-upt},i,k}\ [\frac{mm}{day}]`

        :return: crop water uptake at current day 
        :rtype: numpy.ndarray
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

    @Model.is_required('hydraulic_conductivity_sat', 'zone.soil', U.mm_day)
    def hc(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-hcs},0}\ [\frac{mm}{day}]`

        :return: hydraulic conductivity of saturated soil 
        :rtype: Requirement
        """

    @Model.is_required('porosity', 'zone.soil', U.frac)
    def por(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{por},0}\ [\ ]`

        :return: soil porosity (saturated water content) 
        :rtype: Requirement
        """

    @Model.is_required('thickness', '.__parent__', U.mm)
    def lt(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__'`` (parent layer)

        :math:`s_{\textrm{lt},i,0}\ [m]`

        :return: layer thickness 
        :rtype: Requirement
        """

    @Model.is_required('depth', '.__parent__', U.mm)
    def ld(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__'`` (parent layer)

        :math:`s_{\textrm{ld},i,0}\ [m]`

        :return: depth of bottom layer border 
        :rtype: Requirement
        """

    @Model.is_required('evaporation_pot', 'zone.soil.surface.evapotranspiration', U.mm_day)
    def evp_pot(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.surface.evapotranspiration'``

        :math:`s_{\textrm{W-ep},s,k}\ [\frac{mm}{day}]`

        :return: potential soil evaporation 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Intialization of the water amount (derived from a priori moisture and 
        layer thickness) and other random outputs (zero arrays).

        :param epoch: intialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))

        self.amount = self.moisture * self.lt.value
        self.infiltrated = self._zv.copy()
        self.percolation = self._zv.copy()
        self.evaporation = self._zv.copy()
        self.uptake = self._zv.copy()

    def update(self, epoch):
        """
        The following computations are performed

        * travel time of percolating water ``ttp`` - [R1]_ (equ. 2:3.2.4)
        * percolation ``perc = exc * (1. - np.exp(-1. / ttp))`` (``exc`` being the amount of water exceeding the field capacity)
        * in the case that the remaining water amount exceeds soil porosity, the corresponding excess is added to percolation
        * update of water amount and moisture
        * evaporative demand at soil layer upper and bottom boundary - [R1]_ (equ. 2:2.3.16a)
        * evaporative demand in the layer - [R1]_ (equ. 2:2.3.16b)
        * first correction term - [R1]_ (equ. 2:2.3.18, 2:2.3.19)
        * second correction term - [R1]_ (equ. 2:2.3.20)
        * update water amount and moisture

        NOTE: the tuning variable `esco` from [R1]_ (equ. 2:2.3.17) is omitted

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        ########################################################################
        # Percolation
        fcmm = self.fc.value * self.lt.value
        satmm = self.por.value * self.lt.value
        wpmm = self.wp.value * self.lt.value
        self.amount += self.infiltrated
        exc = self.amount - fcmm
        exc = np.where(exc >= 0.0, exc, 0)
        ttp = (satmm - fcmm) / self.hc.value  # [1] equ. 2:3.2.4
        self.percolation = exc * (1. - np.exp(-1. / ttp))
        # ensure that water amount/content does not exceed layer thickness
        watemp = self.amount - self.percolation
        addperc = np.where(watemp > satmm, watemp - satmm, 0.0)
        self.percolation += addperc
        # update of water amount and moisture content
        self.amount -= self.percolation
        self.moisture = self.amount / self.lt.value

        ########################################################################
        # Evaporation
        edl = self.evp_pot.value * (
            self.ld.value / 
            (self.ld.value + np.exp(2.374 - 0.00713 * self.ld.value))
        )  # evaporative demand at soil layer bottom boundary - [1] equ. 2:2.3.16a
        ud = self.ld.value - self.lt.value  # depth of upper boundary of current layer
        edu = self.evp_pot.value * (ud / (ud + np.exp(2.374 - 0.00713 * ud)))  # evaporative demand at soil layer upper boundary
        ed1 = edl - edu  # evaporative demand in the current layer - [1] equ. 2:2.3.16b
        # NOTE the tuning variable `esco` is omitted herein - [1] equ. 2:2.3.17
        corr1 = np.exp((2.5 * (self.amount - fcmm)) / (fcmm - wpmm))  # [1] equ. 2:2.3.18 + 2:2.3.19
        ed2 = np.where(
            self.amount < fcmm, ed1 * corr1, ed1
        )
        corr2 = 0.8 * (self.amount - wpmm)  # [1] equ. 2:2.3.20
        ed3 = np.where(ed2 > corr2, corr2, ed2)
        self.evaporation = np.where(ed3 > self.amount, self.amount, ed3)  # step to be sure that water amount cannot be negative - should be ensured anyway by `corr2`
        self.amount -= self.evaporation
        self.moisture = self.amount / self.lt.value
