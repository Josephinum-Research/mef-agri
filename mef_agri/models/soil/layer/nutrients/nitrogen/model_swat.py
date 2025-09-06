import numpy as np

from .....base import Model, Quantities as Q, Units as U
from .....requ import Requirement


class N_NO3_NH4_V2009(Model):
    r"""
    Soil layer model for mineral nitrogen which contains the 
    :math:`NO_3^-` and the :math:`NH_4^+`-pool. 
    Nitrification is the only process considered herein - no losses such as 
    nitrate leaching, denitrification or ammonia volatilization.

    kwargs :math:`\rightarrow` :class:`ssc_csm.models.base.Model`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.STATE, U.kg_ha)
    def NO3(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`s_{\textrm{NO}_{3}^-,i,k}\ [\frac{kg}{ha}]`

        :return: amount of nitrate in the soil layer 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.STATE, U.kg_ha)
    def NH4(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`s_{\textrm{NH}_{4}^+,i,k}\ [\frac{kg}{ha}]`

        :return: amount of ammonium in the soil layer 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def NO3_uptake(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{NO}_{3}^{-}-u,i,k}\ [\frac{kg}{ha}]`

        :return: amount of nitrate removed from soil layer by the crop 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def NH4_uptake(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{NH}_{4}^{+}-u,i,k}\ [\frac{kg}{ha}]`

        :return: amount of ammonium removed from soil layer by the crop 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_ha)
    def nitrification(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{N-nit},i,k}\ [\frac{kg}{ha}]`

        :return: amount of ammonium which is removed from the pool at current day (i.e. added to the NO3-pool) 
        :rtype: numpy.ndarray
        """

    @Model.is_required('temperature', '.__parent__.__parent__.temperature', unit=U.degC)
    def ltemp(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__.__parent__.temperature'`` (temperature model of parent layer)

        :math:`s_{\textrm{T-t},i,k}\ [^\circ C]`

        :return: soil layer temperature 
        :rtype: Requirement
        """

    @Model.is_required('amount', '.__parent__.__parent__.water', unit=U.mm)
    def lwa(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__.__parent__.water'`` (water model of parent layer)

        :math:`s_{\textrm{W-a},i,k}\ [mm]`

        :return: water amount of soil layer 
        :rtype: Requirement
        """

    @Model.is_required('thickness', '.__parent__.__parent__', unit=U.mm)
    def lt(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__.__parent__'`` (parent layer)

        :math:`s_{\textrm{lt},i,0}\ [mm]`

        :return: layer thickness 
        :rtype: Requirement
        """

    @Model.is_required('wilting_point', 'zone.soil', unit=U.frac)
    def wp(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-wp},0}\ [\ ]`

        :return: soil moisture at wilting point 
        :rtype: Requirement
        """

    @Model.is_required('field_capacity', 'zone.soil', unit=U.frac)
    def fc(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-fc},0}\ [\ ]`

        :return: soil moisture at field capacity 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Only the super-call is done here.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)

    def update(self, epoch):
        """
        The following computations are performed

        * nitrification temperature factor - [R1]_ (equ. 3:1.3.1)
        * nitrification soil water factor - [R1]_ (equ. 3:1.3.2, 3:1.3.3)
        * nitrification factor - [R1]_ (equ. 3:1.3.6)
        * nitrification rate - [R1]_ (equ. 3:1.3.11 with setting volatilization factor to zero :math:`\Rightarrow` equ. 3:1.3.11 is equal to equ. 3:1.3.8)
        * update N-pools

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        # nitrification temperature factor - [1] equ. 3:1.3.1
        ftemp = 0.41 * ((self.ltemp.value - 5.0) / 10.0)
        ftemp = np.where(self.ltemp.value > 5.0, ftemp, 0.0)
        # nitrification soil water factor - [1] equ. 3:1.3.2 + 3:1.3.3
        wpmm = self.wp.value * self.lt.value
        fcmm = self.fc.value * self.lt.value
        fsw = (self.lwa.value - wpmm) / (0.25 * (fcmm - wpmm))
        check = self.lwa.value < (0.25 * fcmm - 0.75 * wpmm)
        fsw = np.where(check, fsw, 1.0)
        # nitrification factor - [1] equ. 3:1.3.6
        fnit = ftemp * fsw
        # simplification by setting volatilization factor ([1] equ. 3:1.3.7) to 
        # zero - i.e equ. 3:1.3.11 equals 3:1.3.8 in [1]
        self.nitrification = self.NH4 * (1. - np.exp(-fnit))
        # adjusting pools
        self.NO3 += self.nitrification
        self.NH4 -= self.nitrification


class N_NO3_NH4_Norg_V2009(N_NO3_NH4_V2009):
    r"""
    Contrary to :class:`N_NO3_NH4_V2009`, this model additionally considers the 
    organic N-pool and mineralization. Again no losses such as 
    nitrate leaching, denitrification or ammonia volatilization.

    Inherits from :class:`N_NO3_NH4_V2009`

    kwargs :math:`\rightarrow` :class:`ssc_csm.models.base.Model`

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.STATE, U.kg_ha)
    def N_org(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`s_{\textrm{N-org},i,k}\ [\frac{kg}{ha}]` - [R1]_ (section 3:5.1)

        :return: organic N-pool 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def N_org_add(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{N-}\Delta\textrm{o},i,k}\ [\frac{kg}{ha\cdot day}]` - [R1]_ (equ. 3:5.1.2b)

        :return: input from decomposed crop residuals to the organic N-pool 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def mineralized_N(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{N-mo},i,k}\ [\frac{kg}{ha\cdot day}]`

        :return: mineralized N from organic N-pool 
        :rtype: numpy.ndarray
        """

    @Model.is_required('humified_C_res', '.__parent__.carbon', U.kg_haxday)
    def chum_res(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__.carbon'`` (carbon model from parent nutrient model)

        :math:`s_{\textrm{C-rdh},i,k}\ [\frac{kg}{ha\cdot day}]`
        
        :return: amount of decomposed C from crop residuals which is added to the organic C-pool (humus) 
        :rtype: Requirement
        """

    @Model.is_required('CN_org', '.__parent__', U.none)
    def cnorg(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__'`` (parent nutrient model)
        
        :math:`s_{\textrm{CN-o},i,k}\ [\ ]`

        :return: C/N ratio of organic matter in the soil layer 
        :rtype: Requirement
        """

    @Model.is_required('mineralization', '.__parent__.carbon', U.per_day)
    def fmin(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__.carbon'`` (carbon model from parent nutrient model)

        :math:`s_{\textrm{min},i,k}\ [\frac{1}{day}]`

        :return: mineralization rate 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        :func:`N_org_add` is initialized with a zero array.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self.N_org_add = np.zeros((self.model_tree.n_particles,))

    def update(self, epoch):
        r"""
        Additionally to :func:`N_NO3_NH4_V2009.update`, 
        the following computations are performed

        * increase of organic N-pool (proportional to humified C) - [R1]_ (equ. 3:5.1.2b)
        * mineralization of organic N which feeds the :math:`\textrm{NH}_4^+`-pool - [R1]_ (equ. 3:5.1.2b)

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        # increase of organic N-pool is proportional to the humified C from 
        # crop residuals - [1] equ. 3:5.1.2b
        self.N_org_add = self.chum_res.value / self.cnorg.value
        self.N_org += self.N_org_add
        # mineralization of organic N - [1] equ. 3:5.1.2b
        self.mineralized_N = self.fmin.value * self.N_org
        self.N_org -= self.mineralized_N
        # mineralization feeds the NH4-pool according to [1] section 3:5.2
        self.NH4 += self.mineralized_N
