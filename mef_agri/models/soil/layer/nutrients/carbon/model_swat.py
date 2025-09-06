import numpy as np

from .....base import Model, Quantities as Q, Units as U
from .....requ import Requirement


class C_Corg_Cres_V2009(Model):
    r"""
    This model implements the carbon model from [R1]_ section 3:5, which is 
    composed of differential equations. Herein, these differential equations 
    are used directly because they are solved on a daily basis (i.e. 
    numerical integration). This seems to be reasonable compared to [R1]_ 
    section 3:5.3, where the analytical solutions are derived and evaluated 
    at time steps of one year.

    kwargs :math:`\rightarrow` :class:`ssc_csm.models.base.Model`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.STATE, U.kg_ha)
    def C_org(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`s_{\textrm{C-org},i,k}\ [\frac{kg}{ha}]` - [R1]_ (section 3:5.1)

        :return: organic C-pool (~humus) 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.STATE, U.kg_ha)
    def C_res(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`s_{\textrm{C-res},i,k}\ [\frac{kg}{ha}]` - [R1]_ (section 3:5.1)

        :return: C-pool in the crop residues 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def decomposed_C_res(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{C-rd},i,k}\ [\frac{kg}{ha\cdot day}]` - [R1]_ (equ. 3:5.1.1a)

        :return: C decomposed from crop residuals at current day
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def humified_C_res(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{C-rdh},i,k}\ [\frac{kg}{ha\cdot day}]` - [R1]_ (equ. 3:5.1.2a)

        :return: amount of decomposed C from crop residuals which is added to the organic C-pool (humus) 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.per_day)
    def mineralization(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{min},i,k}\ [\frac{1}{day}]` [R1]_ (equ. 3:5.1.7 neglecting tillage factor)

        :return: mineralization rate 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def mineralized_C_org(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{C-om},i,k}\ [\frac{kg}{ha\cdot day}]`

        :return: mineralized C from organic C-pool 
        :rtype: numpy.ndarray
        """

    @Model.is_required('nutrient_cycling_factor', '.__parent__', U.none)
    def fnc(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__'`` (parent nutrient model)

        :math:`s_{\textrm{ncf},i,k}\ [\ ]`

        :return: factor combining soil temperatur and water influence on nutrient cycle 
        :rtype: Requirement
        """

    @Model.is_required('bulk_density', 'zone.soil', U.g_cm3)
    def sbd(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{bd},0}\ [\frac{g}{cm^3}]`

        :return: bulk density of the soil 
        :rtype: Requirement
        """

    @Model.is_required('clay_content', 'zone.soil', U.frac)
    def scl(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{clay},0}\ [\ ]`

        :return: clay content of soil 
        :rtype: Requirement
        """

    @Model.is_required('thickness', '.__parent__.__parent__', U.m)
    def lt(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__.__parent__'`` (parent layer)

        :math:`s_{\textrm{lt},i,0}\ [m]`

        :return: layer thickness 
        :rtype: Requirement
        """

    @Model.is_required('decomposition_res_opt', 'zone.soil', U.per_day)
    def dC_res_opt(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{C-rdo},0}\ [\frac{1}{day}]`

        :return: optimum decomposition rate of crop residual C-pool (percent per day)
        :rtype: Requirement
        """

    @Model.is_required('mineralization_opt', 'zone.soil', U.per_day)
    def min_opt(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{C-omo},0}\ [\frac{1}{day}]`

        :return: optimum mineralization rate from organic C-pool (percent per day)
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
        r"""
        The following computations are performed

        * decomposed C on the current day :math:`s_{\textrm{C-rd},i,k}=s_{\textrm{C-res},i,k}\cdot s_{\textrm{C-rdo},0}\cdot s_{\textrm{ncf},i,k}`
        * reference organic carbon - [R1]_ (equ. 3:5.1.4)
        * auxiliary variable *hx* - [R1]_ (equ. 3:5.1.3b)
        * humified C from crop residuals - [R1]_ (equ. 3:5.1.3a)
        * current mineralization rate - [R1]_ (equ. 3:5.1.7 - neglecting tillage factor)
        * mineralized C - [R1]_ (subtracting term in equ. 3:5.1.2a)

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        ##### decomposition of crop residuals ##################################
        self.decomposed_C_res = self.dC_res_opt.value * self.C_res
        self.decomposed_C_res *= self.fnc.value
        self.C_res -= self.decomposed_C_res

        ##### humification of crop residuals ###################################
        # reference organic carbon - [1] equ. 3:5.1.4
        scc = self.sbd.value * self.lt.value * (0.021 + 0.038 * self.scl.value)  # [t/m2]
        scc *= 1e7  # [kg/ha]
        # auxiliary variable hx - [1] equ. 3:5.1.3b
        hx = 0.09 * (2. - np.exp(-5.5 * self.scl.value))
        # humification rate of crop residuals - [1] equ. 3:5.1.3a
        hr = hx * np.power(1. - (self.C_org / scc), 6.)
        hr = np.where(hr <= 0.18, hr, 0.18)
        self.humified_C_res = hr * self.decomposed_C_res
        self.C_org += self.humified_C_res

        ##### mineralization ###################################################
        # mineralization rate [1] - equ. 3:5.1.7 (tillage factor neglected)
        self.mineralization = self.min_opt.value * self.fnc.value
        self.mineralization *= np.power(self.C_org / scc, 0.5)
        # mineralized C corresponds to the subtracting term  in [1] equ. 3:5.1.2a
        self.mineralized_C_org = self.mineralization * self.C_org
        self.C_org -= self.mineralized_C_org
