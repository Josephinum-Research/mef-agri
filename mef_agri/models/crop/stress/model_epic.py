import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement


class Stress(Model):
    """
    Stress model which considers water-, temperature-, aeration and nitrogen-stress 
    according to [R2]_ .
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ov:np.ndarray = None

    @Model.is_quantity(Q.ROUT, U.frac)
    def water_stress(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{W-str},k}\ [\ ]` - [R2]_ (equ. 45)

        :return: water stress factor
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def aeration_stress(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{astr},k}\ [\ ]` - [R2]_ (equ. 49, 50)

        :return: aeration stress factor 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def temperature_stress(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{tstr},k}\ [\ ]` - [R2]_ (equ. 46)

        :return: temperature stress factor
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def nitrogen_stress(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{N-str},k}\ [\ ]` - [R2]_ (equ. 47)

        :return: nitrogen stress factor
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def growth_constraint(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{gc},k}\ [\ ]` - [R2]_ (last paragraph in "Growth Constraints" > "Biomass")
        
        :return: (biomass) growth constraint (min. value of several stress factors) 
        :rtype: numpy.ndarray
        """

    @Model.is_required('water', 'crop.demand', U.mm_day)
    def wdem(self) -> Requirement:
        r"""
        RQ - ``'water'`` from model with id ``'crop.demand'``

        :math:`c_{\textrm{W-dem},k}\ [\frac{mm}{day}]`

        :return: water demand of crop
        :rtype: Requirement
        """

    @Model.is_required('water', 'crop.uptake', U.mm_day)
    def wupt(self) -> Requirement:
        r"""
        RQ - ``'water'`` from model with id ``'crop.uptake'``

        :math:`c_{\textrm{W-up},k}\ [\frac{mm}{day}]`

        :return: uptake of water at current day
        :rtype: Requirement
        """

    @Model.is_required('moisture_1m', 'zone.soil.supply', U.frac)
    def m1m(self) -> Requirement:
        r"""
        RQ - ``moisture_1m`` from model with id ``'zone.soil.supply'``

        :math:`s_{\textrm{W-m1m},k}\ [\ ]`

        :return: moisture of first meter of soil
        :rtype: Requirement
        """

    @Model.is_required('porosity', 'zone.soil', U.frac)
    def por(self) -> Requirement:
        r"""
        RQ - ``'porosity'`` from model with id ``'zone.soil'``

        :math:`s_{\textrm{por},0}\ [\ ]`

        :return: soil porosity (fraction of total volume)
        :rtype: Requirement
        """

    @Model.is_required('critical_aeration_factor', 'crop', U.none)
    def caf(self) -> Requirement:
        r"""
        RQ - ``'critical_aeration_factor'`` from model with id ``'crop'``

        :math:`c_{\textrm{caf},0}\ [\ ]`

        :return: critical aeration factor of crop
        :rtype: Requirement
        """

    @Model.is_required('temperature', 'zone.soil.surface.temperature', U.degC)
    def stemp(self) -> Requirement:
        r"""
        RQ - ``'temperature'`` from model with id ``'zone.soil.surface.temperature'``

        :math:`s_{\textrm{T-t},s,k}\ [^\circ C]`

        :return: soil surface temperature
        :rtype: Requirement
        """

    @Model.is_required('temperature_base', 'crop.development', U.degC)
    def tb(self) -> Requirement:
        r"""
        RQ - ``temperature_base`` from model with id ``'crop.development'``

        :math:`c_{\textrm{D-tb},0}\ [^\circ C]`

        :return: crop specific base temperature
        :rtype: Requirement
        """

    @Model.is_required('temperature_opt', 'crop.development', U.degC)
    def to(self) -> Requirement:
        r"""
        RQ - ``temperature_opt`` from model with id ``'crop.development'``

        :math:`c_{\textrm{D-to},0}\ [^\circ C]`

        :return: crop specific optimum temperature
        :rtype: Requirement
        """

    @Model.is_required('n_amount_opt', 'crop.demand', U.kg_ha)
    def nao(self) -> Requirement:
        r"""
        RQ - ``'n_amount_opt'`` from model with id ``'crop.demand'``

        :math:`c_{\textrm{N-ao},k}\ [\frac{kg}{ha}]`

        :return: optimal N-amount in crop biomass
        :rtype: Requirement
        """

    @Model.is_required('nitrogen_sum', 'crop.uptake', U.kg_ha)
    def nsum(self) -> Requirement:
        r"""
        RQ - ``'nitrogen_sum'`` from model with id ``'crop.uptake'``

        :math:`c_{\textrm{N-ups},k}\ [\frac{kg}{ha}]`

        :return: sum of nitrogen uptake of current crop/vegetation period
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Initialization of stress factors with one vectors (i.e. no stress)

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._ov = np.ones((self.model_tree.n_particles))
        self.aeration_stress = self._ov.copy()
        self.temperature_stress = self._ov.copy()
        self.water_stress = self._ov.copy()
        self.growth_constraint = self._ov.copy()

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`water_stress`
        * :func:`aeration_stress`
        * :func:`temperature_stress`
        * :func:`nitrogen_stress`
        * :func:`growth_constraint`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        if np.min(self.wdem.value) < 1e-8:
            self.water_stress = self._ov.copy()
        else:
            self.water_stress = self.wupt.value / self.wdem.value
        
        wsat = (self.m1m.value / self.por.value - self.caf.value) * 1e3
        wsat = np.where(wsat >= 0.0, wsat, 0.0)
        self.aeration_stress = 1.0 - (wsat / (wsat + np.exp(-1.291 - 56.1 * wsat)))
        tstr = np.sin(0.5 * np.pi * (
            (self.stemp.value - self.tb.value) / (self.to.value - self.tb.value)
        ))
        c1, c2, c3 = tstr < 0.0, (tstr >= 0.0) & (tstr <= 1.0), tstr > 1.0
        self.temperature_stress = np.select([c1, c2, c3], [0.0, tstr, 1.0])

        if np.isin(True, np.isnan(self.nao.value)):
            self.nitrogen_stress = self._ov.copy()
        else:
            ratio1 = self.nsum.value / self.nao.value
            # TODO the following line is a quick-and-dirty approach to suppress 
            # TODO warnings > need to be tested if it is better to check  
            # TODO the nitrogen uptake and adjust the uptake rates accordingly
            ratio1 = np.where(ratio1 <= 1.0, ratio1, 1.0)
            nscl = 2.0 * (1.0 - ratio1)
            add1 = np.exp(3.39 - 10.93 * nscl)
            denom = nscl + add1
            ratio2 = nscl / denom
            self.nitrogen_stress = 1.0 - ratio2

        strs = np.vstack((
            np.atleast_2d(self.water_stress),
            np.atleast_2d(self.aeration_stress),
            np.atleast_2d(self.temperature_stress),
            np.atleast_2d(self.nitrogen_stress)
        )).T
        self.growth_constraint = np.min(strs, axis=1)
