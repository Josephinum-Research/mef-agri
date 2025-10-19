import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement


class Leaves(Model):
    """
    Model which computes the leaf area index according to [R2]_ (no biomass, 
    ...).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._laild:np.ndarray = None  # value of lai when leaf decline starts
        self._hufl_prv:np.ndarray = None  # daily change of leaves heat-unit-factor

    @Model.is_quantity(Q.STATE, U.none)
    def lai(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`c_{\textrm{L-lai},k}\ [\ ]` - [R2]_ (equ. 7)

        :return: current leaf area index 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def lai_rate_pot(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{L-lrp},k}\ [\ ]` - [R2]_ (equ. 8)

        :return: potential daily increase of lai 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def lai_max(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{L-lmx},0}\ [\ ]` - [R2]_ (equ. 8, table 2)

        :return: max. possible value of lai 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def hui_leaf_decline(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{L-hld},0}\ [\ ]` - [R2]_ (equ. 10, table 2)

        :return: heat unit index at which leaf decline starts 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def lai_regr_coeff(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{L-lc},0}\ [\ ]` - [R2]_ (equ. 10, table 2)

        :return: regression coefficient of lai computation at leaf decline stage 
        :rtype: numpy.ndarray
        """

    @Model.is_required('heat_unit_index', 'crop.development', U.none)
    def hui(self) -> Requirement:
        r"""
        RQ - ``'heat_unit_index'`` from model with id ``'crop.development'``

        :math:`c_{\textrm{D-hui},k}\ [\ ]`

        :return: current heat unit index 
        :rtype: Requirement
        """

    @Model.is_required('heat_unit_factor_leaves', 'crop.development', U.none)
    def hufl(self) -> Requirement:
        r"""
        RQ - ``'heat_unit_factor_leaves'`` from model with id ``'crop.development'``

        :math:`c_{\textrm{L-huf},k}\ [\ ]`

        :return: heat unit factor for leaves at current day
        :rtype: Requirement
        """

    @Model.is_required('growth_constraint', 'crop.stress', U.frac)
    def gc(self) -> Requirement:
        r"""
        RQ - ``'growth_constraint'`` from model with id ``'crop.stress'``

        :math:`c_{\textrm{gc},k}\ [\ ]`

        :return: crop biomass growth constraint
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Initialize class-intern variables and :func:`lai_rate_pot` with a zero 
        vector.

        :param epoch: intialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        zv = np.zeros((self.model_tree.n_particles,))

        self._laild = zv.copy()
        self._laild[:] = np.nan
        self.lai_rate_pot = zv.copy()

    def update(self, epoch):
        r"""
        The following computations are performed

        * daily change of :func:`hufl`
        * :func:`lai_rate_pot`
        * compute boolean/condition array ``cld = self.hui.value > self.hui_leaf_decline``
        * save value of :func:`lai` at the day when leaf decline starts (i.e. when a realization of :func:`hui` exceeds a realization of :func:`hui_leaf_decline`, or where ``cld`` contains ``True``)
        * two lai values are computed

            * one before leaf decline ``lai1`` :math:`\rightarrow` [R2]_ (equ. 7)
            * one after leaf decline has started ``lai2`` :math:`\rightarrow` [R2]_ (equ. 10)
            * depending on ``cld``, ``lai1`` or ``lai2`` is set for :func:`lai`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        # potential increase of lai
        self.cond_hufl()
        dhufl = self.hufl.value - self._hufl_prv  # daily change of leaves heat-unit factor [R1]_ equ. 8
        self.lai_rate_pot = dhufl * self.lai_max * (
            1. - np.exp(5. * (self.lai - self.lai_max))
        ) * np.sqrt(self.gc.value)
        self._hufl_prv = self.hufl.value.copy()
        # determine value of lai at the day when hui exceeds hui_leaf_decline
        cld = self.hui.value > self.hui_leaf_decline
        self._laild = np.where(
            cld & np.isnan(self._laild), self.lai, self._laild
        )
        # two lai computations, one before leaf decline and one afterwards
        lai1 = self.lai + self.lai_rate_pot
        aux = (1. - self.hui.value) / (1. - self.hui_leaf_decline)
        aux = np.where(aux >= 0.0, aux, 0.0)
        lai2 = self._laild * np.power(aux, self.lai_regr_coeff)
        # computation of lai
        self.lai = np.where(cld, lai2, lai1)

    @Model.is_condition
    def cond_hufl(self) -> None:
        """
        Ensures that current value of :func:`hufl` is not lower than the value 
        of the previous day (otherwise negative values of the difference lead to 
        problems)
        """
        if self._hufl_prv is None:
            self._hufl_prv = self.hufl.value.copy()
        self.hufl.value = np.where(
            self.hufl.value >= self._hufl_prv, self.hufl.value, self._hufl_prv
        )

    @Model.is_condition
    def cond_lai(self) -> None:
        r"""
        Ensures that :func:`lai` stays in the range 
        :math:`c_{\textrm{L-lai},k} \in [0.0, c_{\textrm{L-lmx},0}]`
        """
        self.lai = np.where(self.lai >= 0.0, self.lai, 0.0)
        self.lai = np.where(self.lai <= self.lai_max, self.lai, self.lai_max)
