import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement
from ....evaluation.stats_utils import DISTRIBUTIONS


class Uptake(Model):
    INITIAL_STATE_VALUES = {
        'nitrogen_sum': {
            'value': 0.0,
            'distr': {
                'distr_id': DISTRIBUTIONS.GAMMA_1D,
                'std': 0.001
            }
        }
    }

    @Model.is_quantity(Q.STATE, U.kg_ha)
    def nitrogen_sum(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{N-ups},k}\ [\frac{kg}{ha}]`

        :return: sum of nitrogen uptake of current crop/vegetation period
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def water(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{W-up},k}\ [\frac{mm}{day}]`

        :return: uptake of water at current day (equal to supplied water by soil :func:`wsup`)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_ha)
    def nitrogen(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{N-up},k}\ [\frac{kg}{ha\cdot day}]`

        :return: uptake of nitrogen at current day (equal to supplied nitrogen by soil :func:`nsup`)
        :rtype: numpy.ndarray
        """

    @Model.is_required('water', 'zone.soil.supply', U.mm_day)
    def wsup(self) -> Requirement:
        r"""
        RQ - ``'water'`` from model with id ``'zone.soil.supply'``

        :math:`s_{\textrm{W-avl},k}\ [\frac{mm}{day}]`

        :return: water supplied by soil
        :rtype: Requirement
        """

    @Model.is_required('nitrogen', 'zone.soil.supply', U.kg_haxday)
    def nsup(self) -> Requirement:
        r"""
        RQ - ``'nitrogen'`` from model with id ``'zone.soil.supply'``

        :math:`s_{\textrm{N-avl},k}\ [\frac{kg}{ha\cdot day}]`
        
        :return: amount of nitrogen which is available for the crop
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

    def initialize(self, epoch):
        """
        Initialization of :func:`nitrogen_sum` with zero vector.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self.nitrogen_sum = np.zeros((self.model_tree.n_particles,))

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`water`
        * :func:`nitrogen`
        * :func:`nitrogen_sum`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        aux = np.isin(True, np.isnan(self.wsup.value))
        if aux:
            self.water = np.zeros((self.model_tree.n_particles,))
        else:
            self.water = self.wsup.value.copy()

        self.nitrogen = self.nsup.value.copy()
        self.nitrogen_sum += self.nitrogen

    @Model.is_condition
    def cond_nsum(self) -> None:
        self.nitrogen_sum = np.where(
            self.nitrogen_sum >= 0.0, self.nitrogen_sum, 0.0
        )
