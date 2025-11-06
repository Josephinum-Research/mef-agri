import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement
from ....farming import crops
from ....evaluation.stats_utils import DISTRIBUTIONS


class Roots(Model):
    r"""
    Model which computes root growth and biomass according to [R2]_ .

    kwargs :math:`\rightarrow` :class:`mef_agri.models.base.Model`
    """
    DEFAULT_PARAM_VALUES = {
        crops.maize.__name__: {
            'depth_max': {
                'value': 2.0,
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.15,
                    'lb': 1.3,
                    'ub': 2.1
                }
            }
        },
        crops.winter_wheat.__name__: {
            'depth_max': {
                'value': 2.0,
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.15,
                    'lb': 1.3,
                    'ub': 2.1
                }
            }
        }
    }
    INITIAL_STATE_VALUES = {
        'biomass': {
            'value': 0.0,  # [t/ha]
            'distr': {
                'distr_id': DISTRIBUTIONS.GAMMA_1D,
                'std': 0.01
            }
        }
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.STATE, U.t_ha)
    def biomass(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`c_{\textrm{R-bm},k}\ [\frac{t}{ha}]`

        :return: root biomass 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.m)
    def depth(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{R-d},k}\ [m]` - [R2]_ (equ. 14)

        :return: current rooting depth 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.t_haxday)
    def biomass_rate(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{R-}\Delta\textrm{bm},k}\ [\frac{t}{ha\cdot day}]` - [R2]_ (equ. 12)

        :return: daily increase of root biomass 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.m)
    def depth_max(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{R-dm},0}\ [m]` - [R2]_ (equ. 14, table 2)

        :return: max. attainable rooting depth of crop 
        :rtype: numpy.ndarray
        """

    @Model.is_required('biomass_rate', 'crop', U.t_haxday)
    def bmtr(self) -> Requirement:
        r"""
        RQ - ``'biomass_rate'`` from model with id ``'crop'``

        :math:`c_{\Delta\textrm{bm},k}\ [\frac{t}{ha\cdot day}]`

        :return: increase of total biomass
        :rtype: Requirement
        """

    @Model.is_required('biomass', 'crop', U.t_ha)
    def bmt(self) -> Requirement:
        r"""
        RQ - ``'biomass'`` from model with id ``'crop'``

        :math:`c_{\textrm{bm},k}\ [\frac{t}{ha}]`

        :return: total dry biomass (aboveground + roots) 
        :rtype: Requirement
        """

    @Model.is_required('heat_unit_index', 'crop.development', U.none)
    def hui(self) -> Requirement:
        r"""
        RQ - ``'heat_unit_index'`` from model with id ``'crop.development'``

        :math:`c_{\textrm{D-hui},k}\ [\ ]`

        :return: current heat unit index 
        :rtype: Requirement
        """

    @Model.is_required('rooting_depth_max', 'zone.soil', U.m)
    def rdmax(self) -> Requirement:
        r"""
        RQ - ``'rooting_depth_max'`` from model with id ``'zone.soil'``

        :math:`s_{\textrm{rdm},0}\ [m]`

        :return: max. possible rooting depth in soil 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Initialize :func:`biomass_rate` and :func:`depth` with zero vectors and 
        ensure, that initial root biomass :func:`biomass` does not exceed the 
        total crop biomass :func:`ssc_csm.models.crop.model_epic.Crop.biomass`.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        zv = np.zeros((self.model_tree.n_particles,))
        self.biomass_rate = zv.copy()
        self.depth = zv.copy()
        self.biomass = np.where(
            self.biomass <= self.bmt.value, self.biomass, self.bmt.value
        )

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`biomass_rate`
        * :func:`biomass`
        * :func:`depth`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        self.biomass_rate = self.bmtr.value * (0.4 - 0.2 * self.hui.value)
        self.biomass += self.biomass_rate
        self.depth = 2.5 * self.depth_max * self.hui.value
        self.depth = np.where(
            self.depth <= self.rdmax.value, self.depth, self.rdmax.value
        )

    @Model.is_condition
    def cond_biomass(self) -> None:
        self.biomass = np.where(self.biomass >= 0., self.biomass, 0.0)
