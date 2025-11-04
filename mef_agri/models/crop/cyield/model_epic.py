import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement
from ....farming import crops
from ....evaluation.stats_utils import DISTRIBUTIONS

class Yield(Model):
    r"""
    Simplified yield model which uses only equations 15, 16 and 17 from [R2]_ 
    to compute yield biomass. If water stress should be considered (i.e. 
    equations 62 and 63 should be incorporated in yield biomass computation), 
    see :class:`Yield_Stressed`.

    kwargs :math:`\rightarrow` see :class:`mef_agri.models.base.Model`
    """
    DEFAULT_PARAM_VALUES = {
        crops.winter_wheat.__name__: {
            'harvest_index_max': {
                'value': 0.42,
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.02,
                    'lb': 0.4,
                    'ub': 0.5
                }
            }
        }
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None

    @Model.is_quantity(Q.ROUT, U.frac)
    def harvest_index(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{Y-hi},k}\ [\ ]` - [R2]_ (equ. 16)

        [R2]_ is not totally clear about this quantity, i.e. the time-variable 
        harvest index. It is computed with equ. 16 but in equ. 15, the harvest 
        index at maturity :func:`harvest_index_mat` is used to compute biomass. 
        In this case, the time-variable harvest index would not be used at all.
        Thus, here it will be used in equ. 15 to compute biomass.

        :return: harvest index 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.t_ha)
    def biomass(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{Y-bm},k}\ [\frac{t}{ha}]` - [R2]_ (equ. 15)

        :return: yield biomass 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def heat_unit_factor(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{Y-huf},k}\ [\ ]` - [R2]_ (equ. 17)

        :return: heat unit factor for yield/harvest index 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.frac)
    def harvest_index_max(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`y_{\textrm{hi-max},0}\ [\ ]` - [1] (equ. 16, table 2)

        :return: max. attainable harvest index of crop 
        :rtype: numpy.ndarray
        """

    @Model.is_required('biomass_aboveground', 'crop', U.t_ha)
    def bmag(self) -> Requirement:
        r"""
        RQ - ``'biomass_aboveground'`` from model with id ``'crop'``

        :math:`c_{\textrm{bma},k}\ [\frac{t}{ha}]`

        :return: aboveground biomass
        :rtype: Requirement
        """

    @Model.is_required('heat_unit_index', 'crop.development', U.frac)
    def hui(self) -> Requirement:
        r"""
        RQ - ``'heat_unit_index'`` from model with id ``'crop.development'``

        :math:`c_{\textrm{D-hui},k}\ [\ ]`

        :return: current heat unit index
        :rtype: Requirement
        """

    @Model.is_required('winter_dormancy', 'crop.development', U.boolean)
    def wdorm(self) -> Requirement:
        r"""
        RQ - ``'winter_dormancy'`` from model with id ``'crop.development'``

        :math:`c_{\textrm{D-wd},k}\ [\ ]` (boolean value)

        :return: indicator for winter dormancy period
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Initialization of random outputs with zero vectors

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))
        self.heat_unit_factor = self._zv.copy()
        self.harvest_index = self._zv.copy()
        self.biomass = self._zv.copy()

    def update(self, epoch):
        """
        The following computations are performed

        * if ``self.wdorm.value == True``

            * :func:`heat_unit_factor` set to zero vector
            * :func:`harvest_index` and :func:`biomass` remain unchanged

        * else

            * :func:`heat_unit_factor`
            * :func:`harvest_index`
            * :func:`biomass`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        if self.wdorm.value:
            self.heat_unit_factor = self._zv.copy()
            return
        
        self.heat_unit_factor = self.hui.value / (
            self.hui.value + np.exp(6.5 - 10.0 * self.hui.value)
        )
        self.harvest_index = self.harvest_index_max * self.heat_unit_factor
        self.biomass = self.harvest_index * self.bmag.value


class Yield_Stressed(Yield):
    """
    Yield model wich considers water stress in the harvest-index-computation - 
    i.e. incorporating equations 62 and 63 from [R2]_ .
    """
    DEFAULT_PARAM_VALUES = {
        crops.winter_wheat.__name__: {
            'harvest_index_max': {
                'value': 0.42,
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.02,
                    'lb': 0.4,
                    'ub': 0.5
                }
            },
            'water_stress_influence': {
                'value': 0.01,
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.002,
                    'lb': 0.0,
                    'ub': 1.0
                }
            }
        }
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @Model.is_quantity(Q.ROUT, U.none)
    def heat_unit_factor(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{Y-huf},k}\ [\ ]` - [R2]_ (equ. 63)

        :return: heat unit factor for yield/harvest index 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def harvest_index(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{Y-hi},k}\ [\ ]` - [R2]_ (equ. 62)

        :return: harvest index 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def water_stress_influence(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{Y-ws},0}\ [\ ]` - [R2]_ (equ. 62, table 2)

        :return: crop specific water stress influence factor on yield
        :rtype: numpy.ndarray
        """

    @Model.is_required('water_stress', 'crop.stress', U.frac)
    def wstrs(self) -> Requirement:
        r"""
        RQ - ``'water_stress'`` from model with id ``'crop.stress'``

        :math:`c_{\textrm{W-str},k}\ [\ ]`

        :return: water stress indicator
        :rtype: Requirement
        """

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`heat_unit_factor`
        * :func:`harvest_index`
        * :func:`Yield.biomass`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        
        # heat unit factor for yield
        hufy = np.sin((np.pi / 2.0) * ((self.hui.value - 0.3) / 0.3))
        chufy = (self.hui.value >= 0.3) & (self.hui.value <= 0.9)
        self.heat_unit_factor = np.where(chufy, hufy, 0.0)

        # change of harvest index
        dhi = -self.harvest_index_mat * (
            1.0 - (1.0 / (
                1.0 + self.water_stress_influence * self.heat_unit_factor * (
                    0.9 - self.wstrs.value
                )
            ))
        )

        # harvest index and yield biomass
        self.harvest_index += dhi
        self.biomass = self.harvest_index * self.bmag.value
