from copy import deepcopy
import numpy as np

from ..base import Model, Quantities as Q
from ..utils import Units as U
from ..requ import Requirement
from .demand.model_epic import Demand as DemandEPIC
from .uptake.model_epic import Uptake as UptakeEPIC
from .stress.model_epic import Stress as StressEPIC
from .development.model_epic import (
    Development as DevelEPIC, Development_Dormancy as DevelDormEPIC
)
from .development.model_ceres import Development as DevelBBCH
from .leaves.model_epic import Leaves as LeavesEPIC
from .roots.model_epic import Roots as RootsEPIC
from .cyield.model_epic import (
    Yield as YieldEPIC, Yield_Stressed as YieldStrEPIC
)
from ...farming import crops
from ...evaluation.stats_utils import DISTRIBUTIONS

"""
[1]
Zhou, X. and Liu, H. and Li, L.
Estimation of water interception of winter wheat canopy under sprinkler irrigation using UAV image data
Water, Vol. 16, No. 24
2024
https://doi.org/10.3390/w16243609

"""

class Crop_Simple(Model):
    """
    This class acts as root-model containing several child models (i.e. 
    ``model_name='crop'`` is passed to the parent class 
    :class:`mef_agri.models.base.Model`).

    It uses 

    * :class:`mef_agri.models.crop.development.model_epic.Development` as development-model
    * :class:`mef_agri.models.crop.cyield.model_epic.Yield` as crop-yield-model

    """
    DEFAULT_PARAM_VALUES = {
        crops.maize.__name__: {
            'water_storage_max': {
                'value': 2.4,  # [ mm ] - deduced from winter wheat value
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.2
                }
            },
            'energy_conversion': {
                'value': 40.0,  # [( kg x m2 ) / (ha x MJ x day)]
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 1.0
                }
            },
            'critical_aeration_factor': {
                'value': 0.85,  # []
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.002,
                    'lb': 0.0,
                    'ub': 1.0
                }
            },
            'height_max': {
                'value': 2.5,
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.1,
                    'lb': 1.15,
                    'ub': 1.5
                }
            },
            'frost_coeff1': {
                'value': 5.01,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.1
                }
            },
            'frost_coeff2': {
                'value': 15.05,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.3
                }
            }
        },
        crops.winter_wheat.__name__: {
            'water_storage_max': {
                'value': 1.2,  # [ mm ] - approx. max. values from [1] figures 4 and 5
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.1
                }
            },
            'energy_conversion': {
                'value': 35.0,  # [( kg x m2 ) / (ha x MJ x day)]
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 1.0
                }
            },
            'critical_aeration_factor': {
                'value': 0.85,  # []
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.002,
                    'lb': 0.0,
                    'ub': 1.0
                }
            },
            'height_max': {
                'value': 1.2,
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.1,
                    'lb': 1.15,
                    'ub': 1.5
                }
            },
            'frost_coeff1': {
                'value': 5.01,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.1
                }
            },
            'frost_coeff2': {
                'value': 15.05,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.3
                }
            }
        }
    }
    INITIAL_STATE_VALUES = {
        'biomass': {
            'value': 0.0,  # [t/ha]
            'distr': {
                'distr_id': DISTRIBUTIONS.GAMMA_1D,
                'std': 0.05
            }
        }
    }

    def __init__(self):
        super().__init__(model_name='crop')
        self._dlprv = None  # daylength difference
        self._zv:np.ndarray = None

    @Model.is_quantity(Q.STATE, U.t_ha)
    def biomass(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`c_{\textrm{bm},k} = c_{\textrm{bm},k-1} + c_{\Delta\textrm{bm},k}\ [\frac{t}{ha}]`

        Monteith (1977) approach is used for biomass increase. Thus, all biomass 
        quantities represent dry matter (see [R6]_ first sentence of Introduction)

        :return: total dry biomass (aboveground + roots)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.t_haxday)
    def biomass_rate(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\Delta\textrm{bm},k}\ [\frac{t}{ha\cdot day}]` - [R2]_ (equ. 4, 44)

        :return: daily increase of total biomass  - [1] equ. 4 and 44
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.t_ha)
    def biomass_aboveground(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{bma},k}\ [\frac{t}{ha}] = c_{\textrm{bm},k} - c_{\textrm{R-bm},k}\ [\frac{t}{ha}]`

        :return: above-ground biomass
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.t_ha)
    def biomass_aboveground_reduction(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\downarrow\Delta\textrm{bma},k}\ [\frac{t}{ha}]`

        :return: reduction of above-ground biomass (only occurs in winter dormancy periods)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.m)
    def height(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{h},k}\ [m]` - [R2]_ (equ. 11)

        :return: crop height 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.MJ_m2xday)
    def radiation_intercepted(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{irad},k}\ [\frac{MJ}{m^2\cdot day}]` - [R2]_ (equ. 3)

        :return: intercepted radiation 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def rf_daylength(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{rfdl},k}\ [\ ]` - [R2]_ (equ. 64)

        :return: reduction factor on above-ground biomass due to daylength
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def rf_frost(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{rffr},k}\ [\ ]` - [R2]_ (equ. 65)

        :return: reduction factor on above-ground biomass due to frost damage
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.mm)
    def water_storage_max(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{W-csm},0}\ [mm]` - [R1]_ (equ. 2:2.1.1)

        :return: amount of water that can be stored in the canopy when leaves are fully developed 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.kgxm2_haxMJxday)
    def energy_conversion(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{e2bm},0}\ [\frac{kg\cdot m^2}{MJ\cdot hat\cdot day}]` - [R2]_ (table 2)

        :return: factor to convert energy to biomass 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def critical_aeration_factor(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{caf},0}\ [\ ]` - [R2]_ (table 2)

        :return: critical aeration factor 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.m)
    def height_max(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{hmax},0}]\ [\ ]` - [R2]_ (table 2)

        :return: max. crop height 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def frost_coeff1(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{frc1},0}\ [\ ]` - [R2]_ (equ. 65, table 2)

        :return: regression coefficient for frost influence on biomass reduction
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.n_degC)
    def frost_coeff2(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{frc2},0}\ [\frac{1}{^\circ C}]` - [R2]_ (equ. 65, table 2)

        :return: regression coefficient for frost influence on biomass reduction
        :rtype: numpy.ndarray
        """

    @Model.is_required('temperature_min', 'zone.atmosphere.weather', U.degC)
    def tmin(self) -> Requirement:
        r"""
        RQ - ``'temperature_min'`` from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{tmin},k}\ [^\circ C]`

        :return: min. temperature at current day
        :rtype: Requirement
        """

    @Model.is_required('daylength', 'zone.atmosphere.daylength', U.h)
    def dl(self) -> Requirement:
        r"""
        RQ - ``'daylength'`` from model with id ``'zone.atmosphere.daylength'``

        :math:`a_{\textrm{dl},k}\ [h]`

        :return: current daylength
        :rtype: Requirement
        """

    @Model.is_required('daylength_min', 'zone.atmosphere.daylength', U.h)
    def dlmin(self) -> Requirement:
        r"""
        RQ - ``'daylength_min'`` from model with id ``'zone.atmosphere.daylength'``

        :math:`a_{\textrm{dlmin},k}\ [h]`

        :return: min. daylength in a year at current site
        :rtype: Requirement
        """

    @Model.is_required('growth_constraint', 'crop.stress', U.frac)
    def gc(self) -> Requirement:
        r"""
        RQ - ``'growth_constraint'`` from model with id ``'crop.stress'``

        :math:`c_{\textrm{gc},k}\ [\ ]`

        :return: biomass growth constraint (min. value of several stress factors) 
        :rtype: Requirement
        """

    @Model.is_required('radiation_sum', 'zone.atmosphere.weather', U.MJ_m2xday)
    def rad(self) -> np.ndarray:
        r"""
        RQ - ``'radiation_sum'`` from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{rad},k}\ [\frac{MJ}{m^2\cdot day}]`

        :return: daily radiation sum 
        :rtype: numpy.ndarray
        """

    @Model.is_required('lai', 'crop.leaves', U.none)
    def lai(self) -> Requirement:
        r"""
        RQ - ``'lai'`` from model with id ``'crop.leaves'``

        :math:`c_{\textrm{L-lai},k}\ [\ ]`

        :return: current leaf area index 
        :rtype: Requirement
        """

    @Model.is_required('heat_unit_index', 'crop.development', U.frac)
    def hui(self) -> Requirement:
        r"""
        RQ - quantity ``'heat_unit_index'`` from model with id ``'crop.development'``

        :math:`c_{\textrm{hui},k}\ [\ ]`

        :return: current heat unit index
        :rtype: Requirement
        """

    @Model.is_required('heat_unit_factor_leaves', 'crop.development', U.none)
    def hufl(self) -> Requirement:
        r"""
        RQ - ``'heat_unit_factor_leaves'`` from model with id ``'crop.development'``

        :math:`c_{\textrm{L-huf},k}\ [\ ]`

        :return: heat unit factor for leaf growth
        :rtype: Requirement
        """

    @Model.is_required('winter_dormancy', 'crop.development', U.boolean)
    def wdorm(self) -> Requirement:
        r"""
        RQ - ``'winter_dormancy'`` from model with id ``'crop.development'``

        :math:`c_{\textrm{wd},k}\ [\ ]` (boolean value)

        :return: flag if winter dormancy is active
        :rtype: Requirement
        """

    @Model.is_required('biomass', 'crop.roots', U.t_ha)
    def rbm(self) -> Requirement:
        r"""
        RQ - ``'biomass'`` from model with id ``'crop.roots'``

        :math:`c_{\textrm{R-bm},k}\ [\frac{t}{ha}]`

        :return: root biomass
        :rtype: Requirement
        """

    ##### CHILD MODELS #########################################################
    @Model.is_child_model(DemandEPIC)
    def demand(self) -> DemandEPIC:
        """
        Child Model

        :return: model to determine crop demands
        :rtype: mef_agri.models.crop.demand.model_epic.Demand
        """

    @Model.is_child_model(UptakeEPIC)
    def uptake(self) -> UptakeEPIC:
        """
        Child Model

        :return: model to determine/map the supplied quantities from the soil (and atmosphere)
        :rtype: mef_agri.models.crop.uptake.model_epic.Uptake
        """

    @Model.is_child_model(StressEPIC)
    def stress(self) -> StressEPIC:
        """
        Child model

        :return: model which determines stress factors
        :rtype: mef_agri.models.crop.stress.model_epic.Stress
        """

    @Model.is_child_model(DevelEPIC)
    def development(self) -> DevelEPIC:
        """
        Child Model

        :return: model which computes the crop development stage
        :rtype: mef_agri.models.crop.development.model_epic.Development
        """

    @Model.is_child_model(DevelBBCH)
    def development_stages(self) -> DevelBBCH:
        """
        Child Model

        NOTE: currently this model is just an "addon", meaning, that it does not 
        influence any other models in the model-tree (still it requires 
        quantities from the zone.atmosphere models).

        :return: model which computes the BBCH stages
        :rtype: mef_agri.models.crop.development.model_ceres.DevelBBCH
        """

    @Model.is_child_model(LeavesEPIC)
    def leaves(self) -> LeavesEPIC:
        """
        Child Model

        :return: model which computes growth of leaves
        :rtype: mef_agri.models.crop.leaves.model_epic.Leaves
        """

    @Model.is_child_model(RootsEPIC)
    def roots(self) -> RootsEPIC:
        """
        Child Model

        :return: model which computes root growth
        :rtype: mef_agri.models.crop.roots.model_epic.Roots
        """

    @Model.is_child_model(YieldEPIC)
    def cyield(self) -> YieldEPIC:
        """
        Child Model

        :return: model to compute yield increase
        :rtype: mef_agri.models.crop.cyield.model_epic.Yield
        """

    def initialize(self, epoch):
        """
        Initialization of random outputs with zero vectors and child models

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))

        self.height = self._zv.copy()
        self.biomass_aboveground = self._zv.copy()
        self.biomass_rate = self._zv.copy()
        self.radiation_intercepted = self._zv.copy()
        self.biomass_aboveground_reduction = self._zv.copy()

        self.uptake.initialize(epoch)
        self.development.initialize(epoch)
        self.development_stages.initialize(epoch)
        self.stress.initialize(epoch)
        self.leaves.initialize(epoch)
        self.roots.initialize(epoch)
        self.cyield.initialize(epoch)
        self.demand.initialize(epoch)

    def update(self, epoch):
        """
        The following computations are performed

        * update :func:`uptake`
        * update :func:`development`
        * :func:`radiation_intercepted`
        * if ``self.wdorm.value == True``

            * set :func:`biomass_rate` to zero vector
            * :func:`rf_daylength`
            * :func:`rf_frost`
            * :func:`biomass_aboveground_reduction`

        * else

            * difference in daylength
            * :func:`biomass_rate`
            * set :func:`rf_daylength`, :func:`rf_frost` and :func:`biomass_aboveground_reduction` to zero vectors

        * :func:`biomass`
        * :func:`height`
        * update :func:`stress`
        * update :func:`leaves`
        * update :func:`roots`
        * update :func:`cyield`
        * update :func:`demand`
        * :func:`biomass_aboveground`
        * in the case of ``self.wdorm.value == True`` > reduce :func:`biomass` and :func:`biomass_aboveground` by :func:`biomass_aboveground_reduction`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        if self._dlprv is None:
            self._dlprv = deepcopy(self.dl.value)

        # models which are updated in the beginning before crop growth is computed
        self.uptake.update(epoch)
        self.development.update(epoch)
        self.development_stages.update(epoch)

        # compute biomass increase
        self.radiation_intercepted = 0.5 * self.rad.value * (
            1. - np.exp(-0.65 * self.lai.value))
        if self.wdorm.value:
            # winter dormancy periods
            self.biomass_rate = self._zv.copy()
            self.rf_daylength = 0.35 * (
                1.0 - (self.dl.value / (self.dlmin.value + 1.0))
            )
            self.rf_frost = -self.tmin.value / (
                -self.tmin.value - np.exp(
                    self.frost_coeff1 + self.frost_coeff2 * self.tmin.value
                )
            )
            rfs = np.vstack((
                np.atleast_2d(self.rf_daylength),
                np.atleast_2d(self.rf_frost)
            ))
            self.biomass_aboveground_reduction = \
                0.5 * self.biomass_aboveground * (
                    1.0 - self.hui.value) * np.max(rfs, axis=1)
        else:
            # standard biomass computations
            ddl = self.dl.value - self._dlprv
            self.biomass_rate = \
                0.001 * self.energy_conversion * np.power(1. + ddl, 3.) * \
                self.radiation_intercepted * self.gc.value
            # setting winter dormancy quantities to zero
            self.rf_daylength = self._zv.copy()
            self.rf_frost = self._zv.copy()
            self.biomass_aboveground_reduction = self._zv.copy()
        
        self._dlprv = deepcopy(self.dl.value)
        self.biomass += self.biomass_rate
        self.height = self.height_max * np.sqrt(self.hufl.value)
        
        # update child models
        self.stress.update(epoch)
        self.leaves.update(epoch)
        self.roots.update(epoch)
        self.cond_root_biomass()
        self.cyield.update(epoch)
        self.demand.update(epoch)

        # computation of aboveground biomass
        self.biomass_aboveground = self.biomass - self.rbm.value
        if self.wdorm.value:
            # influence of winter dormancy on biomass
            self.biomass_aboveground -= self.biomass_aboveground_reduction
            self.biomass -= self.biomass_aboveground_reduction

    @Model.is_condition
    def cond_biomass(self) -> None:
        self.biomass = np.where(self.biomass >= 0.0, self.biomass, 0.)

    @Model.is_condition
    def cond_root_biomass(self) -> None:
        self.rbm.value = np.where(
            self.rbm.value <= self.biomass, self.rbm.value, self.biomass
        )


class Crop_Extended(Crop_Simple):
    """
    This class acts as root-model containing several child models (i.e. 
    ``model_name='crop'`` is passed to the parent class 
    :class:`ssc_csm.models.base.Model`).

    It uses 

    * :class:`ssc_csm.models.crop.development.model_epic.Development_Dormancy` as development-model
    * :class:`ssc_csm.models.crop.cyield.model_epic.Yield_Stressed` as crop-yield-model

    """
    @Model.is_child_model(DevelDormEPIC)
    def development(self) -> DevelDormEPIC:
        """
        Child Model

        :return: model to compute crop development
        :rtype: ssc_csm.models.crop.development.model_epic.Development_Dormancy
        """

    @Model.is_child_model(YieldStrEPIC)
    def cyield(self) -> YieldStrEPIC:
        """
        Child Model

        :return: model to compute yield increase
        :rtype: ssc_csm.models.crop.cyield.model_epic.Yield_Stressed
        """