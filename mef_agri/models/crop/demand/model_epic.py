import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement
from ....farming import crops
from ....evaluation.stats_utils import DISTRIBUTIONS


class Demand(Model):
    DEFAULT_PARAM_VALUES = {
        crops.maize.__name__: {
            'ncoeff1': {
                'value': 0.044,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.002
                }
            },
            'ncoeff2': {
                'value': 0.0164,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.002
                }
            },
            'ncoeff3': {
                'value': 0.0128,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.0005
                }
            }
        },
        crops.winter_wheat.__name__: {
            'ncoeff1': {
                'value': 0.06,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.002
                }
            },
            'ncoeff2': {
                'value': 0.0231,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.002
                }
            },
            'ncoeff3': {
                'value': 0.0134,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.0005
                }
            }
        }
    }

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def water(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{W-dem},k}\ [\ ]`

        :return: water demand of the crop > is set to the potential transpiration from the soil evaporation model
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_haxday)
    def nitrogen(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{N-dem},k}\ [\frac{kg}{ha}]` - [R2]_ (equ. 25)

        :return: nitrogen demand of the crop
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_t)
    def n_concentration_opt(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{N-co},k}\ [\frac{kg}{t}]` - [R2]_ (equ, 25, 26)

        :return: optimal N-concentration in crop biomass
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_ha)
    def n_amount_opt(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{N-ao},k} = c_{\textrm{N-co},k}\cdot c_{\textrm{bm},k} \ [\frac{kg}{ha}]`

        :return: optimal N-amount in crop biomass
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def ncoeff1(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{N-c1},0}\ [\frac{kg}{t}]` - [R2]_ (equ 26, table 2)

        :return: coefficient to determine optimal N-concentration in crop biomass
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def ncoeff2(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{N-c2},0}\ [\frac{kg}{t}]` - [R2]_ (equ 26, table 2)

        :return: coefficient to determine optimal N-concentration in crop biomass
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def ncoeff3(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{N-c3},0}\ [\ ]` - [R2]_ (equ 26, table 2)

        :return: coefficient to determine optimal N-concentration in crop biomass
        :rtype: numpy.ndarray
        """

    @Model.is_required('transpiration_pot', 'zone.soil.surface.evapotranspiration', U.mm_day)
    def transpiration_pot(self) -> Requirement:
        r"""
        RQ - ``'transpiration_pot'`` from model with id ``'zone.soil.surface.evapotranspiration'``

        :math:`s_{\textrm{W-tp},s,k}\ [\frac{mm}{day}]`

        :return: potential transpiration
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

    @Model.is_required('biomass', 'crop', U.t_ha)
    def bmt(self) -> Requirement:
        r"""
        RQ - ``'biomass'`` from model with id ``'crop'``

        :math:`c_{\textrm{bm},k}\ [\frac{t}{ha}]`

        :return: total dry biomass (aboveground + roots) 
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
        r"""
        Initialization of random outputs with zero vectors.

        Except: :math:`c_{\textrm{N-co},0} = c_{\textrm{N-c1},0} + c_{\textrm{N-c2},0}` 
        which is the result when setting :math:`c_{\textrm{D-hui},0} = 0.0` in equ. 26 [R2]_ .

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))
        self.water = self._zv.copy()
        self.nitrogen = self._zv.copy()
        self.n_concentration_opt = self.ncoeff1 + self.ncoeff2  # results when self.hui.value = 0.0 in the corresponding equation in the update-method

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`water`
        * :func:`n_concentration_opt`
        * :func:`n_amount_opt`
        * :func:`nitrogen`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        self.water = self.transpiration_pot.value
        nco_inv = self.ncoeff1 + self.ncoeff2 * np.exp(
            -self.ncoeff3 * self.hui.value
        )
        self.n_concentration_opt = 1. / nco_inv
        self.n_amount_opt = self.bmt.value * self.n_concentration_opt
        self.nitrogen = self.n_amount_opt - self.nsum.value
        self.nitrogen = np.where(self.nitrogen >= 0.0, self.nitrogen, 0.0)
