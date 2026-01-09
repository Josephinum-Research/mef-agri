import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U, PFunction
from ...requ import Requirement
from ....farming import crops
from ....evaluation.stats_utils import DISTRIBUTIONS
from ...utils import PFunction


class Development(Model):
    r"""
    Crop development based on [R11]_ and [R12]_. 
    The equations in [R11]_ table 1 are rearranged such that only the BBCH 
    stages remain as state.
    This model uses the unmodified equations from [R11]_ table 1.
    For the setting of parameter default values, also [R13]_ is used
    """

    DEFAULT_PARAM_VALUES = {
        crops.winter_wheat.__name__: {
            'vern_days_func': {
                'fdef': {  # [R12]_ figure 2
                    'ftype': PFunction.FTYPE.PIECEWISE_LINEAR,
                    'values-x': [-5., 0., 8., 15.],
                    'values-y': [0., 1., 1., 0.],
                    'distr-x': {
                        'distr_id': DISTRIBUTIONS.NORMAL_1D,
                        'std': 0.3
                    },
                    'distr-y': {
                        'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                        'std': 0.03,
                        'lb': 0.,
                        'ub': 1.
                    }
                }
            },
            'temperature_base': {
                'value': 0.,
                'distr': {
                    'distr_id': DISTRIBUTIONS.NORMAL_1D,
                    'std': 0.5
                }
            },
            'vern_days_requ': {
                'value': 5.,  # [R12]_ table 3 + [R13]_ table A2
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 1.
                }
            },
            'sens_photoperiod': {
                'value': 2.5,  # [R11]_ table 2
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 3.
                }
            },
            'tt_sowing2emergence': {
                'value': 100.,  # [R11]_ table 2
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 10.
                }
            },
            'phyllochron': {
                'value': 100.,  # [R11]_ table 2
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 5.
                }
            },
            'tt_filling2maturity': {
                'value': 10.,
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 10.
                }
            }

        }
    }
    INITIAL_STATE_VALUES = {
        'bbch': {
            'value': 0.,
            'distr': {
                'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                'std':0.5,
                'lb': 0.,
                'ub': 99.
            }
        },
        'vern_days_sum': {
            'value': 0.,
            'distr': {
                'distr_id': DISTRIBUTIONS.GAMMA_1D,
                'std': 0.5
            }
        }
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None
        self._ov:np.ndarray = None

    @Model.is_quantity(Q.STATE, U.none)
    def bbch(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`c_{\textrm{bbch},k}\ [\ ]` - [R11]_ (table 1)

        :return: BBCH stage
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.STATE, U.none)
    def vern_days_sum(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`c_{\textrm{D-vds},k}\ [day]` - [R12]_ (equ. 2)

        :return: accumulated vernalization days
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.per_day)
    def bbch_rate(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\Delta\textrm{bbch},k}\ [\frac{1}{day}]` - [R11]_ (table 1)

        :return: daily change/rate of BBCH stage
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def f_vernalization(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{D-fv},k}\ [\ ]` - [R12]_ (equ. 2, equ. 3)

        :return: vernalization factor
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def f_photoperiod(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{D-fp},k}\ [\ ]` - [R12]_ (equ. 4)

        :return: photoperiod/daylength factor
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PFUNC, U.none)
    def vern_days_func(self) -> PFunction:
        r"""
        MQ - Parameter-Function

        [R12]_ (fig. 2)

        :return: function to compute rate of vernalization days
        :rtype: PFunction
        """

    @Model.is_quantity(Q.PARAM, U.day)
    def vern_days_requ(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-vdr},0}\ [day]` - [R12]_ (equ. 2)

        :return: required vernalization days
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def sens_photoperiod(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-spp},0}\ [\ ]` - [R12]_ (equ. 4)
        
        NOTE: It is assumed, that the divisor (fixed value of 10.000) in 
        [R12]_ (equ. 4) has the units [ h x h ] and this parameter is unitless, 
        such that :func:`f_photoperiod` is also unitless.

        :return: sensitivity to photoperiod/daylength
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.degC)
    def temperature_base(self) -> np.ndarray:
        """
        MQ - Parameter

        :return: crop-specific base temperature
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.degCday)
    def tt_sowing2emergence(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-tt9},0}\ [^\circ C\cdot day]` - [R11]_ (table 1 + 2)

        :return: thermal time from sowing to emergence
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.degCday)
    def phyllochron(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-phy},0}\ [^\circ C\cdot day]` - [R11]_ (table 1 + 2)

        :return: phyllochron is degree-days per leaf (also a thermal time)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.degCday)
    def tt_filling2maturity(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-tt5},0}\ [^\circ C\cdot day]` - [R11]_ (table 1 + 2)

        :return: thermal time from grain filling to maturity
        :rtype: numpy.ndarray
        """

    @Model.is_required('temperature_mean', 'zone.atmosphere.weather', U.degC)
    def tavg(self) -> Requirement:
        r"""
        RQ - ``'temperature_mean'`` from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{temp},k}\ [^\circ C]`

        :return: daily mean temperature
        :rtype: Requirement
        """

    @Model.is_required('daylength', 'zone.atmosphere.daylength', U.h)
    def dl(self) -> Requirement:
        r"""
        RQ - ``'daylength'`` from model with id ``'zone.atmosphere.daylength'``

        :math:`a_{\textrm{dl},k}\ [h]`

        :return: current daylength at site
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Initialization of random outputs.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))
        self._ov = np.zeros((self.model_tree.n_particles))
        self.bbch_rate = self._zv.copy()
        self.f_vernalization = self._ov.copy()
        self.f_photoperiod = self._ov.copy()

    def update(self, epoch):
        """
        The following computations are performed

        * update :func:`vern_days_sum` through :func:`vern_days_func`
        * :func:`f_vernalization`
        * :func:`f_photoperiod`
        * :func:`bbch_rate` - each case in [R11]_ table1 is computed and then correctly seleceted depending on :func:`bbch`
        * update :func:`bbch` through adding appropriate :func:`bbch_rate`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        # influence of vernalization and photoperiod/daylength
        self.vern_days_sum += self.vern_days_func(self.tavg.value)
        self.f_vernalization = 1. - (self.vern_days_requ / 50.)
        self.f_vernalization += (self.vern_days_sum / self.vern_days_requ)
        self.f_photoperiod = 1. - (self.sens_photoperiod / 10000.) * np.power(
            20. - self.dl.value, 2.
        )
        self.cond_fv_fp()

        # conditions to choose appropriate equation for bbch-rate computation
        c1 = self.bbch < 5.
        c2 = (self.bbch >= 5.) & (self.bbch < 30.)
        c3 = (self.bbch >= 30.) & (self.bbch < 40.)
        c4 = (self.bbch >= 40.) & (self.bbch < 57.)
        c5 = (self.bbch >= 57.) & (self.bbch < 71.)
        c6 = (self.bbch >= 71.) & (self.bbch < 90.)
        c7 = self.bbch >= 90.
        cs = [c1, c2, c3, c4, c5, c6, c7]
        # bbch-rates
        t_trunc = self.tavg.value - self.temperature_base
        t_trunc_thresh = np.maximum(self._zv, t_trunc)
        d1 = (t_trunc_thresh / self.tt_sowing2emergence) * 5.
        d2 = (t_trunc_thresh / ((400. / 95.) * self.phyllochron)) * 12.5  # NOTE 12.5 => 0.5 (due to two growth stages in this case) * 25 (bbch range in this case)
        d2 *= np.minimum(self.f_vernalization, self.f_photoperiod)
        d3 = (t_trunc_thresh / (3. * self.phyllochron)) * 10.
        d4 = (t_trunc_thresh / (2. * self.phyllochron)) * 17.
        d5 = (t_trunc_thresh / 200.) * 14.
        d6 = (np.maximum(self._zv, t_trunc - 1.) / ((self.tt_filling2maturity + 21.5) / 0.05)) * 19.
        d7 = (t_trunc_thresh / 250.) * 10.
        ds = [d1, d2, d3, d4, d5, d6, d7]
        self.bbch_rate = np.select(cs, ds)
        self.bbch += self.bbch_rate

    @Model.is_condition
    def cond_bbch(self) -> None:
        """
        Ensures that :func:`bbch` stays in the range [0, 99].
        """
        self.bbch = np.where(self.bbch >= 0., self.bbch, 0.)
        self.bbch = np.where(self.bbch <= 99., self.bbch, 99.)

    @Model.is_condition
    def cond_vdays(self) -> None:
        """
        Ensures that :func:`vern_days_sum` stays positive.
        """
        self.vern_days_sum = np.where(
            self.vern_days_sum >= 0., self.vern_days_sum, 0.
        )

    @Model.is_condition
    def cond_fv_fp(self) -> None:
        """
        Ensures, that :func:`f_vernalization` and :func:`f_photoperiod` stay in 
        the range [0, 1]
        """
        self.f_vernalization = np.where(
            self.f_vernalization >= 0.0, self.f_vernalization, 0.0
        )
        self.f_vernalization = np.where(
            self.f_vernalization <= 1.0, self.f_vernalization, 1.0
        )
        self.f_photoperiod = np.where(
            self.f_photoperiod >= 0.0, self.f_photoperiod, 0.0
        )
        self.f_photoperiod = np.where(
            self.f_photoperiod <= 1.0, self.f_photoperiod, 1.0
        )
