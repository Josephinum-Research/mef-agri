import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U, PFunction
from ...requ import Requirement


class Development(Model):
    r"""
    Crop development based on [R11]_ and [R12]_. 
    This model uses the unmodified equations for :func:`growth_stage` and 
    :func:`bbch` from [R11]_ (table 1).

    In contrast to [R11]_ (table 1), the :func:`growth_stage` is parameterized 
    such that it starts from zero.
    The BBCH equations from [R11]_ (table 1) which require the 
    :func:`growth_stage` as input are adjusted accordingly.
    """

    DEFAULT_PARAM_VALUES = {}  # TODO
    INITIAL_STATE_VALUES = {}  # TODO

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO

    @Model.is_quantity(Q.STATE, U.none)
    def growth_stage(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`c_{\textrm{D-gs},k}\ [\ ]` - [R11]_ (table 1)

        :return: growth stage
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def bbch(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{bbch},k}\ [\ ]` - [R11]_ (table 1)

        :return: BBCH stages
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

    @Model.is_quantity(Q.ROUT, U.none)
    def vern_days_sum(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{D-vds},k}\ [day]` - [R12]_ (equ. 2)

        :return: accumulated vernalization days
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PFUNC)
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

        :math:`c_{\textrm{D-tt1},0}\ [^\circ C\cdot day]` - [R11]_ (table 1 + 2)

        :return: thermal time from sowing to emergence
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.degCday)
    def phyllochron(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-tt2},0}\ [^\circ C\cdot day]` - [R11]_ (table 1 + 2)

        :return: phyllochron (also a thermal time)
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
        super().initialize(epoch)
        # TODO

    def update(self, epoch):
        super().update(epoch)

        self.vern_days_sum += self.vern_days_func(self.tavg.value)
        self.f_vernalization = 1. - (self.vern_days_requ / 50.)
        self.f_vernalization += (self.vern_days_sum / self.vern_days_requ)
        self.f_photoperiod = 1. - (self.sens_photoperiod / 10000.) * np.power(
            20. - self.dl.value
        )

        # TODO implement unmodified equations from [R11]_ (table 1)

    @Model.is_condition
    def cond_fv_fp(self) -> None:
        # TODO
        pass
