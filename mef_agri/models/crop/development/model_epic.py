import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement


class Development(Model):
    r"""
    Model which computes crop development based on [R2]_. Winter dormancy is 
    not considered (i.e. ``self.winter_dormancy`` is always set to 
    ``False``). If this is required, see :class:`Development_Dormancy`.

    kwargs :math:`\rightarrow` see :class:`mef_agri.models.base.Model`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None
        self.winter_dormancy:bool = None

    @Model.is_quantity(Q.STATE, U.frac)
    def heat_unit_index(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`c_{\textrm{D-hui},k} = \frac{\sum_k{c_{\textrm{D-dhu},k}}}{c_{\textrm{D-phu},0}}\ [\ ]` - [R2]_ (equ. 2)

        :return: current heat unit index 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.degC)
    def heat_units_rate(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{D-dhu},k}\ [^\circ C]` - [R2]_ (equ. 1)

        :return: daily increase of heat units 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def heat_unit_factor_leaves(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`c_{\textrm{L-huf},k}\ [\ ]` - [R2]_ (equ. 9)

        :return: heat unit factor for leaf growth 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.DOUT, U.boolean)
    def winter_dormancy(self) -> bool:
        r"""
        MQ - Deterministic Output

        :math:`c_{\textrm{D-wd},k}\ [\ ]` (boolean value) - [R2]_ (section Model Description > Growth Constraints > Winter Dormancy)

        :return: indicator for winter dormancy period
        :rtype: bool
        """
    
    @Model.is_quantity(Q.PARAM, U.degC)
    def temperature_base(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-tb},0}\ [^\circ C]` - [R2]_ (equ. 1, table 2)

        :return: crop-specific base temperature for heat-unit computation
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.degC)
    def temperature_opt(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-to},0}\ [^\circ C]` - [R2]_ (equ. 46, table 2)

        :return: crop-specific optimum temperature regarding temperature stress
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.degC)
    def heat_units_pot(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-phu},0}\ [^\circ C]` - [R2]_ (equ. 2, table 2)

        :return: potential heat units for maturity 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def hufl_coeff1(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{D-lc1},0}\ [\ ]` - [R2]_ (equ. 9, table 2)

        :return: regression coefficient for leaves heat-unit-factor 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def hufl_coeff2(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{C-lc2},0}\ [\ ]` - [R2]_ (equ. 9, table 2)
        
        :return: regression coefficient for leaves heat-unit-factor 
        :rtype: numpy.ndarray
        """

    @Model.is_required('temperature_max', 'zone.atmosphere.weather', U.degC)
    def tmax(self) -> Requirement:
        r"""
        RQ - ``'temperature_max'`` from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{tmax},k}\ [^\circ C]`

        :return: daily max. temperature  
        :rtype: Requirement
        """

    @Model.is_required('temperature_min', 'zone.atmosphere.weather', U.degC)
    def tmin(self) -> Requirement:
        r"""
        RQ - ``'temperature_min'`` from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{tmin},k}\ [^\circ C]` 

        :return: daily min. temperature 
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
        Initialization of random outputs with zero vectors.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))
        self.heat_units_rate = self._zv.copy()
        self.heat_unit_factor_leaves = self._zv.copy()
        self.heat_unit_factor_leaves_rate = self._zv.copy()
        self.winter_dormancy = False

    def update(self, epoch):
        """
        The following computations are performed

        * call :func:`update_dormancy`
        * if ``self.winter_dormancy == True``

            * set :func:`heat_units_rate` to zero vector
            * :func:`heat_unit_index` and :func:`heat_unit_factor_leaves` remain unchanged

        * else

            * :func:`heat_units_rate`
            * :func:`heat_unit_index`
            * :func:`heat_unit_factor_leaves`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        self.update_dormancy()

        if self.winter_dormancy:
            self.heat_units_rate = self._zv.copy()
            # hui and hufl remain unchanged (therefore lai, crop height and root depth does not increase)
            # biomass increase is suppressed in the corresponding model as it cannot be controlled via variables of this model
            return
        
        self.heat_units_rate = \
            0.5 * (self.tmax.value + self.tmin.value) - self.temperature_base
        self.heat_units_rate = np.where(
            self.heat_unit_index < 1.0, self.heat_units_rate, 0.0
        )
        self.heat_unit_index += self.heat_units_rate / self.heat_units_pot
        self.cond_hui()
        # heat unit factor for leaves
        self.heat_unit_factor_leaves = (
            self.heat_unit_index / (self.heat_unit_index + np.exp(
                self.hufl_coeff1 - self.hufl_coeff2 * self.heat_unit_index
            ))
        )        

    def update_dormancy(self) -> None:
        """
        Method which has to be implemented in a child class of 
        :class:`Development`. It is called in :func:`Development.update`.
        """
        pass

    @Model.is_condition
    def cond_hui(self) -> None:
        r"""
        Ensures that :func:`heat_unit_index` stays in the range of 
        :math:`c_{\textrm{D-hui},k} \in [0.0, 1.0]`
        """
        self.heat_unit_index = np.where(
            self.heat_unit_index >= 0.0, self.heat_unit_index, 0.0
        )
        self.heat_unit_index = np.where(
            self.heat_unit_index <= 1.0, self.heat_unit_index, 1.0
        )


class Development_Dormancy(Development):
    r"""
    This model considers winter dormancy periods. According to [R2]_, it is 
    present, when the current daylength is within :math:`\pm` one hour around 
    the minimum possible daylength at the current site.

    Inherits from :class:`Development`

    kwargs :math:`\rightarrow` see :class:`mef_agri.models.base.Model`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dlmin:float = None

    def update_dormancy(self):
        """
        Determine winter dormancy periods based on current daylength and 
        minimum possible daylength at current site.
        """
        if self._dlmin is None:
            self._dlmin = self.model_tree.get_quantity(
                'daylength_min', 'zone.atmosphere.daylength', U.h
            )
        if self.dl.value < (self._dlmin + 1.0):
            self.winter_dormancy = True
        else:
            self.winter_dormancy = False
