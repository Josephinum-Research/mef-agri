import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from .inca_obs import WeatherINCA


class Weather_V2009(WeatherINCA):
    r"""
    This model inherits from 
    :class:`ssc_csm.models.atmosphere.weather.inca_obs.WeatherINCA` and computes 
    further quantities from the INCA observations based on [R1]_ .

    Used constants
    
    * ``SPECIFIC_HEAT = 1.013e-3`` - :math:`\frac{MJ}{kg\cdot\ ^\circ C}`
    * ``GRAVITY = 9.80665`` - :math:`\frac{m}{s^2}`
    * ``MOLAR_MASS_DRY_AIR = 0.02896968`` - :math:`\frac{kg}{mol}`
    * ``SEALVL_STD_TEMP = 288.15`` - :math:`^\circ K`
    * ``UNIV_GAS_CONST = 8.314462618`` - :math:`\frac{J}{mol\cdot ^\circ K}`

    """

    SPECIFIC_HEAT = 1.013e-3  # [ ( MJ ) / (kg x degC) ]
    GRAVITY = 9.80665  # [ ( m ) / ( s2 ) ]
    MOLAR_MASS_DRY_AIR = 0.02896968  # [ ( kg ) / ( mol ) ]
    SEALVL_STD_TEMP = 288.15  # [ degK ]
    UNIV_GAS_CONST = 8.314462618  # [ ( J ) / (mol x degK) ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._h:float = None  # site elevation [m]

    @Model.is_quantity(Q.PARAM, U.degC)
    def temperature_mean_annual(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`a_{\bar{\textrm{t}}}\ [^\circ C]`

        :return: mean annual temperature 
        :rtype: numpy.ndarray
        """
    
    @Model.is_quantity(Q.ROUT, U.kPa)
    def vapor_pressure_sat(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{vprs},k}\ [kPa]` - [R1]_ (equ. 1:2.3.2)

        :return: saturated vapor pressure 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kPa)
    def vapor_pressure(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{vpr},k}\ [kPa]` - [R1]_ (equ. 1:2.3.3)

        :return: actual vapor pressure 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kPa_degC)
    def slope_vpr_sat(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{slp-vpr}}\ [\frac{kPa}{^\circ C}]` - [R1]_ (equ. 1:2.3.4)

        :return: slope of the saturated vapor pressure curve :math:`\frac{\delta a_{\textrm{vprs},k}}{\delta a_{\textrm{temp},k}}`
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.MJ_kg)
    def latent_heat(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{lhv},k}\ [\frac{MJ}{kg}]` - [R1]_ (equ. 1:2.3.6)

        :return: latent heat of vaporization 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kPa)
    def pressure(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{apr},k}\ [kPa]` - https://en.wikipedia.org/wiki/Barometric_formula

        :return: atmospheric pressure at site elevation (computed with barometric formula)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kPa_degC)
    def psychrometric_const(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{psyc},k}\ [\frac{kPa}{^\circ C}]` - [R1]_ (equ. 1:2.3.7)

        :return: psychrometric constant at site elevation 
        :rtype: numpy.ndarray
        """

    def initialize(self, epoch):
        """
        Get site height/elevation from model-tree.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._h = self.model_tree.get_model('zone').height

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`vapor_pressure_sat`
        * :func:`vapor_pressure`
        * :func:`slope_vpr_sat`
        * :func:`latent_heat`
        * :func:`pressure`
        * :func:`psychrometric_const`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        # saturated vapor pressure [1] equ. 1:2.3.2
        self.vapor_pressure_sat = np.exp(
            (16.78 * self.temperature_mean - 116.0) / 
            (self.temperature_mean + 237.3)
        )
        # actual vapor pressure [1] equ. 1:2.3.3
        self.vapor_pressure = self.humidity_mean * self.vapor_pressure_sat
        # slope of the saturated vapor pressure (with respect to temperature)
        # [1] equ. 1:2.3.4
        self.slope_vpr_sat = (
            (4098.0 * self.vapor_pressure_sat) / 
            np.power(self.temperature_mean + 237.3, 2.0)
        )
        # latent heat of vaporization [1] equ. 1:2.3.6
        self.latent_heat = 2.501 - 2.361e-3 * self.temperature_mean
        # pressure is computed with https://en.wikipedia.org/wiki/Barometric_formula
        self.pressure = self.pressure_msl_mean * np.exp(
            (-self.GRAVITY * self._h * self.MOLAR_MASS_DRY_AIR) / 
            (self.SEALVL_STD_TEMP * self.UNIV_GAS_CONST)
        )
        # psychrometric constant [1] equ. 1:2.3.7
        self.psychrometric_const = (
            (self.SPECIFIC_HEAT * self.pressure) / (0.622 * self.latent_heat)
        )
