import numpy as np

from ...base import Model, Quantities as Q, Units as U


class WeatherINCA(Model):
    """
    This model acts as INCA weather observation "container". No computations 
    are done within this model but it can be used as super-class to perform 
    further meterological computations based on the weather observations.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height_temp_obs:float = None
        self.height_wind_obs:float = None
        self.height_hum_obs:float = None

    @Model.is_quantity(Q.OBS, U.degC)
    def temperature_min(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{tmin},k}\ [^\circ C]`

        :return: minimum daily temperature 2 m above surface - [R5]_ (fig. 4.7) 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.degC)
    def temperature_max(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{tmax},k}\ [^\circ C]`

        :return: maximum daily temperature 2 m above surface - [R5]_ (fig. 4.7)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.degC)
    def temperature_mean(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{temp},k}\ [^\circ C]`

        :return: mean daily temperature 2 m above surface [R5]_ (fig. 4.7) 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.degC)
    def dewpoint_mean(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{dewp},k}\ [^\circ C]`

        :return: mean daily dewpoint temperature 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.mm_m2xday)
    def precipitation_sum(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{prec},k}\ [\frac{mm}{m^2\cdot day}]`

        :return: daily precipitation sum 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.Wh_m2xday)
    def radiation_sum(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{rad},k}\ [\frac{Wh}{m^2\cdot day}]`

        :return: daily radiation sum 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.frac)
    def humidity_min(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{hmin},k}\ [\ ]`

        :return: minimum value of daily humidity 2 m above surface - [R5]_ (fig. 5.1.5)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.frac)
    def humidity_max(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{hmax},k}\ [\ ]`

        :return: maximum value of daily humidity 2 m above surface [R5]_ (fig. 5.1.5) 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.frac)
    def humidity_mean(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{hum},k}\ [\ ]`

        :return: mean value of daily humidity 2 m above surface [R5]_ (fig. 5.1.5) 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.m_s)
    def wind_speed_mean(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{wsp},k}\ [\frac{m}{s}]`

        :return: mean absolute and directionless value of daily wind speed 10 m above surface [R5]_ (fig. 5.2.3) 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.OBS, U.kPa)
    def pressure_msl_mean(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`a_{\textrm{pmsl},k}\ [kPa]`

        :return: mean value of daily mean-sea-level-pressure 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.DOUT, U.m)
    def height_temp_obs(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{h-tmp}}\ [m]` - [R5]_ (fig. 4.7) 

        :return: reference height of the temperature observations 
        :rtype: float
        """

    @Model.is_quantity(Q.DOUT, U.m)
    def height_hum_obs(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{h-hum}}\ [m]` - [R5]_ (fig. 5.1.5)

        :return: reference height of the humidity observations 
        :rtype: float
        """

    @Model.is_quantity(Q.DOUT, U.m)
    def height_wind_obs(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{h-wnd}}\ [m]` - [R5]_ (fig. 5.2.3) 

        :return: reference height of the wind observations 
        :rtype: float
        """

    def initialize(self, epoch):
        """
        Initialization of the heights of observed temperature, wind and humidity.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self.height_temp_obs = 2.0  # 2 m above surface [2] - fig. 4.7
        self.height_hum_obs = 2.0  # 2 m above surface [2] - fig. 5.1.5
        self.height_wind_obs = 10.0  # 10 m above surface [2] - fig. 5.2.3
