import numpy as np

from ...base import Model, Quantities as Q, Units as U
from ...requ import Requirement


EARTH_ROTATION_VELOCITY = 0.2618  # [rad/h]
SOLAR_CONSTANT = 4.921  # [MJ/m2/h]
BOLTZMANN_CONSTANT = 4.903e-9  # [MJ / (m^2 x K^4 x day)]
CC_A, CC_B = 1.2, -0.2  # coefficients for cloud cover adjustment (General values of table 1:1-3 in [1])
EM_A, EM_B = 0.39, -0.158  # coefficients for net emittence computation (General values of table 1:1-3 in [1])


def eccentricity_correction(doy:int) -> float:
    """
    According to [R1]_ (equ. 1:1.1.1)

    :param doy: n-th day of the year
    :type doy: int
    :return: eccentricity correction [ ]
    :rtype: float
    """
    return 1.0 + 0.033 * np.cos((2 * np.pi * doy) / 365)


def extraterrestrial_radiation(
        exc_corr:float, dayl:float, sd:float, lat:float
    ) -> float:
    r"""
    According to [R1]_ (equ. 1:1.2.5)

    :param exc_corr: eccentricity correction [ ]
    :type exc_corr: float
    :param dayl: daylength [h]
    :type dayl: float
    :param sd: solar declination [rad]
    :type sd: float
    :param lat: site latitude [rad]
    :type lat: float
    :return: extraterrestrial radiation :math:`\frac{MJ}{m^2\cdot day}`
    :rtype: float
    """
    aux = EARTH_ROTATION_VELOCITY * dayl * 0.5  # sunrise hour is half daylength
    term1 = (24 / np.pi) * SOLAR_CONSTANT * exc_corr
    term2 = aux * np.sin(sd) * np.sin(lat)
    term2 += np.cos(sd) * np.cos(lat) * np.sin(aux)
    return term1 * term2


class Radiation_V2009(Model):
    r"""
    Model to compute radiation-related quantities.

    kwargs :math:`\rightarrow` :class:`mef_agri.models.base.Model`

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lat:float = None

    @Model.is_quantity(Q.DOUT, U.none)
    def eccentricity_correction(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{ecorr},k}\ [\ ]` - [R1]_ (equ. 1:1.1.1)

        :return: eccentricity correction due to elliptic orbit of earth
        :rtype: float
        """

    @Model.is_quantity(Q.DOUT, U.MJ_m2xday)
    def extraterrestrial_radiation(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{exrad},k}\ [\frac{MJ}{m^2\cdot day}]` - [R1]_ (equ. 1:1.2.5)

        :return: extraterrestrial radiation (i.e. before atmosphere influence)
        :rtype: float
        """

    @Model.is_quantity(Q.ROUT, U.MJ_m2xday)
    def net_shortwave_radiation(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{nsrad},k}\ [\frac{MJ}{m^2\cdot day}]` - [R1]_ (equ. 1:1.2.12)

        :return: net shortwave radiation is a fraction of the observed daily incoming radiation 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.MJ_m2xday)
    def net_longwave_radiation(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{nlrad},k}\ [\frac{MJ}{m^2\cdot day}]` - [R1]_ (equ. 1:1.2.18)

        :return: net longwave radiation 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.MJ_m2xday)
    def net_radiation(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`a_{\textrm{nrad},k}\ [\frac{MJ}{m^2\cdot day}]` - [R1]_ (equ. 1:1.2.11)

        :return: daily net radiation 
        :rtype: numpy.ndarray
        """

    @Model.is_required('radiation_sum', 'zone.atmosphere.weather', U.MJ_m2xday)
    def radiation(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{rad},k}\ [\frac{MJ}{m^2\cdot day}]`

        :return: observed daily incoming radiation 
        :rtype: Requirement
        """

    @Model.is_required('vapor_pressure', 'zone.atmosphere.weather', U.kPa)
    def vapor_pressure(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{vpr},k}\ [kPa]`

        :return: current vapor pressure
        :rtype: Requirement
        """

    @Model.is_required('temperature_mean', 'zone.atmosphere.weather', U.degK)
    def temperature(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{temp},k}\ [^\circ K]`

        :return: mean air temperature 
        :rtype: Requirement
        """

    @Model.is_required('daylength', 'zone.atmosphere.daylength', U.h)
    def daylength(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.daylength'``

        :math:`a_{\textrm{dl},k}`

        :return: daylength 
        :rtype: Requirement
        """

    @Model.is_required('solar_declination', 'zone.atmosphere.daylength', U.rad)
    def solar_declination(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.daylength'``

        :math:`a_{\textrm{sd},k}\ [rad]`

        :return: solar declination 
        :rtype: Requirement
        """

    @Model.is_required('albedo', 'zone.soil.surface', U.frac)
    def albedo(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.surface'``

        :math:`s_{\textrm{alb},s,k}\ [\ ]`

        :return: albedo at current day (considering crops)
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Get the site latitude.

        :param epoch: initialzation epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._lat = self.model_tree.get_model('zone').latitude

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`Radiation_V2009.eccentricity_correction`
        * :func:`Radiation_V2009.extraterrestrial_radiation`
        * :func:`net_shortwave_radiation`
        * cloud cover adjustment factor - [R1]_ (equ. 1:1.2.19)
        * net emittance - [R1]_ (equ. 1:1.2.20)
        * :func:`net_longwave_radiation`
        * :func:`net_radiation`

        :param epoch: _description_
        :type epoch: _type_
        """
        super().update(epoch)
        doy = epoch.timetuple().tm_yday

        # computation of the extraterrestrial radiation
        self.eccentricity_correction = eccentricity_correction(doy)
        self.extraterrestrial_radiation = extraterrestrial_radiation(
            self.eccentricity_correction, self.daylength.value, 
            self.solar_declination.value, self._lat
        )
        
        # net short wave radiation
        self.net_shortwave_radiation = self.radiation.value * (
            1. - self.albedo.value)
        cca = CC_A * (self.radiation.value / self.extraterrestrial_radiation)
        cca -= CC_B  # cloud cover adjustment factor ([1] equ. 1:1.2.19)
        nem = -(EM_A + EM_B * np.sqrt(self.vapor_pressure.value))  # net emittance ([1] equ. 1:1.2.20)
        self.net_longwave_radiation = (cca * nem * BOLTZMANN_CONSTANT * 
                                       np.power(self.temperature.value, 4.0))
        self.net_radiation = (self.net_shortwave_radiation + 
                              self.net_longwave_radiation)
