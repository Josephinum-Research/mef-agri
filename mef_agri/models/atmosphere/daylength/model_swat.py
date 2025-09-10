import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U


EARTH_ROTATION_VELOCITY = 0.2618  # [rad/h]


def compute_solar_declination(doy:int) -> float:
    """
    :param doy: n-th day of the year
    :type doy: int
    :return: solar declination [rad]
    :rtype: float
    """
    return np.arcsin(0.4 * np.sin(((2 * np.pi) / 365.0) * (doy - 82)))


def compute_daylength(sd:float, lat:float) -> float:
    """
    :param sd: solar declination [rad]
    :type sd: float
    :param lat: geographic latitude [rad]
    :type lat: float
    :return: daylength [h]
    :rtype: float
    """
    cosarg = -np.tan(sd) * np.tan(lat)
    if np.abs(cosarg) > 1.0:
            msg = 'No sunrise/sunset at the provided site - this case is not '
            msg += 'considered yet!'
            raise ValueError(msg)
    return (2 * np.arccos(cosarg)) / EARTH_ROTATION_VELOCITY


class Daylength_V2009(Model):
    r"""
    This model computes daylength related quantities. They are all 
    deterministic outputs because they only depend on the site latitude.

    kwargs :math:`\rightarrow` :class:`mef_agri.models.base.Model`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lat:float = None  # mean latitude of considered field

    @Model.is_quantity(Q.DOUT, U.rad)
    def solar_declination(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{sd},k}\ [rad]` - [R1]_ (equ. 1:1.1.2)

        The solar declination is the earth's latittude at which incoming solar 
        rays are normal to the earth's surface. It is :math:`0^\circ` in the 
        spring and fall equinox and approaches :math:`\pm 23.5^\circ` 
        (summer and winter).

        :return: solar declination 
        :rtype: float
        """

    @Model.is_quantity(Q.DOUT, U.h)
    def sunrise_hour(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{srh},k}\ [h]` - [R1]_ (equ. 1:1.1.4)

        :return: hour of sunrise 
        :rtype: float
        """

    @Model.is_quantity(Q.DOUT, U.h)
    def sunset_hour(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{ssh},k}=-a_{\textrm{srh},k}\ [h]` - [R1]_ (equ. 1:1.1.5)

        :return: hour of sunset 
        :rtype: float
        """

    @Model.is_quantity(Q.DOUT, U.h)
    def daylength(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{dl},k}=2\cdot a_{\textrm{srh},k}` - [R1]_ (equ. 1:1.1.6)

        :return: daylength 
        :rtype: float
        """

    @Model.is_quantity(Q.DOUT, U.h)
    def daylength_min(self) -> float:
        r"""
        MQ - Deterministic Output

        :math:`a_{\textrm{dlmin},k}`

        :return: minimum daylength in a year at current site
        :rtype: float
        """

    def initialize(self, epoch):
        """
        Here, :func:`daylength_min` is computed.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._lat = self.model_tree.get_model('zone').latitude

        # compute min. daylength
        dls = []
        for doy in np.arange(0, 366, 1):
            dls.append(
                compute_daylength(
                    compute_solar_declination(doy), self._lat
                )
            )
        self.daylength_min = np.min(np.array(dls))

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`solar_declination`
        * :func:`daylength`
        * :func:`sunrise_hour`
        * :func:`sunset_hour`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        doy = epoch.timetuple().tm_yday

        self.solar_declination = compute_solar_declination(doy)
        self.daylength = compute_daylength(self.solar_declination, self._lat)
        self.sunrise_hour = self.daylength / 2.
        self.sunset_hour = -self.sunrise_hour
