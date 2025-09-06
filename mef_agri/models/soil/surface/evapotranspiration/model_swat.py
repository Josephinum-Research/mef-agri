import numpy as np

from ....base import Model, Quantities as Q, Units as U
from ....requ import Requirement


class Evapotranspiration_V2009(Model):
    r"""
    This evapotranspiration model uses the Penman-Monteith method to compute 
    potential evapotranspiration.

    In [R1]_ either soil evaporation or (crop) transpiration occurs.
    This approach seems to be unrealistic especially in early growing stages of 
    the crop (much bare soil area). 
    Here, the soil cover index will be used to split the remaining potential 
    evapotranspiration (i.e. after evaporating the canopy water) into 
    transpiration and soil evaporation. 
    Snow covers will be neglected for now.

    Used constants

    * ``SOIL_HEAT_FLUX = 0.0`` - [R1]_ (section 2:2.2.1.1)
    * ``KARMAN_CONST = 0.41`` - [R1]_ (section 2:2.2.1.2)

    kwargs :math:`\rightarrow` :class:`ssc_csm.models.base.Model`
    """
    SOIL_HEAT_FLUX = 0.0  # [R1]_ section 2:2.2.1.1
    KARMAN_CONST = 0.41  # [R1]_ section 2:2.2.1.2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def evapotranspiration_pot(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-etp},s,k}\ [\frac{mm}{day}]` - [R1]_ (equ. 2:2.2.2)

        :return: potential evapotranspiration 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def transpiration_pot(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-tp},s,k}\ [\frac{mm}{day}]`

        :return: potential transpiration 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def evaporation_pot(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-ep},s,k}\ [\frac{mm}{day}]`

        :return: potential evaporation from soil 
        :rtype: numpy.ndarray
        """
    
    @Model.is_quantity(Q.ROUT, U.s_m)
    def resistance_air(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{ra},s,k}\ [\frac{s}{m}]` - [R1]_ (equ. 2:2.2.3)

        :return: diffusion resistance of the air layer 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.s_m)
    def resistance_crop(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{rc},s,k}\ [\frac{s}{m}]`

        :return: diffusion resistance of crop canopy 
        :rtype: numpy.ndarray
        """

    @Model.is_required('latent_heat', 'zone.atmosphere.weather', U.MJ_kg)
    def lh(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{lhv},k}\ [\frac{MJ}{kg}]`

        :return: latent heat of vaporization 
        :rtype: Requirement
        """

    @Model.is_required('slope_vpr_sat', 'zone.atmosphere.weather', U.kPa_degC)
    def slp(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{ssvpc}}\ [\frac{kPa}{^\circ C}]`

        :return: slope of the saturated vapor pressure curve (d_vpr_sat / d_temp) 
        :rtype: Requirement
        """

    @Model.is_required('psychrometric_const', 'zone.atmosphere.weather', U.kPa_degC)
    def psyc(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{psyc},k}\ [\frac{kPa}{^\circ C}]`

        :return: psychrometric constant at site elevation 
        :rtype: Requirement
        """

    @Model.is_required('vapor_pressure_sat', 'zone.atmosphere.weather', U.kPa)
    def vpr_sat(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{vprs},k}\ [kPa]`

        :return: saturated vapor pressure 
        :rtype: Requirement
        """

    @Model.is_required('vapor_pressure', 'zone.atmosphere.weather', U.kPa)
    def vpr(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{vpr},k}\ [kPa]`

        :return: actual vapor pressure 
        :rtype: Requirement
        """

    @Model.is_required('height_hum_obs', 'zone.atmosphere.weather', U.cm)
    def hh(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{h-hum}}\ [cm]`

        :return: ref. height of the humidity observations 
        :rtype: Requirement
        """

    @Model.is_required('height_wind_obs', 'zone.atmosphere.weather', U.cm)
    def hw(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{h-wnd}}\ [cm]`

        :return: ref. height of the wind observations 
        :rtype: Requirement
        """

    @Model.is_required('wind_speed_mean', 'zone.atmosphere.weather', U.m_s)
    def ws(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{wsp},k}\ [\frac{m}{s}]`

        :return: mean daily wind speed 
        :rtype: Requirement
        """

    @Model.is_required('radiation_sum', 'zone.atmosphere.weather', U.W_m2)
    def rad(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{rad},k}\ [\frac{W}{m^2}]`

        :return: daily radiation sum 
        :rtype: Requirement
        """

    @Model.is_required('temperature_mean', 'zone.atmosphere.weather', U.degC)
    def temp(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{temp},k}\ [^\circ C]`

        :return: daily mean temperature 
        :rtype: Requirement
        """

    @Model.is_required('net_radiation', 'zone.atmosphere.radiation', U.MJ_m2xday)
    def nrad(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.radiation'``

        :math:`a_{\textrm{nrad},k}\ [\frac{MJ}{m^2\cdot day}]`

        :return: daily net radiation 
        :rtype: Requirement
        """

    @Model.is_required('soil_cover_index', 'zone.soil.surface', U.none)
    def sci(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.surface'``

        :math:`s_{\textrm{sci},s,k}\ [\ ]`

        :return: soil cover index 
        :rtype: Requirement
        """

    @Model.is_required('water_canopy', 'zone.soil.surface.water', U.mm)
    def cw(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.surface.water'``

        :math:`s_{\textrm{W-c},k}\ [mm]`

        :return: amount of water stored in the canopy 
        :rtype: Requirement
        """

    @Model.is_required('height', 'crop', U.cm)
    def hc(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop'``

        :math:`c_{\textrm{h},k}\ [cm]`

        :return: current crop height 
        :rtype: Requirement
        """

    @Model.is_required('lai', 'crop.leaves', U.none)
    def lai(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop.leaves'``

        :math:`c_{\textrm{lai},k}\ [\ ]`

        :return: leaf area index 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Initialization of :func:`evapotranspiration_pot`, 
        :func:`evaporation_pot` and :func:`transpiration_pot` with zero arrays.

        :param epoch: intialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))
        self.evapotranspiration_pot = self._zv.copy()
        self.evaporation_pot = self._zv.copy()
        self.transpiration_pot = self._zv.copy()

    def update(self, epoch):
        """
        The following computations are performed

        * :func:`resistance_air`

            * if a crop is present

                * roughness length of momentum transfer - [R1]_ (equ. 2:2.2.4, 2:2.2.5)
                * roughness length of vapor transfer - [R1]_ (equ. 2:2.2.6)
                * zero plane displacement of the wind profile - [R1]_ (equ. 2:2.2.7)
                * check the :math:`\ln`-arguments in the computation of :func:`resistance_air` - 

                    * if the min. value is lower than ``1e-8``, :func:`resistance_air` will be computed for a reference crop - [R1]_ (equ. 2:2.2.20)
                    * else - [R1]_ (equ. 2:2.2.3)

            * if no crop is present, :func:`resistance_air` will be computed for a reference crop - [R1]_ (equ. 2:2.2.20)

        * :func:`resistance_crop`

            * absorbed photosynthetically active radiation (``apar``) - [R4]_ (equ. 6, section Discussion)
            * stomatal resistance with regression function - [R4]_ (approximate conversion value of 4.0 from fig. 1)
            * :func:`resistance_crop` - [R1]_ (equ. 2:2.2.8)

        * combined term of Penman-Monteith method ``ct = 1710.0 - 6.85 * self.temp.value``
        * :func:`evapotranspiration_pot`
        * evaporation of canopy water
        * splitting remaining evapotranspiration into potential transpiration and potential evaporation from soil using :func:`sci`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        cp = self.model_tree.get_model('zone').crop_rotation.crop_present

        ########################################################################
        # POTENTIAL EVAPOTRANSPIRATION (PENMAN-MONTEITH METHOD)
        if cp:
            # roughness length for momentum transfer [cm] - [R1]_ equ. 2:2.2.4 + 2:2.2.5
            zom = np.where(
                self.hc.value <= 200.0, 
                0.123 * self.hc.value,
                0.058 * np.power(self.hc.value, 1.19)
            )
            if np.min(zom) < 1e-8:
                self.resistance_air = 114. / self.ws.value
            else:
                # roughness length of vapor transfer [cm] - [R1]_ equ. 2:2.2.6
                zov = 0.1 * zom
                # zero plane displacement of the wind profile [cm] - [R1]_ equ. 2:2.2.7
                dwf = (2.0 / 3.0) * self.hc.value
                # check ln-args of air layer
                ln1 = (self.hw.value - dwf) / zom
                ln2 = (self.hh.value - dwf) / zov
                lns = np.min(np.concatenate((ln1, ln2), axis=0))
                if np.min(lns) < 1e-8:
                    self.resistance_air = 114. / self.ws.value  # air resistance for reference crop - [1] equ. 2:2.2.20
                else:
                    # air layer resistance [s/m] - [1] equ. 2:2.2.3
                    self.resistance_air = (
                        (np.log(ln1) * np.log(ln2)) /
                        (np.power(self.KARMAN_CONST, 2.0) * self.ws.value)
                    )
        else:
            self.resistance_air = 114. / self.ws.value  # air resistance for reference crop - [1] equ. 2:2.2.20
        if cp:
            # absorbed photosynthetically active radiation [mumol photons / (m2 x s)] - [R4]_ equ. 6
            # observed radiation in [W/m2] - [R4]_ section "Discussion", first paragraph
            apar = 0.47 * (1. - 0.1) * self.rad.value
            # in [R4]_ the canopy resistance r_c is directly derived from APAR but
            # here the computed value of r_c is multiplied with 4.0, which is the
            # approximate conversion value in [R4]_ fig. 1 to get the stomatal 
            # resistance. Then [R1]_ equ. 2:2.2.8 is used to compute r_c. Thus, also 
            # lai influences r_c (contrary to [R4]_)
            rs = 4. * (0.001094 + 8.87e-6 * apar - 2.29e-9 * apar * apar)  # stomatal resistance
            denom = (0.5 * self.lai.value)
            denom = np.where(denom >= 1e-6, denom, 1e-6)
            self.resistance_crop = rs / denom
        else:
            self.resistance_crop = self._zv.copy()
        # computation of the combined term of Penman-Monteith
        ct = 1710.0 - 6.85 * self.temp.value
        # potential evapotranspiration - [1] equ. 2:2.2.2
        self.evapotranspiration_pot = (
            (
                self.slp.value * self.nrad.value + 
                self.psyc.value * ct * 
                ((self.vpr_sat.value - self.vpr.value) / self.resistance_air)
            ) / 
            (
                self.lh.value * (self.slp.value + self.psyc.value * 
                (1. + (self.resistance_crop / self.resistance_air)))
            )
        )

        ########################################################################
        # POTENTIAL TRANSPIRATION AND EVAPORATION
        # in [R1]_ either soil evaporation or (crop) transpiration occurs
        # this approach seems to be unrealistic especially in early growing 
        # stages of the crop (much bare soil area). Here, the soil cover index 
        # will be used to split the remaining potential evapotranspiration (i.e.
        # after evaporating the canopy water) into transpiration and soil 
        # evaporation. Snow covers will be neglected for now.
        # there is no check if a crop is present, because water_canopy `cw` and 
        # soil_cover_index `sci` already consider the cases "crop" or "no crop"

        # evaporation of canopy water
        cwtemp = self.cw.value - self.evapotranspiration_pot
        self.cw.value = np.where(cwtemp >= 0.0, cwtemp, 0.0)
        evp_rem = np.where(cwtemp < 0.0, -1.0 * cwtemp, 0.0)
        # splitting remaining pot. evapotranspiration into pot. transpiration 
        # and pot. evaporation from soil
        self.transpiration_pot = (1. - self.sci.value) * evp_rem
        sevp_pot = self.sci.value * evp_rem
        denom = sevp_pot + self.transpiration_pot
        denom = np.where(denom >= 1e-4, denom, 1e-4)
        sevp_corr = (sevp_pot * evp_rem) / denom
        self.evaporation_pot = np.min(  # [1] equ. 2:2.3.9
            np.vstack((np.atleast_2d(sevp_pot), np.atleast_2d(sevp_corr))),
            axis=0
        )

