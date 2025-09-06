import numpy as np

from ....base import Model, Quantities as Q, Units as U
from ....requ import Requirement


class Water_CNM_V2009(Model):
    r"""
    This surface water model corresponds to the SCS curve number procedure 
    outlined in [R1]_. In its current version, the curve number is roughly 
    approximated by introducing it as a hyper-parameter and considering 
    crop cover (i.e. an approximation of [R1]_ table 2:1-1). The retention 
    parameter is computed by considering only the water content in the soil 
    profile ([R1]_ equ. 2:1.1.6), i.e. no evapotranspiration and frozen top 
    soil layer.

    kwargs :math:`\rightarrow` :class:`ssc_csm.models.base.Model`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zv:np.ndarray = None

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def precipitation_soil(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{prec},s,k}\ [\frac{mm}{day}]`

        :return: amount of precipitation reaching the soil surface
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm)
    def water_canopy(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-c},k}\ [mm]`

        :return: amount of water stored in the canopy 
        :rtype: numpy.ndarray
        """
    
    @Model.is_quantity(Q.ROUT, U.mm)
    def retention(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{ret},s,k}\ [mm]` - [R1]_ (equ. 2:1.1.6)

        :return: retention parameter 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def runoff(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-ro},s,k}\ [\frac{mm}{day}]` - [R1]_ (equ. 2:1.1.3)

        :return: runoff rate at current day 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def infiltration(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{inf},s,k}\ [\frac{mm}{day}]`

        :return: infiltration rate at current day 
        :rtype: numpy.ndarray
        """

    @Model.is_required('precipitation_sum', 'zone.atmosphere.weather', U.mm_m2xday)
    def prec(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmosphere.weather'``

        :math:`a_{\textrm{prec},k}\ [\frac{mm}{day}]`

        :return: daily precipitation sum 
        :rtype: Requirement
        """

    @Model.is_required('water_soil', 'zone.soil', U.mm)
    def sw(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-a},k}\ [mm]`

        :return: water amount in the soil profile 
        :rtype: Requirement
        """

    @Model.is_required('curve_number', 'zone.soil', U.none)
    def cn(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{cn},k}\ [\ ]`

        :return: curve number considering crops
        :rtype: Requirement
        """

    @Model.is_required('rooting_depth_max', 'zone.soil', U.mm)
    def rdm(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{rdm},0}\ [m]`

        :return: maximum rootable depth of the soil 
        :rtype: Requirement
        """

    @Model.is_required('wilting_point', 'zone.soil', U.frac)
    def wp(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-wp},0}\ [\ ]`

        :return: soil moisture at wilting point 
        :rtype: Requirement
        """

    @Model.is_required('field_capacity', 'zone.soil', U.frac)
    def fc(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-fc},0}\ [\ ]`

        :return: soil moisture at field capacity 
        :rtype: Requirement
        """
    
    @Model.is_required('porosity', 'zone.soil', U.frac)
    def por(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{por},0}\ [\ ]`

        :return: soil porosity (saturated water content) 
        :rtype: Requirement
        """

    @Model.is_required('water_storage_max', 'crop', U.mm)
    def can_max(self) -> Requirement:
        r"""
        RQ - from mode with id ``'crop'``

        :math:`c_{\textrm{W-csm},0}\ [mm]`

        :return: max. water storage of canopy 
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

    @Model.is_required('lai_max', 'crop.leaves', U.none)
    def lai_max(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop.leaves'``

        :math:`c_{\textrm{laimx},0}\ [\ ]`

        :return: max. attainable lai 
        :rtype: Requirement
        """

    def initialize(self, epoch):
        """
        Initialization of :func:`water_canopy` and :func:`runoff` with zero 
        arrays.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))
        self.water_canopy = self._zv.copy()
        self.runoff = self._zv.copy()

    def update(self, epoch):
        """
        The following computations are performed

        * canopy water storage - [R1]_ (section 2:2.1)
        * :func:`precipitation_soil`
        * curve number values for wilting point and field capacity - [R1]_ (equ. 2:1.1.4, 2:1.1.5)
        * retention values for wilting point and field capacity - [R1]_ (equ. 2:1.1.2)
        * shape parameters of retention parameter
        * retention parameter - [R1]_ (equ. 2:1.1.6)
        * :func:`runoff` and :func:`infiltration`

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        cp = self.model_tree.get_model('zone').crop_rotation.crop_present

        ########################################################################
        # canopy water storage - [R1]_ section 2:2.1
        if cp:
            can_day = self.can_max.value * (self.lai.value / self.lai_max.value)
            wctemp = self.water_canopy +  self.prec.value
            cwexc = wctemp > can_day
            self.water_canopy = np.where(cwexc, can_day, wctemp)
            self.precipitation_soil = np.where(cwexc, wctemp - can_day, 0.0)
        else:
            self.water_canopy = self._zv.copy()
            self.precipitation_soil = self.prec.value

        ########################################################################
        # compute retention parameter
        aux1 = 100.0 - self.cn.value
        # computation of curve number values for wilting point and 
        # field capacity - [1] equs. 2:1.1.4 and 2:1.1.5
        cn1 = self.cn.value - ((20.0 * aux1) / 
            (100.0 - self.cn.value + np.exp(2.533 - 0.0636 * aux1)))
        cn3 = self.cn.value * np.exp(0.00673 * aux1)
        # computation of retention values for wilting point and 
        # field capacity - [1] equ. 2:1.1.2 with cn1 and cn3
        smax = 25.4 * ((1000.0 / cn1) - 10.0)
        s3 = 25.4 * ((1000.0 / cn3) - 10.0)
        # shape parameters of [1] equ. 2:1.1.6
        fc = self.fc.value * self.rdm.value
        sat = self.por.value * self.rdm.value
        aux2 = np.log((fc / (1.0 - (s3 / smax))) - fc)
        aux3 = np.log((sat / (1.0 - (2.54 / smax))) - sat)
        w2 = (aux2 - aux3) / (sat - fc)
        w1 = aux2 + w2 * fc
        # compute retention paramter - [1] equ. 2:1.1.6
        sw = self.sw.value - (self.wp.value * self.rdm.value)
        aux4 = np.exp(w1 - w2 * sw)
        self.retention = smax * (1.0 - (sw / (sw + aux4)))

        ########################################################################
        # compute runoff and infiltration
        nom = self.precipitation_soil - 0.2 * self.retention
        rotemp = np.power(
            self.precipitation_soil - 0.2 * self.retention, 2.0
        ) / (
            self.precipitation_soil + 0.8 * self.retention
        )
        self.runoff = np.where(nom >= 0.0, rotemp, 0.0)
        self.infiltration = self.precipitation_soil - self.runoff
