import numpy as np

from ..base import Model, Quantities as Q, Units as U
from ..requ import Requirement
from ..utils_soil import init_layer_dimensions
from .layer.model_swat import Layer_V2009
from .surface.model_swat import Surface_V2009
from .supply.model_epic import Supply as Supply_EPIC


class DEFAULT_VALUES:
    r"""
    Class which contains default values for quantities needed in `Soil_V2009`.

    **Bare soil albedo**

    Albedo values are taken from [R3]_ table 1 (row with `Fields, bare`).
    The max. value is a little bit decreased (compared to table 1) to be 
    closer to values in the sand and clay rows of table 1.

    Included default values (class variables):

    * *ALBEDO_DRY* :math:`= 0.22\ [\ ]`
    * *ALBEDO_WET* :math:`= 0.12\ [\ ]`
    * *ALBEDO_MEAN* :math:`= 0.17\ [\ ]`

    **Soil properties**

    Included default values (class variables):

    * *PARTICLE_DENSITY* :math:`= 2.65\ [\frac{g}{cm^3}]` - [R1]_ pp. 147
    * *CLAYVALS* :math:`= [0.03,\ 0.22,\ 0.47]\ [\ ]` - [R1]_ table 2:3-1
    * *WPVALS* :math:`= [0.02,\ 0.05,\ 0.2]\ [\ ]` - [R1]_ table 2:3-1
    * *FCVALS* :math:`= [0.06,\ 0.29,\ 0.41]\ [\ ]` - [R1]_ table 2:3-1
    * *SATVALS* :math:`= [0.4,\ 0.5,\ 0.6]\ [\ ]` - [R1]_ table 2:3-1

    *PARTICLE_DENSITY* is used in the `porosity_from_bulk_density()` function. 
    The other values are of type ``numpy.ndarray`` and are used to interpolate 
    the wilting point, field capacity and/or saturation from given clay content 
    (i.e. *CLAYVALS* corresponds to the x-values and *WPVALS*, *FCVALS* and 
    *SATVALS* are three variants of the y-values in interpolation algorithms).

    """
    ALBEDO_DRY = 0.22
    ALBEDO_WET = 0.12
    ALBEDO_MEAN = 0.17
    PARTICLE_DENSITY = 2.65  # [g/cm3] - [R1]_ paragraph below equ. 2:3.1.3
    # The following values are from - [R1]_ table 2:3-1
    CLAYVALS = np.array([0.03, 0.22, 0.47])
    FCVALS = np.array([0.06, 0.29, 0.41])
    WPVALS = np.array([0.02, 0.05, 0.20])
    SATVALS = np.array([0.4, 0.5, 0.6])


def porosity_from_bulk_density(bulk_dens):
    r"""
    Compute soil porosity from bulk density - [R1]_ equ. 2:3.1.3

    :param bulk_dens: soil bulk density :math:`s_{\textrm{bd},0}\ [\frac{g}{cm^3}]`
    :type bulk_dens: float or numpy.ndarray
    :return: soil porosity :math:`s_{\textrm{por},0}\ [\ ]` (fraction of total soil volume)
    :rtype: float or numpy.ndarray
    """
    return 1.0 - (bulk_dens / DEFAULT_VALUES.PARTICLE_DENSITY)  # [R1]_ equ. 2:3.1.3


def wilting_point_from_clay_content(clay_cont):
    r"""
    Compute water content at wilting point :math:`s_{\textrm{wp},0}\ [\ ]` from 
    given clay content with linear interpolation (``numpy.interp()``) using 
    the values from [R1]_ table 2:3-1.

    :param clay_cont: soil clay content :math:`s_{\textrm{clay},0}\ [\ ]` (percent of solids)
    :type clay_cont: float or numpy.ndarray
    :return: water content at wilting point :math:`s_{\textrm{wp},0}\ [\ ]` (fraction of total soil volume)
    :rtype: float or numpy.ndarray
    """
    return np.interp(clay_cont, DEFAULT_VALUES.CLAYVALS, DEFAULT_VALUES.WPVALS)


def field_capacity_from_clay_content(clay_cont):
    r"""
    Compute water content at field capacity :math:`s_{\textrm{fc},0}\ [\ ]` from 
    given clay content with linear interpolation (``numpy.interp()``) using 
    the values from [R1]_ table 2:3-1.

    :param clay_cont: soil clay content :math:`s_{\textrm{clay},0}\ [\ ]` (percent of solids)
    :type clay_cont: float or numpy.ndarray
    :return: water content at field capacity :math:`s_{\textrm{fc},0}\ [\ ]` (fraction of total soil volume)
    :rtype: float or numpy.ndarray
    """
    return np.interp(clay_cont, DEFAULT_VALUES.CLAYVALS, DEFAULT_VALUES.FCVALS)


def saturation_from_clay_content(clay_cont):
    r"""
    Compute saturated water content which equals soil porosity 
    :math:`s_{\textrm{por},0}\ [\ ]` from given clay content with linear 
    interpolation (``numpy.interp()``) using the values from [R1]_ table 2:3-1.

    :param clay_cont: soil clay content :math:`s_{\textrm{clay},0}\ [\ ]` (percent of solids)
    :type clay_cont: float or numpy.ndarray
    :return: saturation/porosity :math:`s_{\textrm{por},0}\ [\ ]` (fraction of total soil volume)
    :rtype: float or numpy.ndarray
    """
    return np.interp(clay_cont, DEFAULT_VALUES.CLAYVALS, DEFAULT_VALUES.SATVALS)


class Soil_V2009(Model):
    r"""

    Herein, soil porosity and water content at saturation are assumed to be 
    equal, which is also stated in [R1]_ section 2:3.1 in the paragraph above 
    table 2:3-1. Thus, this model only contains ``self.porosity`` as soil 
    property (random output) which is computed with ``self.clay_content`` 
    (hyper parameter) using ``saturation_from_clay_content()`` function (i.e. 
    values from table 2:3-1 [R1]_). Equ. 2:3.1.3 from [R1]_ is not used in this 
    model. 
    
    ``self.field_capacity`` and ``self.wilting_point`` (both random outputs) are 
    also computed using ``self.clay_content`` with the functions 
    ``wilting_point_from_clay_content()`` and ``field_capacity_from_clay_content()`` 
    which are also based on the values contained in table 2:3-1 [R1]_ .

    kwargs :math:`\rightarrow` :class:`mef_agri.models.base.Model`

    """
    LAYER_MODEL_NAMES = ['layer01', 'layer02', 'layer03', 'layer04', 'layer05']
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            self.layer01, self.layer02, self.layer03, self.layer04, self.layer05
        ]
        self._zv:np.ndarray = None

    ############################################################################
    # HYPER PARAMETERS (derived from external datasources, e.g. ebod or set by 
    # the user)
    @Model.is_quantity(Q.HPARAM, U.frac)
    def clay_content(self) -> np.ndarray:
        r"""
        MQ - Hyper Parameter

        :math:`s_{\textrm{clay},0}\ [\ ]`

        :return: clay content of soil as percent of solids
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.HPARAM, U.frac)
    def sand_content(self) -> np.ndarray:
        r"""
        MQ - Hyper Parameter

        :math:`s_{\textrm{sand},0}\ [\ ]`

        :return: sand content of soil as percent of solids
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.HPARAM, U.mm_day)
    def hydraulic_conductivity_sat(self) -> np.ndarray:
        r"""
        MQ - Hyper Parameter

        :math:`s_{\textrm{W-hcs},0}\ [\frac{mm}{day}]`

        :return: hydraulic conductivity of saturated soil 
        :rtype: numpy.ndarray
        """
    
    @Model.is_quantity(Q.HPARAM, U.m)
    def rooting_depth_max(self) -> np.ndarray:
        r"""
        MQ - Hyper Parameter

        :math:`s_{\textrm{rdm},0}\ [m]`

        :return: maximum rootable depth of the soil 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.HPARAM, U.g_cm3)
    def bulk_density(self) -> np.ndarray:
        r"""
        MQ - Hyper Parameter

        :math:`s_{\textrm{bd},0}\ [\frac{g}{cm^3}]`

        :return: bulk density of the soil
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.HPARAM, U.per_day)
    def decomposition_res_opt(self) -> np.ndarray:
        r"""
        MQ - Hyper Parameter

        :math:`s_{\textrm{C-rdo},0}\ [\frac{1}{day}]`

        :return: optimum decomposition rate of crop residual C-pool in percent per day - [R1]_ equ. 3:5.1.1.a
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.HPARAM, U.per_day)
    def mineralization_opt(self) -> np.ndarray:
        r"""
        MQ - Hyper Parameter

        :math:`s_{\textrm{C-omo},0}\ [\frac{1}{day}]`

        :return: optimum mineralization rate from organic C-pool in percent per day - [R1]_ equ. 3:5.1.7
        :rtype: numpy.ndarray
        """

    ############################################################################
    # HYPER PARAMETERS which are internally computed
    @Model.is_quantity(Q.ROUT, U.none)
    def curve_number_bare(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{cnb},0}\ [\ ]`

        :return: curve number of bare soil - [R1]_ table 2:1-1
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def porosity(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{por},0}\ [\ ]`

        :return: soil porosity as fraction of total soil volume  - computed with :func:`saturation_from_clay_content`
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def wilting_point(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-wp},0}\ [\ ]`

        :return: soil moisture at wilting point - computed with :func:`wilting_point_from_clay_content`
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.frac)
    def field_capacity(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-fc},0}\ [\ ]`

        :return: soil moisture at field capacity - computed with :func:`field_capacity_from_clay_content`
        :rtype: numpy.ndarray
        """

    ############################################################################
    # RANDOM OUTPUTS DERIVED FROM HYPER PARAMETERS
    @Model.is_quantity(Q.ROUT, U.none)
    def albedo(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{alb},k}\ [\ ]`

        The actual albedo value is adjusted by soil moisture of the first soil 
        layer. Sand and clay content of the soil are not considered, as the 
        differences are minor (similar values in sand and clay rows of table 1
        [R3]_ )

        :return: bare soil albedo adjusted by moisture of first soil layer
        :rtype: numpy.ndarray
        """
    
    @Model.is_quantity(Q.ROUT, U.none)
    def curve_number(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{cn},k}\ [\ ]`

        :return: curve number considering crops - [R1]_ table 2:1-1
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def hydraulic_conductivity_eff(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-hce},k}\ [\frac{mm}{day}]`

        :return: effective hydraulic conductivity - [R1]_ equ. 2:1.2.4
        :rtype: numpy.ndarray
        """

    ############################################################################
    # OTHER RANDOM OUTPUTS
    @Model.is_quantity(Q.ROUT, U.mm)
    def damping_depth(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{T-dd},k}\ [m]`

        :return: current damping depth of the soil - [R1]_ equ. 1:1.3.8
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def water_change(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-}\Delta\textrm{ta},k}\ [\frac{mm}{day}]`

        :return: change of the overall water amount at current day
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm)
    def water_overall(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-ta},k}\ [mm]`

        :return: current overall water amount (soil profile + canopy)
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm)
    def water_soil(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-a},k}\ [mm]`

        :return: current water amount in the soil profile
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def water_percolated(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-per},k}\ [\frac{mm}{day}]`

        :return: water that percolated from the soil profile into deeper layers at current day (i.e. water loss)
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def water_evaporated(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-evp},k}\ [\frac{mm}{day}]`

        :return: water that evaporated from the soil profile at current day (i.e. water loss)
        :rtype: np.ndarry
        """

    @Model.is_quantity(Q.ROUT, U.mm_day)
    def water_loss(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-l},k}\ [\frac{mm}{day}]`

        :return: water losses from the soil (surface-runoff + evaporation + percolation + crop-uptake)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.mm)
    def water_balance(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{W-bal},k}\ [mm]`

        :return: quantity to control water computations
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_ha)
    def NO3(self) -> np.ndarray:
        r"""
        MQ - Random output

        :math:`s_{\textrm{NO}_3^-,k}\ [\frac{kg}{ha}]`

        :return: sum of nitrate in the soil layers
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_ha)
    def NH4(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{NH}_4^+,k}\ [\frac{kg}{ha}]`

        :return: sum of ammonium in the soil layers
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_ha)
    def N_org(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{N-org},k}\ [\frac{kg}{ha}]`

        :return: sum of organic nitrogen in the soil layers
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_ha)
    def C_org(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{C-org},k}\ [\frac{kg}{ha}]`

        :return: sum of organic carbon (humus) in the soil layers
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.kg_ha)
    def C_res(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{C-res},k}\ [\frac{kg}{ha}]`

        :return: sum of carbon contained in crop residues in the soil layers
        :rtype: numpy.ndarray
        """

    ############################################################################
    # CHILD MODELS
    @Model.is_child_model(Layer_V2009)
    def layer01(self) -> Layer_V2009:
        """
        Child Model

        :return: first soil layer
        :rtype: Layer_V2009
        """

    @Model.is_child_model(Layer_V2009)
    def layer02(self) -> Layer_V2009:
        """
        Child Model
        
        :return: second soil layer
        :rtype: Layer_V2009
        """
        
    @Model.is_child_model(Layer_V2009)
    def layer03(self) -> Layer_V2009:
        """
        Child Model
        
        :return: third soil layer
        :rtype: Layer_V2009
        """

    @Model.is_child_model(Layer_V2009)
    def layer04(self) -> Layer_V2009:
        """
        Child Model
        
        :return: fourth soil layer
        :rtype: Layer_V2009
        """
    
    @Model.is_child_model(Layer_V2009)
    def layer05(self) -> Layer_V2009:
        """
        Child Model
        
        :return: fifth soil layer
        :rtype: Layer_V2009
        """

    @Model.is_child_model(Surface_V2009)
    def surface(self) -> Surface_V2009:
        """
        Child Model
        
        :return: soil surface model
        :rtype: Surface_V2009
        """

    ############################################################################
    # REQUIRED QUANTITIES
    @Model.is_required('water_canopy', 'zone.soil.surface.water', unit=U.mm)
    def wcan(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.surface.water'``

        :math:`s_{\textrm{W-c},k}\ [mm]`

        :return: amount of water stored in the canopy
        :rtype: Requirement
        """

    @Model.is_required('runoff', 'zone.soil.surface.water', unit=U.mm_day)
    def wro(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil.surface.water'``

        :math:`s_{\textrm{W-ro},k}\ [\frac{mm}{day}]`

        :return: runoff rate at current day 
        :rtype: Requirement
        """

    @Model.is_required('water', 'crop.uptake', unit=U.mm_day)
    def wupt(self) -> Requirement:
        r"""
        RQ - from model with id ``'crop.uptake'``

        :math:`c_{\textrm{W-upt},k}\ [\frac{mm}{day}]`

        :return: uptake of water at current day 
        :rtype: Requirement
        """

    @Model.is_required('precipitation_sum', 'zone.atmosphere.weather', unit=U.mm_m2xday)
    def prec(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.atmoshpere.weather'``

        :math:`a_{\textrm{prec},k}\ [\frac{mm}{m^2\cdot day}]`

        :return: daily precipitation sum 
        :rtype: Requirement
        """

    ############################################################################
    # METHODS
    def initialize(self, epoch):
        """
        Initialization of quantities and child models.
        The following random outputs act as hyper-parameters and are also 
        initialized within this method

        * :func:`albedo`
        * :func:`curve_number_bare`
        * :func:`wilting_point`
        * :func:`field_capacity`
        * :func:`porosity`
        
        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self._zv = np.zeros((self.model_tree.n_particles,))
        self._ov = np.ones((self.model_tree.n_particles,))
        # initialize soil albedo
        self._albmean = DEFAULT_VALUES.ALBEDO_MEAN * self._ov
        self._albdry = DEFAULT_VALUES.ALBEDO_DRY * self._ov
        self._albwet = DEFAULT_VALUES.ALBEDO_WET * self._ov
        self.albedo = self._albmean.copy()

        # initialize curve number of bare soil
        # mean value in [1] table 2:1-1 (row "bare soil")
        self.curve_number_bare = 86.0 + np.random.normal(
            loc=0.0, scale=3.0, size=self.model_tree.n_particles
        )

        # initialize outputs derived from hyper-parameters
        self.wilting_point = wilting_point_from_clay_content(self.clay_content)
        self.field_capacity = field_capacity_from_clay_content(self.clay_content)
        self.porosity = saturation_from_clay_content(self.clay_content)
        # variables for the adjustment of soil albedo
        dwpfc = self.field_capacity - self.wilting_point
        self._m025 = self.wilting_point + 0.25 * dwpfc
        self._m075 = self.wilting_point + 0.75 * dwpfc

        # initialize surface
        self.surface.initialize(epoch)
        # initialize layer dimensions
        self = init_layer_dimensions(self, self.rooting_depth_max)
        # intialize layer models
        for layer in self.layers:
            layer.initialize(epoch)

        self.initialize_supply(epoch)
        # initialize overall water amount and nutrients in soil profile
        self.water_soil = self._zv.copy()
        self.NO3, self.NH4 = self._zv.copy(), self._zv.copy()
        self.N_org = self._zv.copy()
        self.C_org, self.C_res = self._zv.copy(), self._zv.copy()
        for layer in self.layers:
            lwmid = layer.model_id + '.water'
            lnmid = layer.model_id + '.nutrients.nitrogen'
            lcmid = layer.model_id + '.nutrients.carbon'
            self.water_soil += self.model_tree.get_quantity(
                'amount', lwmid, unit=U.mm
            )
            self.NO3 += self.model_tree.get_quantity('NO3', lnmid, unit=U.kg_ha)
            self.NH4 += self.model_tree.get_quantity('NH4', lnmid, unit=U.kg_ha)
            self.N_org += self.model_tree.get_quantity(
                'N_org', lnmid, unit=U.kg_ha
            )
            self.C_org += self.model_tree.get_quantity(
                'C_org', lcmid, unit=U.kg_ha
            )
            self.C_res += self.model_tree.get_quantity(
                'C_res', lcmid, unit=U.kg_ha
            )
        self.water_evaporated = self._zv.copy()
        self.water_percolated = self._zv.copy()
        self.water_overall = self.water_soil.copy()
        self.water_change = self._zv.copy()
        self.water_loss = self._zv.copy()

    def update(self, epoch):
        """
        The following computations are performed

        * update the bare soil albedo :func:`albedo`
        * :func:`curve_number` equals :func:`curve_number_bare` if no crop is present, otherwise it is reduced by 10 (this value is an approximation derived from [R1]_ table 2:1-1)
        * :func:`hydraulic_conductivity_eff` [R1]_ equ. 2:1.2.4
        * update model :func:`surface`
        * :func:`damping_depth` [R1]_ equ. 1:1.3.6
        * update layer models (infiltration into first layer equals infiltrating water from the surface, for the other layers it is the percolated water from the above layer)
        * call :func:`update_supply`
        * compute overall water and nutrient amounts
        * compute water balance
        
        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        cp = self.model_tree.get_model('zone').crop_rotation.crop_present

        ########################################################################
        # update bare soil albedo
        cs_alb = [
            self.layer01.water.moisture <= self._m025,
            (self._m025 < self.layer01.water.moisture) & 
            (self.layer01.water.moisture < self._m075),
            self.layer01.water.moisture >= self._m075
        ]
        self.albedo = np.select(
            cs_alb, [self._albdry, self._albmean, self._albwet]
        )

        ########################################################################
        # update quantities derived from hyper parameters
        # the curve number of bare soil is reduced by 10 if a crop is present
        # the value 10 is an approximation derived from [1] table 2:1-1
        if cp:
            self.curve_number = self.curve_number_bare - 10.0
        else:
            self.curve_number = self.curve_number_bare.copy()
        # effective hydraulic conductivity - [1] equ. 2:1.2.4
        nom = 56.82 * np.power(self.hydraulic_conductivity_sat, 0.286)
        denom = 1.0 + 0.051 * np.exp(0.062 * self.curve_number)
        self.hydraulic_conductivity_eff = (nom / denom) - 2.0

        ########################################################################
        # update surface model
        self.surface.update(epoch)

        ########################################################################
        # computation of damping depth which is necessary for the soil temp.
        # max. damping depth [R1]_ equ. 1:1.3.6
        aux1 = np.exp(-5.63 * self.bulk_density)
        ddmax = 1000.0 + (
            (2500.0 * self.bulk_density) / (self.bulk_density + 686.0 * aux1)
        )
        # scaling factor [1] equ. 1:1.3.7
        scf = self.water_soil / (
            (0.356 - 0.144 * self.bulk_density) * self.rooting_depth_max
        )
        # current damping depth [1] equ. 1:1.3.8
        self.damping_depth = ddmax * np.exp(
            np.log(500. / ddmax) * np.power((1 - scf) / (1 + scf), 2.0)
        )

        ########################################################################
        # update layers
        first_layer = True
        for layer in self.layers:
            if first_layer:
                winfl = self.model_tree.get_quantity(
                    'infiltration', 'zone.soil.surface.water', unit=U.mm_day
                )
                first_layer = False

            # infiltration into layer and update
            lwmid = layer.model_id + '.water'
            self.model_tree.set_quantity(
                'infiltrated', lwmid, winfl, unit=U.mm_day
            )
            layer.update(epoch)

            # update infiltration water for next layer
            winfl = self.model_tree.get_quantity(
                'percolation', lwmid, unit=U.mm_day
            )

        self.water_percolated = winfl.copy()

        ########################################################################
        # determine water, nutrient, ... supply of soil for the crop
        self.update_supply(epoch)

        ########################################################################
        # determine overall water and nutrient amounts after crop supply 
        # computation
        self.water_soil = self._zv.copy()
        self.water_evaporated = self._zv.copy()
        self.NO3, self.NH4 = self._zv.copy(), self._zv.copy()
        self.N_org = self._zv.copy()
        self.C_org, self.C_res = self._zv.copy(), self._zv.copy()
        for layer in self.layers:
            lwmid = layer.model_id + '.water'
            lnmid = layer.model_id + '.nutrients.nitrogen'
            lcmid = layer.model_id + '.nutrients.carbon'
            self.water_soil += self.model_tree.get_quantity(
                'amount', lwmid, unit=U.mm
            )
            self.water_evaporated += self.model_tree.get_quantity(
                'evaporation', lwmid, unit=U.mm_day
            )
            self.NO3 += self.model_tree.get_quantity('NO3', lnmid, unit=U.kg_ha)
            self.NH4 += self.model_tree.get_quantity('NH4', lnmid, unit=U.kg_ha)
            self.N_org += self.model_tree.get_quantity(
                'N_org', lnmid, unit=U.kg_ha
            )
            self.C_org += self.model_tree.get_quantity(
                'C_org', lcmid, unit=U.kg_ha
            )
            self.C_res += self.model_tree.get_quantity(
                'C_res', lcmid, unit=U.kg_ha
            )

        ########################################################################
        # water balance
        self.water_change = self.water_soil + self.wcan.value
        self.water_change -= self.water_overall
        self.water_overall += self.water_change
        self.water_loss = self.wro.value + self.water_evaporated
        self.water_loss += self.water_percolated
        if cp:
            if not np.isin(True, np.isnan(self.wupt.value)):
                self.water_loss += self.wupt.value
        self.water_balance = self.water_change - (
            self.prec.value - self.water_loss)

    def initialize_supply(self, epoch) -> None:
        """
        Method which has to be implemented in a child class of 
        :class:`Soil_V2009`.
        It is called in 
        :func:`ssc_csm.models.soil.model_swat.Soil_V2009.initialize`.

        Thus, it is possible to develop different variants of supply mechanisms 
        (water, nutrients) within the same soil model.
        The reason is, that the supply processes are likely defined in the crop 
        growth models in the literature but logically they belong to the soil 
        model.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        pass

    def update_supply(self, epoch) -> None:
        """
        Method which has to be implemented in a child class of 
        :class:`Soil_V2009`.
        It is called in 
        :func:`ssc_csm.models.soil.model_swat.Soil_V2009.update`.

        Thus, it is possible to develop different variants of supply mechanisms 
        (water, nutrients) within the same soil model.
        The reason is, that the supply processes are likely defined in the crop 
        growth models in the literature but logically they belong to the soil 
        model.

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        pass


class Soil_V2009_EPIC(Soil_V2009):
    r"""
    ``Soil_V2009`` model which implements the supply process from the EPIC crop 
    growth model [R2]_.

    kwargs :math:`\rightarrow` :class:`ssc_csm.models.base.Model`
    """
    WUDF_DEF = 2.0

    @Model.is_quantity(Q.HPARAM, U.none)
    def water_use_distribution_factor(self) -> np.ndarray:
        r"""
        MQ - Hyper-Parameter

        :math:`s_{\textrm{W-udf},0}\ [\ ]` [R2]_ equ. 19 - 22

        :return: water use distribution factor 
        :rtype: numpy.ndarray
        """

    @Model.is_child_model(Supply_EPIC)
    def supply(self) -> Supply_EPIC:
        """
        Child Model

        :return: model to determine supplied water, nutrients, ... for the crop
        :rtype: :class:`ssc_csm.models.soil.supply.model_epic.Supply`
        """

    def initialize_supply(self, epoch):
        """
        Initializes ``water_use_distribution_factor`` if not already set and 
        calls :func:`ssc_csm.models.soil.supply.model_epic.Supply.initialize`.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        if self.water_use_distribution_factor is None:
            self.water_use_distribution_factor = self.WUDF_DEF * np.ones(
                (self.model_tree.n_particles,)
            )
        self.supply.initialize(epoch)

    def update_supply(self, epoch):
        """
        Calls :func:`ssc_csm.models.soil.supply.model_epic.Supply.update`.

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        self.supply.update(epoch)
