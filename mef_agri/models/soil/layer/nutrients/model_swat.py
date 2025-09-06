import numpy as np

from ....base import Model, Quantities as Q, Units as U
from ....requ import Requirement
from .nitrogen.model_swat import N_NO3_NH4_V2009, N_NO3_NH4_Norg_V2009
from .carbon.model_swat import C_Corg_Cres_V2009


class Nutrients_N_V2009(Model):
    r"""
    Nutrient model which contains a nitrogen model with a :math:`NO_3^-`,
    a :math:`NH_4^+` pool and nitrification process.

    kwargs :math:`\rightarrow` :class:`ssc_csm.models.base.Model`

    """
    @Model.is_child_model(N_NO3_NH4_V2009)
    def nitrogen(self) -> N_NO3_NH4_V2009:
        """
        Child Model

        :return: nitrogen model which considers NO3, NH4 and nitrification
        :rtype: N_NO3_NH4_V2009
        """

    def initialize(self, epoch):
        """
        Initialization of the nitrogen model.

        :param epoch: intialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self.nitrogen.initialize(epoch)

    def update(self, epoch):
        """
        Update of the nitrogen model

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)
        self.nitrogen.update(epoch)


class Nutrients_C_N_V2009(Model):
    @Model.is_quantity(Q.STATE, U.none)
    def CN_res(self) -> np.ndarray:
        r"""
        MQ - State

        :math:`s_{\textrm{CN-r},i,k}\ [\ ]`

        :return: C/N ratio of crop residues in the soil layer 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def CN_org(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{CN-o},i,k}\ [\ ]` - [R1]_ (equ. 3:5.1.5)

        :return: C/N ratio of organic matter in the soil layer 
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def nutrient_cycling_factor(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`s_{\textrm{ncf},i,k}\ [\ ]` - [R1]_ (equ. 3:1.2.1, 3:1.2.2)

        :return: factor combining soil temperatur and water influence on nutrient cycle 
        :rtype: numpy.ndarray
        """

    @Model.is_required('temperature', '.__parent__.temperature', unit=U.degC)
    def ltemp(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__.temperature'`` (temperature model of parent layer)

        :math:`s_{\textrm{T-t},i,k}\ [^\circ C]`

        :return: soil layer temperature 
        :rtype: Requirement
        """

    @Model.is_required('amount', '.__parent__.water', unit=U.mm)
    def lwa(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__.water'`` (water model of parent layer)

        :math:`s_{\textrm{W-a},i,k}\ [mm]`

        :return: water amount of soil layer 
        :rtype: Requirement
        """

    @Model.is_required('thickness', '.__parent__', unit=U.mm)
    def lt(self) -> Requirement:
        r"""
        RQ - from model with id ``'.__parent__'`` (parent layer)

        :math:`s_{\textrm{lt},i,0}\ [mm]`

        :return: layer thickness 
        :rtype: Requirement
        """

    @Model.is_required('field_capacity', 'zone.soil', unit=U.frac)
    def fc(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{W-fc},0}\ [\ ]`

        :return: soil moisture at field capacity 
        :rtype: Requirement
        """

    @Model.is_required('bulk_density', 'zone.soil', unit=U.kg_m3)
    def bd(self) -> Requirement:
        r"""
        RQ - from model with id ``'zone.soil'``

        :math:`s_{\textrm{bd},0}\ [\frac{g}{cm^3}]`

        :return: bulk density of the soil 
        :rtype: Requirement
        """

    @Model.is_required('NO3', '.nitrogen', unit=U.kg_ha)
    def no3(self) -> Requirement:
        r"""
        RQ - from model with id ``'.nitrogen'`` (child nitrogen model)

        :math:`s_{\textrm{NO}_{3}^{-},i,k}\ [\frac{kg}{ha}]`

        :return: amount of nitrate in the soil layer 
        :rtype: Requirement
        """

    @Model.is_required('NH4', '.nitrogen', unit=U.kg_ha)
    def nh4(self) -> Requirement:
        r"""
        RQ - from model with id ``'.nitrogen'`` (child nitrogen model)

        :math:`s_{\textrm{NH}_{4}^{+},i,k}\ [\frac{kg}{ha}]`

        :return: amount of ammonium in the soil layer 
        :rtype: Requirement
        """

    @Model.is_child_model(N_NO3_NH4_Norg_V2009)
    def nitrogen(self) -> N_NO3_NH4_Norg_V2009:
        """
        Child Model

        :return: nitrogen model which considers NO3, NH4 and nitrification as well as organic N-pool, mineralization and residue decomposition
        :rtype: N_NO3_NH4_Norg_V2009
        """

    @Model.is_child_model(C_Corg_Cres_V2009)
    def carbon(self) -> C_Corg_Cres_V2009:
        """
        Child Model

        :return: carbon model containing C-pools for humus and crop residuals
        :rtype: C_Corg_Cres_V2009
        """

    def initialize(self, epoch):
        """
        Initialization of nitrogen and carbon model.

        :param epoch: initialization epoch
        :type epoch: datetime.date
        """
        super().initialize(epoch)
        self.carbon.initialize(epoch)
        self.nitrogen.initialize(epoch)

    def update(self, epoch):
        """
        The carbon and nitrogen models are updated and the following 
        computations are performed

        * nutrient cycling temp. factor - [R1]_ (equ. 3:1.2.1)
        * nutrient cycling water factor - [R1]_ (equ. 3:1.2.2)
        * combined effect ``ncf = np.sqrt(ncf_t * ncf_w)``
        * CN-ratio of organic matter - [R1]_ (equ. 3:5.1.5)

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        super().update(epoch)

        ##### NUTRIENT CYCLING FACTOR ##########################################
        # nutrient cycling temperature factor - [R1]_ equ. 3:1.2.1
        ftemp = 0.9 * (self.ltemp.value / (self.ltemp.value + np.exp(
            9.93 - 0.312 * self.ltemp.value
        ))) + 0.1
        ftemp = np.where(ftemp >= 0.1, ftemp, 0.1)
        # nutrient cycling water factor - [1] equ. 3:1.2.2
        fwatr = self.lwa.value / (self.fc.value * self.lt.value)
        fwatr = np.where(fwatr >= 0.05, fwatr, 0.05)
        # combined effect
        self.nutrient_cycling_factor = np.sqrt(ftemp * fwatr)

        ##### CN RATIO OF ORGANIC MATTER #######################################
        # implementation of [1] equ. 3:5.1.5
        c1 = 8. * self.bd.value * self.lt.value * 1e-4  # concentration [mg/kg] to amount [kg/ha]
        nmin = self.no3.value + self.nh4.value
        ccres = 1. - (1. / (1. + np.power(self.CN_res / 110., 3.0)))
        cnmin = 1. + (1. / (1. + np.power(nmin / c1, 3.0)))
        self.CN_org = 8.5 + 2.7 * ccres * cnmin
        self.CN_org = np.where(self.CN_org >= 8.5, self.CN_org, 8.5)
        self.CN_org = np.where(self.CN_org <= 14., self.CN_org, 14.)

        ##### UPDATING SUB-MODELS ##############################################
        self.carbon.update(epoch)
        self.nitrogen.update(epoch)
