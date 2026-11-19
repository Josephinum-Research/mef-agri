import numpy as np

from ...base import Model, Quantities as Q
from ...utils import Units as U
from ...requ import Requirement
from ....farming.crops import Crops
from ....evaluation.stats_utils import DISTRIBUTIONS as D


class StemLodging(Model):
    GRAVITY_CONST = 9.80885
    AIR_DENSITY = ...
    DEFAULT_PARAM_VALUES = {
        Crops.WinterWheat.NAME: {
            'height_canopy_mass_center': {
                'value': 0.5,  # guess
                'distr': {
                    'distr_id': D.GAMMA_1D,
                    'std': 0.05
                }
            },
            'stems_per_plant': {
                'value': 3.2,  # baker-paper
                'distr': {
                    'distr_id': D.GAMMA_1D,
                    'std': 0.1
                }
            },
            'young_modulus': {
                'value': ...,
                'distr': {
                    'distr_id': D.GAMMA_1D,
                    'std': 1.
                }
            },
            'second_moment_area': {
                'value': ...,
                'distr': {
                    'distr_id': D.GAMMA_1D,
                    'std': ...
                }
            },
            'yield_stress': {
                'value': 3e6,  # [Pa] baker-paper
                'distr': {
                    'distr_id': D.GAMMA_1D,
                    'std': 1.
                }
            },
            'natural_frequency': {
                'value': 0.95,  # [Hz] baker-paper
                'distr': {
                    'distr_id': D.GAMMA_1D,
                    'std': 0.05
                }
            },
            'damping_coeff': {
                'value': 0.1,  # [ ] baker-paper
                'distr': {
                    'distr_id': D.NORMAL_1D,
                    'std': 0.003
                }
            },
            'shear_area': {
                'value': 0.025,  # [m2] baker-paper
                'distr': {
                    'distr_id': D.GAMMA_1D,
                    'std': 0.002
                }
            }
        }
    }

    @Model.is_quantity(Q.PARAM, U.Pa)
    def young_modulus(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{S-ym},0}\ [Pa]`

        :return: young modulus of stem (ratio of compressive force per area and axial strain/relative deformation)
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.m4)
    def second_moment_area(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{S-ma},0}\ [m^4]`

        :return: second moment of area of stem
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.Pa)
    def yield_stress(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{S-ys},0}\ [Pa]`

        :return: stem yield stress (i.e. force per area)
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.Hz)
    def natural_frequency(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{S-nf},0}\ [Hz]`

        :return: natural frequency of plant/stem when driven by wind
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def damping_coeff(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{S-dc},0}\ [\ ]`

        :return: generalized damping coefficient of plant/stem when driven by wind
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.m2)
    def shear_area(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{S-sa},0}\ [m^2]`

        :return: shear area of plant/stem
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.frac)
    def height_canopy_mass_center(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{hcmc},0}\ [\ ]`

        :return: fraction of plant height where the center of canopy mass is located
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.PARAM, U.none)
    def stems_per_plant(self) -> np.ndarray:
        r"""
        MQ - Parameter

        :math:`c_{\textrm{S-pp},0}\ [\ ]`

        :return: number of stems per plant
        :rtype: np.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def lodging_bottom(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :return: indicator for lodging at the bottom of the plant which corresponds to 10% of crop-height (yes if it exceeds 1)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def lodging_middle(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :return: indicator for lodging at the middle of the plant which corresponds to 50% of crop-height (yes if it exceeds 1)
        :rtype: numpy.ndarray
        """

    @Model.is_quantity(Q.ROUT, U.none)
    def lodging_top(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :return: indicator for lodging at the top of the plant which corresponds to 90% of crop-height (yes if it exceeds 1)
        :rtype: numpy.ndarray
        """

    @Model.is_required('height', 'crop', U.m)
    def hcrop(self) -> Requirement:
        """
        RQ - TODO

        :return: _description_
        :rtype: Requirement
        """

    @Model.is_required('biomass', 'crop.leaves', U.kg_ha)
    def bmlvs(self) -> Requirement:
        """
        RQ - TODO

        :return: _description_
        :rtype: Requirement
        """

    @Model.is_required('biomass', 'crop.cyield', U.kg_ha)
    def bmyld(self) -> Requirement:
        """
        RQ - TODO

        :return: _description_
        :rtype: Requirement
        """

    @Model.is_required('wind_turbulence', 'zone.atmosphere.weather', U.none)
    def wturb(self) -> Requirement:
        """
        RQ - TODO

        :return: _description_
        :rtype: Requirement
        """

    @Model.is_required('wind_speed_mean', 'zone.atmosphere.weather', U.m_s)
    def wvel(self) -> Requirement:
        """
        RQ - TODO

        :return: _description_
        :rtype: Requirement
        """

    @Model.is_required('radius', 'crop.stem', U.m)
    def srad(self) -> Requirement:
        """
        RQ - TODO

        :return: _description_
        :rtype: Requirement
        """

    @Model.is_required('thickness', 'crop.stem', U.m)
    def sthick(self) -> Requirement:
        """
        RQ - TODO

        :return: _description_
        :rtype: Requirement
        """

    @Model.is_required('density', 'crop', U.n_m2)
    def pdens(self) -> Requirement:
        """
        RQ - TODO

        :return: number of plants per unit area
        :rtype: Requirement
        """

    def update(self, epoch):
        super().update(epoch)
        
        # height of canopy mass center in [m]
        hcmc = self.height_canopy_mass_center * self.hcrop.value
        # biomass of canopy (leaves + yield)
        bmcan = self.bmlvs.value + self.bmyld.value
        # auxiliary variables to compute lodging indicator
        anom = bmcan * self.GRAVITY_CONST * np.power(hcmc, 2)
        adenom = self.young_modulus * self.second_moment_area
        adenom *= self.stems_per_plant * self.pdens.value
        alpha = anom / adenom
        aux1 = np.power(self.natural_frequency * 2. * np.pi, 2) * (
            hcmc / self.GRAVITY_CONST
        )
        aux2 = self.AIR_DENSITY * self.shear_area * self.wvel * self.wvel * hcmc
        aux2 *= 0.5
        aux3 = 1 + 6.86 * self.wturb.value * np.power(
            1. + 0.087 * (np.pi / (4. * self.damping_coeff)), 
            0.5
        )
        # lodging moment per plant
        ml = np.pi * self.yield_stress * (np.power(self.srad.value, 3.) / 4.)
        ml *= 1. - np.power(
            (self.srad.value - self.sthick.value) / self.srad.vlaue,
            4.
        ) * self.stems_per_plant

        # function to compute lodging indicator at different heights
        def lodging_indicator(frac_ch:float) -> np.ndarray:
            """
            :param frac_ch: relative position on crop along stem (i.e. fraction of crop height where lodging indicator is computed)
            :type frac_ch: float
            :return: lodging indicator
            :rtype: np.ndarray
            """
            aux4 = alpha * frac_ch
            aux5 = np.cos(aux4) - np.sin(aux4) / np.tan(alpha)
            # moment induced by mean wind speed
            mwm = ((1. + aux1) / aux1) * aux2 * aux5
            # moment induced by wind (also considering fluctuation/turbulence)
            mw = mwm * aux3
            return mw / ml

        self.lodging_bottom = lodging_indicator(0.1)
        self.lodging_middle = lodging_indicator(0.5)
        self.lodging_top = lodging_indicator(0.9)
