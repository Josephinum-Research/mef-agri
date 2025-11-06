import numpy as np
from datetime import date

from .base import Zoning
from ..interfaces.geosphere_austria.inca_interfaces import (
    INCA_AtmosphereSWAT_JRV1
)
from ..interfaces.ebod_austria.ebod_interfaces import EBOD_SoilSWAT_JRV1
from ..interfaces.planetary_computer.lai.interfaces_inrae import (
    PCSentinel2_NNET10
)
from ..interfaces.jr_management.interface import Management_JRV01
from ...farming.tasks.zoning import Zoning as ZoningTask
from ...utils.gis import mean_latitude_wgs84
from ...models.zone.model_jr import ZoneJR_V1
from ...data.jr_management.interface import ManagementInterface
from ...models.base import Quantities as Q
from ...models.utils_soil import distribute_nutrient_amount
from ...models.soil.model_swat import (
    wilting_point_from_clay_content, field_capacity_from_clay_content, 
    Soil_V2009_EPIC
)
from ..stats_utils import DISTRIBUTIONS
from ..eval_def import EvaluationDefinitions as ED


class ZoningJR_V01(Zoning):
    def __init__(self, prj):
        super().__init__(prj, ZoneJR_V1)
        self.add_interface(INCA_AtmosphereSWAT_JRV1)
        self.add_interface(EBOD_SoilSWAT_JRV1)
        self.add_interface(PCSentinel2_NNET10)
        self.add_interface(Management_JRV01)

        self._zd:date = None

    @property
    def zoning_date(self) -> date:
        """
        :return: set date of zoning task which should be used to specify evaluation definitions
        :rtype: datetime.date
        """
        return self._zd
    
    @zoning_date.setter
    def zoning_date(self, val):
        self._zd = val

    def determine_zones(self, field_name):
        tasks = self.project.get_data(
            self.zoning_date, dids=ManagementInterface.DATA_SOURCE_ID,
            fields=field_name
        )
        ztask:ZoningTask = tasks[
            field_name][ManagementInterface.DATA_SOURCE_ID][ZoningTask.__name__]
        zones = {}
        for zname in ztask.layer_ids:
            ics = np.where(ztask[zname])
            ics = ics[1:] if ztask.layer_index == 0 else ics[:2]
            ics_hom = np.vstack((
                np.atleast_2d(ics[1]), 
                np.atleast_2d(ics[0]), 
                np.ones((1, len(ics[0])))
            ))
            gcs = (ztask.transformation @ ics_hom)[:2, :]
            lat = mean_latitude_wgs84(gcs, ztask.crs)
            zones[zname] = {'lat': lat, 'gcs': gcs}

        return zones
    
    def init_soil_moisture(self, ed:ED, frac:float, std:float) -> ED:
        """
        Initialize soil moisture of soil layers.

        :param ed: instance of `EvaluationDefinitions`
        :type ed: mef_agri.evaluation.eval_def.EvaluationDefinitions
        :param frac: fraction of the range between wilting point and field capacity which is filled with water
        :type frac: float
        :param std: standard deviation of ``frac``
        :type std: float
        """
        for zid in ed['zone_models'].keys():
            # derive field capacity and wilting point for current zone
            clayc = ed.get_qinfos_from_zone_model(
                zid, Q.PARAM, 'zone.soil', 'clay_content'
            )['value']
            wp = wilting_point_from_clay_content(clayc)
            fc = field_capacity_from_clay_content(clayc)

            # update the qinfo dict for the corresponding state
            for lname in Soil_V2009_EPIC.LAYER_MODEL_NAMES:
                mname = 'zone.soil.' + lname + '.water'
                sminfo = ed.get_qinfos_from_zone_model(
                    zid, Q.STATE, mname, 'moisture'
                )
                sminfo['value'] = round(wp + frac * (fc - wp), 3)
                sminfo['epoch'] = date.fromisoformat(sminfo['epoch'])
                sminfo['distr'] = {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': std
                }
                ed.set_zone_qinfos(
                    zid, Q.STATE, mname, 'moisture', sminfo
                )

        return ed

    def init_temp_values(self, ed:ED, std:float) -> ED:
        r"""
        Initialize soil layer temperatures with temparature observations from 
        ``'zone.atmosphere.weather'``. First an annual mean air temperature is 
        computed which is used afterwards for the soil layers.

        :param ed: instance of `EvaluationDefinitions`
        :type ed: mef_agri.evaluation.eval_def.EvaluationDefinitions
        :param std: standard deviation of the annual mean air temperature in :math:`[^{\circ}C]`
        :type std: float
        """
        for zid in ed['zone_models'].keys():
            # compute mean value of daily tempearture means in used data
            wobs = ed['zone_models'][zid]['observation']
            wobs = wobs['zone.atmosphere.weather']
            tavg = np.array(wobs['temperature_mean']['values']).mean()

            # set the annual mean temperature
            tainfo = ed.get_qinfos_from_zone_model(
                zid, Q.PARAM, 'zone.atmosphere.weather', 
                'temperature_mean_annual'
            )
            tainfo['value'] = round(tavg, 1)
            tainfo['epoch'] = date.fromisoformat(tainfo['epoch'])
            tainfo['distr'] = {
                'distr_id': DISTRIBUTIONS.NORMAL_1D,
                'std': std
            }
            ed.set_zone_qinfos(
                zid, Q.PARAM, 'zone.atmosphere.weather', 
                'temperature_mean_annual', tainfo
            )

            # set the initial soil temperature values
            for lname in Soil_V2009_EPIC.LAYER_MODEL_NAMES:
                mname = 'zone.soil.' + lname + '.temperature'
                stinfo = ed.get_qinfos_from_zone_model(
                    zid, Q.STATE, mname, 'temperature'
                )
                stinfo['value'] = round(tavg, 1)
                stinfo['epoch'] = date.fromisoformat(stinfo['epoch'])
                stinfo['distr'] = {
                    'distr_id': DISTRIBUTIONS.NORMAL_1D,
                    'std': std
                }
                ed.set_zone_qinfos(
                    zid, Q.STATE, mname, 'temperature', stinfo
                )
        return ed

    def init_n_amounts(
            self, ed:ED, no3:float, no3_std:float, nh4:float, nh4_std:float,
        ) -> ED:
        r"""
        Initialize amounts of nitrate and ammonia in the soil laters using 
        :func:`ssc_csm.models.utils_soil.distribute_nutrient_amount`.

        :param ed: instance of `EvaluationDefinitions`
        :type ed: mef_agri.evaluation.eval_def.EvaluationDefinitions
        :param no3: amount of :math:`NO_{3}^{-}` in :math:`[\frac{kg}{ha}]`
        :type no3: float
        :param no3_std: standard deviation of ``no3``
        :type no3_std: float
        :param nh4: amount of :math:`NH_{4}^{+}` in :math:`[\frac{kg}{ha}]`
        :type nh4: float
        :param nh4_std: standard deviation of ``nh4``
        :type nh4_std: float
        """
        frs = distribute_nutrient_amount(len(Soil_V2009_EPIC.LAYER_MODEL_NAMES))
        for zid in ed['zone_models'].keys():
            for lname, fr in zip(Soil_V2009_EPIC.LAYER_MODEL_NAMES, frs):
                mname = 'zone.soil.' + lname + '.nutrients.nitrogen'

                no3info = ed.get_qinfos_from_zone_model(
                    zid, Q.STATE, mname, 'NO3'
                )
                no3info['value'] = no3 * fr
                no3info['epoch'] = date.fromisoformat(no3info['epoch'])
                no3info['distr'] = {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': no3_std
                }
                ed.set_zone_qinfos(
                    zid, Q.STATE, mname, 'NO3', no3info
                )

                nh4info = ed.get_qinfos_from_zone_model(
                    zid, Q.STATE, mname, 'NH4'
                )
                nh4info['value'] = nh4 * fr
                nh4info['epoch'] = date.fromisoformat(nh4info['epoch'])
                nh4info['distr'] = {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': nh4_std
                }
                ed.set_zone_qinfos(
                    zid, Q.STATE, mname, 'NH4', nh4info
                )
        return ed

    def init_org_amounts(
            self, ed:ED, corg_frac:float, norg_frac:float, c_std_scale:float,
            n_std_scale:float
        ) -> ED:
        r"""
        Initialize organic carbon and organic nitrogen in the soil layers using 
        :func:`ssc_csm.models.utils_soil.distribute_nutrient_amount`. First the 
        amount of organic carbon is computed in :math:`\frac{kg}{ha}` by using 
        bulk density and max. rooting depth. From this value, organic nitrogen 
        computed by using ``norg_frac``.

        :param ed: instance of `EvaluationDefinitions`
        :type ed: mef_agri.evaluation.eval_def.EvaluationDefinitions
        :param corg_frac: fraction of the soil which represents organic carbon
        :type corg_frac: float
        :param norg_frac: fraction of organic carbon which represents organic nitrogen
        :type norg_frac: float
        :param c_std_scale: value which is multiplied with the amount of organic carbon in current soil layer to derive the absolute standard deviation 
        :type c_std_scale: float
        :param n_std_scale: value which is multiplied with the amount of organic nitrogen in current soil layer to derive the absolute standard deviation
        :type n_std_scale: float
        """
        frs = distribute_nutrient_amount(len(Soil_V2009_EPIC.LAYER_MODEL_NAMES))
        for zid in ed['zone_models'].keys():
            sbd = ed.get_qinfos_from_zone_model(
                zid, Q.PARAM, 'zone.soil', 'bulk_density'
            )['value']
            rdm = ed.get_qinfos_from_zone_model(
                zid, Q.PARAM, 'zone.soil', 'rooting_depth_max'
            )['value']
            corg = corg_frac * sbd * rdm * 1e4 * 1e3  # 1e4 ... m2 > ha / 1e3 ... t > kg
            norg = norg_frac * corg
            for lname, fr in zip(Soil_V2009_EPIC.LAYER_MODEL_NAMES, frs):
                mname = 'zone.soil.' + lname + '.nutrients.carbon'
                corginfo = ed.get_qinfos_from_zone_model(
                    zid, Q.STATE, mname, 'C_org'
                )
                corginfo['value'] = corg * fr
                corginfo['epoch'] = date.fromisoformat(corginfo['epoch'])
                corginfo['distr'] = {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': c_std_scale * corg * fr
                }
                ed.set_zone_qinfos(
                    zid, Q.STATE, mname, 'C_org', corginfo
                )

                mname = 'zone.soil.' + lname + '.nutrients.nitrogen'
                norginfo = ed.get_qinfos_from_zone_model(
                    zid, Q.STATE, mname, 'N_org'
                )
                norginfo['value'] = norg * fr
                norginfo['epoch'] = date.fromisoformat(norginfo['epoch'])
                norginfo['distr'] = {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': n_std_scale * norg * fr
                }
                ed.set_zone_qinfos(
                    zid, Q.STATE, mname, 'N_org', norginfo
                )
        return ed

    def set_decomposition_rates(self, ed:ED) -> ED:
        """
        Setting default values for decomposition rate of crop residues and 
        mineralization rate according to [R1]_ .

        :param ed: instance of `EvaluationDefinitions`
        :type ed: mef_agri.evaluation.eval_def.EvaluationDefinitions
        """
        for zid in ed['zone_models'].keys():
            ie = ed.get_qinfos_from_zone_model(
                zid, Q.PARAM, 'zone.soil', 
                'clay_content'
            )['epoch']
            dro = {
                'value': 0.05,
                'epoch': date.fromisoformat(ie),
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 0.005,
                    'lb': 0.0, 
                    'ub': 1.0
                }
            }
            mo = {
                'value': 0.0001,
                'epoch': date.fromisoformat(ie),
                'distr': {
                    'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                    'std': 5e-5,
                    'lb': 0.0,
                    'ub': 1.0
                }
            }
            ed.set_zone_qinfos(
                zid, Q.PARAM, 'zone.soil', 'decomposition_res_opt',dro
            )
            ed.set_zone_qinfos(
                zid, Q.PARAM, 'zone.soil', 'mineralization_opt', mo
            )
        return ed

    def init_cropres_amounts(
            self, ed:ED, cres:float, cres_std:float, cnres:float, 
            cnres_std:float
        ) -> ED:
        r"""
        Initialize the amount of crop residues and the corresponding C/N ratio. 
        The distribution to the layers is done with 
        :func:`ssc_csm.models.utils_soil.distribute_nutrient_amount`.

        :param ed: instance of `EvaluationDefinitions`
        :type ed: mef_agri.evaluation.eval_def.EvaluationDefinitions
        :param cres: amount of crop residues in :math:`\frac{kg}{ha}`
        :type cres: float
        :param cres_std: standard deviation of ``cres``
        :type cres_std: float
        :param cnres: C/N ratio of the crop residuals
        :type cnres: float
        :param cnres_std: standard deviation of ``cnres``
        :type cnres_std: float
        """
        frs = distribute_nutrient_amount(len(Soil_V2009_EPIC.LAYER_MODEL_NAMES))
        for zid in ed['zone_models'].keys():
            for lname, fr in zip(Soil_V2009_EPIC.LAYER_MODEL_NAMES, frs):
                mname = 'zone.soil.' + lname + '.nutrients.carbon'
                
                cresinfo = ed.get_qinfos_from_zone_model(
                    zid, Q.STATE, mname, 'C_res'
                )
                cresinfo['value'] = cres * fr
                cresinfo['epoch'] = date.fromisoformat(cresinfo['epoch'])
                cresinfo['distr'] = {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': cres_std
                }
                ed.set_zone_qinfos(
                    zid, Q.STATE, mname, 'C_res', cresinfo
                )

                mname = 'zone.soil.' + lname + '.nutrients'
                cnrinfo = ed.get_qinfos_from_zone_model(
                    zid, Q.STATE, mname, 'CN_res'
                )
                cnrinfo['value'] = cnres
                cnrinfo['epoch'] = date.fromisoformat(cnrinfo['epoch'])
                cnrinfo['distr'] = {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': cnres_std
                }
                ed.set_zone_qinfos(
                    zid, Q.STATE, mname, 'CN_res', cnrinfo
                )
        return ed

    def init_wudf(self, ed:ED, value:float, std:float) -> ED:
        """
        TODO

        :param ed: instance of `EvaluationDefinitions`
        :type ed: mef_agri.evaluation.eval_def.EvaluationDefinitions
        :param value: _description_
        :type value: float
        :param std: _description_
        :type std: float
        """
        for zid in ed['zone_models'].keys():
            winfo = ed.get_qinfos_from_zone_model(
                zid, Q.PARAM, 'zone.soil', 
                'water_use_distribution_factor'
            )
            winfo['epoch'] = date.fromisoformat(winfo['epoch'])
            winfo['value'] = value
            winfo['distr'] = {
                'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                'std': std,
                'lb': 1.0,
                'ub': 10.0,
                'sample': False
            }
            ed.set_zone_qinfos(
                zid, Q.PARAM, 'zone.soil', 
                'water_use_distribution_factor', winfo
            )
        return ed
