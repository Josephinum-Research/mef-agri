import numpy as np
from copy import deepcopy

from .. import Interface
from ...stats_utils import DISTRIBUTIONS as D
from ....models.base import Quantities as Q
from ....models.utils import Units as U
from ....data.geosphere_austria.inca.inca import INCALABELS


class INCA2SWAT_V1(Interface):
    """
    Interface which determines observations and parameters for 
    :class:`mef_agri.models.atmosphere.weather.model_swat.Weather_V2009` from 
    inca-data (geosphere-austria).
    """
    @Interface.model_quantity
    def temperature_min(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def temperature_mean(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def temperature_max(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def dewpoint_mean(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def precipitation_sum(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def radiation_sum(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def humidity_min(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def humidity_max(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def humidity_mean(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def wind_speed_mean(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def pressure_msl_mean(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def temperature_mean_annual(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def height_temp_obs(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def height_hum_obs(self) -> Interface.MQDefinition:
        pass

    @Interface.model_quantity
    def height_wind_obs(self) -> Interface.MQDefinition:
        pass

    def process_data(self, data):
        self.temperature_min.qtype = Q.OBS
        self.temperature_min.unit = U.degC
        self.temperature_max.qtype = Q.OBS
        self.temperature_max.unit = U.degC
        self.temperature_mean.qtype = Q.OBS
        self.temperature_mean.unit = U.degC
        self.dewpoint_mean.qtype = Q.OBS
        self.dewpoint_mean.unit = U.degC
        self.precipitation_sum.qtype = Q.OBS
        self.precipitation_sum.unit = U.mm_m2xday
        self.radiation_sum.qtype = Q.OBS
        self.radiation_sum.unit = U.Wh_m2xday
        self.humidity_min.qtype = Q.OBS
        self.humidity_min.unit = U.frac
        self.humidity_max.qtype = Q.OBS
        self.humidity_max.unit = U.frac
        self.humidity_mean.qtype = Q.OBS
        self.humidity_mean.unit = U.frac
        self.wind_speed_mean.qtype = Q.OBS
        self.wind_speed_mean.unit = U.m_s
        self.pressure_msl_mean.qtype = Q.OBS
        self.pressure_msl_mean.unit = U.kPa
        self.temperature_mean_annual.qtype = Q.PARAM
        self.temperature_mean_annual.unit = U.degC

        # compute model quantities from zoned values
        epochs = []
        mdata = {
            'tmin': [], 'tmax': [], 'tmean': [], 'dewp': [], 'prec': [], 
            'radi': [], 'hmin': [], 'hmax': [], 'hmean': [], 'wmean': [],
            'pmean': []
        }
        for epoch, zvals in data.items():
            epochs.append(epoch)
            tvals = list(zvals[INCALABELS.TEMPERATURE].values())
            hvals = list(zvals[INCALABELS.HUMIDITY].values())
            wse = list(zvals[INCALABELS.WIND_EAST].values())
            wsn = list(zvals[INCALABELS.WIND_NORTH].values())
            mdata['tmin'].append(np.nanmin(tvals))
            mdata['tmax'].append(np.nanmax(tvals))
            mdata['tmean'].append(np.nanmean(tvals))
            mdata['hmin'].append(np.nanmin(hvals))
            mdata['hamx'].append(np.nanmax(hvals))
            mdata['hmean'].append(np.nanmean(hvals))
            mdata['dewp'].append(np.nanmean(
                list(zvals[INCALABELS.DEWPOINT].values)
            ))
            mdata['prec'].append(np.nansum(
                list(zvals[INCALABELS.PRECIPITATION].values())
            ))
            mdata['radi'].append(np.nansum(
                list(zvals[INCALABELS.RADIATION].values())
            ))
            mdata['wmean'].append(np.nanmean(
                np.power(np.power(wse, 2.) + np.power(wsn, 2.), 0.5).tolist()
            ))
            mdata['wmean'].append(np.nanmean(
                list(zvals[INCALABELS.PRESSURE].values())
            ))

        # set observation values and distributions
        self.temperature_min.values = np.round(mdata['tmin'], 2).tolist()
        self.temperature_min.distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }
        self.temperature_max.values = np.round(mdata['tmax'], 2).tolist()
        self.temperature_max.distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }
        self.temperature_mean.values = np.round(mdata['tmean'], 2).tolist()
        self.temperature_mean.distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }
        self.dewpoint_mean.values = np.round(mdata['dewp'], 2).tolist()
        self.dewpoint_mean.distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }
        # NOTE precipitation is not converted from kg/m2 (INCA) to mm/m2 
        # (i.e. l/m2) as the density is ~1 kg/l (range from ~0.9998 to ~0.995 
        # depending on temperature)
        self.precipitation_sum.values = np.round(mdata['prec'], 1).tolist()
        self.precipitation_sum.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 1.
        }
        # NOTE radiation sum is not converted because an hourly value for W_m2 
        # is equal to Wh_m2 and due to the summation (~integration) over a day
        # (i.e. 24 hours) the result has unit Wh_m2xday
        self.radiation_sum.values = np.round(mdata['radi'], 3).tolist()
        self.radiation_sum.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 5.
        }
        self.wind_speed_mean.values = np.round(mdata['wmean'], 2).tolist()
        self.wind_speed_mean.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.3
        }
        self.humidity_min.values = np.round(
            U.convert(np.array(mdata['hmin']), U.perc, U.frac), 3
        ).tolist()
        self.humidity_min.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.003
        }
        self.humidity_max.values = np.round(
            U.convert(np.array(mdata['hamx']), U.perc, U.frac), 3
        ).tolist()
        self.humidity_max.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.003
        }
        self.humidity_mean.values = np.round(
            U.convert(np.array(mdata['hmean']), U.perc, U.frac), 3
        ).tolist()
        self.humidity_mean.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.003
        }
        self.pressure_msl_mean.values = np.round(
            U.convert(np.array(mdata['pmean'], U.Pa, U.kPa)), 4
        ).tolist()
        self.pressure_msl_mean.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 1.
        }

        # set mean annual temperature value and distribution
        self.temperature_mean_annual.values = np.mean(
            self.temperature_mean.values
        )
        self.temperature_mean_annual.distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }

        # define observation heights as parameters
        self.height_temp_obs.qtype = Q.PARAM
        self.height_temp_obs.unit = U.m
        self.height_temp_obs.values = 2.
        self.height_temp_obs.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.01
        }
        self.height_hum_obs.qtype = Q.PARAM
        self.height_hum_obs.unit = U.m
        self.height_hum_obs.values = 2.
        self.height_hum_obs.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.01
        }
        self.height_wind_obs.qtype = Q.PARAM
        self.height_wind_obs.unit = U.m
        self.height_wind_obs.values = 10.
        self.height_wind_obs.distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.01
        }

        # set model-ids for all quantities and epochs for the observations
        for mqname in self.model_quantities:
            mq = getattr(self, mqname)
            setattr(mq, 'model', 'zone.atmosphere.weather')
            if getattr(mq, 'qtype') == Q.OBS:
                setattr(mq, 'epochs') = deepcopy(epochs)
