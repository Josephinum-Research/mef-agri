import numpy as np
from copy import deepcopy

from .. import Interface
from ...zoning.utils import ZONE_VALUE_METHOD
from ...stats_utils import DISTRIBUTIONS as D
from ....models.base import Quantities as Q
from ....models.utils import Units as U
from ....data.geosphere_austria.inca.inca import INCALABELS
from ....data.geosphere_austria.inca.interface import INCAInterface


class INCA2SWAT_V1(Interface):
    def __init__(self):
        super().__init__()
        self.data_source_id = INCAInterface.data_source_id
        self.zone_value_method = ZONE_VALUE_METHOD.MEDIAN

        # initialize observations
        self._obs:list[str] = [
            ['temperature_min', U.degC], 
            ['temperatuer_mean', U.degC], 
            ['temperature_max', U.degC],
            ['dewpoint_mean', U.degC], 
            ['precipitation_sum', U.mm_m2xday], 
            ['radiation_sum', U.Wh_m2xday], 
            ['humidity_min', U.frac], 
            ['humidity_mean', U.frac], 
            ['humidity_max', U.frac], 
            ['wind_speed_mean', U.m_s],
            ['pressure_msl_mean', U.kPa]
        ]
        for obs in self._obs:
            self.add_quantity(obs[0], Q.OBS, obs[1], model_hints='weather')

        # initialize parameters
        self.add_quantity(
            'temperature_mean_annual', Q.PARAM, U.degC, model_hints='weather'
        )
        self.add_quantity(
            'height_temp_obs', Q.PARAM, U.m, model_hints='weather'
        )
        self.add_quantity(
            'height_hum_obs', Q.PARAM, U.m, model_hints='weather'
        )
        self.add_quantity(
            'height_wind_obs', Q.PARAM, U.m, model_hints='weather'
        )

    def process_data(self, data):
        # process observations
        qdefs = {}
        for obs in self._obs:
            qdefs[obs[0]] = self.get_qdef(obs[0], Q.OBS)
        for epoch, zvals in data.items():
            for qdef in qdefs.values():
                qdef.epochs.append(epoch)

            tvals = list(zvals[INCALABELS.TEMPERATURE].values())
            hvals = list(zvals[INCALABELS.HUMIDITY].values())
            wse = list(zvals[INCALABELS.WIND_EAST].values())
            wsn = list(zvals[INCALABELS.WIND_NORTH].values())
            qdefs['temperature_min'].values.append(round(np.nanmin(tvals), 2))
            qdefs['temperature_mean'].values.append(round(np.nanmean(tvals), 2))
            qdefs['temperature_max'].values.append(round(np.nanmax(tvals), 2))
            qdefs['humidity_min'].values.append(round(np.nanmin(hvals), 3))
            qdefs['humidity_mean'].values.append(round(np.nanmean(hvals), 3))
            qdefs['humidity_max'].values.append(round(np.nanmax(hvals), 3))
            qdefs['dewpoint_mean'].values.append(round(np.nanmean(
                list(zvals[INCALABELS.DEWPOINT].values())
            ), 2))
            # NOTE precipitation is not converted from kg/m2 (INCA) to mm/m2 
            # (i.e. l/m2) as the density is ~1 kg/l (range from ~0.9998 to 
            # ~0.995 depending on temperature)
            qdefs['precipitation_sum'].values.append(round(np.nansum(
                list(zvals[INCALABELS.PRECIPITATION].values())
            ), 1))
            # NOTE radiation sum is not converted because an hourly value for 
            # W_m2 is equal to Wh_m2 and due to the summation (~integration) 
            # over a day (i.e. 24 hours) the result has unit Wh_m2xday
            qdefs['radiation_sum'].values.append(round(np.nansum(
                list(zvals[INCALABELS.RADIATION].values())
            ), 3))
            qdefs['wind_speed_mean'].values.append(round(np.nanmean(
                np.power(np.power(wse, 2.) + np.power(wsn, 2.), 0.5)
            ), 2))
            qdefs['pressure_msl_mean'].values.append(round(np.nanmean(
                list(zvals[INCALABELS.PRESSURE].values())
            ), 4))

        qdefs['temperature_min'].distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }
        qdefs['temperature_mean'].distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }
        qdefs['temperature_max'].distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }
        qdefs['humidity_min'].distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.003
        }
        qdefs['humidity_mean'].distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.003
        }
        qdefs['humidity_max'].distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.003
        }
        qdefs['precipitation_sum'].distributions = {
            'distr_id': D.GAMMA_1D, 'std': 1.
        }
        qdefs['radiation_sum'].distributions = {
            'distr_id': D.GAMMA_1D, 'std': 5.
        }
        qdefs['dewpoint_mean'].distributions = {
            'distr_id': D.NORMAL_1D, 'std': 1.
        }
        qdefs['wind_speed_mean'].distributions = {
            'distr_id': D.GAMMA_1D, 'std': 0.3
        }
        qdefs['pressure_msl_mean'].distributions = {
            'distr_id': D.GAMMA_1D, 'std': 1.
        }
        for obs in self._obs:
            self.set_qdef(obs[0], Q.OBS, qdefs[obs[0]])

        # process parameters
        tma = self.get_qdef('temperature_mean_annual', Q.PARAM)
        tma.values.append(round(np.mean(qdefs['temperature_mean'].values), 2))
        tma.distributions = {'distr_id': D.NORMAL_1D, 'std': 1.}
        self.set_qdef('temperature_mean_annual', Q.PARAM, tma)

        ht = self.get_qdef('height_temp_obs', Q.PARAM)
        ht.values.append(2.)
        ht.distributions = {'distr_id': D.GAMMA_1D, 'std': 0.01}
        self.set_qdef('height_temp_obs', Q.PARAM, ht)

        hh = self.get_qdef('height_hum_obs', Q.PARAM)
        hh.values.appen(2.)
        hh.distributions = {'distr_id': D.GAMMA_1D, 'std': 0.01}
        self.set_qdef('height_hum_obs', Q.PARAM, hh)

        hw = self.get_qdef('height_wind_obs', Q.PARAM)
        hw.values.append(2.)
        hw.distributions = {'distr_id': D.GAMMA_1D, 'std': 0.01}
        self.set_qdef('height_wind_obs', Q.PARAM, hw)
