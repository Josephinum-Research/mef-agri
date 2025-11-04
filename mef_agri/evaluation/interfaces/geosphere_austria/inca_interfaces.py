import numpy as np
from datetime import date
from copy import deepcopy

from ....data.geosphere_austria.inca.interface import INCAInterface
from ....data.geosphere_austria.inca.inca import INCALABELS
from ....utils.raster import GeoRaster
from ..base import EvalInterface as Interface
from ...stats_utils import DISTRIBUTIONS
from ...eval_def import EvaluationDefinitions
from ....models.base import Quantities


CONV_WPM2_2_MJPM2PDAY = 0.0864  # Table 3 from https://www.fao.org/3/X0490E/x0490e07.htm#chapter%203%20%20%20meteorological%20data


def inca_dict_2_zone_values(inca:dict[GeoRaster], gcs:np.ndarray) -> dict:
    """
    Compute the mean values from an dictionary containing the inca `GeoRasters`s 
    with hourly data within a zone represented by the geo-coordinates `gcs`.

    :param inca: inca data for a specific day
    :type inca: dict[GeoRaster]
    :param gcs: geo-coordinates - shape has to be (2, n), where n ... number of points
    :type gcs: np.ndarray
    :return: dictionary containing the mean values for the considered zone
    :rtype: dict
    """
    # NOTE maybe consider variance propagation/distribution of processed values (i.e. multivariate distributions)

    prec = inca[INCALABELS.PRECIPITATION]
    rows, cols = prec.get_raster_coordinates_array(*gcs)
    return {
        INCALABELS.PRECIPITATION: np.nanmean(
            prec.raster[:, rows, cols], axis=1
        ),
        INCALABELS.TEMPERATURE: np.nanmean(
            inca[INCALABELS.TEMPERATURE].raster[:, rows, cols], axis=1
        ),
        INCALABELS.RADIATION: np.nanmean(
            inca[INCALABELS.RADIATION].raster[:, rows, cols], axis=1
        ),
        INCALABELS.PRESSURE: np.nanmean(
            inca[INCALABELS.PRESSURE].raster[:, rows, cols], axis=1
        ),
        INCALABELS.HUMIDITY: np.nanmean(
            inca[INCALABELS.HUMIDITY].raster[:, rows, cols], axis=1
        ),
        INCALABELS.WIND_EAST: np.nanmean(
            inca[INCALABELS.WIND_EAST].raster[:, rows, cols], axis=1
        ),
        INCALABELS.WIND_NORTH: np.nanmean(
            inca[INCALABELS.WIND_NORTH].raster[:, rows, cols], axis=1
        ),
        INCALABELS.DEWPOINT: np.nanmean(
            inca[INCALABELS.DEWPOINT].raster[:, rows, cols], axis=1
        )
    }


class INCA_AtmosphereSWAT_JRV1(Interface):
    def __init__(self):
        super().__init__(INCAInterface.DATA_SOURCE_ID)

    def process_data(
            self, edefs:EvaluationDefinitions, rasters:dict[GeoRaster], 
            gcs:np.ndarray, epoch:date, zid:str
        ) -> EvaluationDefinitions:
        wdata = self._zdata_2_model(inca_dict_2_zone_values(rasters, gcs))
        for qname, val in wdata.items():
            qinfo = deepcopy(val)
            qinfo['epoch'] = epoch
            edefs.set_qinfos(edefs['zmodel'], Quantities.OBS, 
                'zone.atmosphere.weather', qname, qinfo, zid=zid)
        return edefs

    @staticmethod
    def _zdata_2_model(zdata:dict[np.ndarray]) -> dict:
        """
        Compute the values which are necessary for the crop growth model consisting 
        of wofost and jr components.
        The returned dictionary contains the following values

        * temperature_min [degC]
        * temperature_max [degC]
        * temperature_mean [degC]
        * precipitation_sum [mm/m2/day]
        * radiation_sum [Wh/m2/day]
        * humidity_mean []
        * humidity_min []
        * humidity_max []
        * dewpoint_mean [degC]
        * wind_speed_mean [m/s]
        * pressure_msl_mean [kPa]

        :param values: dictionary containing hourly inca data for one day
        :type values: dict
        :return: dictionary containing values
        :rtype: dict
        """
        # NOTE maybe consider variance propagation/distribution of processed values (i.e. multivariate distributions)

        humi_h = zdata[INCALABELS.HUMIDITY] * 1e-2  # [percent] -> []
        pres_h = zdata[INCALABELS.PRESSURE] * 1e-3  # [Pa] -> [kPa]
        we = np.nanmean(zdata[INCALABELS.WIND_EAST])  # [m/s]
        wn = np.nanmean(zdata[INCALABELS.WIND_NORTH])  # [m/s]

        ret = {
            'temperature_min': {
                'value': round(np.nanmin(zdata[INCALABELS.TEMPERATURE]), 2),
                'distr': {
                    'distr_id': DISTRIBUTIONS.NORMAL_1D,
                    'std': 1.0
                }
            },
            'temperature_max': {
                'value': round(np.nanmax(zdata[INCALABELS.TEMPERATURE]), 2),
                'distr': {
                    'distr_id': DISTRIBUTIONS.NORMAL_1D,
                    'std': 1.0
                }
            },
            'precipitation_sum': {
                'value': round(np.nansum(zdata[INCALABELS.PRECIPITATION]), 2),  # NOTE precipitation is not converted from kg/m2/h (INCA) to mm/m2/h (i.e. l/m2/h) as the density is ~1 kg/l (range from ~0.9998 to ~0.995 depending on temperature)
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 2.0
                }
            },
            'radiation_sum': {
                'value': round(np.nansum(zdata[INCALABELS.RADIATION]), 3),
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 20.0
                }
            },
            'humidity_min': {
                'value': round(np.nanmin(humi_h), 3),
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.02
                }
            },
            'humidity_max': {
                'value': round(np.nanmax(humi_h), 3),
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.02
                }
            },
            'humidity_mean': {
                'value': round(np.nanmean(humi_h), 3),
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.02
                }
            },
            'dewpoint_mean': {
                'value': round(np.nanmean(zdata[INCALABELS.DEWPOINT]), 2),
                'distr': {
                    'distr_id': DISTRIBUTIONS.NORMAL_1D,
                    'std': 1.0
                }
            },
            'wind_speed_mean': {
                'value': round(np.sqrt(we * we + wn * wn), 2),
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 0.2
                }
            },
            'pressure_msl_mean': {
                'value': round(np.nanmean(pres_h), 4),
                'distr': {
                    'distr_id': DISTRIBUTIONS.GAMMA_1D,
                    'std': 1.0
                }
            }
        }
        ret['temperature_mean'] = {
            'value': round(0.5 * (
                ret['temperature_min']['value'] + 
                ret['temperature_max']['value']
            ), 2),
            'distr': {
                'distr_id': DISTRIBUTIONS.NORMAL_1D,
                'std': 1.0
            }
        }
        return ret
