import numpy as np
from copy import deepcopy
from datetime import date, timedelta

#from .....data.planetary_computer.sentinel2.interface import Sentinel2Interface
from .....data.planetary_computer.sentinel2.sentinel2 import (
    ImageSentinel2, SCLMAP
)
from ....stats_utils import DISTRIBUTIONS
from ...base import EvalInterface as Interface
from .....models.base import Quantities as Q
from .model_inrae import NNET10_PARAMS
from ...utils import nn_params_dict2model


class PCSentinel2_NNET10(Interface):
    def __init__(self):
        super().__init__()
        self.data_source_id = 'sentinel-2_planetary-computer'  # NOTE this is a workaround, better to fix the next line
        #self.data_source_id = Sentinel2Interface.DATA_SOURCE_ID
        self._hps_set = {}

    def process_data(self, edefs, rasters, gcs, epoch, zid):
        if not str(zid) in self._hps_set.keys():
            self._hps_set[str(zid)] = False
        if not self._hps_set[str(zid)]:
            # epoch of hyper parameters is one day before start of evaluation
            nnhps = nn_params_dict2model(NNET10_PARAMS, epoch - timedelta(days=1))
            for qname, qinfo in nnhps.items():
                edefs.set_zone_qinfos(
                    zid, Q.PARAM, 'zone.sentinel2_lai', qname, qinfo
                )
            self._hps_set[str(zid)] = True

        if rasters is None:
            # no image available for current epoch
            return edefs
        
        img = rasters['sentinel-2-image']
        zvals = self.s2img_2_zone_values(img, gcs)
        if not zvals:
            # no vegetation pixels available for current epoch
            return edefs
        s2obs = self._s2_zvals_2_model(zvals, epoch)
        for qname, qinfo in s2obs.items():
            edefs.set_zone_qinfos(
                zid, Q.OBS, 'zone.sentinel2_lai', qname, qinfo
            )

        return edefs

    @staticmethod
    def s2img_2_zone_values(img:ImageSentinel2, gcs:np.ndarray) -> dict:
        newimg = deepcopy(img)
        rows, cols = img.get_raster_coordinates_array(*gcs)
        veg = img['SCL'] == SCLMAP.VEGETATION
        if True in veg.flatten():
            newimg['B03'][~veg] = np.nan
            newimg['B04'][~veg] = np.nan
            newimg['B08'][~veg] = np.nan
            newimg['SUN_AZIMUTH'][~veg] = np.nan
            newimg['SUN_ZENITH'][~veg] = np.nan
            newimg['VIEW_AZIMUTH_MEAN'][~veg] = np.nan
            newimg['VIEW_ZENITH_MEAN'][~veg] = np.nan
            b03med = np.nanmedian(newimg['B03'][0, rows, cols].flatten())
            b04med = np.nanmedian(newimg['B04'][0, rows, cols].flatten())
            b08med = np.nanmedian(newimg['B08'][0, rows, cols].flatten())
            samed = np.nanmedian(newimg['SUN_AZIMUTH'][0, rows, cols].flatten())
            szmed = np.nanmedian(newimg['SUN_ZENITH'][0, rows, cols].flatten())
            vamed = np.nanmedian(newimg['VIEW_AZIMUTH_MEAN'][0, rows, cols].flatten())
            vzmed = np.nanmedian(newimg['VIEW_ZENITH_MEAN'][0, rows, cols].flatten())

            vals = [b03med, b04med, b08med, samed, szmed, vamed, vzmed]
            if True in np.isnan(vals):
                return {}
            return {
                'b03': b03med, 'b04': b04med, 'b08': b08med,
                'sun_azimuth': samed, 'sun_zenith': szmed,
                'view_azimuth': vamed, 'view_zenith': vzmed,
            }
        else:
            return {}
        
    @staticmethod
    def _s2_zvals_2_model(zvals:dict, epoch:date) -> dict:
        ret = {}
        
        for key, val in zvals.items():
            di = {'distr_id': DISTRIBUTIONS.TRUNCNORM_1D}
            
            if key in ('b03', 'b04', 'b08'):
                cval = round(float(val), 4)
                di['std'] = 3e-3 * (1. + cval * 4.)
                di['lb'] = 0.
                di['ub'] = 1.
            else:
                # for angles the number comma digits is set to 5 because in this 
                # case, the resulting resolution on the ground is ~0.14 [m]
                # which is appropriate considering a pixel size of 10 m
                cval = round(float(val), 5)
                di['lb'] = 0.,
                di['ub'] = 360.
                di['std'] = 5e-4  # on ground this corresponds to ~7 m

            ret[key] = {
                'value': cval,
                'epoch': epoch,
                'distr': di
            }
        
        return ret
