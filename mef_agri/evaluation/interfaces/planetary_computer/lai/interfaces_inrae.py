import numpy as np
from copy import deepcopy
from datetime import date, timedelta

from .....data.planetary_computer.sentinel2.interface import Sentinel2Interface
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
        super().__init__(Sentinel2Interface.DATA_SOURCE_ID)
        self._hps_set:bool = False

    def process_data(self, rasters, gcs, edefs, epoch, zid):
        if not self._hps_set:
            # epoch of hyper parameters is one day before start of evaluation
            nnhps = nn_params_dict2model(NNET10_PARAMS, epoch - timedelta(days=1))
            for qname, qinfo in nnhps.items():
                edefs.set_qinfos(
                    edefs['zmodel'], Q.HPARAM, 'zone.sentinel2_lai', qname, qinfo,
                    zid=zid
                )
            self._hps_set = True

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
            edefs.set_qinfos(
                edefs['zmodel'], Q.OBS, 'zone.sentinel2_lai', qname, qinfo, 
                zid=zid
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
            return {
                'b03': np.nanmedian(newimg['B03'][0, rows, cols].flatten()),
                'b04': np.nanmedian(newimg['B04'][0, rows, cols].flatten()),
                'b08': np.nanmedian(newimg['B08'][0, rows, cols].flatten()),
                'sun_azimuth': np.nanmedian(
                    newimg['SUN_AZIMUTH'][0, rows, cols].flatten()
                ),
                'sun_zenith': np.nanmedian(
                    newimg['SUN_ZENITH'][0, rows, cols].flatten()
                ),
                'view_azimuth': np.nanmedian(
                    newimg['VIEW_AZIMUTH_MEAN'][0, rows, cols].flatten()
                ),
                'view_zenith': np.nanmedian(
                    newimg['VIEW_ZENITH_MEAN'][0, rows, cols].flatten()
                ),
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
