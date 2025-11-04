import numpy as np
from datetime import date
from copy import deepcopy

from ....data.ebod_austria.interface import EbodInterface
from ....data.ebod_austria.ebod import EbodRaster
from ....utils.raster import GeoRaster
from ..base import EvalInterface as Interface
from ....models.base import Quantities
from ...eval_def import EvaluationDefinitions
from .ptfs import pedotransfer_ebod_swat_jrv1


def ebod_raster_2_zone_values(ebod:EbodRaster, gcs:np.ndarray) -> dict:
    """
    Compute mode (i.e. value which most often occurs) of the ebod values within 
    the zone represented by `gcs`.

    :param ebod: raster of ebod values for whole considered aoi
    :type ebod: EbodRaster
    :param gcs: geo-coordinates - shape has to be (2, n), where n ... number of points
    :type gcs: np.ndarray
    :return: dictionary containing mode of the values in the ebod layers
    :rtype: dict
    """
    rows, cols = ebod.get_raster_coordinates_array(*gcs)
    ret = {}
    for layer in ebod.layer_ids:
        data = ebod[layer][0, rows, cols]
        dmax, dmin = np.max(data), np.min(data)
        counts, borders = np.histogram(
            data, bins=int(dmax - dmin + 1), range=[dmin - 0.5, dmax + 0.5]
        )
        values = [dmin - 1 + int(d) for d in np.cumsum(np.diff(borders))]
        ret[layer] = values[np.argmax(counts)]

    return ret


class EBOD_SoilSWAT_JRV1(Interface):
    PTF_FILENAME = 'ptf_jrv01.json'

    def __init__(self):
        super().__init__(EbodInterface.DATA_SOURCE_ID)
        self.time_independent = True

    def process_data(
            self, edefs:EvaluationDefinitions, rasters:dict[GeoRaster], 
            gcs:np.ndarray, epoch:date, zid:str
        ) -> EvaluationDefinitions:
        sdata = pedotransfer_ebod_swat_jrv1(
            ebod_raster_2_zone_values(list(rasters.values())[0], gcs)
        )
        for qname, val in sdata.items():
            qinfo = deepcopy(val)
            qinfo['epoch'] = epoch
            edefs.set_qinfos(
                edefs['zmodel'], Quantities.PARAM, qinfo['qmodel'],
                qinfo['qname'], qinfo, zid=zid
            )

        return edefs
