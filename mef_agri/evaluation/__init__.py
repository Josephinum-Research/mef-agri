import datetime as dt

from .interfaces import Interface
from .db import EvalDB
from .zoning import Zoning
from ..models.base import Model
from ..data.project import ProjectData


class EvalSetup(object):
    def __init__(self, edb:EvalDB, prj:ProjectData, intfs:list[Interface]):
        self._edb:EvalDB = edb
        self._prj:ProjectData = prj
        self._intfs = {}
        for intf in intfs:
            self._intfs[intf.data_source_id] = intf

    def set_up_eval(self, field_name:str, zoning:Zoning):
        data = self._prj.get_data(
            zoning.valid_from, zoning.valid_until, 
            dids=list(self._intfs.keys()), fields=field_name
        )
        for did, ifdata in data.items():
            self._intfs[did].process_data(ifdata[field_name], zoning)


"""
NOTE >>> how to derive geocoordinates from zoning

ics = np.where(ztask[zname])
ics = ics[1:] if ztask.layer_index == 0 else ics[:2]
ics_hom = np.vstack((
    np.atleast_2d(ics[1]), 
    np.atleast_2d(ics[0]), 
    np.ones((1, len(ics[0])))
))
gcs = (ztask.transformation @ ics_hom)[:2, :]

NOTE >>> how to get raster coordinates from other georaster from geocoordinates

rows, cols = georaster.get_raster_coordinates_array(*gcs)

TODO 
>>> above approach only works reasonably if zoning has higher resolution than data-source
>>> otherwise, pixels from the data-source are not used at all

"""
