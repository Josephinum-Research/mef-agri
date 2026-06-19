import numpy as np
from datetime import date
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, MultiPolygon

from ...utils.misc import PixelUnits
from ...utils.raster import GeoRaster
from ...utils.gis import bbox_from_aoi
from ..tasks import Task


class Zoning(GeoRaster):
    """
    Inherits from :class:`mef_agri.utils.raster.GeoRaster`.

    The :func:`raster` attribute has to contain a layer for each 
    zone, where pixels within a zone are indicated with ones and the remaining 
    pixels contain zeros. 
    Thus, each layer can be used as mask in the following way 
    ``arr[np.where(zoning['zone_id'])] = value``.
    Additionally there is one layer which contains all zones, i.e. incremented 
    integer values starting from 1.
    These integer values "minus one" can be used as index to retrieve the 
    corresponding zone-label from :func:`layer_ids`.
    An appropriate integer-datatype from numpy will be used depending on the 
    number of zones.

    :func:`date_begin` indicates the start of the validity of the zoning and 
    :func:`valid_until` indicates its end.
    """
    def __init__(self):
        super().__init__()
        self._meta['valid_from'] = None
        self._meta['valid_until'] = None

    @property
    def valid_from(self) -> date:
        """
        :return: epoch until when the zoning is valid
        :rtype: datetime.date
        """
        return date.fromisoformat(self._meta['valid_from'])
    
    @valid_from.setter
    def valid_from(self, val):
        self._meta['valid_from'] = Task._check_date(val).isoformat()

    @property
    def valid_until(self) -> date:
        """
        :return: epoch until when the zoning is valid
        :rtype: datetime.date
        """
        return date.fromisoformat(self._meta['valid_until'])
    
    @valid_until.setter
    def valid_until(self, val):
        self._meta['valid_until'] = Task._check_date(val).isoformat()

    def _set_crs(self, geom:GeoDataFrame | Polygon | MultiPolygon, crs) -> None:
        if isinstance(geom, GeoDataFrame):
            self.crs = geom.crs.to_epsg()
        else:
            if crs is None:
                msg = 'If `field_geom` is not a `geopandas.GeoDataFrame`, `crs` '
                msg += 'has to be provided too!'
                raise ValueError(msg)
            else:
                self.crs = int(crs)

    @classmethod
    def one_zone(
        cls, 
        field_geom:GeoDataFrame | Polygon | MultiPolygon,
        crs:int=None
    ):
        rstr = cls()
        rstr._set_crs(field_geom, crs)
        rstr.units = PixelUnits.INT8
        rstr.nodata_value = 0
        rstr.layer_index = 0
        # raster and layer specific stuff
        rstr.raster = np.array([[[1]]], type=np.int8)
        rstr.raster_shape = (1, 1, 1)
        rstr.layer_ids = ['zone-1']
        # computaton of transformation matrix and bounds
        bbox = bbox_from_aoi(field_geom)
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        rstr.transformation = np.array([
            [scale, 0., bbox[0]],
            [0., -scale, bbox[3]],
            [0., 0., 1.]
        ])
        rstr.bounds = (bbox[0], bbox[3] - scale, bbox[0] + scale, bbox[3])
        return rstr

    @classmethod
    def from_vector(
        cls, 
        field_geom:GeoDataFrame | Polygon | MultiPolygon, 
        objres:float, 
        vfile:str
    ):
        pass

    @classmethod
    def regular_grid(
        cls, 
        field_geom:GeoDataFrame | Polygon | MultiPolygon, 
        objres:float,
        crs:int=None
    ):
        rstr = cls()
        rstr._set_crs(field_geom, crs)
        # TODO

