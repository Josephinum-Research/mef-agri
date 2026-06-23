import numpy as np
import geopandas as gpd
from math import ceil
from datetime import date
from shapely.geometry import Polygon, MultiPolygon

from ...utils.misc import PixelUnits
from ...utils.raster import GeoRaster
from ...utils.rv_manipulation import RasterVectorIntersection
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

    def _set_crs(self, geom:gpd.GeoDataFrame | Polygon | MultiPolygon, crs) -> None:
        if isinstance(geom, gpd.GeoDataFrame):
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
        field_geom:gpd.GeoDataFrame | Polygon | MultiPolygon,
        crs:int=None
    ):
        """
        Represent field as one zone, i.e. on square which covers the field.
        The square is defined as following

        * upper-left corner equals the the upper-left coordinate of the field's bounding box
        * the side length equals the longer edge of the field's bounding box

        :param field_geom: geometry describing the field
        :type field_geom: GeoDataFrame | Polygon | MultiPolygon
        :param crs: epsg code, defaults to None
        :type crs: int, optional
        :return: initialized Zoning object
        :rtype: mef_agri.farming.zoning.Zoning
        """
        rstr = cls()
        rstr.layer_index = 0
        rstr._set_crs(field_geom, crs)
        rstr.units = PixelUnits.INT8
        rstr.nodata_value = 0
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
        field_geom:gpd.GeoDataFrame | Polygon | MultiPolygon, 
        objres:float, 
        vfile:str,
        crs:int=None
    ):
        rstr = cls()
        rstr.layer_index = 0
        rstr._set_crs(field_geom, crs)
        gdf = gpd.read_file(vfile)
        if (gdf.crs.to_epsg() != rstr.crs):
            gdf.to_crs(epsg=rstr.crs, inplace=True)
        # convert provided geometries to zones and create raster
        aux = Zoning.regular_grid(field_geom, objres, crs=rstr.crs)
        for tpl in gdf.itertuples():
            zgeom = getattr(tpl, gdf.geometry.name)
            # check if zone is in the field or at least intersects it
            if not field_geom.contains(zgeom):
                if field_geom.intersects(zgeom):
                    zgeom = field_geom.intersection(zgeom)
                else:
                    continue
            # create zone layers
            # TODO ensure that zgeom is a Polygon
            rvi = RasterVectorIntersection(zgeom, aux)
            rvi.compute()
            # TODO using rvi.assignment as mask to create zone layers

    @classmethod
    def regular_grid(
        cls, 
        field_geom:gpd.GeoDataFrame | Polygon | MultiPolygon, 
        objres:float,
        crs:int=None
    ):
        """
        Zones correspond to a regular grid overlaying the field, where each 
        zone is a square with side length being ``objres``.
        The upper-left corner of the regular grid is equal to the upper-left 
        coordinates of the field's bounding box.
        The layer index will be set to ``0`` (i.e. layers are accessible by 
        the first index of the 3D raster).

        :param field_geom: geometry representing the field
        :type field_geom: GeoDataFrame | Polygon | MultiPolygon
        :param objres: extent of the zones (i.e. squares)
        :type objres: float
        :param crs: epsg code, defaults to None
        :type crs: int, optional
        :return: initialized Zoning object
        :rtype: mef_agri.farming.zoning.Zoning
        """
        rstr = cls()
        rstr.layer_index = 0
        rstr._set_crs(field_geom, crs)
        # number of pixels and appropriate data type
        bbox = bbox_from_aoi(field_geom)
        dimx, dimy = bbox[2] - bbox[0], bbox[3] - bbox[1]
        npx, npy = ceil(dimx / objres), ceil(dimy / objres)
        np = npx * npy
        if (np <= (2 ** 8)):
            rstr.units = PixelUnits.INT8
        elif (np <= (2 ** 16)):
            rstr.units = PixelUnits.INT16
        elif (np <= (2 ** 32)):
            rstr.units = PixelUnits.INT32
        else:
            msg = 'More than {} zones are not supported!'
            raise ValueError(msg.format(2 ** 32))
        rstr.nodata_value = 0
        # 
        rstr.raster_shape = (np, npy, npx)
        rstr.raster = np.zeros((np, npy, npx))
        zids = []
        ix, iy = 0, 0
        for i in range(np):
            zids.append('zone-{}'.format(i + 1))
            rstr.raster[i, iy, ix] = 1
            if (ix == (npx - 1)):
                ix = 0
                iy += 1
            else:
                ix += 1
            i += 1
        rstr.layer_ids = zids
        rstr.bounds = [
            bbox[0], bbox[3] - objres * npy, bbox[0] + objres * npx, bbox[3]
        ]
        rstr.transformation = np.array([
            [objres, 0., bbox[0]],
            [0., -objres, bbox[3]],
            [0., 0., 1.]
        ])
        return rstr
