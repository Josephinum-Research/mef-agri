import numpy as np
import geopandas as gpd
from math import ceil
from datetime import date
from shapely.geometry import Polygon, MultiPolygon

from ...utils.misc import PixelUnits
from ...utils.raster import GeoRaster
from ...utils.rv_manipulation import RasterVectorIntersection
from ...utils.gis import bbox_from_aoi
from ...farming.tasks import Task


class Zoning(GeoRaster):
    """
    Inherits from :class:`mef_agri.utils.raster.GeoRaster`.

    The :func:`raster` attribute has to contain a layer for each 
    zone, where pixels within a zone are indicated with ones and the remaining 
    pixels contain zeros. 
    Thus, each layer can be used as mask in the following way 
    ``arr[np.where(zoning['zone_id'])] = value``.
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

    def zones_to_gdf(
            self, incl:list[str]=None, excl:list[str]=None
        ) -> gpd.GeoDataFrame:
        if incl is None:
            zids = self.layer_ids
        else:
            zids = incl
        if excl is not None:
            zids = [zid for zid in zids if zid not in excl]

        gdf = self.to_gdf()
        data = {'geometry': [], 'zone_id': []}
        for zid in zids:
            zgdf = gdf[gdf[zid] == 1]
            data['geometry'].append(zgdf.geometry.union_all())
            data['zone_id'].append(zid)
        return gpd.GeoDataFrame(data=data, crs=self.crs)

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
        :param crs: epsg code (has to be provided if ``geom`` not an instance of ``geopandas.GeoDataFrame``), defaults to None
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
        rstr.raster = np.array([[[1]]], dtype=np.int8)
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
    def regular_grid(
        cls, 
        field_geom:gpd.GeoDataFrame | Polygon | MultiPolygon, 
        objres:float,
        crs:int=None,
        thresh:float=1e-6
    ):
        """
        Zones correspond to a regular grid overlaying the field, where each 
        zone is a square with side length being ``objres``.
        The upper-left corner of the regular grid is equal to the upper-left 
        coordinates of the field's bounding box.
        :func:`layer_index` will be set to ``0`` (i.e. layers are accessible by 
        the first index of the 3D raster).
        The last layer indicates if a zone touches the field (pixel-value = 1) 
        or not (pixel-value = 0).
        Correspondingly, the layer-ids are `zone-<i>` if the zone thouches the 
        field or `outside-<i>` if not.

        :param field_geom: geometry representing the field
        :type field_geom: GeoDataFrame | Polygon | MultiPolygon
        :param objres: extent of the zones (i.e. squares)
        :type objres: float
        :param crs: epsg code, defaults to None
        :type crs: int, optional
        :param thresh: threshold which is used to indicate if a pixel touches the field, defaults to ``1e-6``
        :type thresh: float, optional
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
        npa = npx * npy
        if (npa <= (2 ** 8)):
            rstr.units = PixelUnits.INT8
        elif (npa <= (2 ** 16)):
            rstr.units = PixelUnits.INT16
        elif (npa <= (2 ** 32)):
            rstr.units = PixelUnits.INT32
        else:
            msg = 'More than {} zones are not supported!'
            raise ValueError(msg.format(2 ** 32))
        rstr.nodata_value = 0

        # defining the georaster
        rstr.bounds = [
            bbox[0], bbox[3] - objres * npy, bbox[0] + objres * npx, bbox[3]
        ]
        rstr.transformation = np.array([
            [objres, 0., bbox[0]],
            [0., -objres, bbox[3]],
            [0., 0., 1.]
        ])
        rstr.raster_shape = (npa + 1, npy, npx)
        rstr.raster = np.zeros((npa + 1, npy, npx), dtype=rstr.units)

        # create an additional layer indicating, if a pixel touches the field
        rvi = RasterVectorIntersection(field_geom, rstr)
        rvi.fraction = thresh
        rvi.compute()
        rstr.raster[-1:, :, :] = rvi.assignment

        zids = []
        ix, iy = 0, 0
        inan, iins = 0, 0
        for i in range(npa):
            if rstr.raster[-1, iy, ix]:
                zids.append(f'zone_{iins + 1}')
                iins += 1
            else:
                zids.append(f'outside_{inan + 1}')
                inan += 1
            
            rstr.raster[i, iy, ix] = 1
            if (ix == (npx - 1)):
                ix = 0
                iy += 1
            else:
                ix += 1
            i += 1
        zids.append('touches_field')
        rstr.layer_ids = zids

        return rstr

    @classmethod
    def from_vector(
        cls, 
        field_geom:gpd.GeoDataFrame | Polygon | MultiPolygon, 
        objres:float, 
        vfile:str,
        crs:int=None,
        intersect_fraction:float=1.0
    ):
        """
        Create zoning-georaster from provided absolute filepath ``vfile`` which 
        represents the zones as vector geometries (i.e. polygons or 
        multi-polygons).
        ``vfile`` has to be a file which can be read by 
        :func:`geopandas.read_file`.
        Zones/vector geometries will be removed if they do not intersect the 
        ``field_geom`` and the geodataframe containing the remaining zones is 
        available via the ``zone_geoms`` attribute of the returned georaster.
        Each zone/vector geometry will be converted to one layer which pixels 
        contains a 1 if the intersection of the these pixels with a zone 
        exceed ``intersect_fraction``.
        The georeference of the created georaster is derived from the bounding 
        box of ``field_geom`` and provided ``objres``.

        :param field_geom: geometry representing the field
        :type field_geom: GeoDataFrame | Polygon | MultiPolygon
        :param objres: extent of the zones (i.e. squares)
        :type objres: float
        :param vfile: absolute path to file containing zones/vector geometries
        :type vfile: str
        :param crs: epsg code, defaults to None
        :type crs: int, optional
        :param intersect_fraction: intersection percentage which is used to decide if a pixel belongs to a zone/vector geometry, defaults to 1.0
        :type intersect_fraction: float, optional
        :return: zoning instance
        :rtype: Zoning
        """
        rstr = cls()
        rstr.layer_index = 0
        rstr._set_crs(field_geom, crs)
        rstr.units = PixelUnits.INT16
        rstr.nodata_value = 0

        gdf = gpd.read_file(vfile)
        if (gdf.crs.to_epsg() != rstr.crs):
            gdf.to_crs(epsg=rstr.crs, inplace=True)
        # remove vectors/polygons which do not touch the field
        if isinstance(field_geom, gpd.GeoDataFrame):
            fgeom = field_geom.geometry.union_all()
        else:
            fgeom = field_geom
        gdf['touches_field'] = gdf.intersects(fgeom)
        gdf = gdf[gdf['touches_field'] == True]
        setattr(rstr, 'zone_geoms', gdf)

        bbox = bbox_from_aoi(field_geom)
        dimx, dimy = bbox[2] - bbox[0], bbox[3] - bbox[1]
        npx, npy = ceil(dimx / objres), ceil(dimy / objres)

        rstr.bounds = [
            bbox[0], bbox[3] - objres * npy, bbox[0] + objres * npx, bbox[3]
        ]
        rstr.transformation = np.array([
            [objres, 0., bbox[0]],
            [0., -objres, bbox[3]],
            [0., 0., 1.]
        ])
        rstr.raster_shape = (len(gdf), npy, npx)
        rstr.raster = np.zeros(rstr.raster_shape, dtype=rstr.units)

        rvi = RasterVectorIntersection(gdf, rstr)
        rvi.nodata_value = rstr.nodata_value
        rvi.fraction = intersect_fraction
        rvi.compute()

        lids = []
        for i in range(len(gdf)):
            ixs = np.where(rvi.assignment == i + 1)
            rstr.raster[i, ixs[1], ixs[2]] = 1
            lids.append(f'zone_{i + 1}')
        rstr.layer_ids = lids

        return rstr
