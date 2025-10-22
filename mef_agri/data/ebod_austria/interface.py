import os
import requests
import numpy as np
import geopandas as gpd
import mapbox_vector_tile as mvt
from shapely.geometry import Polygon
from copy import deepcopy

from ...utils.gis import latlon2tilexy, tilecoords2latlon, bbox_from_gdf
from .ebod import EBOD_PROPERTIES, EbodRaster
from ..interface import DataInterface


class EbodInterface(DataInterface):
    DATA_SOURCE_ID = 'soil_ebod'
    URL_META = 'https://bodenkarte.at/data/bodenkarte-tiles.json'
    URL_TILES = 'https://bodenkarte.at/data/bodenkarte-tiles/'
    META_MINZOOM = 'minzoom'
    META_MAXZOOM = 'maxzoom'
    ID_EBOD = 'ebod'
    TILES_FILENAME = 'tiles'


    def __init__(
            self, obj_res:float=10.0, zoom:int=15, 
            exclude_ebod_properties:list[str]=None
        ) -> None:
        """
        Interface to download soil information from 
        https://bodenkarte.at
        """
        super().__init__(obj_res=obj_res)

        self._zoom:int = zoom
        if exclude_ebod_properties is None:
            self._excl:list[str] = [
                'bofo_id', 'kurzbezeichnung', 'gruenlandwert'
            ]
        else:
            self._excl:list[str] = exclude_ebod_properties

        self._meta:dict = {}
        self._min_zoom:int = 6  # value read from URL_META on 2023-09-07
        self._max_zoom:int = 15  # value read from URL_META on 2023-09-07
        try:
            self._meta = requests.get(self.URL_META).json()
            self._min_zoom = int(self._meta[self.META_MINZOOM])
            self._max_zoom = int(self._meta[self.META_MAXZOOM])
        except:
            pass

    @property
    def object_resolution(self) -> float:
        return self._ores
    
    @object_resolution.setter
    def object_resolution(self, val):
        if isinstance(val, float):
            self._ores = val
        elif isinstance(val, int):
            self._ores = float(val)
        else:
            msg = '`object_resolution` has to be provided as `int` or `float`!'
            raise ValueError(msg)
        
    @property
    def zoom_level(self) -> int:
        return self._zoom
    
    @zoom_level.setter
    def zoom_level(self, val):
        if isinstance(val, float):
            self._zoom = int(val)
        elif isinstance(val, int):
            self._zoom = val
        else:
            msg = '`zoom_level` has to be provided as `int` or `float`!'
            raise ValueError(msg)
        
    @property
    def exclude_ebod_properties(self) -> list[str]:
        return self._excl
    
    @exclude_ebod_properties.setter
    def exclude_ebod_properties(self, val):
        if isinstance(val, list):
            notstr = False
            for entry in val:
                if isinstance(entry, str) and entry in EBOD_PROPERTIES.keys():
                    continue
                notstr = True
            if notstr:
                msg = 'Entries of provided list are not strings or not part of '
                msg += 'available ebod properties!'
                raise ValueError(msg)
            self._excl = val
        else:
            msg = '`exclude_ebod_properties` has to be a list of available '
            msg += 'ebod properties (i.e. strings)!'
            raise ValueError(msg)

    def add_prj_data(self, aoi, tstart, tstop):
        """
        Method from parent-class which is called within 
        :func:`mef_agri.data.project.Project.add_data` to save ebod data
        in the project folder structure.

        :param aoi: area of interest for which data is requested
        :type aoi: geopandas.GeoDataFrame
        :param tstart: not used for ebod data
        :type tstart: datetime.date
        :param tstop: not used for ebod data
        :type tstop: datetime.date
        """
        tiles = self.get_tile_data(aoi, zoom=self.zoom_level)
        ebod = EbodRaster.from_vector_tile_geodataframe(
            tiles, aoi,
            self.object_resolution,
            self.exclude_ebod_properties
        )
        self.save_ebod_raster(ebod, overwrite=False)
        return tstart, tstop

    def get_prj_data(self, epoch):
        """
        Method from parent-class which is called within
        `mef_agri.data.project.Project.get_data` to load ebod data from 
        the project folder structure

        :param epoch: not used for ebod data
        :type epoch: datetime.date
        :return: dictionary containing raster representation of ebod data
        :rtype: dict[GeoRaster]
        """
        soil = EbodRaster()
        fpath = os.path.join(self.project_directory, self.save_directory)
        soil.load_geotiff(fpath)
        return {'ebod-raster': soil}

    def get_tile_data(self, aoi:gpd.GeoDataFrame, zoom:int) -> gpd.GeoDataFrame:
        """
        Download data as vector tiles from ebod. The result will be provided as 
        `geopandas.GeoDataFrame` in the crs defined in `aoi`.
        Vector tiles from ebod are defined according to 
        https://docs.mapbox.com/data/tilesets/guides/vector-tiles-standards/
        and will be converted or collapsed into a GeoDataFrame.

        :param aoi: defining the area of interest
        :type aoi: gpd.GeoDataFrame
        :param zoom: zoom level
        :type zoom: int
        :return: GeoDataFrame information from vector tiles
        :rtype: gpd.GeoDataFrame
        """
        def append_properties(feat, data, coords, tcs, zoom, extent):
            data['geometry'].append(self._plgn2wgs84(
                coords, tcs, zoom, extent
            ))
            for key, val in EBOD_PROPERTIES.items():
                insval = np.nan
                if key in feat['properties'].keys():
                    insval = feat['properties'][key]
                data[val].append(insval)
            return data

        gdfr = deepcopy(aoi)
        tcrs = aoi.crs.to_epsg()
        if not tcrs == 4326:
            gdfr.to_crs(4326, inplace=True)
        bbox = bbox_from_gdf(gdfr)
        tx1, ty1 = latlon2tilexy(bbox[3], bbox[0], zoom)
        tx2, ty2 = latlon2tilexy(bbox[1], bbox[2], zoom)

        data = {}
        for key in list(EBOD_PROPERTIES.values()) + ['geometry']:
            data[key] = []
        for txi in np.arange(tx1 - 1, tx2 + 2):
            for tyi in np.arange(ty1 - 1, ty2 + 2):
                vdata = self._request(txi, tyi, zoom)
                extent = vdata['bodenform_mpoly']['extent']
                for feat in vdata['bodenform_mpoly']['features']:
                    geom = feat['geometry']['coordinates']
                    if feat['geometry']['type'] == 'Polygon':
                        data = append_properties(
                            feat, data, geom[0], [txi, tyi], zoom, extent
                        )
                    elif feat['geometry']['type'] == 'MultiPolygon':
                        for coords in geom:
                            data = append_properties(
                                feat, data, coords[0], [txi, tyi], zoom, extent
                            )
                    else:
                        # other geometry type which is not considered here until now
                        pass

        gdf_tiles = gpd.GeoDataFrame(data, crs=4326)
        if tcrs != 4326:
            gdf_tiles.to_crs(tcrs, inplace=True)

        return gdf_tiles
    
    def save_tile_data(self, gdf:gpd.GeoDataFrame) -> None:
        self._check_dirs()
        spath = os.path.join(
            self.project_directory, self.save_directory, 
            self.ID_EBOD + '.geojson'
        )
        gdf.to_file(spath, driver='GeoJSON')

    def save_ebod_raster(self, ebodr:EbodRaster, overwrite:bool=False) -> None:
        self._check_dirs()
        ebodr.save_geotiff(
            os.path.join(self.project_directory, self.save_directory), 
            overwrite=overwrite
        )

    def _request(self, tx:int, ty:int, zoom:int) -> dict:
        url = self.URL_TILES + '{}/{}/{}.pbf'.format(zoom, tx, ty)
        pbf = requests.get(url).content
        vdata = mvt.decode(pbf)
        vdata['tile_coordinates'] = [tx, ty]
        vdata['zoom'] = zoom
        return vdata
        
    
    @staticmethod
    def _plgn2wgs84(
            coords:list[list[int, int]], tcoords:list[int, int], zoom:int, 
            extent:int
        ) -> Polygon:
        lat, lon = tilecoords2latlon(
            coords, tcoords[0], tcoords[1], zoom, extent
        )
        lonlat = np.vstack((np.atleast_2d(lon), np.atleast_2d(lat))).T
        return Polygon(lonlat)
