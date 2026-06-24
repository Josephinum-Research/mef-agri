import os
import requests
import numpy as np
import geopandas as gpd
import mapbox_vector_tile as mvt
from shapely.geometry import Polygon
from copy import deepcopy

from ...utils.gis import latlon2tilexy, tilecoords2latlon, bbox_from_gdf
from .ebod import EBOD_PROPERTIES, EbodRaster
from ..interface import Interface


class EbodInterface(Interface):
    REQU_DATA_V1 = {
        'url_meta': 'https://bodenkarte.at/data/bodenkarte-tiles.json',
        'url_tiles': 'https://bodenkarte.at/data/bodenkarte-tiles/',
        'key_meta_maxzoom': 'maxzoom',
        'key_bf': 'bodenform_mpoly',
        'key_ext': 'extent',
        'key_feat': 'features',
        'key_geom': 'geometry',
        'key_coords': 'coordinates',
        'key_type': 'type',
        'key_props': 'properties'
    }

    def __init__(self):
        super().__init__()

        # properties from parent class
        self.data_source_id = 'soil_ebod'
        self.description = """
            Interface to get data from https://bodenkarte.at
        """
        self.georaster_class = EbodRaster
        self.static_data = True

        # private attributes
        self._objres:float = 10.
        self._excl:list[str] = ['bofo_id', 'kurzbezeichnung', 'gruenlandwert']
        self._zoom = 15
        self._rs = self.REQU_DATA_V1
        self._maxzoom = None
        self._ebodmd = None
        self._requgdf = None
        self._requrstr = None
        self._aoirstr = None

    ############################################################################
    # PROPERTIES
    @property
    def request_specifications(self) -> dict:
        """
        Settable

        :return: Specifications to perform request and extract necessary data from request-response, see ``REQU_DATA_V1`` in :class:`EbodInterface`
        :rtype: dict
        """
        return self._rs
    
    @request_specifications.setter
    def request_specifications(self, value):
        self._rs = value

    @property
    def object_resolution(self) -> float:
        """
        Settable

        :return: object resolution which is applied when rasterizing the vector tiles, defaults to ``10.`` [m]
        :rtype: float
        """
        return self._objres
    
    @object_resolution.setter
    def object_resolution(self, value):
        self._objres = value

    @property
    def apply_zoom(self) -> int:
        """
        Setter

        Zoom level which should be used to request vector tiles and when 
        rasterizing them.
        This value will only be applied if the metadata-request to ebod failes 
        or `maxzoom` is not available in the requested metadata.

        :return: zoom-level, defaults to ``15``
        :rtype: int
        """
        return self._zoom
    
    @apply_zoom.setter
    def apply_zoom(self, value):
        self._zoom = value

    @property
    def exclude_properties(self) -> list[str]:
        """
        Setter

        possible values: keys of ``mef_agri.data.ebod_austria.ebod.EBOD_PROPERTIES``

        :return: ebod-properties which should not be used for rasterization, defaults to ``['bofo_id', 'kurzbezeichnung', 'gruenlandwert']``
        :rtype: list[str]
        """
        return self._excl
    
    @exclude_properties.setter
    def exclude_properties(self, value):
        self._excl = value

    @property
    def ebod_metadata(self) -> dict:
        """
        :return: requested metadata
        :rtype: dict
        """
        return self._ebodmd
    
    @property
    def ebod_data_gdf(self) -> gpd.GeoDataFrame:
        """
        Columns of the GeoDataFrame correspond to the ebod-properties specified 
        in ``mef_agri.data.ebod_austria.ebod.EBOD_DATA``.
        The rows correspond to the vector features of all tiles which intersect 
        with the provided area-of-interest.

        :return: requested ebod-data
        :rtype: geopandas.GeoDataFrame
        """
        return self._requgdf
    
    @property
    def ebod_data_raster(self) -> EbodRaster:
        """
        :return: rasterized :func:`ebod_data_gdf` - the ebod-properties are the layers of the raster
        :rtype: EbodRaster
        """
        return self._requrstr
    
    ############################################################################
    # METHODS
    def request_metadata(self):
        """
        Request metadata from ebod.
        """
        try:
            self._ebodmd = requests.get(self._rs['url_meta']).json()
            self._maxzoom = int(self._ebodmd[self._rs['key_meta_maxzoom']])
        except:
            self._maxzoom = self.apply_zoom

    def request_data(self, aoi:gpd.GeoDataFrame, zoom:int=None) -> None:
        """
        Download data as vector tiles from ebod. 
        Vector tiles from ebod are defined according to 
        https://docs.mapbox.com/data/tilesets/guides/vector-tiles-standards/.

        :param aoi: defining the area of interest and crs of resulting ebod data
        :type aoi: gpd.GeoDataFrame
        :param zoom: overrides :func:`apply_zoom` or maxzoom from :func:`ebod_metadata` if provided - defaults to None
        :type zoom: int, optional
        """
        def append_properties(feat, data, coords, tcs, zoom, extent):
            data['geometry'].append(self._plgn2wgs84(
                coords, tcs, zoom, extent
            ))
            for key, val in EBOD_PROPERTIES.items():
                insval = np.nan
                if key in feat[self._rs['key_props']].keys():
                    insval = feat[self._rs['key_props']][key]
                data[val].append(insval)
            return data
        
        if zoom is None:
            zoom = self._maxzoom

        self._aoirstr = deepcopy(aoi)
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
                bf = vdata[self._rs['key_bf']]
                extent = bf[self._rs['key_ext']]
                for feat in bf[self._rs['key_feat']]:
                    geom = feat[self._rs['key_geom']]
                    coords = geom[self._rs['key_coords']]
                    if geom[self._rs['key_type']] == 'Polygon':
                        data = append_properties(
                            feat, data, coords[0], [txi, tyi], zoom, extent
                        )
                    elif geom[self._rs['key_type']] == 'MultiPolygon':
                        for coord in coords:
                            data = append_properties(
                                feat, data, coord[0], [txi, tyi], zoom, extent
                            )
                    else:
                        # other geometry type which is not considered here until now
                        pass

        gdf_tiles = gpd.GeoDataFrame(data, crs=4326)
        if tcrs != 4326:
            gdf_tiles.to_crs(tcrs, inplace=True)
        self._requgdf = gdf_tiles

    def rasterize(self, objres:float=None) -> None:
        """
        Rasterize :func:`ebod_data_gdf`

        :param objres: overrides :func:`object_resolution` if provided, defaults to None
        :type objres: float, optional
        """
        self._requrstr = EbodRaster.from_vector_tile_geodataframe(
            self.ebod_data_gdf, self._aoirstr, objres, self.exclude_properties
        )

    @Interface.add_data_task
    def prj_add_ebod(self):
        self.progress = 'starting requests'

        self.request_metadata()
        self.request_data(self.aoi, self._maxzoom)
        self.rasterize()
        self.ebod_data_raster.save_geotiff(
            self.directory, overwrite=True, compress=False
        )

        self.progress = 'successfully saved ebod-data'

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
