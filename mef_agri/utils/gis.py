import numpy as np
import math as m
import rasterio as rio
import rasterio.windows as riowdw
from rasterio.transform import Affine
from shapely.geometry import Polygon, MultiPolygon
from geopandas import GeoDataFrame


EARTH_SPHERE_RADIUS = 6371000.0  # [m]


################################################################################
# CRS STUFF
################################################################################
def get_epsg_from_rio_crs(crs:rio.CRS) -> int:
    """
    Get the EPSG code as integer from a crs of type 
    `rasterio.CRS`.

    :param crs: crs defined by rasterio
    :type crs: rio.CRS
    :return: epsg code
    :rtype: int
    """
    epsg = crs.to_epsg()
    if epsg is None:
        return map_proj4_to_epsg(crs.data)
    else:
        return epsg


def map_proj4_to_epsg(dprj:dict) -> int:
    """
    Map a crs with proj4 definition to epsg code. `dprj` has to have at least 
    the following keys: 'proj', 'datum'

    :param dprj: proj4 definition `rasterio.CRS.data`
    :type dprj: dict
    :raises ValueError: raised if `dprj` lacks necessary keys
    :return: epsg code
    :rtype: int
    """
    neckeys = ['proj', 'datum']
    for key in neckeys:
        if not key in dprj.keys():
            errmsg = 'Provided dict does not contain neccessary keys: '
            errmsg = [errmsg + ', ' + nk for nk in neckeys]
            raise ValueError(errmsg)
    
    if dprj['proj'] == 'utm':
        if not 'zone' in dprj.keys():
            errmsg = 'If projection is utm a zone key has to be provided in '
            errmsg += 'the data-dict!'
            raise ValueError(errmsg)
        if dprj['datum'] == 'WGS84':
            if dprj['zone'] == 32:
                return 32632
            elif dprj['zone'] == 33:
                return 32633


################################################################################
# CONVERSION STUFF
################################################################################
def imgshape_from_rasterio(arr:np.ndarray) -> np.ndarray:
    return np.moveaxis(arr, 0, 2)


def imgshape_to_rasterio(arr:np.ndarray) -> np.ndarray:
    return np.moveaxis(arr, 2, 0)


def trafo_from_affine(trafo:Affine) -> np.ndarray:
    return np.array([
        [trafo.a, trafo.b, trafo.c],
        [trafo.d, trafo.e, trafo.f],
        [trafo.g, trafo.h, trafo.i]
    ])


def affine_from_trafo(trafo:np.ndarray) -> Affine:
    return Affine(
        a=trafo[0, 0], b=trafo[0, 1], c=trafo[0, 2],
        d=trafo[1, 0], e=trafo[1, 1], f=trafo[1, 2],
        g=trafo[2, 0], h=trafo[2, 1], i=trafo[2, 2]
    )


################################################################################
# BBOX STUFF
################################################################################
def bbox2polygon(xmin:float, ymin:float, xmax:float, ymax:float) -> Polygon:
    """
    Create `shapely.geometry.Polygon` from given bounding box.

    :param xmin: min. x-coordinate
    :type xmin: float
    :param ymin: min. y-coordinate
    :type ymin: float
    :param xmax: max. x-coordinate
    :type xmax: float
    :param ymax: max. y-coordinate
    :type ymax: float
    :return: geometry representing the bounding box
    :rtype: Polygon
    """
    p1 = (xmin, ymin)
    p2 = (xmin, ymax)
    p3 = (xmax, ymax)
    p4 = (xmax, ymin)
    return Polygon(shell=(p1, p2, p3, p4))


def bbox_from_gdf(gdf:GeoDataFrame) -> tuple[float, float, float, float]:
    """
    Derive bounding box from given `geopandas.GeoDataFrame`.

    :param gdf: GeoDataFrame containing the individual geometries
    :type gdf: GeoDataFrame
    :return: bounding box containing all geometries within `gdf`
    :rtype: tuple[float, float, float, float]
    """
    dfb = gdf.bounds
    return (
        dfb.minx.min(), dfb.miny.min(), dfb.maxx.max(), dfb.maxy.max()
    )


def bbox_from_aoi(aoi) -> tuple[int, int, int, int]:
    """
    Derive bounding box from provided area-of-interest which can an instance of 
    the following three class

    * `geopandas.GeoDataFrame`
    * `shapely.geometry.Polygon`
    * `shapely.geometry.MultiPolygon`

    :param aoi: area-of-interest
    :type aoi: geometry-type (see docstring)
    :return: bounding box
    :rtype: tuple[int, int, int, int]
    """
    if isinstance(aoi, GeoDataFrame):
        return bbox_from_gdf(aoi)
    elif isinstance(aoi, Polygon):
        return aoi.bounds
    elif isinstance(aoi, MultiPolygon):
        return bbox_from_gdf(GeoDataFrame({'geometry': list(aoi)}))
    else:
        raise NotImplementedError()


def get_rio_window_from_bbox(
        transform:Affine, bbox_wdw:list[float, float, float, float]
    ) -> riowdw.Window:
    trafo = np.linalg.inv(trafo_from_affine(transform))
    ulobj = np.array([[bbox_wdw[0], bbox_wdw[3], 1.0]]).T
    lrobj = np.array([[bbox_wdw[2], bbox_wdw[1], 1.0]]).T
    ul = (trafo @ ulobj).astype('int').flatten()[:2]
    lr = (trafo @ lrobj).astype('int').flatten()[:2]
    return riowdw.Window(ul[0], ul[1], lr[0] - ul[0] + 1, lr[1] - ul[1] + 1)


def get_bbox_from_rio_window(
        transform:Affine, wndw:riowdw.Window
    ) -> list[float, float, float, float]:
    trafo = trafo_from_affine(transform)
    ulimg = np.array([[wndw.col_off, wndw.row_off, 1]]).astype('float').T
    lrimg = np.array([[
        wndw.col_off + wndw.width,
        wndw.row_off + wndw.height,
        1
    ]]).astype('float').T
    ul = (trafo @ ulimg).flatten()[:2]
    lr = (trafo @ lrimg).flatten()[:2]
    return [ul[0], lr[1], lr[0], ul[1]]


################################################################################
# TILE STUFF
################################################################################
def latlon2tilexy(lat:float, lon:float, zoom:int) -> tuple[int, int]:
    """
    Convert lat/lon (WGS84) to x- and y-tile-coordinates according to: 
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

    :param lat: latitude WGS84 [deg]
    :type lat: float
    :param lon: longitude WGS84 [deg]
    :type lon: float
    :param zoom: zoom level
    :type zoom: int
    :return: x- and y-tile-coordinates
    :rtype: tuple[int, int]
    """
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)
    ty = int((1.0 - m.asinh(m.tan(m.radians(lat))) / m.pi) / 2.0 * n)
    return (tx, ty)

def tilexy2latlon(tx:int, ty:int, zoom:int) -> tuple[float, float]:
    """
    Convert tile coordinates into lat/lon (WGS84). This represents the 
    upper-left corner of the tile.
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

    :param tx: x-coordinate of the tile []
    :type tx: int
    :param ty: y-coordinate of the tile []
    :type ty: int
    :param zoom: zoom level
    :type zoom: int
    :return: WGS84 coordinates of upper left corner of the tile [deg]
    :rtype: tuple[float, float]
    """
    n = 2 ** zoom
    lon = (tx / n) * 360.0 - 180.0
    lat = m.degrees(m.atan(m.sinh(m.pi * (1.0 - (ty / n) * 2.0))))
    return (lat, lon)

def tilecoords2latlon(
        coords:list[list[int, int]], tx:int, ty:int, zoom:int, extent:int
    ) -> tuple[list[float], list[float]]:
    """
    Convert the x- and y-coordinates within a vector tile into lat/lon (WGS84).
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

    :param coords: x- and y-coordinates within the vector tile []
    :type coords: list[list[int, int]]
    :param tx: x-tile-coordinate [] (left edge)
    :type tx: int
    :param ty: y-tile-coordinate [] (top edge)
    :type ty: ibnt
    :param zoom: zoom level
    :type zoom: int
    :param extent: extent of the vector tile
    :type extent: int
    :return: WGS84 coordinates
    :rtype: tuple[list[float], list[float]]
    """

    lat0, lon0 = tilexy2latlon(tx, ty, zoom)
    lat1, lon1 = tilexy2latlon(tx + 1, ty + 1, zoom)
    dlat = (lat0 - lat1) / extent
    dlon = (lon1 - lon0) / extent

    lat, lon = [], []
    for xy in coords:
        lat.append(xy[1] * dlat + lat1)
        lon.append(xy[0] * dlon + lon0)

    return lat, lon
