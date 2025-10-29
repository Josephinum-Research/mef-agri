import numpy as np
from geopandas import GeoSeries, GeoDataFrame
from copy import deepcopy
from shapely.geometry import Polygon, Point

from ..utils.raster import GeoRaster, PixelUnits
from ..utils.misc import set_attributes
from ..utils.rv_manipulation import RasterVectorIntersection


################################################################################
##############################   PARENT CLASS   ################################
################################################################################
class Field(GeoRaster):
    """
    Class which represents a field and is responsible for the zoning 
    (specific zoning algorithm/approach should be implemented in a child 
    class). The `raster` attribute (from parent class `GeoRaster`) holds 
    the zone indices and the `zones` attribute equals the corresponding list 
    (i.e. indices in `raster` equals the index in `zones`) which holds  
    dictionaries with information for each zone and `zone_ids` is a list 
    with same order containing the unique zone ids.

    Zone element in the dictionary of :func:`zones`

    * zname - name of the zone
    * gcs - geo coordinates representing the zone - (2, n) numpy.ndarray
    * lat - mean latitudeo of the zone (see :func:`_get_mean_latitude_wgs84`)

    This class also has the `geo_coordinates` attribute which holds the 
    geo-coordinates of the raster as (2, n) numpy.ndarray.

    Attributes which can be set

    * **n_zones** - number of zones to which the field should be divided (note: child classes may fix this value)
    * **intersection_percentage** - fraction which is used to decide if raster element should be used to represent field

    :param name: name of the field
    :type name: str
    :param height: representative height for the field (WGS84)
    :type height: float
    """
    EPSG_WGS84 = 4326

    def __init__(self, name:str, height:float) -> None:
        super().__init__()
        self._nz:int = 1  # NOTE 0 means, that each zoning element will be treated as a separate zone
        self._ip:float = 1.0  # this value is used in RasterVectorIntersection for the `fraction` attribute
        
        self._height:float = height
        self._name:str = name
        self._zones:list = []
        self._zids:list = []
        self._gcs:np.ndarray = None  # geo-coordinates of pixels inside the provided aoi
        self._plgn:Polygon = None

        # georaster related attributes
        self.units = PixelUnits.INT32
        self.nodata_value = -1
        self.layer_index = 0
        self.layer_ids = ['zones']

    @property
    def n_zones(self) -> int:
        return self._nz
    
    @property
    def intersection_percentage(self) -> float:
        return self._ip

    @property
    def height(self) -> float:
        """
        :return: representative height of the field [m]
        :rtype: float
        """
        return self._height
    
    @property
    def name(self) -> str:
        """
        :return: name of the field
        :rtype: str
        """
        return self._name

    @property
    def polygon(self) -> Polygon:
        """
        :return: shape of the field
        :rtype: Polygon
        """
        return self._plgn
    
    @property
    def geo_coordinates(self) -> np.ndarray:
        """
        :return: geo-coordinates of all raster points - (2, n) numpy.ndarray
        :rtype: np.ndarray
        """
        return self._gcs
    
    @property
    def zones(self) -> list:
        """
        :return: list of dicts with zone-information (details in class documentation)
        :rtype: list
        """
        return self._zones
    
    @property
    def zone_ids(self) -> list:
        """
        :return: list containing unique zone ids
        :rtype: list
        """
        return self._zids
    
    #############################   METHODS   ##################################
    def set_aoi(
            self, plgn:Polygon, crs:int, obj_res:float
        ) -> RasterVectorIntersection:
        """
        This method creates the raster (i.e. setting the corresponding 
        attributes of parent class `GeoRaster`) representing the field from the 
        provided polygon (`plgn`) and desired object-resolution (`obj_res`).

        :param plgn: shapely polygon representing the field
        :type plgn: shapely.gemetry.Polygon
        :param crs: epsg code of coordinate reference system of `plgn`
        :type crs: int
        :param obj_res: desired resolution of the raster representing the field
        :type obj_res: float
        :return: `RasterVectorIntersection` object from `sitespecificcultivation.utils.rv_manipulation`
        :rtype: RasterVectorIntersection
        """

        if isinstance(plgn, GeoSeries):
            plgn = plgn.values[0]
        self._plgn = deepcopy(plgn)
        self.crs = deepcopy(crs)
        self.bounds = plgn.bounds
        self.transformation = np.array([
            [obj_res,  0.0, self._bbox[0]],
            [0.0, -obj_res, self._bbox[3]],
            [0.0, 0.0, 1.0]
        ])
        # TODO check if this holds for hexagons as raster element and also for different reference points
        self.raster_shape = (
            1,
            int((self._bbox[3] - self._bbox[1]) // obj_res) + 1,
            int((self._bbox[2] - self._bbox[0]) // obj_res) + 1
        )

        rvi = RasterVectorIntersection(plgn, self)
        rvi.fraction = self.intersection_percentage
        rvi.compute()
        self._gcs = (self.transformation @ np.vstack((
            np.flipud(rvi.inside_indices), 
            np.ones((1, rvi.inside_indices.shape[1]))
        )))[:2, :]
        return rvi
    
    def determine_zones(self) -> None:
        """
        Mehod which is called to determine the zones.
        Has to be implemented in a child class of `Field`.
        """
        raise NotImplementedError()
    
    def _get_mean_latitude_wgs84(self, gcs:np.ndarray=None) -> float:
        if gcs is None:
            if self._gcs is None:
                msg =  'No geo-coordinates are available for the field, call '
                msg += 'set_aoi() first!'
                raise ValueError(msg)
            gcs = self._gcs
        
        if self._crs != self.EPSG_WGS84:
            gdf = GeoDataFrame(
                {'geometry': [Point(*np.mean(gcs, axis=1))]},
                crs=self._crs
            )
            gdf.to_crs(self.EPSG_WGS84, inplace=True)
            lat = gdf['geometry'].values[0].y
        else:
            lat = np.mean(gcs[1, :])  # second row corresponds to the y-value ("Hochwert")
        
        return lat * (np.pi / 180.0)  # NOTE conversion to [rad], as latitude in WGS84 comes at [deg]
    

################################################################################
#############################   CHILD CLASSES   ################################
################################################################################
class Field_OneZone(Field):
    ZONE_NAME = 'zone'

    def __init__(self, name:str, height:float, **kwargs) -> None:
        super().__init__(name, height, **kwargs)
        self.n_zones = 1

    def determine_zones(self) -> None:
        zentry = {
            'zname': self.ZONE_NAME,
            'lat': self._get_mean_latitude_wgs84(),
            'gcs': self.geo_coordinates
        }
        self._zones = [zentry,]
        self._zids = [self.name + '.' + self.ZONE_NAME,]
        self._rstr = np.zeros(self.raster_shape, dtype=int)
