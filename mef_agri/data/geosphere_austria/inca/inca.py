import datetime
import numpy as np
import geopandas as gpd
from datetime import date

from ....utils.misc import PixelUnits
from ....utils.raster import GeoRaster


class INCALABELS:
    PRECIPITATION = 'RR'
    TEMPERATURE = 'T2M'
    PRESSURE = 'P0'
    HUMIDITY = 'RH2M'
    RADIATION = 'GL'
    WIND_EAST = 'UU'
    WIND_NORTH = 'VV'
    DEWPOINT = 'TD2M'


INCADATA = {
    INCALABELS.PRECIPITATION: {
        'descr': '1-hour precipitation sum',
        'units': '[kg/m2]'
    },
    INCALABELS.TEMPERATURE: {
        'descr': 'air temperature',
        'units': '[degCelsius]'
    },
    INCALABELS.PRESSURE: {
        'descr': 'mean sea-level pressure',
        'units': '[Pa]'
    },
    INCALABELS.HUMIDITY: {
        'descr': 'relative humidity',
        'units': '[percent]'
    },
    INCALABELS.RADIATION: {
        'descr': 'global radiation',
        'units': '[W/m2]'
    },
    INCALABELS.WIND_EAST: {
        'descr': 'wind speed in eastward direction',
        'units': '[m/s]'
    },
    INCALABELS.WIND_NORTH: {
        'descr': 'wind speed in northward direction',
        'units': '[m/s]'
    },
    INCALABELS.DEWPOINT: {
        'descr': 'dew point temparature',
        'units': '[degCelsius]'
    }
}


class IncaGridDaily(GeoRaster):
    EPSG_DATAPRC = 31287  # inca data processing is done in Austria Lambert by geosphere austria
    EPSG_REQUEST = 4326  # requests to inca api have to be done in WGS84
    META_LAYER_ID = 'layers'
    META_DATE_ID = 'date'
    GRID_COORDS_MAXDEV = 1.0  # [m] coordinate variations lower than this value will be assumed to be on the same dimension of the grid
    GRID_OBJRES = 1000.0  # [m] object resolution of the inca grid
    GDF_GEOMETRY_COL = 'geometry'
    GDF_TIMESTAMP_COL = 'timestamps'

    def __init__(self, data_id:str, epoch:date, **kwargs):
        """
        It contains the hourly values from one type of inca-data from geosphere 
        as its bands or channels.
        The `layer_ids`-attribute of parent-class `GeoRaster` corresponds to 
        iso-formatted time strings for the hourly inca weather data.

        The following quantities can be set through `kwargs`

        * `data_descr` - a detailed description of the data
        * `data_units` - units of the data

        :param data_id: inca data identifier (a key from `mef_agri.data.geosphere_austria.inca.inca.INCADATA`)
        :type data_id: str
        :param epoch: date of inca-data, stored in the metadata file (.json) if saved to disk
        :type epoch: datetime.date
        """
        super().__init__()
        self._meta['data_id'] = data_id
        self._meta['epoch'] = epoch.isoformat()

        self._set_metadata(['data_descr', 'data_units'], kwargs)

    @property
    def data_id(self) -> str:
        return self.metadata['data_id']
    
    @property
    def epoch(self) -> date:
        return date.fromisoformat(self.metadata['epoch'])
    
    @classmethod
    def from_points(
        cls, gdf:gpd.GeoDataFrame, inca_data_id:str, epoch:date, target_crs:int
    ) -> None:
        """
        Create grid-data from points with inca data as attribute. The geometry 
        column should be named `geometry` (otherwise change the 
        `GDF_GEOMETRY_COL` attribute) and the timestamp column should be named 
        `timestamps` (otherwise change the `GDF_TIMESTAMP_COL` attribute). The 
        column-name with inca-data should correspond to one of the keys of 
        `mef_agri.data.geosphere_austria.inca.inca.INCADATA`.

        The geodataframe has to be defined in WGS84!!!
        """
        if gdf.crs.to_epsg() != cls.EPSG_REQUEST:
            msg = 'Provided `gdf` has to be defined in WGS84!'
            raise ValueError(msg)

        inca = cls(
            inca_data_id, epoch, 
            data_descr=INCADATA[inca_data_id]['descr'],
            data_units=INCADATA[inca_data_id]['units'],
        )

        # create grid
        # problem: coordinates are not exactly the same in one direction of the 
        # grid but scatter by some centimeters
        # approach: transform from WGS84 [deg, deg] to Austria-Lambert [m, m]
        # (according to geosphere, the 1km-Raster is defined in Austria-Lambert)
        # find minimum x- and maximum y-coordinate (upper-left) and
        # process the points "row-wise" (i.e. in x-direction) by adding 
        # consecutively 1km;
        gdf.to_crs(cls.EPSG_DATAPRC, inplace=True)

        # create numpy array from points
        parr = np.vstack((
            np.atleast_2d(gdf[inca.GDF_GEOMETRY_COL].x.values),
            np.atleast_2d(gdf[inca.GDF_GEOMETRY_COL].y.values)
        ))
        # determine rastershape from parr
        width = inca._det_shape_from_points(parr[0, :].copy())
        height = inca._det_shape_from_points(parr[1, :].copy())
        if 0 in (width, height):
            errmsg = 'Provided points do not represent a 1km x 1km grid - '
            errmsg += 'deviations are bigger than {} m.'.format(
                inca.GRID_COORDS_MAXDEV
            )
            raise ValueError(errmsg)
        depth = len(gdf[inca.GDF_TIMESTAMP_COL].iloc[0])
        inca.layer_index = 0
        inca.raster = np.zeros((depth, height, width), dtype=inca.units)
        inca.raster_shape = (depth, height, width)
        inca.units = PixelUnits.FLOAT64
        inca.nodata_value = np.nan

        # fill values into inca.raster
        ul = np.array([[parr[0, :].min(), parr[1, :].max()]]).T
        uladd = np.zeros((2, 1))
        for i in range(height):
            for j in range(width):
                diffs = parr - (ul + uladd)
                ix = np.argmin(np.linalg.norm(diffs, axis=0))
                inca.raster[:, i, j] = gdf[inca.data_id].iloc[ix]

                uladd[0, 0] += inca.GRID_OBJRES
            uladd[0, 0] = 0.0
            uladd[1, 0] -= inca.GRID_OBJRES

        # set geo-reference for geotiff
        ulcorr = inca.GRID_OBJRES / 2.0
        ul += np.array([[-ulcorr, ulcorr]]).T
        inca.transformation = np.array([
            [inca.GRID_OBJRES, 0.0, ul[0, 0]],
            [0.0, -inca.GRID_OBJRES, ul[1, 0]],
            [0.0, 0.0, 1.0]
        ])
        inca.bounds = (
            ul[0, 0],
            ul[1, 0] - height * inca.GRID_OBJRES,
            ul[0, 0] + width * inca.GRID_OBJRES,
            ul[1, 0]
        )
        inca.crs = gdf.crs.to_epsg()

        # timestamps
        tstrs = list(gdf[inca.GDF_TIMESTAMP_COL])[0]
        inca.layer_ids = []
        for tstr in tstrs:
            inca.layer_ids.append(
                datetime.datetime.fromisoformat(tstr).time().isoformat()
            )

        if target_crs != inca.crs:
            inca.reproject(target_crs)
        return inca
        
    def _det_shape_from_points(self, arr:np.ndarray) -> int:
        arr -= arr.min()
        thresh = self.GRID_OBJRES - self.GRID_COORDS_MAXDEV
        ret = 1
        while arr.max() > thresh:
            ix = np.where(arr > thresh)[0]
            arr[ix] -= self.GRID_OBJRES
            ret += 1
        if arr.max() > self.GRID_COORDS_MAXDEV:
            ret = 0
        return ret
