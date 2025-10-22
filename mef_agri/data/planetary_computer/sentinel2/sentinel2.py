import numpy as np
from datetime import datetime, date, time
from xarray import Dataset

from ....utils.raster import GeoRaster, PixelUnits
from ....utils.gis import trafo_from_affine


class SCLMAP:
    NO_DATA = 0
    DEFECTIVE = 1
    SHADOW_TOPOGRAPHY = 2
    SHADOW_CLOUDS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUDS_MEDPROB = 8
    CLOUDS_HIGHPROB = 9
    THIN_CIRRUS = 10
    SNOW = 11


class ImageSentinel2(GeoRaster):
    META_IMG_TIMESTAMP = 'image-timestamp'
    META_IMG_DATE = 'image-date'
    META_IMG_TIME = 'image-time'

    def __init__(self):
        super().__init__()
        self._dt:datetime = None

    @property
    def epoch(self) -> date:
        return self._dt.date()

    @property
    def timestamp(self) -> time:
        return self._dt.time()

    @classmethod
    def from_xrdataset(cls, xds:Dataset, target_epsg:int=None):
        layers = list(xds.keys())
        img = cls()

        # temporal information
        img._dt = datetime.fromisoformat(
            np.datetime_as_string(xds.time.to_numpy())[0]
        )
        img.metadata[img.META_IMG_TIMESTAMP] = img._dt.isoformat()
        img.metadata[img.META_IMG_DATE] = img._dt.date().isoformat()
        img.metadata[img.META_IMG_TIME] = img._dt.time().isoformat()

        # spatial information
        img.crs = xds.rio.crs.to_epsg()
        img.transformation = trafo_from_affine(xds.rio.transform())
        img.bounds = xds.rio.bounds()
        img.layer_index = 0
        img.layer_ids = layers
        img.units = PixelUnits.FLOAT32
        img.nodata_value = np.nan

        # raster data
        img.raster_shape = (15, *xds.rio.shape)
        img.raster = np.zeros((15, *xds.rio.shape))
        for lname, i in zip(layers, range(len(layers))):
            img.raster[i] = getattr(xds, lname).to_numpy()

        if (target_epsg is not None) and (img.crs != target_epsg):
            img.reproject(target_epsg)
        return img
