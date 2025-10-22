import os
import numpy as np
from xarray import DataArray, Dataset, merge
from geopandas import GeoSeries
from odc import stac as odc_stac
from odc.geo.geobox import AnchorEnum
from odc.geo.xr import xr_reproject
from datetime import datetime
from copy import deepcopy
from shapely.geometry import Polygon
from pystac_client import Client
from planetary_computer import sign_inplace

from ....utils.gis import bbox_from_gdf, bbox2polygon

from ...interface import DataInterface
from .harmonization import harmonization
from .granule_metadata import parse_granule_metadata_lazy, simplified_metadata
from .sentinel2 import ImageSentinel2


DN_NODATA = 0
BOA_QUANTIFY = 10_000
REFLBANDS = [
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'
]
DATABANDS = REFLBANDS + ['SCL']
METABANDS = [
    'SUN_AZIMUTH', 'SUN_ZENITH', 'VIEW_AZIMUTH_MEAN', 'VIEW_ZENITH_MEAN'
]
BANDS_JR = DATABANDS + METABANDS


class Sentinel2Interface(DataInterface):
    DATA_SOURCE_ID = 'sentinel-2_planetary-computer'
    URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'
    SENTINEL2_L2A = 'sentinel-2-l2a'
    CRS_REQUEST = 4326  # WGS84 EPSG Code

    def __init__(
            self, obj_res=10, collections:list[str]=None
        ):
        super().__init__(obj_res)
        self._catalog = Client.open(self.URL, modifier=sign_inplace)
        self._colls = [self.SENTINEL2_L2A,] if collections is None \
                      else collections

    def add_prj_data(self, aoi, tstart, tstop):
        self.add_prj_data_error = False
        self.save_images(self.get_images(aoi, tstart, tstop))
        return tstart, tstop

    def get_images(self, aoi, tstart, tstop) -> list[ImageSentinel2]:
        depsg = aoi.crs.to_epsg()
        if depsg is None:
            errmsg = 'Provide GeoDataFrame with proper CRS and EPSG-code!'
            raise ValueError(errmsg)
        # transform input GeoDataFrame to WGS84 if necessary
        rgdf = None
        if aoi.crs.to_epsg() != self.CRS_REQUEST:
            rgdf = aoi.to_crs(self.CRS_REQUEST)
        else:
            rgdf = deepcopy(aoi)
        trng = [
            datetime(*tstart.timetuple()[:6]), datetime(*tstop.timetuple()[:6])
        ]

        search = self._catalog.search(
            collections=self._colls,
            bbox=bbox_from_gdf(rgdf),
            datetime=trng[0].isoformat() + '/' + trng[1].isoformat()
        )
        items = list(search.items())
        # iterate over requested items (each loop corresponds to an satellite 
        # image for a specific date)
        imgs, dates = [], []
        for item in items:
            try:
                # check if provided bbox or area of interest from gdf lies fully 
                # within the requested geotiff
                # provided satellite images overlap by approximate 10 km, i.e. if 
                # aoi touches more than one image, multiple images will be provided 
                # by planetary computer for the same date
                # ASSUMPTION: aoi lies fully in at least one image (problem if aoi 
                # extent is bigger than 10km and aoi is in overlapping area)
                prd_plgn = Polygon(item.to_dict()['geometry']['coordinates'][0])
                aoi_plgn = bbox2polygon(*bbox_from_gdf(rgdf))
                if not prd_plgn.contains(aoi_plgn):
                    continue

                # check if there is already a saved image for a specfic date (can 
                # happen if aoi is in the intersection area of two temporal 
                # consecutive geotiffs)
                prd_date = item.datetime.isoformat().split('T')[0]
                if prd_date in dates:
                    continue
                dates.append(prd_date)

                print('-) processing image for ' + prd_date)
                hitem = harmonization(item)
                aoi_ser = GeoSeries([aoi.geometry.values[0]], crs=aoi.crs)

                ################################################################
                # fetch all image data resulting in a xarray dataset `ds_refl`
                ds_data = odc_stac.load(
                    items=[hitem,], bands=DATABANDS, geopolygon=aoi_ser, 
                    resolution=self._ores, anchor=AnchorEnum.EDGE, chunks={}
                )

                # map to reflectance values
                arrd = [int(x.datetime.timestamp() * 1000) for x in [hitem,]]
                assert np.all(
                    ds_data.time == np.array(arrd, dtype='datetime64[ms]')
                )
                # all assets of one item are computed by using the same 
                # processing baseline.
                # therefore we don't need to check offset information of every 
                # band. we use the info's of B4 instead.
                offsets = np.array(
                    [hitem.assets['B04'].ext.raster.bands[0].offset],
                    dtype=np.float32
                )
                ds_offs = DataArray(data=offsets, dims=['time'], coords={
                    'time': ds_data.coords['time']
                })
                ds_refl = ds_data.copy()
                for key in ds_refl.data_vars:
                    if key in REFLBANDS:
                        band:DataArray = ds_refl[key]
                        band_as_float = band.astype(np.float32)
                        band_no_data_mapped = band_as_float.where(
                            lambda x: x != DN_NODATA, other=np.nan
                        )
                        band_reflectance = (
                            band_no_data_mapped / BOA_QUANTIFY
                        ) + ds_offs
                        band_harmonized = band_reflectance.where(
                            lambda x: x > 0, other=0.
                        )
                        ds_refl[key] = band_harmonized

                ################################################################
                # fetch common and granule metadata and save angle data to 
                # xarray dataset `ds_meta`
                mdg = parse_granule_metadata_lazy(hitem)
                md = simplified_metadata(mdg)
                ds_meta = Dataset()
                for band in BANDS_JR:
                    di = None
                    if band == 'SUN_ZENITH':
                        di = md.sun_angles.sel(spherical='zenith')
                    elif band == 'SUN_AZIMUTH':
                        di = md.sun_angles.sel(spherical='azimuth')
                    elif band == 'VIEW_ZENITH_MEAN':
                        di = md.view_angles_mean.sel(spherical='zenith')
                    elif band == 'VIEW_AZIMUTH_MEAN':
                        di = md.view_angles_mean.sel(spherical='azimuth')
                    if di is not None:
                        ds_meta[band] = di

                ds_meta = xr_reproject(
                    ds_meta, how=ds_refl.odc.geobox, resampling='nearest', 
                    chunks=ds_refl.odc.geobox.shape
                )
                ds_meta = ds_meta.drop_vars('spatial_ref')

                ################################################################
                # create georaster from xarray dataset
                imgs.append(ImageSentinel2.from_xrdataset(
                    merge([ds_refl, ds_meta]), aoi.crs.to_epsg()
                ))

            except Exception as exc:
                print(exc)
                if str(exc) == 'No information found in metadata for band SCL!':
                    msg = 'Ignoring image for current date and further '
                    msg += 'process the other dates.'
                    print(msg)
                else:
                    msg = 'Stop looping over further dates!'
                    self.add_prj_data_error = True
                    print(msg)
                    break
        
        return imgs
    
    def get_prj_data(self, epoch):
        fpath = os.path.join(
            self.project_directory, self.save_directory, epoch.isoformat()
        )
        if not os.path.exists(fpath):
            return
        img = ImageSentinel2()
        img.load_geotiff(fpath)
        return {'sentinel-2-image': img}
    
    def save_images(self, images:list[ImageSentinel2]) -> None:
        self._check_dirs()
        spath = os.path.join(self.project_directory, self.save_directory)
        for img in images:
            imgp = os.path.join(spath, img.epoch.isoformat())
            if not os.path.exists(imgp):
                os.mkdir(imgp)
            img.save_geotiff(imgp)
