import os
import numpy as np
from xarray import DataArray, Dataset, merge
from geopandas import GeoSeries, GeoDataFrame
from odc import stac as odc_stac
from odc.geo.geobox import AnchorEnum
from odc.geo.xr import xr_reproject
from datetime import datetime, date
from copy import deepcopy
from shapely.geometry import Polygon
from pystac_client import Client
from planetary_computer import sign_inplace

from ....utils.gis import bbox_from_gdf, bbox2polygon

from ...interface import Interface
from .harmonization import harmonization
from .granule_metadata import parse_granule_metadata_lazy, simplified_metadata
from .sentinel2 import ImageSentinel2


class Sentinel2Interface(Interface):
    BAND_DEFS = {
        'reflectance': [
            'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'
        ],
        'data': ['SCL',],
        'meta': [
            'SUN_AZIMUTH', 'SUN_ZENITH', 'VIEW_AZIMUTH_MEAN', 'VIEW_ZENITH_MEAN'
        ]
    }
    REQU_DATA_V1 = {
        'url': 'https://planetarycomputer.microsoft.com/api/stac/v1',
        'collection': 'sentinel-2-l2a',
        'crs': 4326
    }
    PROCESS_DATA = {
        'dn_nodata': 0,
        'boa_quantify': 10_000,
        'ref_band': 'B04',
        'sun_azimuth': BAND_DEFS['meta'][0],
        'sun_zenith': BAND_DEFS['meta'][1],
        'view_azimuth': BAND_DEFS['meta'][2],
        'view_zenith': BAND_DEFS['meta'][3]
    }

    def __init__(self):
        super().__init__()
        self.data_source_id = 'sentinel-2_planetary-computer'
        self.description = """
            Interface to gather data from 
            'https://planetarycomputer.microsoft.com/api/stac/v1'
        """
        self.georaster_class = ImageSentinel2

        self._rs = self.REQU_DATA_V1
        self._bd = self.BAND_DEFS
        self._pd = self.PROCESS_DATA
        self._objres = 10.

    @property
    def request_specifications(self) -> dict:
        """
        Settable

        :return: Specifications to perform request and extract necessary data from request-response, see ``REQU_DATA_V1`` in :class:`Sentinel2Interface`
        :rtype: dict
        """
        return self._rs
    
    @request_specifications.setter
    def request_specifications(self, value):
        self._rs = value

    @property
    def band_definitions(self) -> dict:
        """
        Settable

        :return: band names/identifiers (for the structure of the dict see ``BAND_DEFS`` of :class:`Sentinel2Interface`)
        :rtype: dict
        """
        return self._bd
    
    @band_definitions.setter
    def band_definitions(self, value):
        self._bd = value

    @property
    def process_specifications(self) -> dict:
        """
        Settable

        :return: specifications to process requested images (for the structure of the dict see ``PROCESS_DATA`` of :class:`Sentinel2Interface`)
        :rtype: dict
        """
        return self._pd
    
    @process_specifications.setter
    def process_specification(self, value):
        self._pd = value

    @property
    def object_resolution(self) -> float:
        """
        Settable

        :return: object resolution which is used to unify the band resolutions, defaults to ``10.`` [m]
        :rtype: float
        """
        return self._objres
    
    @object_resolution.setter
    def object_resolution(self, value):
        self._objres = value

    def request_images(
            self, aoi:GeoDataFrame, tstart:date, tstop:date
        ) -> list[ImageSentinel2]:
        """
        Request sentinel-2 images

        :param aoi: one row of geodataframe
        :type aoi: geopandas.GeoDataFrame
        :param tstart: first epoch for which images should be requested
        :type tstart: datetime.date
        :param tstop: last epoch for which images should be requested
        :type tstop: datetime.date
        :return: requested images
        :rtype: list[mef_agri.data.planetary_computer.sentinel2.sentinel2.ImageSentinel2]
        """
        depsg = aoi.crs.to_epsg()
        if depsg is None:
            errmsg = 'Provide GeoDataFrame with proper CRS and EPSG-code!'
            raise ValueError(errmsg)
        # transform input GeoDataFrame to WGS84 if necessary
        rgdf = None
        if aoi.crs.to_epsg() != self._rs['crs']:
            rgdf = aoi.to_crs(self._rs['crs'])
        else:
            rgdf = deepcopy(aoi)
        trng = [
            datetime(*tstart.timetuple()[:6]), datetime(*tstop.timetuple()[:6])
        ]

        catalog = Client.open(self._rs['url'], modifier=sign_inplace)
        search = self._catalog.search(
            collections=[self._rs['collection'],],
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

                print('-) processing image for ' + prd_date)  # TODO
                hitem = harmonization(item)
                aoi_ser = GeoSeries([aoi.geometry.values[0]], crs=aoi.crs)

                ################################################################
                # fetch all image data resulting in a xarray dataset `ds_refl`
                rbands = self._bd['reflectance'] + self._bd['data']
                ds_data = odc_stac.load(
                    items=[hitem,], bands=rbands, geopolygon=aoi_ser, 
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
                    [hitem.assets[self._pd['ref_band']].ext.raster.bands[0].offset],
                    dtype=np.float32
                )
                ds_offs = DataArray(data=offsets, dims=['time'], coords={
                    'time': ds_data.coords['time']
                })
                ds_refl = ds_data.copy()
                for key in ds_refl.data_vars:
                    if key in self._bd['reflectance']:
                        band:DataArray = ds_refl[key]
                        band_as_float = band.astype(np.float32)
                        band_no_data_mapped = band_as_float.where(
                            lambda x: x != self._pd['dn_nodata'], other=np.nan
                        )
                        band_reflectance = (
                            band_no_data_mapped / self._pd['boa_quantify']
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
                allbands = rbands + self._bd['meta']
                for band in allbands:
                    di = None
                    if band == self._pd['sun_zenith']:
                        di = md.sun_angles.sel(spherical='zenith')
                    elif band == self._pd['sun_azimuth']:
                        di = md.sun_angles.sel(spherical='azimuth')
                    elif band == self._pd['view_zenith']:
                        di = md.view_angles_mean.sel(spherical='zenith')
                    elif band == self._pd['view_azimuth']:
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
    
    @Interface.add_data_task
    def prj_add_images(self):
        imgs = self.request_images(self.aoi, *self.timerange)
        for img in imgs:
            ipath = os.path.join(self.directory, img.epoch.isoformat())
            if not os.path.exists(ipath):
                os.mkdir(ipath)
            img.save_geotiff(ipath, overwrite=True, compress=False)
