import os
import requests
import datetime
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from copy import deepcopy
from pandas import concat

from ....utils.gis import bbox_from_gdf, EARTH_SPHERE_RADIUS
from .inca import IncaGridDaily, INCADATA
from ...interface import Interface


class INCAInterface_old(DataInterface):
    DATA_SOURCE_ID = 'inca_geosphere-austria'
    HOST = 'https://dataset.api.hub.geosphere.at'
    VERS = 'v1'
    ID_INCA = 'inca-v1-1h-1km'
    INCA_OUT_FORMAT = 'geojson'
    INCA_STAMPS = 'timestamps'
    INCA_FEATS = 'features'
    INCA_GEOM_LVL1 = 'geometry'
    INCA_GEOM_LVL2 = 'coordinates'
    INCA_DATA_LVL1 = 'properties'
    INCA_DATA_LVL2 = 'parameters'
    INCA_DATA_LVL3 = 'data'

    def __init__(self) -> None:
        """
        Interface to load data from https://data.hub.geosphere.at/
        The output/target crs can be defined either by explicitely setting 
        the `target_crs` attribute or it will be derived from the provided 
        GeoDataFrames specifying the area of interest.
        """
        super().__init__()
        self.url_common:str = self.HOST + '/' + self.VERS + '/'
        self.target_crs:int = None
        self.add_bbox_km:float = 1.1  # in [km]
        self._delta_235959 = datetime.timedelta(
            hours=23, minutes=59, seconds=59
        )

        self.inca_data:list[str] = INCADATA.keys()

    def add_prj_data(self, aoi, tstart, tstop) -> tuple:
        """
        Method from parent-class which is called within 
        `sitespecificcultivation.data.project.Project` to save inca data
        in the project folder structure.

        :param aoi: area of interest for which data is requested
        :type aoi: geopandas.GeoDataFrame
        :param tstart: start date of requested data (inca data is available for this whole day)
        :type tstart: datetime.date
        :param tstop: stop date of requested data (inca data is available for this whole day)
        :type tstop: datetime.date
        """
        self.save_inca_grid_daily(
            self.get_inca_grid_daily(
                self.get_inca_grid_historical(aoi, tstart, tstop)
            )
        )
        return tstart, tstop

    def get_prj_data(self, epoch):
        """
        Method from parent-class which is called within 
        `sitespecificcultivation.data.project.Project` to load inca data from 
        the project folder structure

        :param epoch: date of inca data
        :type epoch: datetime.date
        :return: dictionary with keys being the inca-data-ids and values the corresponding `GeoRaster` containing the hourly data for the provided `epoch`
        :rtype: dict
        """
        dpath = os.path.join(
            self.project_directory, self.save_directory, epoch.isoformat()
        )
        if not os.path.exists(dpath):
            return
        
        ret = {}
        for key, val in INCADATA.items():
            inca = IncaGridDaily(
                data_id=key, epoch=epoch, 
                data_descr=val['descr'], data_units=val['units']
            )
            fpath = os.path.join(dpath, key)
            inca.load_geotiff(fpath)
            ret[key] = inca
        return ret

    def get_inca_grid_historical(
            self, aoi:gpd.GeoDataFrame, tstart:datetime.date, 
            tstop:datetime.date
        ) -> gpd.GeoDataFrame:
        """
        Request inca-data (grid+historical endpoint). Data is provided as 
        geojson in the WGS84 crs from the geosphere-API, which will be returned 
        as `geopandas.GeoDataFrame` from this method. This geodataframe is also 
        defined in WGS84 and the `target_crs` will be applied when creating the 
        `GeoRaster` representation in `self.get_inca_grid_daily()` or when 
        using `self.save_inca_grid_historical(gdf, apply_target_crs=True)`.
        If `self.target_crs` is not set or changed afterwards, it will be the 
        same as contained in `aoi`.

        The reason for staying in WGS84 is, that a projection (e.g. to UTM), 
        distorts the points such that they do not exhibit a 1km x 1km raster 
        anymore and therefore a proper raster representation is not possible.
        Only the projection to Austria Lambert (EPSG 31287) keeps this property 
        as this is the CRS where INCA data is processed by Geosphere Austria.

        :param aoi: GeoDataFrame representing the area of interest
        :type aoi: geopandas.GeoDataFrame
        :param tstart: start time of requested data (data/timestamps starting from 00:00:00 of this day)
        :type tstart: datetime.date
        :param tstop: stop time of requested data (whole day is included in the resulting GeoDataFrame, i.e. data/timestamps end at 23:00:00 of this day)
        :type tstop: datetime.date
        :return: inca-data as geojson (`geopandas.GeoDataFrame`)
        :rtype: gpd.GeoDataFrame
        """
        # adddeg is used to expand the size of the aoi such that it covers at 
        # least a square-km (otherwise nothing is returned from api)
        adddeg = np.rad2deg((self.add_bbox_km * 1000.0) / EARTH_SPHERE_RADIUS)  # [deg]
        if self.target_crs is None:
            self.target_crs = aoi.crs.to_epsg()

        # rearranging bbox because geosphere requires (south, west, north, east)
        # sitespecificcultivation-definition is (west, south, east, north)
        # TODO: !!! adddeg only works for positive latitude and positive longitude !!!
        aoi_src = aoi.to_crs(epsg=IncaGridDaily.EPSG_REQUEST)
        bbox = bbox_from_gdf(aoi_src)
        bbox = (
            bbox[1] - adddeg,
            bbox[0] - adddeg, 
            bbox[3] + adddeg, 
            bbox[2] + adddeg
        )
        bboxstr = ''.join([str(ci) + ',' for ci in bbox])[:-1]

        resp = requests.get(
            self.url_common + 'grid/historical/' + self.ID_INCA,
            params=dict(
                parameters=self.inca_data,
                start=datetime.datetime(
                    tstart.year, tstart.month, tstart.day
                ).isoformat(),
                end=(
                    datetime.datetime(
                        tstop.year, tstop.month, tstop.day
                    ) + self._delta_235959
                ).isoformat(),
                bbox=bboxstr,
                output_format=self.INCA_OUT_FORMAT
            )
        ).json()
        if not self.INCA_FEATS in resp.keys():
            if 'message' in resp.keys():
                print(resp['message'])

        data = {'geometry': [], 'timestamps': []}
        for key in self.inca_data:
            data[key] = []
        for gdata in resp[self.INCA_FEATS]:
            data['geometry'].append(
                Point(gdata[self.INCA_GEOM_LVL1][self.INCA_GEOM_LVL2])
            )
            data['timestamps'].append(resp[self.INCA_STAMPS])
            for key in self.inca_data:
                pvals = gdata[self.INCA_DATA_LVL1][self.INCA_DATA_LVL2]
                data[key].append(
                    pvals[key][self.INCA_DATA_LVL3]
                )

        return gpd.GeoDataFrame(data=data, crs=IncaGridDaily.EPSG_REQUEST)
    
    def get_inca_grid_daily(self, gdf:gpd.GeoDataFrame) -> dict:
        """
        Convert GeoDataFrame from `self.get_inca_grid_historical()` into 
        GeoRasters for each day and each kind of inca data (each having 
        24 layers representing the hourly inca data).

        The GeoDataFrame has to be defined in WGS84 (EPSG 4326)!!!

        :param gdf: GeoDataFrame containing data from geosphere-inca API
        :type gdf: gpd.GeoDataFrame
        :return: dictionary with `sitespecificcultivation.data_sources.geosphere.inca.IncaGridDaily` objects where first level of keys correspond to the dates in `gdf` and second level of keys correspond to the individual inca data types
        :rtype: dict
        """
        if gdf.crs.to_epsg() != IncaGridDaily.EPSG_REQUEST:
            msg = '`gdf` has to be defined in WGS84!'
            raise ValueError(msg)

        inca = {}
        outstr = 'GeoSphereInterface.get_inca_grid_daily() - processing day {}'
        for ix in np.arange(
            start=0, stop=len(gdf['timestamps'].iloc[0]), step=24
        ):
            inca_day = {}
            data = {
                'geometry': list(gdf['geometry'].values),
                'timestamps': np.array(list(gdf['timestamps']))[:, ix:ix + 24].tolist()
            }
            str_day = data['timestamps'][0][0].split('T')[0]
            print(outstr.format(str_day))
            for col in list(self.inca_data):
                idata = deepcopy(data)
                idata[col] = np.array(list(gdf[col]))[:, ix:ix + 24].tolist()
                inca_day[col] = IncaGridDaily.from_points(
                    gpd.GeoDataFrame(idata, crs=gdf.crs, geometry='geometry'),
                    col, datetime.date.fromisoformat(str_day), self.target_crs
                )
            inca[str_day] = inca_day

        return inca

    def save_inca_grid_historical(
            self, gdf:gpd.GeoDataFrame, apply_target_crs:bool=False
        ) -> None:
        self._check_dirs()
        spath = os.path.join(self.project_directory, self.save_directory)
        tstart = gdf['timestamps'].values[0][0]
        tstop = gdf['timestamps'].values[0][0]
        if apply_target_crs:
            gdf.to_crs(self.target_crs, inplace=True)
        gdf.to_file(
            os.path.join(spath, tstart + '__to__' + tstop + '.json'), 
            driver='GeoJSON'
        )

    def save_inca_grid_daily(self, inca:dict) -> None:
        self._check_dirs()
        for day, data in inca.items():
            daypath = os.path.join(
                self.project_directory, self.save_directory, day
            )
            if not os.path.exists(daypath):
                os.mkdir(daypath)
            for key, obs in data.items():
                spath = os.path.join(daypath, key)
                if not os.path.exists(spath):
                    os.mkdir(spath)
                obs.save_geotiff(spath, compress=True, filename=key)


class INCAInterface(Interface):
    REQU_DATA_V1 = {
        'host': 'https://dataset.api.hub.geosphere.at',
        'version': 'v1',
        'id_dtype': 'grid/historical/',
        'id_inca': 'inca-v1-1h-1km',
        'out_format': 'geojson',
        'key_timestamps': 'timestamps',
        'key_features': 'features',
        'key_geom_lvl1': 'geometry',
        'key_geom_lvl2': 'coordinates',
        'key_data_lvl1': 'properties',
        'key_data_lvl2': 'parameters',
        'key_data_lvl3': 'data'
    }

    def __init__(self):
        super().__init__()
        # properties from parent class
        self.data_source_id = 'inca_geosphere-austria'
        self.description = """
            Interface to get data from 
            https://data.hub.geosphere.at/dataset/inca-v1-1h-1km
        """
        self.georaster_class = IncaGridDaily
        self.data_types = list(INCADATA.keys())
        dtd = ''
        for key, val in INCADATA.items():
            dtd += '{}: {}\n'.format(key, val)
        self.data_types_description = dtd

        # inca-specific stuff
        self._rs = self.REQU_DATA_V1
        self._add_bbox_km:float = 1.1  # in [km]
        self._delta_235959 = datetime.timedelta(
            hours=23, minutes=59, seconds=59
        )

    ############################################################################
    # PROPERTIES
    @property
    def request_specifications(self) -> dict:
        """
        Settable

        :return: Specifications to perform request and extract necessary data from request-response
        :rtype: dict
        """
        return self._rs
    
    @request_specifications.setter
    def request_specifications(self, value):
        self._rs = value

    ############################################################################
    # METHODS
    def get_inca_grid_historical(
            self, aoi:gpd.GeoDataFrame=None, timerange:list[datetime.date]=None
        ) -> gpd.GeoDataFrame:
        # adddeg is used to expand the size of the aoi such that it covers at 
        # least a square-km (otherwise nothing is returned from api)
        adddeg = np.rad2deg((self._add_bbox_km * 1000.0) / EARTH_SPHERE_RADIUS)  # [deg]

        # rearranging bbox because geosphere requires (south, west, north, east)
        # sitespecificcultivation-definition is (west, south, east, north)
        # TODO: !!! adddeg only works for positive latitude and positive longitude !!!
        if aoi is None:
            aoi_src = self.aoi.to_crs(epsg=IncaGridDaily.EPSG_REQUEST)
        else:
            aoi_src = aoi.to_crs(epsg=IncaGridDaily.EPSG_REQUEST)
        
        bbox = bbox_from_gdf(aoi_src)
        bbox = (
            bbox[1] - adddeg,
            bbox[0] - adddeg, 
            bbox[3] + adddeg, 
            bbox[2] + adddeg
        )
        bboxstr = ''.join([str(ci) + ',' for ci in bbox])[:-1]

        url = self._rs['host'] + '/' + self._rs['version'] + '/'
        url += self._rs['id_dtype'] + self._rs['id_inca']
        if timerange is None:
            tstart, tstop = self.timerange
        else:
            tstart, tstop = timerange
        resp = requests.get(
            url=url,
            params=dict(
                parameters=self.data_types,
                start=datetime.datetime(
                    tstart.year, tstart.month, tstart.day
                ).isoformat(),
                end=(
                    datetime.datetime(
                        tstop.year, tstop.month, tstop.day
                    ) + self._delta_235959
                ).isoformat(),
                bbox=bboxstr,
                output_format=self._rs['out_format']
            )
        ).json()
        if not self._rs['key_features'] in resp.keys():
            if 'message' in resp.keys():
                print(resp['message'])  # TODO handle prints

        # prepare data-dict for resulting geodataframe
        data = {'geometry': [], 'timestamps': [], 'location': []}
        for key in self.data_types:
            data[key] = []

        # get data from response
        tstmps = resp[self._rs['key_timestamps']]
        iloc = 1
        for gdata in resp[self._rs['key_features']]:
            pi = Point(
                gdata[self._rs['key_geom_lvl1']][self._rs['key_geom_lvl2']]
            )
            data['geometry'] += [pi for i in range(len(tstmps))]
            data['location'] += [iloc for i in range(len(tstmps))]
            data['timestamps'] += tstmps
            iloc += 1
            for key in self.data_types:
                pvals = gdata[
                    self._rs['key_data_lvl1']
                ][
                    self._rs['key_data_lvl2']
                ][
                    key
                ][
                    self._rs['key_data_lvl3']
                ]
                data[key] += pvals

        return gpd.GeoDataFrame(data=data, crs=IncaGridDaily.EPSG_REQUEST)

    ############################################################################
    # TASKS
    @Interface.add_data_task(order=1)
    def prepare_inca_grid_historical(self, **kwargs):
        gdf = self.get_inca_grid_historical()
        ndays = len(gdf) // 24
        if (ndays // self.n_processes) < 2:
            self.n_processes = 1
        perprc, resid = ndays // self.n_processes, ndays % self.n_processes
        locs = gdf['location'].drop_duplicates().values()
        for i in range(self.n_processes):
            prcdf = None
            i1, i2 = i * perprc * 24, (i + 1) * perprc * 24
            if (i + 1) == self.n_processes:
                i2 += resid * 24
            # process grid points/locations
            for loc in locs:
                df = gdf[gdf['location'] == loc].iloc[i1:i2]
                if prcdf is None:
                    prcdf = df
                else:
                    prcdf = concat([prcdf, df], ignore_index=True)
            # save prcdf to a temp-file being available for sub-processes in next task
            # TODO

        ret = {'target_crs': self.aoi.crs.to_epsg()}
        return ret
    
    @Interface.add_data_task(order=2, parallel=True)
    def get_inca_grid_daily(self, gdf:gpd.GeoDataFrame) -> dict:
        # TODO overwork this method to work as parallel task
        """
        Convert GeoDataFrame from `self.get_inca_grid_historical()` into 
        GeoRasters for each day and each kind of inca data (each having 
        24 layers representing the hourly inca data).

        The GeoDataFrame has to be defined in WGS84 (EPSG 4326)!!!

        :param gdf: GeoDataFrame containing data from geosphere-inca API
        :type gdf: gpd.GeoDataFrame
        :return: dictionary with `sitespecificcultivation.data_sources.geosphere.inca.IncaGridDaily` objects where first level of keys correspond to the dates in `gdf` and second level of keys correspond to the individual inca data types
        :rtype: dict
        """
        if gdf.crs.to_epsg() != IncaGridDaily.EPSG_REQUEST:
            msg = '`gdf` has to be defined in WGS84!'
            raise ValueError(msg)

        inca = {}
        outstr = 'GeoSphereInterface.get_inca_grid_daily() - processing day {}'
        for ix in np.arange(
            start=0, stop=len(gdf['timestamps'].iloc[0]), step=24
        ):
            inca_day = {}
            data = {
                'geometry': list(gdf['geometry'].values),
                'timestamps': np.array(list(gdf['timestamps']))[:, ix:ix + 24].tolist()
            }
            str_day = data['timestamps'][0][0].split('T')[0]
            print(outstr.format(str_day))
            for col in list(self.inca_data):
                idata = deepcopy(data)
                idata[col] = np.array(list(gdf[col]))[:, ix:ix + 24].tolist()
                inca_day[col] = IncaGridDaily.from_points(
                    gpd.GeoDataFrame(idata, crs=gdf.crs, geometry='geometry'),
                    col, datetime.date.fromisoformat(str_day), self.target_crs
                )
            inca[str_day] = inca_day

        return inca
