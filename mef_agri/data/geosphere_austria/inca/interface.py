import os
import requests
import datetime
import geopandas as gpd
import numpy as np
from importlib import import_module
from shapely.geometry import Point
from pandas import concat
from shutil import rmtree

from ....utils.gis import bbox_from_gdf, EARTH_SPHERE_RADIUS
from .inca import IncaGridDaily, INCADATA
from ...interface import Interface, Worker


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

        :return: Specifications to perform request and extract necessary data from request-response, see ``REQU_DATA_V1`` of :class:`INCAInterface`
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
        """
        Request inca-data (grid+historical endpoint). 
        Data is provided as geojson in the WGS84 crs from the geosphere-API, 
        which will be returned as `geopandas.GeoDataFrame` from this method.

        The reason for staying in WGS84 is, that a projection (e.g. to UTM), 
        distorts the points such that they do not exhibit a 1km x 1km raster 
        anymore and therefore a proper raster representation is not possible.
        Only the projection to Austria Lambert (EPSG 31287) keeps this property 
        as this is the CRS where INCA data is processed by Geosphere Austria.

        :param aoi: aoi for which data will be requested, defaults to None
        :type aoi: gpd.GeoDataFrame, optional
        :param timerange: timerange for which data will be requested, defaults to None
        :type timerange: list[datetime.date], optional
        :return: geodataframe with n_gridpoints x n_timestamps rows
        :rtype: gpd.GeoDataFrame
        """
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
        ret = {}

        # create temporary folder where geojsons for each process are stored
        tempf = os.path.join(self.directory, 'temp')
        if not os.path.exists(tempf):
            os.mkdir(tempf)

        # create chunks from inca-grid historical data being processed in 
        # next task
        locs = gdf['location'].drop_duplicates().values
        ndays = len(gdf) // 24 // len(locs)
        if (ndays // self.n_processes) < 2:
            self.n_processes = 1
        perprc, resid = ndays // self.n_processes, ndays % self.n_processes
        for i in range(self.n_processes):
            prcdf:gpd.GeoDataFrame = None
            # create indices
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
            # save prcdf to a temp-file being available for sub-processes 
            # in next task
            fpath = os.path.join(tempf, self.process_ids[i] + '.json')
            prcdf.to_file(fpath, driver='GeoJSON')
            ret[self.process_ids[i]] = {
                'target_crs': self.aoi.crs.to_epsg(),
                'gjson_file': fpath,
                'data_cols': self.data_types,
                'loc_col': 'location',
                'time_col': 'timestamps',
                'geom_col': 'geometry',
                'data_dir': self.directory,
                'grclass': IncaGridDaily.__name__,
                'grmodule': IncaGridDaily.__module__
            }

        return ret
    
    @Interface.add_data_task(order=2, parallel=True)
    def process_chunks(self, **kwargs):
        import numpy as np

        gdf = gpd.read_file(kwargs['gjson_file'])
        neps = None
        grclass = getattr(import_module(kwargs['grmodule']), kwargs['grclass'])
        q = kwargs['queue']

        # prepare gdf such that there is only one row for each location
        data = {kwargs['geom_col']: [], kwargs['time_col']: []}
        locs = gdf[kwargs['loc_col']].drop_duplicates().values
        for loc in locs:
            gdfl = gdf[gdf[kwargs['loc_col']] == loc]
            data[kwargs['time_col']].append(
                gdfl[kwargs['time_col']].values.tolist()
            )
            data[kwargs['geom_col']].append(
                gdfl[kwargs['geom_col']].values[0]
            )
            for dcol in kwargs['data_cols']:
                di = gdfl[dcol].values.tolist()
                if neps is None:
                    neps = len(di)
                if not dcol in data.keys():
                    data[dcol] = [di]
                else:
                    data[dcol].append(di)

        # create daily georasters and save them
        epochs = []
        for ix in np.arange(0, neps, 24):
            tsday = data[kwargs['time_col']][0][ix:ix + 24]
            epoch = tsday[0].date()
            tsday = [ts.isoformat() for ts in tsday]
            q.put([
                Worker.QFLAG_MESSAGE, 
                'INCA-Interface: processing epoch {} in process {}'.format(
                    epoch.isoformat(), kwargs['pid']
                )
            ])
            epdir = os.path.join(kwargs['data_dir'], epoch.isoformat())
            if not os.path.exists(epdir):
                os.mkdir(epdir)
            for dcol in kwargs['data_cols']:
                datalocs, tslocs = [], []
                for dataloc in data[dcol]:
                    tslocs.append(tsday)
                    datalocs.append(dataloc[ix:ix + 24])
                dataday = {
                    kwargs['geom_col']: data[kwargs['geom_col']],
                    kwargs['time_col']: tslocs,
                    dcol: datalocs
                }
                gr = grclass.from_points(
                    gpd.GeoDataFrame(dataday, crs=gdf.crs, geometry='geometry'),
                    dcol, epoch, kwargs['target_crs']
                )
                ddir = os.path.join(epdir, dcol)
                if not os.path.exists(ddir):
                    os.mkdir(ddir)
                gr.save_geotiff(ddir, compress=True, filename=dcol)
            epochs.append(epoch.isoformat())
        
        q.put([kwargs['pid'] + '_epochs', epochs])

    @Interface.add_data_task(order=3)
    def finish_processing(self, **kwargs):
        # remove temporary folder from first task
        tempf = os.path.join(self.directory, 'temp')
        if os.path.exists(tempf):
            rmtree(tempf)
        # return processed epochs
        epochs = []
        for pid in self.process_ids:
            epochs += kwargs[pid + '_epochs']
        return [datetime.date.fromisoformat(ep) for ep in epochs]
