import os
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import date, timedelta
from importlib import import_module

from ..interface import DataInterface
from ...farming.tasks.sowing import Sowing
from ...farming.tasks.fertilization import MineralFertilization
from ...farming.tasks.harvest import Harvest
from ...farming.tasks.zoning import Zoning
from ...farming import crops
from ...farming import fertilizers
from ...utils.raster import GeoRaster
from ...utils.rv_manipulation import RasterVectorIntersection
from ...models.utils import Units


sheet_fielddata = 'fielddata'
columns_fielddata = {
    'field_name': 'Feldname',
    'pid': 'PID',  # parcel id which will be treated as a zone of the field
    'crop': 'Kultur',
    'cultivar': 'Sorte',
    'sowing_date': 'AussaatdatumKultur',
    'sowing_amount': 'AussaatstaerkeKultur'  # [kg/ha]
}

sheet_fertilization = 'fertilization'
columns_fertilization = {
    'pid': 'PID',
    'fertilizer': 'Duengerart',
    'fertilizer_amount': 'DuengerMenge',  # [kg/ha]
    'fertilization_date': 'Datum'
}

sheet_harvest = 'harvestplots'
columns_harvest = {
    'pid': 'PID',
    'harvest_date': 'Datum',
    'plength': 'Parzellenlaenge',  # length of the harvest plot [m]
    'pwidth': 'Parzellenbreite',  # width of the harvest plot [m]
    'yield_fm_pp': 'FMKornErtrag',  # fresh mass of yield per harvest plot [kg]
    'moisture': 'Feuchte',  # [%]
}


crop_mapper = {
    'WW': 'winter_wheat',
    'WG': 'winter_barley',
    'KM': 'maize',
    'SB': 'soybean'
}


def read_sheet(fpath:str, sheet_name:str, columns:dict) -> pd.DataFrame:
    df = pd.read_excel(fpath, sheet_name=sheet_name, skiprows=[1])
    for cn in df.columns.values:
        df.rename(columns={cn: cn.strip()}, inplace=True)
    data = {}
    for col in columns.keys():
        data[col] = df[columns[col]].values.tolist()
    return pd.DataFrame(data=data)


class ManagementInterface(DataInterface):
    """
    Interface for field trial and management data which are collected in excel 
    files with a specific format. These excel files should be provided in a 
    subfolder within the overall project folder (name of this subfolder has to 
    be the the same as ``ManagementInterface.DATA_FOLDER``).

    An important note on the excel files: 
    the parcel ids of plots within a field have to be unique, especially over 
    several years (e.g. a parcel with id ``1`` can only appear one time, if it 
    appears year by year, the corresponding plots will not be correctly treated 
    in this interface).

    In the geopackage there has to be one table wich contains the geometries of 
    the plots together with the name of the field, the parcel id and date
    columns (see ``ManagementInterface.GPKG_PTABLE_*`` class variables). From 
    these parcels, the zoning for the vegetation periods will be deduced. Thus, 
    each parcel within a field corresponds to one zone afterwards.
    """
    DATA_SOURCE_ID = 'management_jr'
    DATA_FOLDER = 'management'
    GPKG_PTABLE_NAME = 'parcels'
    GPKG_PTABLE_FIELDCOL = 'fname'
    GPKG_PTABLE_PIDCOL = 'pid'
    GPKG_PTABLE_DBEGCOL = 'date_begin'
    GPKG_PTALBE_DVUCOL = 'valid_until'
    ZONING_INTERSECTION = 2. / 3.  # amount of intersecting area of pixel and zone-geometry which needs to be exceeded to assign a pixel to a zone
    DELTADAYS_SAME_TASK = 7  # if same task types differ by less than these days, they will be combined into one task
    
    def __init__(self, obj_res=10):
        super().__init__(obj_res)
        self._dfs:pd.DataFrame = None
        self._dff:pd.DataFrame = None
        self._dfh:pd.DataFrame = None
        self._fr:GeoRaster = None

    def add_prj_data(self, aoi, tstart, tstop):
        """
        Save 
        
        * :class:`mef_agri.farming.tasks.zoning.Zoning`
        * :class:`mef_agri.farming.tasks.sowing.Sowing`
        * :class:`mef_agri.farming.tasks.harvest.Harvest`
        * :class:`mef_agri.farming.tasks.fertilization.MineralFertilization`
        
        tasks to disk which are between ``tstart`` and ```tstop`.
        For zoning tasks only the attribute ``date_begin`` will be considered (
        i.e. if the attribute ``valid_until`` is outside of the provided date 
        range, it will have no effect).

        :param aoi: area of interest representing currently active field (:func:`current_field`)
        :type aoi: geopandas.GeoDataFrame
        :param tstart: first epoch which should be considered
        :type tstart: datetime.date
        :param tstop: last epoch which should be considered
        :type tstop: datetime.date
        :return: adjusted date range according to first and last task found in the excel-files
        :rtype: tuple
        """
        if self._dfs is None:
            self.load_excel_files()

        cs = (self._dfs['epoch'] >= tstart) & (self._dfs['epoch'] <= tstop)
        cf = (self._dff['epoch'] >= tstart) & (self._dff['epoch'] <= tstop)
        ch = (self._dfh['epoch'] >= tstart) & (self._dfh['epoch'] <= tstop)
        csfn = self._dfs['field_name'] == self.current_field
        cffn = self._dff['field_name'] == self.current_field
        chfn = self._dfh['field_name'] == self.current_field
        dfs = self._dfs[cs & csfn]
        dff = self._dff[cf & cffn]
        dfh = self._dfh[ch & chfn]

        self._fr = GeoRaster.from_gdf_and_objres(
            aoi, self.object_resolution, lix=0, nlayers=1
        )
        self._create_zoning_tasks(tstart, tstop)
        self._create_sowing_tasks(dfs)
        self._create_minfert_tasks(dff)
        self._create_harvest_tasks(dfh)

        allepochs = pd.concat([dfs['epoch'], dff['epoch'], dfh['epoch']])
        return allepochs.min(), allepochs.max()
        
    def get_prj_data(self, epoch):
        """
        Get management/task data for the provided epoch. The returned 
        dictionary contains keys which correspond to the class-names of the 
        available tasks.

        :param epoch: epoch to look for available tasks
        :type epoch: datetime.date
        :return: dictionary with available tasks
        :rtype: dict
        """
        fpath = os.path.join(
            self.project_directory, self.save_directory, epoch.isoformat()
        )
        if not os.path.exists(fpath):
            return
        
        ret = {}
        for folder in os.listdir(fpath):
            tpath = os.path.join(fpath, folder)
            fio = open(os.path.join(tpath, 'metadata.json'), 'r')
            md = json.load(fio)
            fio.close()
            task = getattr(import_module(md['task_module']), md['task_name'])()
            task.load_geotiff(tpath)
            ret[task.task_name] = task

        return ret

    def load_excel_files(self) -> None:
        mddir = os.path.join(self.project_directory, self.DATA_FOLDER)
        for ent in os.listdir(mddir):
            c1 = '.xlsx' in ent  # first condition which should be met by file
            c2 = not bool(ent.split('.xlsx')[-1])  # second condition which should be met by file
            if not c1 or not c2:
                continue

            # load data
            fpath = os.path.join(mddir, ent)
            df1 = read_sheet(fpath, sheet_fielddata, columns_fielddata)
            df2 = read_sheet(fpath, sheet_fertilization, columns_fertilization)
            df3 = read_sheet(fpath, sheet_harvest, columns_harvest)
            # following lines exist becuase i have no clue why excel is so stupid!
            try:
                df3['yield_fm_pp'] = df3['yield_fm_pp'].values.astype(float)
            except:
                newvals = [
                    float(val.replace(',', '.')) 
                    for val in df3['yield_fm_pp'].values
                ]
                df3['yield_fm_pp'] = newvals
            # add field name to df2 and df3
            df2 = df2.join(df1[['field_name', 'pid']].set_index('pid'), on='pid')
            df3 = df3.join(df1[['field_name', 'pid']].set_index('pid'), on='pid')
            # add columns containing date objects
            df1['epoch'] = [
                date.fromisoformat(d) for d in df1['sowing_date'].values
            ]
            df2['epoch'] = [
                date.fromisoformat(d) for d in df2['fertilization_date'].values
            ]
            df3['epoch'] = [
                date.fromisoformat(d) for d in df3['harvest_date'].values
            ]
            # adding loaded data 
            self._dfs = df1 if self._dfs is None else pd.concat(
                [self._dfs, df1]
            )
            self._dff = df2 if self._dff is None else pd.concat(
                [self._dff, df2]
            )
            self._dfh = df3 if self._dfh is None else pd.concat(
                [self._dfh, df3]
            )

    def _create_zoning_tasks(self, tstart:date, tstop:date) -> None:
        # load parcel information from geopackage
        zones = gpd.read_file(
            self.project_ref.database.file_path, 
            layer=self.GPKG_PTABLE_NAME
        )
        zones['epoch'] = [
            date.fromisoformat(d) for d in zones['date_begin'].values
        ]
        zones = zones[(zones['epoch'] >= tstart) & (zones['epoch'] <= tstop)]
        zones = zones[zones[self.GPKG_PTABLE_FIELDCOL] == self.current_field]

        # determine zones from parcels in the project-gpkg-database
        spath = os.path.join(self.project_directory, self.save_directory)
        zts = {'date_begin': [], 'valid_until': [], 'zoning': []}
        for dbeg in zones[self.GPKG_PTABLE_DBEGCOL].unique():
            zi = zones[zones[self.GPKG_PTABLE_DBEGCOL] == dbeg]

            zt = Zoning()
            zt.crs = self._fr.crs
            zt.bounds = self._fr.bounds
            zt.transformation = self._fr.transformation
            zt.date_begin = dbeg
            zt.valid_until = zi[self.GPKG_PTALBE_DVUCOL].values[0]

            rshp = (len(zi), self._fr.raster_shape[1], self._fr.raster_shape[2])
            zr = np.zeros(rshp)
            lids, i = [], 0
            for pid in zi[self.GPKG_PTABLE_PIDCOL].values:
                lids.append(str(pid))
                zrow = zi[zi[self.GPKG_PTABLE_PIDCOL] == pid]
                rvi = RasterVectorIntersection(zrow, self._fr)
                rvi.fraction = self.ZONING_INTERSECTION
                rvi.compute()
                zr[i, :, :] = rvi.assignment[0, :, :]
                i += 1
            
            zt.specify_application(zr, lids, Units.undef, appl_ix=0)
            zts['date_begin'].append(zt.date_begin)
            zts['valid_until'].append(zt.valid_until)
            zts['zoning'].append(zt)
            fpath = os.path.join(spath, dbeg)
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            fpath = os.path.join(fpath, 'zoning')
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            zt.save_geotiff(
                fpath, overwrite=True, compress=True, filename='task'
            )
        self._zts = pd.DataFrame(zts)

    def _create_sowing_tasks(self, df:pd.DataFrame) -> None:
        feps = self._find_task_dates(df)
        spath = os.path.join(self.project_directory, self.save_directory)
        for tpl in feps.itertuples():
            cond1 = df['field_name'] == tpl.field_name
            cond2 = df['epoch'] >= tpl.date_begin
            cond3 = df['epoch'] <= tpl.date_end
            dfsow = df[cond1 & cond2 & cond3]

            # determine crop
            aux1 = dfsow['crop'].unique()
            if len(aux1) > 1:
                msg = 'Different crops on the same field are not supported yet!'
                raise ValueError(msg)
            if aux1.dtype != object and np.isnan(aux1[0]):
                msg = 'Name of sown crop has to be provided!'
                raise ValueError(msg)
            cname = aux1[0].strip()
            if not cname in crop_mapper.keys():
                msg = 'Provided crop abbreviation {} not supported!'
                raise ValueError(msg.format(cname))
            crop = getattr(crops, crop_mapper[cname])()

            # determine cultivar
            aux2 = dfsow['cultivar'].unique()
            if len(aux2) > 1:
                msg = 'Different cultivars on the same field are not supported '
                msg += 'yet!'
            if aux2.dtype != object and np.isnan(aux2[0]):
                cult = 'generic'
            else:
                cult = aux2[0].strip()
                if not cult in crop.cultivars:
                    msg = 'Provided cultivar is not supported!'
                    raise ValueError(msg)
            
            # define sowing task
            sow = Sowing()
            sow.crs = self._fr.crs
            sow.bounds = self._fr.bounds
            sow.transformation = self._fr.transformation
            sow.cultivar = getattr(crop, cult)
            sow.date_begin = tpl.date_begin
            sow.date_end = tpl.date_end
            
            # create application map for sowing amount
            caux1 = self._zts['date_begin'] <= tpl.date_begin
            caux2 = self._zts['valid_until'] >= tpl.date_begin
            zoning = self._zts[caux1 & caux2]
            if len(zoning) > 1:
                msg = 'Found multiple zonings for sowing-task done at {}!'
                raise ValueError(msg.format(tpl.date_begin))
            ztask:Zoning = zoning['zoning'].values[0]
            sarr = np.zeros(self._fr.raster_shape)
            sarr[:] = np.nan
            for ptpl in dfsow.itertuples():
                sarr[np.where(ztask[str(ptpl.pid)])] = ptpl.sowing_amount
            sow.specify_application(
                sarr, sow.avname_sowing_amount, Units.kg_ha, appl_ix=0
            )

            # save sowing task to disk
            fpath = os.path.join(spath, tpl.date_begin.isoformat())
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            fpath = os.path.join(fpath, 'sowing')
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            sow.save_geotiff(
                fpath, overwrite=True, compress=True, filename='task'
            )
            
    def _create_minfert_tasks(self, df:pd.DataFrame) -> None:
        feps = self._find_task_dates(df, mindiff=1)
        spath = os.path.join(self.project_directory, self.save_directory)
        for tpl in feps.itertuples():
            cond1 = df['field_name'] == tpl.field_name
            cond2 = df['epoch'] >= tpl.date_begin
            cond3 = df['epoch'] <= tpl.date_end
            dffer = df[cond1 & cond2 & cond3]

            aux = dffer['fertilizer'].unique()
            if len(aux) > 1:
                msg = 'Only one type of fertilizer is supported per date of '
                msg += 'application!'
                raise ValueError(msg)
            if aux.dtype != object and np.isnan(aux[0]):
                msg = 'Fertilizer name has to be provided!'
                raise ValueError(msg)
            fname = aux[0].strip()

            # define fertilizer task
            fer = MineralFertilization()
            fer.crs = self._fr.crs
            fer.bounds = self._fr.bounds
            fer.transformation = self._fr.transformation
            fer.fertilizer = fname
            fer.date_begin = tpl.date_begin
            fer.date_end = tpl.date_end

            # create application map for fertilizer amount
            caux1 = self._zts['date_begin'] <= tpl.date_begin
            caux2 = self._zts['valid_until'] >= tpl.date_begin
            zoning = self._zts[caux1 & caux2]
            if len(zoning) > 1:
                msg = 'Found multiple zonings for sowing-task done at {}!'
                raise ValueError(msg.format(tpl.date_begin))
            ztask:Zoning = zoning['zoning'].values[0]
            farr = np.zeros(self._fr.raster_shape)
            farr[:] = np.nan
            for ptpl in dffer.itertuples():
                farr[np.where(ztask[str(ptpl.pid)])] = ptpl.fertilizer_amount
            fer.specify_application(
                farr, fer.avname_fertilizer_amount, Units.kg_ha, appl_ix=0
            )

            # save mineral fertilization task to disk
            fpath = os.path.join(spath, tpl.date_begin.isoformat())
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            fpath = os.path.join(fpath, 'minfert')
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            fer.save_geotiff(
                fpath, overwrite=True, compress=True, filename='task'
            )

    def _create_harvest_tasks(self, df:pd.DataFrame) -> None:
        feps = self._find_task_dates(df)
        spath = os.path.join(self.project_directory, self.save_directory)
        # computing yield of fresh mass in kg/ha
        df['yield_fm'] = df['yield_fm_pp'] / (df['plength'] * df['pwidth'])  # yield per square meter
        df['yield_fm'] *= 1e4  # yield kg per hectar
        for tpl in feps.itertuples():
            cond1 = df['field_name'] == tpl.field_name
            cond2 = df['epoch'] >= tpl.date_begin
            cond3 = df['epoch'] <= tpl.date_end
            dfhrv = df[cond1 & cond2 & cond3]

            # define harvest Task
            hrv = Harvest()
            hrv.crs = self._fr.crs
            hrv.bounds = self._fr.bounds
            hrv.transformation = self._fr.transformation
            hrv.date_begin = tpl.date_begin
            hrv.date_end = tpl.date_end

            # create application map for fresh mass yield and moisture
            caux1 = self._zts['date_begin'] <= tpl.date_begin
            caux2 = self._zts['valid_until'] >= tpl.date_begin
            zoning = self._zts[caux1 & caux2]
            if len(zoning) > 1:
                msg = 'Found multiple zonings for sowing-task done at {}!'
                raise ValueError(msg.format(tpl.date_begin))
            ztask:Zoning = zoning['zoning'].values[0]
            hyarr = np.zeros(self._fr.raster_shape)
            hyarr[:] = np.nan
            hmarr = np.zeros(self._fr.raster_shape)
            hmarr[:] = np.nan
            for pid in dfhrv['pid'].unique():
                dfi = dfhrv[dfhrv['pid'] == pid]
                mask = np.where(ztask[str(pid)])
                hyarr[mask] = np.median(dfi['yield_fm'].values)
                hmarr[mask] = np.median(dfi['moisture'].values * 1e-2)  # conversion from percent to fraction
            hrv.specify_application(
                np.concatenate((hyarr, hmarr), axis=0),
                ['yield_fm', 'moisture'],
                [Units.kg_ha, Units.frac],
                appl_ix=0
            )

            # save harvest task to disk
            fpath = os.path.join(spath, tpl.date_begin.isoformat())
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            fpath = os.path.join(fpath, 'harvest')
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            hrv.save_geotiff(
                fpath, overwrite=True, compress=True, filename='task'
            )

    def _find_task_dates(self, df:pd.DataFrame, mindiff=None) -> pd.DataFrame:
        if mindiff is None:
            mindiff = self.DELTADAYS_SAME_TASK
        feps_raw = df[['field_name', 'epoch']].drop_duplicates()
        feps = {'field_name': [], 'date_begin': [], 'date_end': []}
        for tpl in feps_raw.itertuples():
            diffs = np.array(
                [dt.days for dt in (feps_raw['epoch'] - tpl.epoch).values]
            )
            check = (np.abs(diffs) < mindiff)
            feps['field_name'].append(tpl.field_name)
            if not True in (check & (diffs != 0)):
                feps['date_begin'].append(tpl.epoch)
                feps['date_end'].append(tpl.epoch)
            else:
                mind, maxd = np.min(diffs[check]), np.max(diffs[check])
                db = tpl.epoch + timedelta(days=int(mind))
                de = tpl.epoch + timedelta(days=int(maxd))
                feps['date_begin'].append(db)
                feps['date_end'].append(de)
        ret = pd.DataFrame(feps)
        ret.drop_duplicates(inplace=True)
        return ret
