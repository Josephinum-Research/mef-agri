import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

from ..interface import DataInterface
from ...farming.tasks.sowing import Sowing
from ...farming.tasks.fertilization import MineralFertilization
from ...farming.tasks.harvest import Harvest
from ...farming import crops


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

sheet_harvest = 'harvest'
columns_harvest = {
    'pid': 'PID',
    'hid': 'HID',  # id of harvest plot/parcel
    'harvest_date': 'Datum',
    'plength': 'Parzellenlaenge',  # length of the harvest plot [m]
    'pwidth': 'Parzellenbreite',  # width of the harvest plot [m]
    'yield_fm': 'FMKornErtrag',  # fresh mass of yield [g]
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
    data = {}
    for col in columns.keys():
        data[col] = df[columns[col]].values.tolist()
    return pd.DataFrame(data=data)


class ManagementInterface(DataInterface):
    DATA_SOURCE_ID = 'management_jr'
    DATA_FOLDER = 'management'
    GPKG_PARCEL_TABLE = 'parcels'
    DELTADAYS_SAME_TASK = 7  # if same task types differ by less than these days, they will be combined into one task
    
    def __init__(self, obj_res=10):
        super().__init__(obj_res)
        self._dfs:pd.DataFrame = None
        self._dff:pd.DataFrame = None
        self._dfh:pd.DataFrame = None

    def add_prj_data(self, aoi, tstart, tstop):
        if self._dfs is None:
            self.load_excel_files()

        cs = (self._dfs['epoch'] >= tstart) & (self._dfs['epoch'] <= tstop)
        cf = (self._dff['epoch'] >= tstart) & (self._dff['epoch'] <= tstop)
        ch = (self._dfh['epoch'] >= tstart) & (self._dfh['epoch'] <= tstop)
        dfs, dff, dfh = self._dfs[cs], self._dff[cf], self._dfh[ch]

        # TODO create task-rasters

        allepochs = pd.concat([dfs['epoch'], dff['epoch'], dfh['epoch']])
        return allepochs.min(), allepochs.max()
        
    def get_prj_data(self, epoch):
        pass

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
            # add field name to df2 and df3
            df2.join(df1[['field_name', 'pid']].set_index('pid'), on='pid')
            df3.join(df1[['field_name', 'pid']].set_index('pid'), on='pid')
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

    def _create_sowing_tasks(self, df:pd.DataFrame) -> None:
        feps = self._find_task_dates(df)
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
            if np.isnan(aux1[0]):
                msg = 'Crop has to be specified for sowing task!'
                raise ValueError(msg)
            cname = aux1[0].strip()
            if not cname in crop_mapper.keys():
                msg = 'Provided crop abbreviation {} not supported!'
                raise ValueError(msg.format(cname))
            crop = getattr(crops, cname)()

            # determine cultivar
            aux2 = dfsow['cultivar'].unique()
            if len(aux2) > 1:
                msg = 'Different cultivars on the same field are not supported '
                msg += 'yet!'
            if np.isnan(aux2[0]):
                cult = 'generic'
            else:
                cult = aux2[0].strip()
                if not cult in crop.cultivars:
                    msg = 'Provided cultivar is not supported!'
                    raise ValueError(msg)
            
            # define sowing task
            sow = Sowing()
            sow.cultivar = getattr(crop, cult)

            # TODO get polygons from pids
            

    def _create_minfert_tasks(self, df:pd.DataFrame) -> None:
        pass

    def _create_harvest_tasks(self, df:pd.DataFrame) -> None:
        pass

    def _find_task_dates(self, df:pd.DataFrame) -> pd.DataFrame:
        feps_raw = df[['field_name', 'epoch']].drop_duplicates()
        feps = {'field_name': [], 'date_begin': [], 'date_end': []}
        for tpl in feps_raw.itertuples():
            diffs = np.array(
                [dt.days for dt in (feps_raw['epoch'] - tpl.epoch).values]
            )
            check = (np.abs(diffs) < self.DELTADAYS_SAME_TASK)
            feps['field_name'].append(tpl.field_name)
            if not True in (check & (diffs != 0)):
                feps['date_begin'].append(tpl.epoch)
                feps['date_end'].append(tpl.epoch)
            else:
                mind, maxd = np.min(diffs[check]), np.max(diffs[check])
                db = tpl.epoch + timedelta(days=mind)
                de = tpl.epoch + timedelta(days=maxd)
                feps['date_begin'].append(db)
                feps['date_end'].append(de)
        return pd.DataFrame(feps).drop_duplicates(inplace=True)
