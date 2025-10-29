import os
import json
import pandas as pd
from datetime import date

from .task import Task
from ..interface import DataInterface


class ExcelTaskInterface(DataInterface):
    """
    Interface which extracts task information from the JR-Excel-Template.

    !!! NOT FINISHED !!!
    """
    DATA_SOURCE_ID = 'jr_excel_management'
    DATA_FOLDER = 'jr_data'
    DATA_FILENAME = 'jr_data.xlsx'

    FIELD_NAME_COL = 'Feldname'

    def add_prj_data(self, aoi, tstart, tstop):
        dpath = os.path.join(
            self.project_directory, self.DATA_FOLDER, self.DATA_FILENAME
        )
        fdata = pd.read_excel(
            dpath,
            sheet_name='fielddata',
            header=0,
            skiprows=lambda x: x == 1,
            usecols=[1, 2, 3, 4, 5, 6, 7, 8]
        )
        
        field_name = None
        for col in aoi.columns:
            fldn = aoi[col].values[0]
            if not isinstance(fldn, str):
                continue
            if fldn.lower() in fdata[self.FIELD_NAME_COL].values.tolist():
                field_name = fldn.lower()
                break

        if field_name is None:
            msg = 'No field name found in provided aoi/geodataframe which '
            msg += 'matches one of the names in the excel template!'
            raise ValueError(msg)
        
        fdata = fdata[fdata[self.FIELD_NAME_COL] == field_name]

    
    def get_prj_data(self, epoch):
        pass



class JSONTaskInterface(DataInterface):
    TASKDATAFOLDER = 'tasks'
    DATA_SOURCE_ID = 'jr_management'

    def __init__(self):
        """
        Interface to integrate task-data in JR-format into 
        `sitespecificcultivation.data.project.Project`. 
        In the project directory (i.e. the directory where the .gpgk is located)
        a folder will be created according to the class-variable 
        `TASKDATAFOLDER` (if not already present).
        Within this folder, there should be a json file for each field, 
        containing the tasks in the format defined in `.json_def.py`.
        """
        super().__init__()
        self._flds:list[str] = []
        self._check_task_folder()

    def add_prj_data(self, aoi, tstart, tstop) -> tuple:
        """
        Method from parent-class which is called within 
        `sitespecificcultivation.data.project.Project` to save task data
        in the project folder structure.
        The `aoi` geodataframe has to contain the column which is used in 
        `sitespecificcultivation.data.project.Project` to specify the field 
        names. Otherwise a ValueError will be raised.

        :param aoi: area of interest for which data is requested
        :type aoi: geopandas.GeoDataFrame
        :param tstart: not used for ebod data
        :type tstart: datetime.date
        :param tstop: not used for ebod data
        :type tstop: datetime.date
        :return: updated tstart and tstop depending on data availability
        :rtype: tuple
        """

        self._check_task_folder()

        # find the field name in the provided aoi/geodataframe
        field_name = None
        for col in aoi.columns:
            fld = aoi[col].values[0]
            if not isinstance(fld, str):
                continue
            if fld.lower() in self._flds:
                field_name = fld
                break

        if field_name is None:
            msg = 'No field name found in provided aoi/geodataframe which '
            msg += 'matches one of the available json-filenames!'
            raise ValueError(msg)
        
        fjson = os.path.join(
            self.project_directory, self.TASKDATAFOLDER, field_name + '.json'
        )
        if not os.path.exists(fjson):
            fjson = os.path.join(
                self.project_directory, self.TASKDATAFOLDER, 
                field_name.lower() + '.json'
            )
        if not os.path.exists(fjson):
            msg = 'Cannot find a .json file for field {}!'.format(field_name)
            raise ValueError(msg)
        fio = open(fjson, 'r')
        jtasks = json.load(fio)
        fio.close()

        # for each task, a folder will be created in the save-dir containing 
        # the geotiff and metadata
        # TODO later addition of tasks in present vegetation periods not 
        # TODO possible in current version
        spath = os.path.join(self.project_directory, self.save_directory)
        tdates = []
        for jtask in jtasks:
            dtask = date.fromisoformat(jtask[Task.DATEBEGIN_KEY])
            if tstart <= dtask <= tstop:
                task = Task.from_uniform_application(jtask, aoi)
                tpath = os.path.join(
                    spath, jtask[Task.DATEBEGIN_KEY] + '_' + task.task_id
                )
                if not os.path.exists(tpath):
                    os.mkdir(tpath)
                task.save_geotiff(tpath)
                tdates.append(dtask)

        # order tdates and return date-range according to found tasks
        tdates.sort()
        return tdates[0], tdates[-1]

    def get_prj_data(self, epoch:date):
        """
        Method from parent-class which is called within
        `sitespecificcultivation.data.project.Project` to load task data from 
        the project folder structure.
        Resulting dict has the ids of the tasks which have been started on the 
        specified `epoch` as its keys. The values are the corresponding 
        `GeoRaster`s.

        :param epoch: not used for ebod data
        :type epoch: datetime.date
        :return: dictionary containing raster representation of task data
        :rtype: dict[GeoRaster]
        """
        ret = {}

        tpath = os.path.join(self.project_directory, self.save_directory)
        for folder in os.listdir(tpath):
            if not epoch.isoformat() in folder:
                continue

            task = Task()
            task.load_geotiff(os.path.join(tpath, folder))
            ret[task.task_id] = task

        return ret
    
    def _check_task_folder(self) -> None:
        if self.project_directory is None:
            return
        
        tdir = os.path.join(self.project_directory, self.TASKDATAFOLDER)
        if not os.path.exists(tdir):
            os.mkdir(tdir)
            return
        
        for file in os.listdir(tdir):
            if not '.json' in file:
                continue
            fldn = file.split('.json')[0].lower()
            if not fldn in self._flds:
                self._flds.append(fldn)
