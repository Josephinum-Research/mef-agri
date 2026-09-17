import datetime
import numpy as np
import json
import os
from geopandas import GeoDataFrame

from ...utils.raster import GeoRaster, bbox_from_gdf
from ...utils.misc import PixelUnits
from ...data.project import ProjectData, DB


class ProjectTasksExtension(object):
    """
    Class which extends :class:`mef_agri.data.project.ProjectData` to add and 
    get tasks-data.
    """
    TASK_DIRECTORY = 'tasks'

    def __init__(self, prj:ProjectData):
        """
        :param prj: instande of project-data(base)
        :type prj: mef_agri.data.project.ProjectData
        """
        self._prj:ProjectData = prj
        self._tdir = os.path.join(prj.directory, self.TASK_DIRECTORY)
        if not os.path.exists(self._tdir):
            os.mkdir(self._tdir)

    def save_task(self, field_name:str, task:Task) -> None:
        """
        Save a task in the project directory

        :param field_name: name of the field
        :type field_name: str
        :param task: task object
        :type task: Task
        :raises ValueError: if task data is already available for provided ``field_name`` and ``task.date_begin``
        """
        fdir = os.path.join(self._tdir, field_name)
        if not os.path.exists(fdir):
            os.mkdir(fdir)
        
        dstr = task.date_begin.isoformat()
        ddir = os.path.join(fdir, dstr)
        if not os.path.exists(ddir):
            os.mkdir(ddir)

        tname = task.__class__.__name__
        sdir = os.path.join(ddir, tname)
        if os.path.exists(sdir):
            msg = f'A `{tname}`-task is already available for field '
            msg += f'`{field_name}` at `{dstr}`!'
            raise ValueError(msg)
        
        os.mkdir(sdir)
        if not task.set_up:
            task.layer_index = 0
            field = self._prj.fields[
                self._prj.fields[DB.TBL_FIELDS.COL_FIELDNAME] == field_name
            ]
            task.set_up_task(field)
        task.save_geotiff(sdir)

    def get_tasks(
            self, tstart:datetime.date, tstop:datetime.date=None, 
            field_name:str=None
        ) -> dict:
        """
        TODO

        :param tstart: _description_
        :type tstart: datetime.date
        :param tstop: _description_, defaults to None
        :type tstop: datetime.date, optional
        :param field_name: _description_, defaults to None
        :type field_name: str, optional
        :return: _description_
        :rtype: dict
        """
        pass


class DBIntegration(object):
    """
    Class which provides sql-commands to integrate a table into a database 
    containing tasks-information.
    """
    COL_FIELD = 'field'
    COL_EPOCH = 'epoch'
    COL_TNAME = 'task'

    def __init__(self, table_name:str):
        """
        :param table_name: name of the table
        :type table_name: str
        """
        self._tname:str = table_name

    @property
    def sql_table_exists(self) -> str:
        """
        :return: name of the tasks-table in the database, if it exists
        :rtype: str
        """
        sql = 'SELECT name FROM sqlite_master WHERE type=\'table\' AND '
        sql += f'name=\'{self._tname}\';'
        return sql

    @property
    def sql_create(self) -> str:
        """
        :return: sql-command to create the tasks-table
        :rtype: str
        """
        sql = f'CREATE TABLE {self._tname} ({self.COL_FIELD} TEXT, '
        sql += f'{self.COL_EPOCH} TEXT, {self.COL_TNAME} TEXT, PRIMARY KEY '
        sql += f'({self.COL_FIELD}, {self.COL_EPOCH}, {self.COL_TNAME}));'
        return sql
    
    def sql_insert(self, task:Task, field:str) -> str:
        """
        Returns the sql-command to insert a task into the tasks-table

        :param task: task-object
        :type task: mef_agri.farming.tasks.Task
        :param field: name of the field
        :type field: str
        :return: sql-command
        :rtype: str
        """
        sql = f'INSERT INTO {self._tname} '
        sql += f'({self.COL_FIELD}, {self.COL_EPOCH}, {self.COL_TNAME}) VALUES '
        sql += self.sql_insert_tuple(task, field) + ';'
        return sql

    def sql_delete(self, task:Task, field:str) -> str:
        sql = f'DELETE FROM {self._tname} WHERE {self.COL_FIELD}=\'{field}\' '
        sql += f'AND {self.COL_EPOCH}=\'{task.date_begin.isoformat()}\' AND '
        sql += f'{self.COL_TNAME}=\'{task.__class__.__name__}\';'
        return sql
    
    def sql_query(self, field:str) -> str:
        """
        Returns sql-command to query all tasks for a specified field.

        :param field: name of the field
        :type field: str
        :return: sql-command
        :rtype: str
        """
        sql = f'SELECT * FROM {self._tname} WHERE {self.COL_FIELD}=\'{field}\' '
        sql += f'ORDER BY {self.COL_EPOCH} ASC;'
        return sql
    
    def sql_insert_tuple(self, task:Task, field:str) -> str:
        """
        Returns a string containing a sql-valid tuple to be inserted into the 
        tasks-table.

        :param task: task object
        :type task: mef_agri.farming.tasks.Task
        :param field: name of the field
        :type field: str
        :return: string containing the tuple
        :rtype: str
        """
        sql = f'(\'{field}\', \'{task.date_begin.isoformat()}\', '
        sql += f'\'{task.__class__.__name__}\')'
        return sql


################################################################################
# APPLICATION
################################################################################
class Application(object):
    """
    This class represents applications which are part of a :class:`Task`. 
    The reason for splitting applications from tasks is, that in some cases more 
    than one application is performed within one task (e.g. application of two 
    types of fertilizers, two crop protection products or combined sowing and 
    rotary harrow).

    An application is composed of numeric values :class:`NumericValue` (e.g. 
    amount of sown seeds in [kg/ha]) and descriptive values 
    :class:`DescriptiveValue` (e.g. names of sown crop and cultivar).
    """
    ##########################   NESTED-CLASSES   ##############################
    class NumericValue(object):
        def __init__(self):
            self._name:str = None
            self._val:float | np.ndarray = None
            self._unit:str = None
            self._uv:list[str] = None
            self._descr:str = None
            self._ores:float = None
            self._epsg:int = None
            self._bbox:tuple = None

        @property
        def name(self) -> str:
            """
            settable

            :return: name of the numeric application value
            :rtype: str
            """
            return self._name
        
        @name.setter
        def name(self, name):
            self._name = name

        @property
        def description(self) -> str:
            """
            settable
            
            :return: description of the numeric value
            :rtype: str
            """
            if self._descr is None:
                return self.name
            else:
                return self._descr
        
        @description.setter
        def description(self, descr):
            self._descr = descr

        @property
        def valid_units(self) -> list[str]:
            """
            settable
            
            :return: valid values for :func:`unit`
            :rtype: list[str]
            """
            return self._uv
        
        @valid_units.setter
        def valid_units(self, valunits):
            self._uv = valunits
        
        @property
        def unit(self) -> str:
            """
            settable
            
            :return: unit of the application value (see :class:`mef_agri.models.utils.__UNITS__`)
            :rtype: str
            """
            return self._unit
        
        @unit.setter
        def unit(self, unit):
            if self.valid_units is not None:
                if not unit in self.valid_units:
                    msg = 'Unit has to be one of the following values: '
                    msg += ', '.join(self.valid_units)
                    raise ValueError(msg)
            self._unit = unit
    
        @property
        def value(self) -> float | np.ndarray:
            """
            settable
            
            :return: application value itself (numeric value if uniform application, numpy.ndarray if application map)
            :rtype: float | numpy.ndarray
            """
            return self._val
        
        @value.setter
        def value(self, value):
            if isinstance(value, (int, float)):
                self._val = float(value)
            elif isinstance(value, np.ndarray):
                self._val = value
            else:
                msg = '`value` has to be a numeric value (i.e. uniform '
                msg += 'application) or a `numpy.ndarray` representing an '
                msg += 'application map!'
                raise ValueError(msg)

        @property
        def obj_resolution(self) -> float:
            """
            settable
            
            :return: object resolution (i.e pixel dimension) in [m] if :func:`value` is a numpy.ndarray (otherwise ``None``)
            :rtype: float
            """
            return self._ores

        @obj_resolution.setter
        def obj_resolution(self, val):
            self._ores = val

        @property
        def crs(self) -> int:
            """
            settable

            :return: epsg-code of crs if :func:`value` is a numpy.ndarray (otherwise ``None``)
            :rtype: int
            """
            return self._epsg

        @crs.setter
        def crs(self, val):
            self._epsg = val

        @property
        def bbox(self) -> tuple:
            """
            settable

            :return: bounding-box (x_min, y_min, x_max, y_max), if :func:`value` is a numpy.ndarray (otherwise ``None``)
            :rtype: tuple
            """
            return self._bbox

        @bbox.setter
        def bbbox(self, val):
            self._bbox = val

    class DescriptiveValue(object):
        def __init__(self):
            self._name:str = None
            self._value:str = None
            self._descr:str = None

        @property
        def name(self) -> str:
            """
            :return: name of the descriptive value of the application
            :rtype: str
            """
            return self._name

        @name.setter
        def name(self, name):
            self._name = name

        @property
        def description(self) -> str:
            """
            :return: detailed description of the descriptive value
            :rtype: str
            """
            if self._descr is None:
                return self.name
            else:
                return self._descr
        
        @description.setter
        def description(self, descr):
            self._descr = descr

        @property
        def value(self) -> str:
            """
            :return: descriptive value of the application (can be also provided as ``dict``, which will be serialized to a json-string)
            :rtype: str
            """
            return self._value
        
        @value.setter
        def value(self, val):
            if isinstance(val, dict):
                val = json.dumps(val)
            self._value = val

    #####################   Application-Class-stuff   ##########################
    def __init__(self):
        self._props = self.get_properties()
        self._nvals:list = None
        self._nnvs:list[str] = None
        self._dvals:list = None
        self._ndvs:list[str] = None

    @property
    def name(self) -> str:
        """
        :return: name of the application
        :rtype: str
        """
        msg = '`name` of application has to be defined in child class!'
        raise NotImplementedError(msg)

    @property
    def numeric_values(self) -> list[NumericValue]:
        """
        :return: all properties being instances of :class:`NumericValue`
        :rtype: list[NumericValue]
        """
        if None in (self._nvals, self._nnvs):
            self._nnvs, self._nvals = self._loop_props(self.NumericValue)
        return self._nvals

    @property
    def numval_names(self) -> list[str]:
        """
        :return: names of :func:`numeric_values` (same order!)
        :rtype: list[str]
        """
        if None in (self._nvals, self._nnvs):
            self._nnvs, self._nvals = self._loop_props(self.NumericValue)
        return self._nnvs

    @property
    def descriptive_values(self) -> list[DescriptiveValue]:
        """
        :return: all properties being instances of :class:`DescriptiveValue`
        :rtype: list[DescriptiveValue]
        """
        if None in (self._dvals, self._ndvs):
            self._ndvs, self._dvals = self._loop_props(self.DescriptiveValue)
        return self._dvals

    @property
    def descrval_names(self) -> list[str]:
        """
        :return: names of :func:`descriptive_values` (same order!)
        :rtype: list[str]
        """
        if None in (self._dvals, self._ndvs):
            self._ndvs, self._dvals = self._loop_props(self.DescriptiveValue)
        return self._ndvs

    def _loop_props(self, cls) -> tuple[list, list]:
        vnames, vals = [], []
        for prop in self._props:
            if prop in ('name', 'numeric_values', 'descriptive_values'):
                continue
            attr = getattr(self, prop)
            if isinstance(attr, cls):
                vnames.append(prop)
                vals.append(attr)
        return vnames, vals
    
    @staticmethod
    def map_from_file(self, fp:str) -> tuple[int, float, tuple, np.ndarray]:
        """
        Load application map from file

        :param fp: absolute path to the file
        :type fp: str
        :return: crs/epsg, object-resolution [m], bbox (x_min, y_min, x_max, y_max), application map itself (i.e. raster)
        :rtype: tuple[int, float, tuple, np.ndarray]
        """
        raise NotImplementedError()

    @classmethod
    def get_properties(cls) -> list:
        """
        Classmethod
        
        :return: names of all methods decorated with ``@property``
        :rtype: list
        """
        props = []
        def loop_cls(cls):
            if cls.__name__ == Application.__name__:
                return
            for key, val in vars(cls).items():
                if isinstance(val, property):
                    props.append(key)
            loop_cls(cls.__base__)
        loop_cls(cls)
        return props


################################################################################
# TASK
################################################################################
class Task(GeoRaster):
    """
    Basic class for agricultural tasks. In **mef_agri**, tasks are represented 
    as :class:`mef_agri.utils.raster.GeoRaster`, thus also enabling the usage of 
    application maps as input information for task data. 

    A task is composed of several applications :class:`Application` where the 
    corresponding numeric values :class:`Application.NumericValue` represent the 
    layers of the task/georaster where layer-ids are composed of the application 
    name and the name of the numeric value (i.e. the ``name`` attributes of 
    :class:`Application` and :class:`Application.NumericValue`).
    If there is more than one application value which is derived from an 
    application map, it is required, that these values exhibit the same 
    georeference (i.e. same raster-shape, bounding-box, object-resolution, 
    crs/epsg).

    The descriptive values :class:`Application.DescriptiveValue` are stored in 
    the metadata file of the georaster.
    """
    META_APPL_KEY = 'applications'
    META_APPL_NAME = 'application'
    META_APPL_MODULE = 'application_module'
    META_APPL_NVALS = 'numeric_values'
    META_APPL_NVALS_UNIT = 'unit'
    META_APPL_NVALS_LID = 'layer_id'
    META_APPL_NVALS_FROM = 'derived_from'
    META_APPL_DVALS = 'descriptive_values'
    META_APPL_DVALS_VALUE = 'value'

    def __init__(self):
        super().__init__()
        self._apps:list[Application] = []
        self._avshape:tuple = None
        self._avores:float = None
        self._avbbox:tuple = None
        self._avepsg:int = None
        self._set_up:bool = False
        self._meta['date_begin'] = None
        self._meta['date_end'] = None
        self._meta['time_begin'] = None
        self._meta['time_end'] = None
        self._meta['task_name'] = self.__class__.__name__
        self._meta['task_module'] = self.__module__

    @property
    def task_name(self) -> str:
        """
        :return: name of the class representing the task
        :rtype: str
        """
        return self._meta['task_name']
    
    @property
    def task_module(self) -> str:
        """
        :return: module containing class representing the task
        :rtype: str
        """
        return self._meta['task_module']

    @property
    def date_begin(self) -> datetime.date:
        """
        :return: date when task has been started
        :rtype: datetime.date
        """
        return datetime.date.fromisoformat(self._meta['date_begin'])
    
    @date_begin.setter
    def date_begin(self, val):
        self._meta['date_begin'] = self._check_date(val).isoformat()

    @property
    def time_begin(self) -> datetime.time:
        """
        :return: time when task has been started
        :rtype: datetime.time
        """
        return datetime.time.fromisoformat(self._meta['time_begin'])
    
    @time_begin.setter
    def time_begin(self, val):
        self._meta['time_begin'] = self._check_time(val).isoformat()

    @property
    def date_end(self) -> datetime.date:
        """
        :return: date when task has been finished
        :rtype: datetime.date
        """
        return datetime.date.fromisoformat(self._meta['date_end'])
    
    @date_end.setter
    def date_end(self, val):
        self._meta['date_end'] = self._check_date(val).isoformat()

    @property
    def time_end(self) -> datetime.time:
        """
        :return: time when task has been finished
        :rtype: datetime.time
        """
        return datetime.time.fromisoformat(self._meta['time_end'])
    
    @time_end.setter
    def time_end(self, val):
        self._meta['time_end'] = self._check_time(val).isoformat()

    @property
    def applications(self) -> list[Application]:
        """
        :return: applications which belong to the task (i.e. added with :func:`add_application`)
        :rtype: list[Application]
        """
        return self._apps

    @property
    def valid_applications(self) -> tuple | list:
        """
        :return: tuple/list of application classes which instances can be added to the current task
        :rtype: tuple | list
        """
        msg = '`valid_applications` have to be defined in child class!'
        raise NotImplementedError(msg)

    @property
    def set_up(self) -> bool:
        """
        :return: flag if task has already been set up (i.e. :func:`set_up_task` has already been called)
        :rtype: bool
        """
        return self._set_up

    def add_application(self, appl:Application):
        """
        Add an :class:`Application` to the task. 

        :param appl: application which should be added to a task
        :type appl: Application
        :raises ValueError: if ``appl`` is not in :func:`valid_applications` (defined in the child-class definition of :class:`Task`)
        :raises ValueError: if numeric values in ``appl`` which represent an application map have different shape
        """
        if not (appl.__class__ in self.valid_applications):
            msg = 'Provided application is not an instance from '
            msg += '`valid_applications`!'
            raise ValueError(msg)
        for val in appl.numeric_values:
            if isinstance(val.value, np.ndarray):
                if self._avshape is None:
                    self._avshape = val.value.shape
                    self._avores = val.obj_resolution
                    self._avbbox = val.bbox
                    self._avepsg = val.crs
                c1 = self._avshape != val.value.shape
                c2 = self._avores != val.obj_resolution
                c3 = self._avbbox != val.bbox
                c4 = self._avepsg != val.crs
                if True in (c1, c2, c3, c4):
                    msg = 'Georeference of provided application values does not'
                    msg += 'match (either raster-shape, object-resolution, '
                    msg += 'bounding-box or crs/epsg)!'
                    raise ValueError(msg)
        self._apps.append(appl)

    def set_up_task(self, field:GeoDataFrame=None):
        """
        Processing the provided applications (see :func:`add_application`) to 
        create appropriate georaster information.
        Numeric values of applications are converted to layers of the georaster.
        The shape of the layers results in the following cases

        * if all numeric values from all applications are float values, the resulting raster will have (1, 1, n_numvals)
        * if at least one numeric value represents an application map with shape (n, m), then the raster will have (n, m, n_numvals); scalar numeric values will be simply upscaled by creating a (n, m) raster wher all values are equal

        Important note: numeric values which represent an application map and 
        which should be added to a task have to exhibit the same shape (ensured 
        in :func:`add_application`).
        :func:`units` is set to :class:`mef_agri.utils.misc.PixelUnits`.FLOAT32 
        and :func:`nodata_value` to ``numpy.nan``.

        ``field`` is only required, if all numeric application values are 
        scalar, i.e. only uniform applications have been performed in the task.

        :param field: one row of a geodataframe containing field geo-information, defaults to None
        :type field: geopandas.GeoDataFrame, optional
        :raises ValueError: if :func:`layer_index` has not been provided yet
        :raises ValueError: if ``field`` has not exactly one row
        :raises ValueError: if there are redundant applications and/or numeric values
        """
        if self.layer_index is None:
            msg = '`layer_index` not provided yet but it is necessary to set '
            msg += 'up the task/georaster!'
            raise ValueError(msg)
        if not self.META_APPL_KEY in self._meta.keys():
            self._meta[self.META_APPL_KEY] = {}
        
        for appl in self._apps:
            applinfo = {
                self.META_APPL_NAME: appl.__class__.__name__,
                self.META_APPL_MODULE: appl.__class__.__module__,
                self.META_APPL_NVALS: {},
                self.META_APPL_DVALS: {}
            }

            # processing numeric values of current application
            # i.e. becoming GeoRaster-layers
            for numval in appl.numeric_values:
                # processing layer-id
                lid = appl.name + ' -> ' + numval.name
                if lid in self.layer_ids:
                    msg = f'Application-name {appl.name} with value '
                    msg += f'{numval.name} already present in `layer_ids`!'
                    raise ValueError(msg)
                self.layer_ids.append(lid)

                # create layer from numeric value
                valfrom = 'from-application-map'
                if isinstance(numval.value, float):
                    valfrom = 'from-numeric-value'
                    if self._avshape is None:
                        layer = np.array([[[numval.value]]], dtype=np.float32)
                    else:
                        layer = np.ones(
                            self._avshape, dtype=np.float32
                        ) * numval.value
                else:
                    layer = numval.value

                if self.raster is None:
                    self.raster = layer
                else:
                    self.raster = np.concatenate(
                        (self.raster, layer), axis=self.layer_index
                    )
                
                # save information about numeric value to metadata
                # to reconstruct the corresponding application
                applinfo[self.META_APPL_NVALS][numval.name] = {
                    self.META_APPL_NVALS_LID: lid,
                    self.META_APPL_NVALS_UNIT: numval.unit,
                    self.META_APPL_NVALS_FROM: valfrom
                }

            # processing descriptive values of current application
            # i.e. being integrated into GeoRaster-metadata
            for descrval in appl.descriptive_values:
                applinfo[self.META_APPL_DVALS][descrval.name] = descrval.value
                
            # save metadata
            self._meta[self.META_APPL_KEY][appl.name] = applinfo

        # final settings
        if self._avshape is None:
            if (field is None) or (len(field) != 1):
                msg = '`field` has to be a geopandas.GeoDataFrame containing '
                msg += 'exactly one row with field-information!'
                raise ValueError(msg)
            self.crs = field.crs.to_epsg()
            self.bounds = bbox_from_gdf(field)
            ores = max(
                self.bounds[2] - self.bounds[0],
                self.bounds[3] - self.bounds[1]
            )
        else:
            self.crs = self._avepsg
            self.bounds = self._avbbox
            ores = self._avores
        self.transformation = np.array([
            [ores, 0., self.bounds[0]],
            [0., -ores, self.bounds[3]],
            [0., 0., 1.]
        ])
        self.raster_shape = self.raster.shape
        self.units = PixelUnits.FLOAT32
        self.nodata_value = np.nan
        self._set_up = True

    @staticmethod
    def _check_date(val) -> datetime.date:
        if isinstance(val, datetime.datetime):
            return val.date()
        elif isinstance(val, str):
            return datetime.date.fromisoformat(val)
        elif isinstance(val, datetime.date):
            return val
        else:
            msg = 'Provided value cannot be converted to datetime.date!'
            raise ValueError(msg)
        
    @staticmethod
    def _check_time(val) -> datetime.time:
        if isinstance(val, datetime.datetime):
            return val.time()
        elif isinstance(val, str):
            return datetime.time.fromisoformat(val)
        elif isinstance(val, datetime.time):
            return val
        else:
            msg = 'Provided value cannot be converted to datetime.time!'
            raise ValueError(msg)
