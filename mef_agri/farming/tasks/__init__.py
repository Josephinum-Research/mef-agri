import datetime
import numpy as np
import json

from ...utils.raster import GeoRaster
from ...utils.misc import PixelUnits


class DBIntegration(object):
    COL_FIELD = 'field'
    COL_EPOCH = 'epoch'
    COL_TNAME = 'task'

    def __init__(self, table_name:str):
        self._tname:str = table_name

    @property
    def sql_table_exists(self) -> str:
        sql = 'SELECT name FROM sqlite_master WHERE type=\'table\' AND '
        sql += f'name=\'{self._tname}\';'
        return sql

    @property
    def sql_create(self) -> str:
        sql = f'CREATE TABLE {self._tname} ({self.COL_FIELD} TEXT, '
        sql += f'{self.COL_EPOCH} TEXT, {self.COL_TNAME} TEXT, PRIMARY KEY '
        sql += f'({self.COL_FIELD}, {self.COL_EPOCH}, {self.COL_TNAME}));'
        return sql
    
    def sql_insert(self, task:Task, field:str) -> str:
        sql = f'INSERT INTO {self._tname} '
        sql += f'({self.COL_FIELD}, {self.COL_EPOCH}, {self.COL_TNAME}) VALUES '
        sql += self.sql_insert_tuple(task, field) + ';'
        return sql
    
    def sql_query(self, field:str) -> str:
        sql = f'SELECT * FROM {self._tname} WHERE {self.COL_FIELD}=\'{field}\' '
        sql += f'ORDER BY {self.COL_EPOCH} ASC;'
        return sql
    
    def sql_insert_tuple(self, task:Task, field:str) -> str:
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

        @property
        def name(self) -> str:
            """
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
        return self._loop_props(self.NumericValue)

    @property
    def descriptive_values(self) -> list[DescriptiveValue]:
        """
        :return: all properties being instances of :class:`DescriptiveValue`
        :rtype: list[DescriptiveValue]
        """
        return self._loop_props(self.DescriptiveValue)

    def _loop_props(self, cls) -> list:
        ret = []
        for prop in self._props:
            if prop in ('name', 'numeric_values', 'descriptive_values'):
                continue
            attr = getattr(self, prop)
            if isinstance(attr, cls):
                ret.append(attr)
        return ret

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
    :class:`Application` and :class:`Application.NumericValue`)
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

    def add_application(self, appl:Application):
        if not (appl.__class__ in self.valid_applications):
            msg = 'Provided application is not an instance from '
            msg += '`valid_applications`!'
            raise ValueError(msg)
        for val in appl.numeric_values:
            c1 = isinstance(val.value, np.ndarray)
            if (self._avshape is None) and c1:
                self._avshape = val.value.shape
            if c1 and (val.value.shape != self._avshape):
                msg = 'Shapes of provided application maps do not match >>> '
                msg += '{} != {}'.format(self._avshape, appl.value.shape)
                raise ValueError(msg)
        self._apps.append(appl)

    def set_up_task(self):
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
        self.units = PixelUnits.FLOAT32
        self.nodata_value = np.nan

    @staticmethod
    def _check_date(val) -> datetime.date:
        if isinstance(val, datetime.datetime):
            return val.date()
        elif isinstance(val, str):
            dt = datetime.datetime.fromisoformat(val)
            return dt.date()
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
            dt = datetime.datetime.fromisoformat(val)
            return dt.time()
        elif isinstance(val, datetime.time):
            return val
        else:
            msg = 'Provided value cannot be converted to datetime.time!'
            raise ValueError(msg)
