import datetime
import numpy as np

from ...utils.raster import GeoRaster
from ...utils.misc import PixelUnits


class Task(GeoRaster):
    """
    Basic class for agricultural tasks. In **mef_agri**, tasks are represented 
    as :class:`mef_agri.utils.raster.GeoRaster`, thus also enabling the usage of 
    application maps as input information for task data. 
    """
    def __init__(self):
        super().__init__()
        self._meta['date_begin'] = None
        self._meta['date_end'] = None
        self._meta['time_begin'] = None
        self._meta['time_end'] = None

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
    def application_map(self) -> np.ndarray:
        """
        :return: application map as raster data (equal to ``GeoRaster.raster``)
        :rtype: numpy.ndarray
        """
        return self.raster
    
    def specify_application(
            self, appl_val:float | np.ndarray, appl_name:str | list[str],
            appl_unit:str | list[str], appl_ix:int=None
        ) -> None:
        """
        This method sets the following properties of 
        :class:`mef_agri.utils.raster.GeoRaster`

        * ``raster`` (if it is a numeric value, the raster will have the shape ``(1, 1, 1)``)
        * ``raster_shape`` (determined from ``appl_val``)
        * ``units`` (provided value/array will be converted to ``numpy.float32``, i.e. only numeric values are supported)
        * ``nodata_value`` (will be set to ``np.nan``)
        * ``layer_index`` (if not already manually set)
        * ``layer_ids`` (from provided application names ``appl_name``)
        * ``layer_info`` (a dictionary with the provided physical unit(s) in ``appl_unit`` will be available for each ``layer_id``)

        The georeference has to be manually set (i.e. the properties 
        ``crs``, ``bounds``, ``transformation``).
        For unit definitions see :class:`mef_agri.models.utils.__UNITS__`

        :param appl_val: application value - if it is a numeric value, uniform treatment will be assumed - otherwise it has to be a 3-dim `numpy.ndarray` (even when there is only one treatment - see `mef_agri.utils.raster.GeoRaster`)
        :type appl_val: float | numpy.ndarray
        :param appl_name: name of the application value(s)
        :type appl_name: str | list[str]
        :param appl_unit: unit of the application value(s)
        :type appl_unit: str | list[str]
        :param appl_ix: index specifying the dimension of the channels of the application values (if numpy.ndarray), defaults to None
        :type appl_ix: int, optional
        """
        if isinstance(appl_val, float) or isinstance(appl_val, int):
            self.raster = np.array([[[appl_val]]], dtype=np.float32)
            self.raster_shape = (1, 1, 1)
            self.layer_index = 0
        elif isinstance(appl_val, np.ndarray):
            self.raster = appl_val.astype(np.float32)
            self.raster_shape = appl_val.shape
            if self.layer_index is None:
                if appl_ix is None:
                    msg = 'The index specifying the dimension of the channels of '
                    msg += 'the application raster has to be provided (either by '
                    msg += 'setting `Task.layer_index` attribute or by providing '
                    msg += 'the `appl_ix` argument of the '
                    msg += '`Task.specify_application` method)!'
                    raise ValueError(msg)
                self.layer_index = appl_ix
        else:
            msg = '`appl_val` has to be a number/scalar or numpy.ndarray!'
            raise ValueError(msg)
        self.units = PixelUnits.FLOAT32
        self.nodata_value = np.nan
        
        if isinstance(appl_name, str):
            appl_name = [appl_name,]
        if isinstance(appl_unit, str):
            appl_unit = [
                appl_unit for i in range(self.raster_shape[self.layer_index])
            ]
        if len(appl_name) != self.raster_shape[self.layer_index]:
            msg = 'number of application names has to match the number of '
            msg += 'applications, i.e. number of layers in `appl_val`!'
            raise ValueError(msg)
        if len(appl_unit) != self.raster_shape[self.layer_index]:
            msg = 'number of application units has to match the number of '
            msg += 'applications, i.e. number of layers in `appl_val`!'
            raise ValueError(msg)
        
        self.layer_ids = appl_name
        li = {}
        for ikey, ival in zip(appl_name, appl_unit):
            li[ikey] = ival
        self.layer_infos['units'] = li
    
    @classmethod
    def get_properties(cls) -> list:
        """
        Classmethod
        
        :return: names of all methods decorated with ``@property``
        :rtype: list
        """
        props = []
        def loop_cls(cls):
            if cls.__name__ == GeoRaster.__name__:
                return
            for key, val in vars(cls).items():
                if isinstance(val, property):
                    props.append(key)
            loop_cls(cls.__base__)
        loop_cls(cls)
        return props

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
