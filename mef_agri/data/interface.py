import os
from geopandas import GeoDataFrame
from datetime import date

from ..utils.raster import GeoRaster


class DataInterface(object):
    """
    Base class for data interfaces which are used to make data from different 
    data sources available as :class:`mef_agri.utils.raster.GeoRaster`-objects.
    If a child class should be used in :class:`mef_agri.data.project.Project`,
    there should be no required input arguments in the constructor of the child 
    class.

    :param obj_res: desired object-resolution (must not necessarily be used in the child classes), defaults to 10.0
    :type obj_res: float, optional
    """
    DATA_SOURCE_ID = None

    def __init__(self, obj_res:float=10.0) -> None:
        self._pd:str = None
        self._sd:str = None
        self._ores:float = obj_res
        self.add_prj_data_error:bool = False

    @property
    def object_resolution(self) -> float:
        """
        :return: desired object resolution of the resulting data
        :rtype: float
        """
        return self._ores
    
    @object_resolution.setter
    def object_resolution(self, val):
        if isinstance(val, float):
            self._ores = val
        elif isinstance(val, int):
            self._ores = float(val)
        else:
            msg = '`object_resolution` has to be provided as `int` or `float`!'
            raise ValueError(msg)

    @property
    def project_directory(self) -> str:
        """
        :return: directory, where all data for a specific project is located (absolute path)
        :rtype: str
        """
        return self._pd
    
    @project_directory.setter
    def project_directory(self, directory:str):
        self._pd = directory

    @property
    def save_directory(self) -> str:
        """
        :return: directory within `project_directory` to save data to (relative path)
        :rtype: str
        """
        return self._sd
    
    @save_directory.setter
    def save_directory(self, directory:str):
        self._sd = directory

    def add_prj_data(self, aoi:GeoDataFrame, tstart:date, tstop:date) -> tuple:
        """
        Add data to the data-folder in the project directory for the provided 
        `aoi`. This method should return a tuple containing updated `tstart` 
        and `tstop`. An update of `tstart` and `tstop` is necessary if this 
        duration is long but data from the corresponding interface is only 
        available on certain days. In this case, it would not be possible 
        anymore to add further data in this date-range without manually changing 
        the date-range in the ``data_available``-table in the geopackage.

        :param aoi: area of interest (i.e. a field)
        :type aoi: GeoDataFrame
        :param tstart: first day of data
        :type tstart: date
        :param tstop: last day of data
        :type tstop: date
        :return: updated tstart and tstop depending on data availability
        :rtype: tuple
        """
        # NOTE implement this method in child class to enable interaction
        # NOTE with `.project.Project`-class
        pass

    def get_prj_data(self, epoch:date) -> dict[GeoRaster]:
        # NOTE implement this method in child class to enable interaction
        # NOTE with `.project.Project`-class
        pass

    def _check_dirs(self) -> None:
        if None in (self.project_directory, self.save_directory):
            msg = '`self.project_directory` and `self.save_directory` have to '
            msg += 'be set!'
            raise ValueError(msg)
        if not os.path.exists(self.project_directory):
            msg = 'Provided `self.project_directory` does not exist!'
            raise ValueError(msg)
        spath = os.path.join(self.project_directory, self.save_directory)
        if not os.path.exists(spath):
            os.mkdir(spath)