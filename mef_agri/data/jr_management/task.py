import numpy as np
import geopandas as gpd
from datetime import date

from ssc_utils.raster import GeoRaster, PixelUnits
from ssc_utils.gis import bbox_from_gdf


class Task(GeoRaster):
    class TASKS:
        class SOWING:
            TASKID = 'sowing'
            CROP_KEY = 'crop'
            CULTIVAR_KEY = 'cultivar'

        class HARVEST:
            TASKID = 'harvest'
            RESIDUES_REMOVED_KEY = 'residues_removed'
            RESIDUES_CNRATIO_KEY = 'residues_cnratio'

        class MULCHING:
            TASKID = 'mulching'

        class FERTILIZATION:
            TASKID = 'fertilization'
            FERTILIZER_KEY = 'fertilizer'

        class GROWTH_REGULATION:
            TASKID = 'growth_regulation'
            PRODUCT_KEY = 'product'

    TASKID_KEY = 'task_id'
    FIELDNAME_KEY = 'field_name'
    DATEBEGIN_KEY = 'date_begin'
    DATEEND_KEY = 'date_end'
    APPLVAL_KEY = 'application_value'
    VALDESCR_KEY = 'value_description'
    VALUNIT_KEY = 'value_unit'

    def __init__(self):
        """
        Class which represents a task as `GeoRaster`. The raster contains the 
        values for applied resources (e.g. sowing or fertilization). All other 
        information is collected in the `metadata`-dict.
        """
        super().__init__()
        # task properties go into the meta-dict
        self._meta:dict = {
            self.TASKID_KEY: None,
            self.FIELDNAME_KEY: None,
            self.DATEBEGIN_KEY: None,
            self.DATEEND_KEY: None,
        }

    @property
    def task_id(self) -> str:
        """
        :return: id of the task (see task-ids in `Task.TASKS`)
        :rtype: str
        """
        return self._meta[self.TASKID_KEY]
    
    @task_id.setter
    def task_id(self, val):
        self._meta[self.TASKID_KEY] = val

    @property
    def date_begin(self) -> date:
        """
        :return: start date of the task
        :rtype: datetime.date
        """
        return date.fromisoformat(self._meta[self.DATEBEGIN_KEY])
    
    @date_begin.setter
    def date_begin(self, val):
        if isinstance(val, date):
            val = val.isoformat()
        elif isinstance(val, str):
            pass
        else:
            msg = '`date_begin` should be of type `datetime.date` or `str`!'
            raise ValueError(msg)
        self._meta[self.DATEBEGIN_KEY] = val
        
    @property
    def date_end(self) -> date:
        """
        :return: date when task is finished
        :rtype: datetime.date
        """
        return date.fromisoformat(self._meta[self.DATEEND_KEY])
    
    @date_end.setter
    def date_end(self, val):
        if isinstance(val, date):
            val = val.isoformat()
        elif isinstance(val, str):
            pass
        else:
            msg = '`date_end` should be of type `datetime.date` or `str`!'
            raise ValueError(msg)
        self._meta[self.DATEEND_KEY] = val

    @classmethod
    def from_jr_excel_template(cls):
        pass

    @classmethod
    def from_application_map(cls):
        raise NotImplementedError()
    
    @classmethod
    def from_uniform_application(
        cls, appl_info:dict, field:gpd.GeoDataFrame
    ):
        """
        Generate `Task` instance from a uniform application/task. The resulting 
        raster date will have shape of (1, 1, 1) and its extend corresponds to 
        the bounding-box of the field.

        :param appl_info: dictionary containing task infos (see `sitespecificcultivation.data.jr_management.json_def`)
        :type appl_info: dict
        :param field: geodataframe-representation of the field
        :type field: gpd.GeoDataFrame
        :return: `Task` instance
        :rtype: Task
        """
        tsk = cls()

        # georaster stuff
        tsk.crs = field.crs.to_epsg()
        tsk.bounds = bbox_from_gdf(field)
        tsk.layer_ids = [tsk.TASKID_KEY]
        tsk.layer_index = 0
        tsk.units = PixelUnits.FLOAT64
        tsk.nodata_value = np.nan
        tsk.raster_shape = (1, 1, 1)
        tsk.raster = np.ones(tsk.raster_shape).astype(PixelUnits.FLOAT64)
        tsk.transformation = np.array([
            [tsk.bounds[2] - tsk.bounds[0], 0.0, tsk.bounds[0]],
            [0.0, -(tsk.bounds[3] - tsk.bounds[1]), tsk.bounds[3]],
            [0.0, 0.0, 1.0]
        ])

        # task stuff
        tsk.task_id = appl_info[tsk.TASKID_KEY]
        if tsk.APPLVAL_KEY in list(appl_info.keys()):
            if appl_info[tsk.APPLVAL_KEY] is None:
                tsk.raster[:] = np.nan
            else:
                tsk.raster *= appl_info[tsk.APPLVAL_KEY]
            tsk.metadata[tsk.VALDESCR_KEY] = appl_info[tsk.VALDESCR_KEY]
            tsk.metadata[tsk.VALUNIT_KEY] = appl_info[tsk.VALUNIT_KEY]
        tsk.date_begin = date.fromisoformat(appl_info[tsk.DATEBEGIN_KEY])
        if not tsk.DATEEND_KEY in appl_info.keys() or \
            appl_info[tsk.DATEEND_KEY] is None:
            tsk.date_end = date.fromisoformat(appl_info[tsk.DATEBEGIN_KEY])
        else:
            tsk.date_end = date.fromisoformat(appl_info[tsk.DATEEND_KEY])
        tsk.metadata[tsk.FIELDNAME_KEY] = appl_info[tsk.FIELDNAME_KEY]
        
        # if task equals sowing, crop and cultivar will be saved to metadata
        if tsk.task_id == tsk.TASKS.SOWING.TASKID:
            tsk.metadata[tsk.TASKS.SOWING.CROP_KEY] = \
                appl_info[tsk.TASKS.SOWING.CROP_KEY]
            tsk.metadata[tsk.TASKS.SOWING.CULTIVAR_KEY] = \
                appl_info[tsk.TASKS.SOWING.CULTIVAR_KEY]
        # if task equals fertilization, fertilizer type will be saved to 
        # metadata
        elif tsk.task_id == tsk.TASKS.FERTILIZATION.TASKID:
            tsk.metadata[tsk.TASKS.FERTILIZATION.FERTILIZER_KEY] = \
                appl_info[tsk.TASKS.FERTILIZATION.FERTILIZER_KEY]
        # if task equals growth regulation, growth regulator product will be 
        # saved to metadata
        elif tsk.task_id == tsk.TASKS.GROWTH_REGULATION.TASKID:
            tsk.metadata[tsk.TASKS.GROWTH_REGULATION.PRODUCT_KEY] = \
                appl_info[tsk.TASKS.GROWTH_REGULATION.PRODUCT_KEY]

        return tsk
