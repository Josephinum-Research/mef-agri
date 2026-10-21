import datetime as dt
from copy import deepcopy
from inspect import isclass

from ..utils.gpkg import Geopackage, SQLTable, ForeignKey, GeometryType


class DBSchema(object):
    def __init__(self):
        self._fk_te = ForeignKey()
        self._fk_te.fkey_columns = ['eid']
        self._fk_te.ftable = self.table_evals.name
        self._fk_te.ftable_columns = self.table_evals.primary_key
        self._fk_te.on_delete = 'CASCADE'
        self._fk_te.on_update = 'CASCADE'

    @property
    def table_zones(self) -> SQLTable:
        tbl = SQLTable()
        tbl.name = 'zones'
        tbl.columns = {
            'zid': 'INT',  # NOTE incrementing integer
            'field': 'TEXT',
            'zoning': 'TEXT',  # NOTE identifier to find specific zoning in folder structure
            'zone_name': 'TEXT',
            'height': 'REAL'
        }
        tbl.primary_key = ['zid']
        tbl.geom = GeometryType.MULTIPOLYGON
        return tbl

    @property
    def table_evals(self) -> SQLTable:
        tbl = SQLTable()
        tbl.name = 'evaluations'
        tbl.columns = {
            'eid': 'INT',  # NOTE incrementing integer
            'zid': 'INT',
            'epoch_start': 'TEXT',
            'epoch_end': 'TEXT',
            'zone_model': 'TEXT',
            'zone_model_module': 'TEXT',
            'info': 'TEXT'
        }
        tbl.primary_key = ['eid']
        fk_tz = ForeignKey()
        fk_tz.fkey_columns = ['zid']
        fk_tz.ftable = self.table_zones.name
        fk_tz.ftable_columns = self.table_zones.primary_key
        fk_tz.on_delete = 'SET NULL'
        fk_tz.on_update = 'CASCADE'
        tbl.foreign_keys = [fk_tz]
        return tbl

    @property
    def table_obs_def(self) -> SQLTable:
        tbl = SQLTable()
        tbl.name = 'obs_def'
        tbl.columns = {
            'eid': 'INT',
            'name': 'TEXT',
            'model': 'TEXT',
            'epoch': 'TEXT',
            'value': 'REAL',
            'distr': 'TEXT'
        }
        tbl.primary_key = ['eid', 'name', 'model', 'epoch']
        tbl.foreign_keys = [deepcopy(self._fk_te)]
        return tbl

    @property
    def table_states_def(self) -> SQLTable:
        tbl = SQLTable()
        tbl.name = 'states_def'
        tbl.columns = {
            'eid': 'INT',
            'name': 'TEXT',
            'model': 'TEXT',
            'epoch': 'TEXT',
            'value': 'REAL',
            'distr': 'TEXT'
        }
        tbl.primary_key = ['eid', 'name', 'model', 'epoch']
        tbl.foreign_keys = [deepcopy(self._fk_te)]
        return tbl

    @property
    def table_params_def(self) -> SQLTable:
        tbl = SQLTable()
        tbl.name = 'params_def'
        tbl.columns = {
            'eid': 'INT',
            'name': 'TEXT',
            'model': 'TEXT',
            'epoch': 'TEXT',
            'value': 'REAL',
            'distr': 'TEXT'
        }
        tbl.primary_key = ['eid', 'name', 'model', 'epoch']
        tbl.foreign_keys = [deepcopy(self._fk_te)]
        return tbl

    @property
    def table_pfuncs_def(self) -> SQLTable:
        tbl = SQLTable()
        tbl.name = 'pfuncs_def'
        tbl.columns = {
            'eid': 'INT',
            'name': 'TEXT',
            'model': 'TEXT',
            'epoch': 'TEXT',
            'fdef': 'TEXT',
        }
        tbl.primary_key = ['eid', 'name', 'model', 'epoch']
        tbl.foreign_keys = [deepcopy(self._fk_te)]
        return tbl

    @property
    def table_croprot(self) -> SQLTable:
        tbl = SQLTable()
        tbl.name = 'crop_rotation'
        tbl.columns = {
            'eid': 'INT',
            'crop_model': 'TEXT',
            'crop_model_module': 'TEXT',
            'epoch_start': 'TEXT',
            'epoch_end': 'TEXT'
        }
        tbl.primary_key = ['eid', 'epoch_start']
        tbl.foreign_keys = [deepcopy(self._fk_te)]
        return tbl

    
class EvalDB(Geopackage):
    """
    This class does the whole DB stuff for the (multiprocessing) evaluation
    of agricultural models. 
    The database contains exhaustive data and results to ensure, that 
    information do not has to be searched in parts of the code or elsewhere.
        
    Each run of an estimation gets an id (automatically incremented) which 
    is the PK of the evaluations-table. 
    Each run of an estimation is representative for on zone in the zones-table 
    (FK `zid` in the evaluations-table). 
    From the zones-table content it is possible to load the corresponding 
    zoning-georaster from the folder structure.

    The *_def-tables contain information about the model-quantities
    the observations, states, parameters and pfunctions. 
    Each of these quantities belongs to a specific model (see 
    `mef_agri.models`) and its name has to be unique (only) within this 
    model.
    For the states, params, pfunctions, the epoch will usually be the 
    initial epoch. 
    For all of this quantities, it is necessary to provide a value and the 
    distribution information (see`mef_agri.evaluation.stats_utils` for more 
    details).
    """
    def __init__(self, save_dir, file_name):
        """
        constructor

        :param save_dir: absolute path of directory where evaluation-database is/should be located
        :type save_dir: str
        :param file_name: name of the database-file
        :type file_name: str
        """
        super().__init__(save_dir, file_name)
        self._dbs:DBSchema = DBSchema()
        self.tables = {
            self._dbs.table_zones, self._dbs.table_evals, 
            self._dbs.table_croprot, self._dbs.table_states_def, 
            self._dbs.table_params_def, self._dbs.table_pfuncs_def,
            self._dbs.table_obs_def
        }

    def insert_eval_data(
            self, zid:int, epoch_start:dt.date, model, 
            epoch_end:dt.date|None=None, info:str|None=None
        ):
        """
        Insert data into the evaluations-table.
        ``eid`` will be automatically incremented from the rows already being 
        present in the evaluations table

        :param zid: PK of zones-table
        :type zid: int
        :param epoch_start: first epoch of evaluation
        :type epoch_start: datetime.date
        :param model: name of the top-level model (i.e. the zone-model)
        :type model: mef_agri.models.zone.base.Zone
        :param epoch_end: epoch when evaluation is stopped, defaults to None
        :type epoch_end: datetime.date | None, optional
        :param info: information about the evaluation run, defaults to None
        :type info: str | None, optional
        """
        if epoch_end is None:
            epoch_end = 'null'
        if info is None:
            info = 'null'

        if isclass(model):
            mmodule = model.__module__
            mname = model.__name__
        else:
            mmodule = model.__class__.__module__
            mname = model.__class__.__name__

        eid = self.number_of_rows(self._dbs.table_evals.name) + 1
        self.insert_data(
            self._dbs.table_evals.name,
            eid=eid, 
            zid=zid, 
            epoch_start=epoch_start.isoformat(),
            epoch_end=epoch_end, 
            zone_model=mname, 
            zone_model_module=mmodule,
            info=info
        )
