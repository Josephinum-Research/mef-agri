import datetime as dt
import json
from copy import deepcopy
from inspect import isclass

from ..utils.gpkg import Geopackage, SQLTable, ForeignKey, GeometryType
from ..utils.misc import check_date
from .zoning.data import ZoningData
from .zoning import Zoning


class DBSchema(object):
    def __init__(self):
        self._fk_te = ForeignKey()
        self._fk_te.fkey_columns = ['eid']
        self._fk_te.ftable = self.table_evals.name
        self._fk_te.ftable_columns = self.table_evals.primary_key
        self._fk_te.on_delete = 'CASCADE'
        self._fk_te.on_update = 'CASCADE'

        self._tzones = SQLTable()
        self._tzones.name = 'zones'
        self._tzones.columns = {
            'zid': 'INT',  # NOTE incrementing integer
            'field': 'TEXT',
            'zoning': 'TEXT',  # NOTE identifier to find specific zoning in folder structure
            'zone_name': 'TEXT',
            'height': 'REAL'
        }
        self._tzones.primary_key = ['zid']
        self._tzones.geom = GeometryType.MULTIPOLYGON

        self._tevals = SQLTable()
        self._tevals.name = 'evaluations'
        self._tevals.columns = {
            'eid': 'INT',  # NOTE incrementing integer
            'zid': 'INT',
            'epoch_start': 'TEXT',
            'epoch_end': 'TEXT',
            'zone_model': 'TEXT',
            'zone_model_module': 'TEXT',
            'info': 'TEXT'
        }
        self._tevals.primary_key = ['eid']
        fk_tz = ForeignKey()
        fk_tz.fkey_columns = ['zid']
        fk_tz.ftable = self.table_zones.name
        fk_tz.ftable_columns = self.table_zones.primary_key
        fk_tz.on_delete = 'SET NULL'
        fk_tz.on_update = 'CASCADE'
        self._tevals.foreign_keys = [fk_tz]

        self._tdobs = SQLTable()
        self._tdobs.name = 'obs_def'
        self._tdobs.columns = {
            'eid': 'INT',
            'name': 'TEXT',
            'model': 'TEXT',
            'epoch': 'TEXT',
            'value': 'REAL',
            'distr': 'TEXT'
        }
        self._tdobs.primary_key = ['eid', 'name', 'model', 'epoch']
        self._tdobs.foreign_keys = [deepcopy(self._fk_te)]

        self._tdstates = SQLTable()
        self._tdstates.name = 'states_def'
        self._tdstates.columns = {
            'eid': 'INT',
            'name': 'TEXT',
            'model': 'TEXT',
            'epoch': 'TEXT',
            'value': 'REAL',
            'distr': 'TEXT'
        }
        self._tdstates.primary_key = ['eid', 'name', 'model', 'epoch']
        self._tdstates.foreign_keys = [deepcopy(self._fk_te)]

        self._tdparams = SQLTable()
        self._tdparams.name = 'params_def'
        self._tdparams.columns = {
            'eid': 'INT',
            'name': 'TEXT',
            'model': 'TEXT',
            'epoch': 'TEXT',
            'value': 'REAL',
            'distr': 'TEXT'
        }
        self._tdparams.primary_key = ['eid', 'name', 'model', 'epoch']
        self._tdparams.foreign_keys = [deepcopy(self._fk_te)]

        self._tdpfuncs = SQLTable()
        self._tdpfuncs.name = 'pfuncs_def'
        self._tdpfuncs.columns = {
            'eid': 'INT',
            'name': 'TEXT',
            'model': 'TEXT',
            'epoch': 'TEXT',
            'fdef': 'TEXT',
        }
        self._tdpfuncs.primary_key = ['eid', 'name', 'model', 'epoch']
        self._tdpfuncs.foreign_keys = [deepcopy(self._fk_te)]

        self._tcrpr = SQLTable()
        self._tcrpr.name = 'crop_rotation'
        self._tcrpr.columns = {
            'eid': 'INT',
            'crop_model': 'TEXT',
            'crop_model_module': 'TEXT',
            'epoch_start': 'TEXT',
            'epoch_end': 'TEXT'
        }
        self._tcrpr.primary_key = ['eid', 'epoch_start']
        self._tcrpr.foreign_keys = [deepcopy(self._fk_te)]
    
    @property
    def table_zones(self) -> SQLTable:
        return self._tzones
    
    @property
    def table_evals(self) -> SQLTable:
        return self._tevals
    
    @property
    def table_obs_def(self) -> SQLTable:
        return self._tdobs
    
    @property
    def table_states_def(self) -> SQLTable:
        return self._tdstates
    
    @property
    def table_params_def(self) -> SQLTable:
        return self._tdparams
    
    @property
    def table_pfuncs_def(self) -> SQLTable:
        return self._tdpfuncs
    
    @property
    def table_croprot(self) -> SQLTable:
        return self._tcrpr

    
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
    DBNAME = 'eval.gpkg'

    def __init__(self, prjdir:str):
        """
        constructor

        :param save_dir: absolute path of directory where evaluation-database is/should be located
        :type save_dir: str
        :param file_name: name of the database-file
        :type file_name: str
        """
        super().__init__(prjdir, self.DBNAME)
        self._dbs:DBSchema = DBSchema()
        self.tables = {
            self._dbs.table_zones.name: self._dbs.table_zones, 
            self._dbs.table_evals.name: self._dbs.table_evals, 
            self._dbs.table_croprot.name: self._dbs.table_croprot, 
            self._dbs.table_states_def.name: self._dbs.table_states_def, 
            self._dbs.table_params_def.name: self._dbs.table_params_def, 
            self._dbs.table_pfuncs_def.name: self._dbs.table_pfuncs_def,
            self._dbs.table_obs_def.name: self._dbs.table_obs_def  
        }
        self._tdefs = [
            self._dbs.table_croprot.name, self._dbs.table_obs_def.name,
            self._dbs.table_states_def.name, self._dbs.table_params_def.name,
            self._dbs.table_pfuncs_def.name
        ]
        self._ddefs:dict = None
        self._ddefs = dict.fromkeys(self._tdefs, None)
        for tname in self._tdefs:
            self._reset_defdata(tname)

        self._zd:ZoningData = ZoningData(prjdir)

    @property
    def zoning_data(self) -> ZoningData:
        """
        :return: object to handle zoning georasters in the project folder structure
        :rtype: ZoningData
        """
        return self._zd
    
    ############################################################################
    # evaluation and zoning stuff
    ############################################################################
    def save_zoning(self, field_name:str, zoning:Zoning):
        pass
        # TODO

    def insert_eval_data(
            self, zid:int, epoch_start:dt.date|str, model, 
            epoch_end:dt.date|str|None=None, info:str|None=None
        ):
        """
        Insert data into the evaluations-table.
        ``eid`` will be automatically incremented from the rows already being 
        present in the evaluations table

        :param zid: PK of zones-table
        :type zid: int
        :param epoch_start: first epoch of evaluation
        :type epoch_start: datetime.date | str
        :param model: name of the top-level model (i.e. the zone-model)
        :type model: mef_agri.models.zone.base.Zone
        :param epoch_end: epoch when evaluation is stopped, defaults to None
        :type epoch_end: datetime.date | str | None, optional
        :param info: information about the evaluation run, defaults to None
        :type info: str | None, optional
        :return: new evaluation id
        :rtype: int
        """
        epoch_start = check_date(epoch_start)
        if epoch_end is None:
            epoch_end = 'null'
        else:
            epoch_end = check_date(epoch_end).isoformat()
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
        return eid

    ############################################################################
    # data definition stuff
    ############################################################################
    def add_crop_rotation(
            self, eid:int, crop_model, epoch_start:dt.date|str, 
            epoch_end:dt.date|str
        ):
        """
        Add crop rotation data to queued data which will be inserted into the 
        crop_rotation-table.

        :param eid: evaluation id
        :type eid: int
        :param crop_model: model representing the crop
        :type crop_model: object or class
        :param epoch_start: sowing/planting date
        :type epoch_start: datetime.date | str
        :param epoch_end: harvesting/mulching date
        :type epoch_end: datetime.date | str
        """
        if isclass(crop_model):
            mmodule = crop_model.__module__
            mname = crop_model.__name__
        else:
            mmodule = crop_model.__class__.__module__
            mname = crop_model.__class__.__name__
        self._add_defdata(
            self._dbs.table_croprot.name,
            eid=eid,
            crop_model=mname,
            crop_model_module=mmodule,
            epoch_start=check_date(epoch_start).isoformat(),
            epoch_end=check_date(epoch_end).isoformat()
        )

    def add_observation_def(
            self, eid:int, oname:str, omodel:str, epoch:dt.date|str, 
            value:float, distr:dict
        ) -> None:
        """
        Add observation definition to queued data which will be inserted into 
        the obs_def-table.

        :param eid: evaluation id
        :type eid: int
        :param oname: name of the observation within the `model`
        :type oname: str
        :param model: `Model.model_id` of the model which the observation belongs to
        :type model: str
        :param epoch: epoch which the observation belongs to
        :type epoch: datetime.date
        :param value: value of the observation
        :type value: float
        :param distr: dictionary containing information about distribution
        :type distr: dict
        """
        self._add_defdata(
            self._dbs.table_obs_def.name,
            eid=eid,
            name=oname,
            model=omodel,
            epoch=check_date(epoch).isoformat(),
            value=value,
            distr=json.dumps(distr)
        )

    def add_states_def(
            self, eid:int, sname:str, smodel:str, epoch:dt.date|str, 
            value:float, distr:dict
        ) -> None:
        """
        add state definition to queued data which will be inserted into the 
        states_def-table

        :param eid: evaluation id
        :type eid: int
        :param sname: name of the state within the `model`
        :type sname: str
        :param smodel: `Model.model_id` of the model which the state belongs to
        :type smodel: str
        :param epoch: epoch which the state belongs to
        :type epoch: datetime.date
        :param value: value of the state
        :type value: float
        :param distr: dictionary containing information about distribution
        :type distr: dict
        """
        self._add_defdata(
            self._dbs.table_states_def.name,
            eid=eid,
            name=sname,
            model=smodel,
            epoch=check_date(epoch).isoformat(),
            value=value,
            distr=json.dumps(distr)
        )

    def add_param_def(
            self, eid:int, pname:str, pmodel:str, epoch:dt.date|str, 
            value:float, distr:dict
        ) -> None:
        """
        Add parameter definition to queued data which will be inserted into the 
        params_def-table

        :param eid: evaluation id
        :type eid: int
        :param pname: name of the parameter within the `model`
        :type pname: str
        :param pmodel: `Model.model_id` of the model which the parameter belongs to
        :type pmodel: str
        :param epoch: epoch which the parameter belongs to
        :type epoch: datetime.date
        :param value: value of the parameter
        :type value: float
        :param distr: dictionary containing information about distribution
        :type distr: dict
        """
        self._add_defdata(
            self._dbs.table_params_def.name,
            eid=eid,
            name=pname,
            model=pmodel,
            epoch=check_date(epoch).isoformat(),
            value=value,
            distr=json.dumps(distr)
        )

    def add_pfunc_def(
            self, eid:int, fname:str, fmodel:str, epoch:dt.date|str, fdef:dict
        ) -> None:
        """
        Add parameter-function definition to queued data which will be inserted 
        into the pfuncs_def-table.

        :param eid: evaluation id
        :type eid: int
        :param fname: name of the pfunction within the `model`
        :type fname: str
        :param fmodel: `Model.model_id` of the model which the pfunction belongs to
        :type fmodel: str
        :param epoch: epoch which the pfunction belongs to
        :type epoch: datetime.date
        :param fdef: dictionary containing information about the pfunction
        :type fdef: dict
        """
        self._add_defdata(
            self._dbs.table_pfuncs_def.name,
            eid=eid,
            name=fname,
            model=fmodel,
            epoch=check_date(epoch).isoformat(),
            fdef=json.dumps(fdef)
        )

    def insert_crop_rotation(self):
        """
        Insert queued crop rotation data into the crop_rotation-table.
        """
        self._insert_defdata(self._dbs.table_croprot.name)

    def insert_obs_def(self):
        """
        Insert queued observation definitions into the obs_def-table
        """
        self._insert_defdata(self._dbs.table_obs_def.name)

    def insert_states_def(self):
        """
        Insert queued state definitions into the states_def-table
        """
        self._insert_defdata(self._dbs.table_states_def.name)

    def insert_params_def(self):
        """
        Insert queued parameter definitions into the params_def-table
        """
        self._insert_defdata(self._dbs.table_params_def.name)

    def insert_pfuncs_def(self):
        """
        Insert queued parameter-function definitions into the pfuncs_def-table
        """
        self._insert_defdata(self._dbs.table_pfuncs_def.name)

    def _reset_defdata(self, tname:str):
        self._ddefs[tname] = dict.fromkeys(
            self.tables[tname].columns.keys(), 
            []
        )

    def _add_defdata(self, tname:str, **kwargs):
        for col in self._ddefs[tname].keys():
            if col in kwargs.keys():
                self._ddefs[tname].append(kwargs[col])
            else:
                self._ddefs[tname].append('null')

    def _insert_defdata(self, tname:str):
        self.insert_data(tname, **self._ddefs[tname])
        self._reset_defdata(tname)
