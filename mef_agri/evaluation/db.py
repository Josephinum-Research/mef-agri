import sqlite3
import os
import json
import numpy as np
import pandas as pd
from datetime import date
from copy import deepcopy
from inspect import isclass
from importlib import import_module

# TODO include later from ..data.eval_def import EvaluationDefinitions
from ..models.base import Quantities


class EvaluationDB(object):
    ############################################################################
    # commands to set up db-schema
    CREATE_EVAL =  'CREATE TABLE evaluations ('
    CREATE_EVAL += 'eid INT, epoch_start TEXT, epoch_end TEXT, crs INT,'
    CREATE_EVAL += 'zmodel TEXT, zmodel_module TEXT, eval_info TEXT, '
    CREATE_EVAL += 'PRIMARY KEY (eid)'
    CREATE_EVAL += ');'

    CREATE_ZONE =  'CREATE TABLE zones ('
    CREATE_ZONE += 'zid INT, eid INT, zname TEXT, latitude REAL, height REAL, '  # NOTE WGS84 latitude
    CREATE_ZONE += 'PRIMARY KEY (zid), '
    CREATE_ZONE += 'FOREIGN KEY (eid) REFERENCES evaluations(eid), '
    CREATE_ZONE += 'UNIQUE (eid, zname)'
    CREATE_ZONE += ');'

    CREATE_GCS =  'CREATE TABLE zone_geo_coordinates ('
    CREATE_GCS += 'zid INT, x REAL, y REAL, '
    CREATE_GCS += 'FOREIGN KEY (zid) REFERENCES zones(zid), '
    CREATE_GCS += 'UNIQUE (zid, x, y)'
    CREATE_GCS += ');'

    CREATE_DOBS =  'CREATE TABLE obs_def('
    CREATE_DOBS += 'zid INT, name TEXT, model TEXT, '
    CREATE_DOBS += 'epoch TEXT, value REAL, distr TEXT, '
    CREATE_DOBS += 'FOREIGN KEY (zid) REFERENCES zones(zid), '
    CREATE_DOBS += 'UNIQUE (zid, name, model, epoch)'
    CREATE_DOBS += ');'

    CREATE_DSTATES =  'CREATE TABLE states_def ('
    CREATE_DSTATES += 'zid INT, name TEXT, model TEXT, '
    CREATE_DSTATES += 'epoch TEXT, value REAL, distr TEXT, '
    CREATE_DSTATES += 'FOREIGN KEY (zid) REFERENCES zones(zid), '
    CREATE_DSTATES += 'UNIQUE (zid, name, model, epoch)'
    CREATE_DSTATES += ');'

    CREATE_DHPARAMS =  'CREATE TABLE hparams_def ('
    CREATE_DHPARAMS += 'zid INT, name TEXT, model TEXT, '
    CREATE_DHPARAMS += 'epoch TEXT, value REAL, distr TEXT, '
    CREATE_DHPARAMS += 'FOREIGN KEY (zid) REFERENCES zones(zid), '
    CREATE_DHPARAMS += 'UNIQUE (zid, name, model, epoch)'
    CREATE_DHPARAMS += ');'

    CREATE_DHPFUNCS =  'CREATE TABLE hpfuncs_def ('
    CREATE_DHPFUNCS += 'zid INT, name TEXT, model TEXT, epoch TEXT, fdef TEXT, '
    CREATE_DHPFUNCS += 'FOREIGN KEY (zid) REFERENCES zones(zid), '
    CREATE_DHPFUNCS += 'UNIQUE (zid, name, model, epoch)'
    CREATE_DHPFUNCS += ');'

    CREATE_CROPROT =  'CREATE TABLE crop_rotation('
    CREATE_CROPROT += 'zid INT, cmodel TEXT, cmodel_module TEXT, '
    CREATE_CROPROT += 'epoch_start TEXT, epoch_end TEXT, '
    CREATE_CROPROT += 'PRIMARY KEY (zid, epoch_start), '
    CREATE_CROPROT += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_CROPROT += ');'

    ############################################################################
    # commands for EstimationWorker
    GET_EDATA = 'SELECT * FROM evaluations WHERE eid={};'
    GET_ZDATA = 'SELECT * FROM zones WHERE eid={};'
    GET_ZGCS = 'SELECT x, y FROM zone_geo_coordinates WHERE zid={};'
    GET_STATES_DEF = 'SELECT name, model, value, epoch, distr FROM states_def '
    GET_STATES_DEF += 'WHERE zid={} AND epoch=\'{}\';'
    GET_HPARAMS_DEF = 'SELECT name, model, value, epoch, distr FROM hparams_def'
    GET_HPARAMS_DEF += ' WHERE zid={} AND epoch=\'{}\';'
    GET_HPFUNCS_DEF = 'SELECT name, model, epoch, fdef FROM hpfuncs_def '
    GET_HPFUNCS_DEF += 'WHERE zid={} AND epoch=\'{}\';'
    GET_OBS_DEF = 'SELECT name, model, value, epoch, distr FROM obs_def '
    GET_OBS_DEF += 'WHERE zid={} AND epoch=\'{}\';'
    GET_CROPROT =  'SELECT cmodel, cmodel_module, epoch_start, epoch_end FROM '
    GET_CROPROT += 'crop_rotation WHERE zid={};'

    ############################################################################
    EVAL_TABLE_NAMES = {
        Quantities.STATE: 'states_eval',
        Quantities.OBS: 'obs_eval',
        Quantities.HPARAM: 'hparams_eval',
        Quantities.HPFUNC: 'hpfuncs_eval',
        Quantities.ROUT: 'out_eval',
        Quantities.DOUT: 'out_eval'
    }
    DEF_TABLE_NAMES = {
        Quantities.STATE: 'states_def',
        Quantities.HPARAM: 'hparams_def',
        Quantities.HPFUNC: 'hpfuncs_def',
        Quantities.OBS: 'obs_def'
    }

    # methods of EvaluationDB
    def __init__(self, directory:str, dbname:str) -> None:
        """
        This class does the whole DB stuff for the (multiprocessing) evaluation  
        of `.estimation.py`. The database contains exhaustive data and results  
        to ensure, that information do not has to be searched in parts of the 
        code or elsewhere.
        
        Each run of an estimation gets an id (automatically incremented) which 
        is the PK of the evaluations-table. Each zone of 
        the considered field(s) is stored in the DB in two tables. The 
        zones-table contains the zone-id (automatically incremented) which 
        represent the PK and the FK of most of the other tables to assign the 
        inputs and outputs of the estimation to the correct evaluation-run and 
        zone. The zone_geo_coordinates-table holds the coordinates of the raster 
        elements (info in evaluations-table) which can be used for the 
        visualization of the results.

        The *_def-tables contain information about the quantities of inference 
        the observations, states, hyper-parameters and hp-functions. Each of 
        these quantities belongs to a specific model (see 
        `sitespecificcultivation.inference.models`) and its name has to be 
        unique (only) within this model. For the states, hparams, hp-functions 
        the epoch will usually be the initial epoch. For all of this quantities, 
        it is necessary to provide a value and the distribution information 
        (see`sitespecificcultivation.inference.stats_utils` for more details).

        The *_eval-tables contain the evaluated values for the corresponding 
        quantities. All model-outputs are considered. Thus, all 
        values (value-column) with same `zid`, name (of the quantity), model 
        (which the quantity belongs to) and epoch represent the corresponding 
        probability distribution.

        :param directory: directory where DB is located
        :type directory: str
        :param dbname: name of the DB (i.e. the sqlite-file)
        :type dbname: str
        """
        self._path = os.path.join(directory, dbname)
        dbexist = os.path.exists(self._path)
        self._conn = sqlite3.connect(self._path)
        self._curs = self._conn.cursor()
        if not dbexist:
            self._curs.execute(self.CREATE_EVAL)
            self._curs.execute(self.CREATE_ZONE)
            self._curs.execute(self.CREATE_GCS)
            self._curs.execute(self.CREATE_DOBS)
            self._curs.execute(self.CREATE_DSTATES)
            self._curs.execute(self.CREATE_DHPARAMS)
            self._curs.execute(self.CREATE_EOBS)
            self._curs.execute(self.CREATE_ESTATES)
            self._curs.execute(self.CREATE_EHPARAMS)
            self._curs.execute(self.CREATE_EOUT)
            self._curs.execute(self.CREATE_DHPFUNCS)
            self._curs.execute(self.CREATE_EHPFUNCS)
            self._curs.execute(self.CREATE_CROPROT)
            self._curs.fetchall()
            self._conn.commit()

        self._reset_insert_dobs()
        self._reset_insert_dstates()
        self._reset_insert_dhparams()
        self._reset_insert_eobs()
        self._reset_insert_estates()
        self._reset_insert_ehparams()
        self._reset_insert_eout()
        self._reset_insert_dhpfuncs()
        self._reset_insert_ehpfuncs()
        self._reset_insert_cropr()

        self._last_exc:str = 'No exception occurred.'  # temporary variable for last raised exception
        self._script_io = None
        self._scriptp = None
        self._eid:int = None

    @property
    def database_path(self) -> str:
        """
        :return: path to DB
        :rtype: str
        """
        return self._path
    
    @property
    def connection(self) -> sqlite3.Connection:
        """
        :return: connection-object to communicate with the database
        :rtype: sqlite3.Connection
        """
        return self._conn

    @property
    def last_exception(self) -> str:
        """
        :return: last occured exception when communicating with the DB
        :rtype: str
        """
        ret = deepcopy(self._last_exc)
        self._last_exc = 'No exception ocurred.'
        return ret
    
    @property
    def evaluation_id(self) -> int:
        """
        :return: eval-id which should be used to set and get data from eval-db
        :rtype: int
        """
        return self._eid
    
    @evaluation_id.setter
    def evaluation_id(self, val):
        if not isinstance(val, int):
            msg = 'evaluation-id has to be an integer!'
            raise ValueError(msg)
        eids = self.get_data_frame('SELECT eid FROM evaluations;')
        if not val in eids['eid'].values.tolist():
            msg = 'evaluation-id has to be present in the `evaluations`-table!'
            raise ValueError(msg)
        self._eid = val

    ############################################################################
    # EVALUATION AND ZONE STUFF
    ############################################################################
    def insert_eval_data(
            self, epoch_start:date, model, crs:int,
            epoch_end:date=None, eval_info:str=None
        ) -> bool:
        """
        Insert data into evaluations-table and set `self.evaluation_id` to the 
        newly created eval-id.
        
        Note: the edge-length [m] of the raster element is not needed because it 
        can be derived from the geo-coordinates (i.e. with known reference 
        point, the edge length can be computed).

        :param epoch_start: first epoch of evaluation
        :type epoch_start: date
        :param model: name of the top-level model (i.e. the zone-model)
        :type model: sitespecificcultivation.inference.models.zone.base.Zone
        :param crs: coordinate reference system (necessary for definition of geo-coordinates in the zone_geo_coordinates-table)
        :type crs: int
        :param epoch_end: epoch when to stop the evaluation, defaults to None
        :type epoch_end: date, optional
        :param eval_info: information about the evaluation, defaults to None
        :type eval_info: str, optional
        :return: flag if sql-execution has been successfull
        :rtype: bool
        """
        if epoch_end is None: 
            epoch_end = 'null'
        if eval_info is None:
            eval_info = 'null'
        
        if isclass(model):
            mmodule = model.__module__
            mname = model.__name__
        else:
            mmodule = model.__class__.__module__
            mname = model.__class__.__name__

        ret = self.execute_sql_command('SELECT * FROM evaluations;')
        new_eid = len(ret) + 1

        sqlc =  'INSERT INTO evaluations ('
        sqlc += 'eid, epoch_start, epoch_end, zmodel, zmodel_module, crs, '
        sqlc += 'eval_info) VALUES ('
        sqlc += '{}, \'{}\', \'{}\', \'{}\', \'{}\', {}, \'{}\');'
        sqlc = sqlc.format(
            new_eid, epoch_start, epoch_end, mname, mmodule, crs, eval_info
        )

        try:
            self.execute_sql_command(sqlc)
            self._eid = new_eid
            return True
        except Exception as exc:
            self._last_exc = str(exc)
            return False
        
    def get_eval_data(self, eid:int=None) -> pd.DataFrame:
        """
        Provides all data from evaluations-table belonging to the provided `eid`

        :param eid: evaluation id, defaults to None
        :type eid: int, optional
        :return: DataFrame with corresponding row of evaluations-table
        :rtype: pd.DataFrame
        """
        if eid is None:
            if self._eid is None:
                msg = '`eid` has to be provided if `self.evaluation_id` is not '
                msg += 'set yet!'
                raise ValueError(msg)
            eid = self._eid
        return self.get_data_frame(self.GET_EDATA.format(eid))
    
    def _check_eval_id(self, eid:int) -> int:
        if eid is None:
            if self._eid is None:
                msg = '`eid` has to be provided if `self.evaluation_id` is not '
                msg += 'set yet!'
                raise ValueError(msg)
            eid = self._eid
        return eid
    
    def check_zone_availability(self, eid:int, zone_name:str) -> bool:
        """
        Check if zone for provided `eid` and `zone_name` is alraedy available in 
        the database.

        :param eid: evaluation id
        :type eid: int
        :param zone_name: name of the zone
        :type zone_name: str
        :return: flag if available or not
        :rtype: bool
        """
        eid = self._check_eval_id(eid)
        # check if zone is already in the db for the provided zone name
        ret = self.execute_sql_command(
            'SELECT * FROM zones WHERE eid={} and zname=\'{}\';'.format(
                eid, zone_name
            )
        )
        if len(ret) == 0:
            return False
        else:
            return True
        
    def insert_zone(
            self, zone_name:str, latitude:float, height:float, gcs:np.ndarray, 
            eid:int=None
        ) -> bool:
        """
        Insert data into the zones- and the zone_geo_coordinates-table. In the 
        case of a new zone (checked with provided `zone_name` and evaluation-id 
        - either provided with `eid` or the currently active 
        `self.evaluation_id` will be used) a new entry will be created in the 
        `zones`-table with unique zone id (i.e. incremented integer) and the 
        coordinates are inserted in the `zone_geo_coordinates`-table.

        :param zone_name: name of zone
        :type zone_name: str
        :param latitude: mean latitude of the zone
        :type latitude: float
        :param height: mean height of the zone
        :type height: float
        :param gcs: geo-coordinates representing the zone
        :type gcs: (2, n) np.ndarray
        :param eid: evaluation id, defaults to None
        :type eid: int, optional
        :return: flag if sql-executions have been successfull
        :rtype: bool
        """
        if self.check_zone_availability(eid, zone_name):
            return True
        
        ret = self.execute_sql_command('SELECT * FROM zones;')
        new_zid = len(ret) + 1
        eid = self._check_eval_id(eid)

        sqlc =  'INSERT INTO zones (zid, eid, zname, latitude, height'
        sqlc += ') VALUES ({}, {}, \'{}\', {}, {});'
        sqlc = sqlc.format(new_zid, eid, zone_name, latitude, height)

        try:
            self.execute_sql_command(sqlc)
        except Exception as exc:
            self._last_exc = str(exc)
            return False

        sqlc = 'INSERT INTO zone_geo_coordinates (zid, x, y) VALUES '
        for coords in gcs.T:
            sqlc += '({}, {}, {}),'.format(new_zid, coords[0], coords[1])
        sqlc = sqlc[:-1] + ';'

        try:
            self.execute_sql_command(sqlc)
            return True
        except Exception as exc:
            self._last_exc = str(exc)
            return False
        
    def get_zones_data(self, eid:int=None) -> pd.DataFrame:
        """
        Provides the data of all zones (i.e. from zones-table) related to the 
        provided `eid`.

        :param eid: evaluation id, defaults to None
        :type eid: int, optional
        :return: DataFrame with corresponding rows of zones-table
        :rtype: pd.DataFrame
        """
        eid = self._check_eval_id(eid)
        return self.get_data_frame(self.GET_ZDATA.format(eid))
    
    def get_zone_id(self, zone_name:str, eid:int=None) -> int:
        """
        Get the zone id which corresponds to the provided evaluation id and 
        zone name.

        :param zone_name: name of the zone
        :type zone_name: str
        :param eid: evaluation id, defaults to None
        :type eid: int, optional
        :return: zone id
        :rtype: int
        """
        eid = self._check_eval_id(eid)
        ret =  self.execute_sql_command(
            'SELECT zid FROM zones WHERE eid={} AND zname=\'{}\';'.format(
                eid, zone_name
            )
        )
        return ret[0][0]
    
    def get_zone_gcs(self, zid:int) -> pd.DataFrame:
        """
        Provides the geo-coordinates of the specific zone from the 
        zone_geo_coordinates-table defined by the zone-id.

        :param zid: name of the zone
        :type zid: str
        :return: DataFrame with rows containing the geo-coordinates
        :rtype: pd.DataFrame
        """
        return self.get_data_frame(self.GET_ZGCS.format(zid))
    
    ############################################################################
    # QUANTITY DEFINITIONS AND CROP ROTATION
    ############################################################################
    def add_crop_rotation(
            self, zid:int, crop_model, epoch_start:date, epoch_end:date
        ) -> None:
        """
        Adds a tuple with a crop-definition to the sql-string which is used in 
        the `insert_crop_rotation()`-method.

        :param zid: id of the zone
        :type zid: str
        :param crop_model: model representing the crop
        :type crop_model: object or class
        :param epoch_start: sowing/planting date
        :type epoch_start: date
        :param epoch_end: harvesting/mulching date
        :type epoch_end: date
        """
        if isclass(crop_model):
            mmodule = crop_model.__module__
            mname = crop_model.__name__
        else:
            mmodule = crop_model.__class__.__module__
            mname = crop_model.__class__.__name__

        self._icropr += \
            '({}, \'{}\', \'{}\', \'{}\', \'{}\'),'.format(
                zid, mname, mmodule, epoch_start.isoformat(), 
                epoch_end.isoformat()
            )
    
    def add_observation_def(
            self, zid:int, oname:str, omodel:str, epoch:date, value:float, 
            distr:dict
        ) -> None:
        """
        Adds a tuple with an observation definition to the sql-string which is 
        used in the `insert_obs_def()`-method.

        :param zid: zone id
        :type zid: int
        :param oname: name of the observation within the `model`
        :type oname: str
        :param model: `Model.model_id` of the model which the observation belongs to
        :type model: str
        :param epoch: epoch which the observation belongs to
        :type epoch: date
        :param value: value of the observation
        :type value: float
        :param distr: dictionary containing information about distribution
        :type distr: dict
        """
        self._idobs += \
            '({}, \'{}\', \'{}\', \'{}\', {}, \'{}\'),'.format(
            zid, oname, omodel, epoch.isoformat(), value, json.dumps(distr)
        )

    def add_states_def(
            self, zid:int, sname:str, smodel:str, epoch:date, value:float, 
            distr:dict
        ) -> None:
        """
        Adds a tuple with a state definition to the sql-string which is 
        used in the `insert_states_def()`-method.

        :param zid: zone id
        :type zid: int
        :param sname: name of the state within the `model`
        :type sname: str
        :param smodel: `Model.model_id` of the model which the state belongs to
        :type smodel: str
        :param epoch: epoch which the state belongs to
        :type epoch: date
        :param value: value of the state
        :type value: float
        :param distr: dictionary containing information about distribution
        :type distr: dict
        """
        self._idsts += \
            '({}, \'{}\', \'{}\', \'{}\', {}, \'{}\'),'.format(
            zid, sname, smodel, epoch.isoformat(), value, json.dumps(distr)
        )

    def add_hparam_def(
            self, zid:int, hname:str, hmodel:str, epoch:date, value:float, 
            distr:dict
        ) -> None:
        """
        Adds a tuple with an hyper-parameter definition to the sql-string which 
        is used in the `insert_hparams_def()`-method.

        :param zid: zone id
        :type zid: int
        :param hname: name of the hyper-parameter within the `model`
        :type hname: str
        :param hmodel: `Model.model_id` of the model which the hyper-parameter belongs to
        :type hmodel: str
        :param epoch: epoch which the hyper-parameter belongs to
        :type epoch: date
        :param value: value of the hyper-parameter
        :type value: float
        :param distr: dictionary containing information about distribution
        :type distr: dict
        """
        self._idhps += \
            '({}, \'{}\', \'{}\', \'{}\', {}, \'{}\'),'.format(
            zid, hname, hmodel, epoch.isoformat(), value, json.dumps(distr)
        )

    def add_hpfunc_def(
            self, zid:int, fname:str, fmodel:str, epoch:date, fdef:dict
        ) -> None:
        """
        Adds a tuple with an hp-function definition to the sql-string which 
        is used in the `insert_hpfuncs_def()`-method.

        :param zid: zone id
        :type zid: int
        :param fname: name of the hp-function within the `model`
        :type fname: str
        :param fmodel: `Model.model_id` of the model which the hp-function belongs to
        :type fmodel: str
        :param epoch: epoch which the hp-function belongs to
        :type epoch: date
        :param fdef: dictionary containing information about the hp-function
        :type fdef: dict
        """
        self._idhpf += \
            '({}, \'{}\', \'{}\', \'{}\', \'{}\'),'.format(
            zid, fname, fmodel, epoch.isoformat(), json.dumps(fdef)
        )
        
    def insert_hpfuncs_def(self) -> bool:
        try:
            self.execute_sql_command(self._idhpf[:-1] + ';')
            self._reset_insert_dhpfuncs()
            return True
        except Exception as exc:
            self._last_exc = str(exc)
            return False
        
    def insert_crop_rotation(self) -> bool:
        try:
            self.execute_sql_command(self._icropr[:-1] + ';')
            self._reset_insert_cropr()
            return True
        except Exception as exc:
            self._last_exc = str(exc)
            return False

    def insert_obs_def(self) -> bool:
        """
        Insert the added observation definitions with `add_obs_def()` into the 
        DB and reset the corresponding sql-string.

        :return: flag if sql-execution has been successfull
        :rtype: bool
        """
        try:
            self.execute_sql_command(self._idobs[:-1] + ';')
            self._reset_insert_dobs()
            return True
        except Exception as exc:
            self._last_exc = str(exc)
            return False

    def insert_states_def(self) -> bool:
        """
        Insert added state definitions with `add_states_def()` into the DB and 
        reset the corresponding sql-string.

        :return: flag if sql-execution has been successfull
        :rtype: bool
        """
        try:
            self.execute_sql_command(self._idsts[:-1] + ';')
            self._reset_insert_dstates()
            return True
        except Exception as exc:
            self._last_exc = str(exc)
            return False
    
    def insert_hparams_def(self) -> bool:
        """
        Insert added hyper-parameter definitions with `add_hparams_def()` into 
        the DB and reset the corresponding sql-string.

        :return: flag if sql-execution has been successfull
        :rtype: bool
        """
        try:
            self.execute_sql_command(self._idhps[:-1] + ';')
            self._reset_insert_dhparams()
            return True
        except Exception as exc:
            self._last_exc = str(exc)
            return False
        
    def get_crop_rotation(self, zid:int) -> pd.DataFrame:
        """
        Get crop rotation data for specified zone.

        :param zid: id of the zone
        :type zid: int
        :return: crop rotation information
        :rtype: pd.DataFrame
        """
        return self.get_data_frame(self.GET_CROPROT.format(zid))
        
    def get_states_def(self, zid:int, epoch:date) -> pd.DataFrame:
        """
        Get state definitions for the provided zone-id and 
        epoch as DataFrame (all columns of the states_def-table).

        :param zid: id of the zone
        :type zid: str
        :param epoch: epoch of state definition
        :type epoch: date
        :return: state definitions
        :rtype: pd.DataFrame
        """
        return self.get_data_frame(
            self.GET_STATES_DEF.format(zid, epoch.isoformat())
        )
    
    def get_hparams_def(self, zid:int, epoch:date) -> pd.DataFrame:
        """
        Get hyper-parameter definitions for the provided zone-id, 
        and epoch as DataFrame (all columns of the hparams_def-table).

        :param zid: id of the zone
        :type zid: int
        :param epoch: epoch of hyper-parameter definition
        :type epoch: date
        :return: hyper-parameter definitions
        :rtype: pd.DataFrame
        """
        return self.get_data_frame(
            self.GET_HPARAMS_DEF.format(zid, epoch.isoformat())
        )
    
    def get_hpfuncs_def(self, zid:int, epoch:date) -> pd.DataFrame:
        """
        Get hp-function definitions for the provided zone-id, 
        and epoch as DataFrame (all columns of the hparams_def-table).

        :param zid: id of the zone
        :type zid: int
        :param epoch: epoch of hp-function definition
        :type epoch: date
        :return: hp-function definitions
        :rtype: pd.DataFrame
        """
        return self.get_data_frame(
            self.GET_HPFUNCS_DEF.format(zid, epoch.isoformat())
        )
    
    def get_obs_def(self, zid:int, epoch:date) -> pd.DataFrame:
        """
        Get observation definitions for the provided zone-id and
        epoch as DataFrame (all columns of the obs_def-table).

        :param zid: id of the zone
        :type zid: str
        :param epoch: epoch of the observation definition
        :type epoch: date
        :return: observation definitions
        :rtype: pd.DataFrame
        """
        return self.get_data_frame(
            self.GET_OBS_DEF.format(zid, epoch.isoformat())
        )
    
    def get_quantity_def(
            self, qname:str, qtype:str, qmodel:str, zid:int=None
        ) -> pd.DataFrame:
        cmd = 'SELECT value, epoch FROM {} WHERE name=\'{}\' AND model=\'{}\''
        cmd = cmd.format(self.DEF_TABLE_NAMES[qtype], qname, qmodel)
        if zid is not None:
            cmd += ' AND zid={}'.format(zid)
        cmd += ';'
        return self.get_data_frame(cmd)
    
    ############################################################################
    # EVALUATED QUANTITIES
    ############################################################################    
    def add_states_eval(
            self, zid:int, epoch:date, sname:str, smodel:str, value:np.ndarray, 
            discrete:bool=False
        ) -> None:
        """
        Add evaluated state for specific epoch to the corresponding 
        INSERT-command.

        :param zid: id of the zone
        :type zid: int
        :param epoch: evaluation epoch
        :type epoch: date
        :param sname: name of the state within `model`
        :type sname: str
        :param smodel: `Model.model_id` of the model which contains the `sname`
        :type smodel: str
        :param value: evaluated value of the state
        :type value: float or numpy.ndarray
        """
        self._iests = self._add_eval(
            self._iests, zid, epoch, sname, smodel, value, discrete=discrete
        )
        
    def add_hparams_eval(
            self, zid:int, epoch:date, hname:str, hmodel:str, value:np.ndarray, 
            discrete:bool=False
        ) -> None:
        """
        Add evaluated hyper-parameter for specific epoch to the corresponding 
        INSERT-command

        :param zid: id of the zone
        :type zid: str
        :param epoch: evaluation epoch
        :type epoch: date
        :param hname: name of the hyper-parameter within `model`
        :type hname: str
        :param hmodel: `Model.model_id` of the model which contains the `hname`
        :type hmodel: str
        :param value: evaluated value of the hyper-parameter
        :type value: numpy.ndarray
        """
        self._iehps = self._add_eval(
            self._iehps, zid, epoch, hname, hmodel, value, discrete=discrete
        )

    def add_hpfuncs_eval(
            self, zid:int, epoch:date, fname:str, fmodel:str, value:np.ndarray, 
            discrete:bool=False
        ) -> None:
        """
        Add evaluated hp-function for specific epoch to the corresponding 
        INSERT-command

        :param zid: id of the zone
        :type zid: int
        :param epoch: evaluation epoch
        :type epoch: date
        :param fname: name of the hyper-parameter within `model`
        :type fname: str
        :param hmodel: `Model.model_id` of the model which contains the `fname`
        :type hmodel: str
        :param value: evaluated value of the hp-function
        :type value: numpy.ndarray
        """
        self._iehpf = self._add_eval(
            self._iehpf, zid, epoch, fname, fmodel, value, discrete=discrete
        )
        
    def add_obs_eval(
            self, zid:int, epoch:date, oname:str, omodel:str, value:np.ndarray, 
            discrete:bool=False
        ) -> None:
        """
        Add evaluated observation for specific epoch to the corresponding 
        INSERT-command.

        :param zid: id of the zone
        :type zid: int
        :param epoch: evaluation epoch
        :type epoch: date
        :param oname: name of the observation within `model`
        :type oname: str
        :param omodel: `Model.model_id` of the model which contains the `oname`
        :type omodel: str
        :param value: evaluated value of the observation
        :type value: numpy.ndarray
        """
        self._ieobs = self._add_eval(
            self._ieobs, zid, epoch, oname, omodel, value, discrete=discrete
        )
        
    def add_out_eval(
            self, zid:int, epoch:date, oname:str, omodel:str, value:np.ndarray, 
            discrete:bool=False
        ) -> None:
        """
        Add evaluated model output quantities for specific epoch to the 
        corresponding INSERT-command.
        `value` can be numeric or boolean (converted to 0 or 1) - strings and 
        `None` will be neglected.

        :param zid: id of the zone
        :type zid: str
        :param epoch: evaluation epoch
        :type epoch: date
        :param oname: name of the output within `model`
        :type oname: str
        :param omodel: `Model.model_id` of the model which contains the `oname`
        :type omodel: str
        :param value: evaluated value of the output
        :type value: numpy.ndarray
        """
        self._ieout = self._add_eval(
            self._ieout, zid, epoch, oname, omodel, value, discrete=discrete
        )
    
    def _add_eval(
        self, icmd:str, zid:int, epoch:date, qname:str, qmodel:str, 
        value:np.ndarray, discrete:bool=False
    ) -> str:
        pass
        
    def insert_states_eval_cmd(self) -> str:
        """
        :return: sql-command to insert evaluated states
        :rtype: str
        """
        ret = self._iests[:-1] + ';'
        self._reset_insert_estates()
        return ret
        
    def insert_hparams_eval_cmd(self) -> str:
        """
        :return: sql-command to insert evaluated hyper parameters (i.e. sampled)
        :rtype: str
        """
        ret = self._iehps[:-1] + ';'
        self._reset_insert_ehparams()
        return ret

    def insert_obs_eval_cmd(self) -> str:
        """
        :return: sql-command to insert evaluated observations (i.e. sampled)
        :rtype: str
        """
        ret = self._ieobs[:-1] + ';'
        self._reset_insert_eobs()
        return ret
    
    def insert_out_eval_cmd(self) -> str:
        """
        :return: sql-command to insert evaluated outputs
        :rtype: str
        """
        ret = self._ieout[:-1] + ';'
        self._reset_insert_eout()
        return ret
    
    def insert_hpfuncs_eval_cmd(self) -> str:
        ret = self._iehpf[:-1] + ';'
        self._reset_insert_ehpfuncs()
        return ret
    
    def get_quantity_eval(
            self, qname:str, qtype:str, qmodel:str, zid:int=None
        ) -> pd.DataFrame:
        cmd = 'SELECT value, epoch FROM {} WHERE name=\'{}\' AND model=\'{}\''
        cmd = cmd.format(self.EVAL_TABLE_NAMES[qtype], qname, qmodel)
        if zid is not None:
            cmd += ' AND zid={}'.format(zid)
        cmd += ';'
        return self.get_data_frame(cmd)
    
    ############################################################################
    # SQL SCRIPT STUFF
    ############################################################################    
    def create_sql_script(self, directory:str, filename:str) -> None:
        """
        Opens/Creates a file handler (`self._script_io`) into which sql-commands 
        can be written with the `self.add_cmd_to_script()`-method.

        :param directory: absolute path, where sql-script should be saved to
        :type directory: str
        :param filename: filename of the sql-script
        :type filename: str
        """
        self._scriptp = os.path.join(directory, filename)
        self._script_io = open(self._scriptp, 'w')

    def add_cmd_to_script(self, cmd:str) -> None:
        """
        Append a sql-command to file handler (`self._script_io` created with
        `self.create_sql_script()`-method).

        :param cmd: sql-command
        :type cmd: str
        """
        self._script_io.write(cmd + '\n')

    def close_script(self) -> None:
        """
        Close the file handler representing a sql-script (`self._script_io` 
        created with `self.create_sql_script()`-method).
        """
        self._script_io.close()

    def execute_script(self) -> None:
        """
        Execute a sql-script which only contains sql-commands per line (empty 
        lines are also allowed). sql-script can be any text-based format (e.g. 
        .txt or .sql)

        :param directory: absolute path to the sql-script
        :type directory: str
        :param filename: filename of the sql-script
        :type filename: str
        """
        fio = open(self._scriptp, 'r')
        cmds = fio.read()
        fio.close()
        self._scriptp = None

        for cmd in cmds.split('\n'):
            if not cmd:
                continue
            self.execute_sql_command(cmd)
        self.connection.commit()

    ############################################################################
    # EXECUTABLES
    ############################################################################
    def execute_sql_command(self, cmd:str) -> list:
        """
        Execute any sql-command with `self._curs.execute(sql_command)`

        :param cmd: sql command
        :type cmd: str
        :return: list containing the requested tuples
        :rtype: list
        """
        try:
            self._curs.execute(cmd)
            return self._curs.fetchall()
        except Exception as exc:
            self._last_exc = str(exc)
            return None
    
    def get_data_frame(self, cmd:str) -> pd.DataFrame:
        """
        Converts the list of tuples (result of database-request with
        `self._curs.execute(sql_command)`) to a pandas.DataFrame

        :param cmd: sql command
        :type cmd: str
        :return: DataFrame with result from sql-query
        :rtype: pd.DataFrame
        """
        res = self.execute_sql_command(cmd)
        if res is None:
            return pd.DataFrame()
        else:
            descr = self._curs.description
            return pd.DataFrame(res, columns=[col[0] for col in descr])
    
    ############################################################################
    # INTERNAL STUFF
    ############################################################################
    def _reset_insert_dobs(self) -> None:
        self._idobs:str = 'INSERT INTO obs_def '
        self._idobs += '(zid, name, model, epoch, value, distr) VALUES '

    def _reset_insert_dstates(self) -> None:
        self._idsts:str = 'INSERT INTO states_def '
        self._idsts += '(zid, name, model, epoch, value, distr) VALUES'

    def _reset_insert_dhparams(self) -> None:
        self._idhps:str = 'INSERT INTO hparams_def '
        self._idhps += '(zid, name, model, epoch, value, distr) VALUES'

    def _reset_insert_dhpfuncs(self) -> None:
        self._idhpf:str = 'INSERT INTO hpfuncs_def '
        self._idhpf += '(zid, name, model, epoch, fdef) VALUES '

    def _reset_insert_ehpfuncs(self) -> None:
        self._iehpf:str = 'INSERT INTO hpfuncs_eval '
        self._iehpf += '(zid, name, model, epoch, value) VALUES '

    def _reset_insert_eobs(self) -> None:
        self._ieobs:str = 'INSERT INTO obs_eval '
        self._ieobs += '(zid, name, model, epoch, value) VALUES '

    def _reset_insert_estates(self) -> None:
        self._iests:str = 'INSERT INTO states_eval '
        self._iests += '(zid, name, model, epoch, value) VALUES '

    def _reset_insert_ehparams(self) -> None:
        self._iehps:str = 'INSERT INTO hparams_eval '
        self._iehps += '(zid, name, model, epoch, value) VALUES '

    def _reset_insert_eout(self) -> None:
        self._ieout:str = 'INSERT INTO out_eval '
        self._ieout += '(zid, name, model, epoch, value) VALUES '

    def _reset_insert_cropr(self) -> None:
        self._icropr:str = 'INSERT INTO crop_rotation ('
        self._icropr += 'zid, cmodel, cmodel_module, '
        self._icropr += 'epoch_start, epoch_end) VALUES '

    ############################################################################
    # CLASS METHODS
    ############################################################################
    @classmethod
    def from_eval_def(cls, dbdir:str, dbname:str, edefs):
        def loop_quantities(
                zid:int, edb:EvaluationDB, qtype:str, defs:dict
            ) -> EvaluationDB:
            for qmodel, qinfos in defs.items():
                for qname, qinfo in qinfos.items():
                    if qtype == Quantities.HPFUNC:
                        epoch = date.fromisoformat(qinfo['epoch'])
                        edb.add_hpfunc_def(
                            zid, qname, qmodel, epoch, qinfo['fdef']
                        )
                    elif qtype == Quantities.STATE:
                        epoch = date.fromisoformat(qinfo['epoch'])
                        edb.add_states_def(
                            zid, qname, qmodel, epoch, qinfo['value'], 
                            qinfo['distr']
                        )
                    elif qtype == Quantities.HPARAM:
                        epoch = date.fromisoformat(qinfo['epoch'])
                        edb.add_hparam_def(
                            zid, qname, qmodel, epoch, qinfo['value'], 
                            qinfo['distr']
                        )
                    elif qtype == Quantities.OBS:
                        for ep, obs in zip(qinfo['epochs'], qinfo['values']):
                            epoch = date.fromisoformat(ep)
                            edb.add_observation_def(
                                zid, qname, qmodel, epoch, obs, qinfo['distr']
                            )
            return edb

        from ..models.base import Quantities
        qtypes = [
            Quantities.STATE, Quantities.OBS, Quantities.HPARAM,
            Quantities.HPFUNC
        ]

        edb = cls(dbdir, dbname)
        zmodel = getattr(import_module(edefs['zmodel_module']), edefs['zmodel'])
        edb.insert_eval_data(
            edefs['epoch_start'], zmodel, edefs['add_info']['crs'], 
            epoch_end=edefs['epoch_end'], eval_info=edefs['eval_info']
        )

        for zone_name, qinfos in edefs[edefs['zmodel']].items():
            lat = edefs['add_info']['zones'][zone_name]['latitude']
            gcs = np.array(edefs['add_info']['zones'][zone_name]['gcs'])
            edb.insert_zone(
                zone_name, lat, edefs['add_info']['height'], gcs
            )

            zid = edb.get_zone_id(zone_name)
            for qtype in qtypes:
                edb = loop_quantities(zid, edb, qtype, qinfos[qtype])

            for crinfo in edefs['crop_rotation']:
                cmodel = getattr(
                    import_module(crinfo['cmodel_module']), crinfo['cmodel']
                )
                estart = date.fromisoformat(crinfo['epoch_start'])
                estop = date.fromisoformat(crinfo['epoch_end'])
                edb.add_crop_rotation(
                    zid, cmodel, estart, estop
                )
                for qtype in qtypes:
                    edb = loop_quantities(
                        zid, edb, qtype, edefs[crinfo['cmodel']][qtype]
                    )

        edb.insert_crop_rotation()
        edb.insert_states_def()
        edb.insert_hparams_def()
        edb.insert_obs_def()
        edb.insert_hpfuncs_def()
        edb.connection.commit()
        return edb
    

class EvalDB_AllParticles(EvaluationDB):
    CREATE_EOBS =  'CREATE TABLE obs_eval ('
    CREATE_EOBS += 'zid INT, name TEXT, model TEXT, epoch TEXT, value REAL, '
    CREATE_EOBS += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_EOBS += ');'

    CREATE_ESTATES =  'CREATE TABLE states_eval ('
    CREATE_ESTATES += 'zid INT, name TEXT, model TEXT, epoch TEXT, value REAL,'
    CREATE_ESTATES += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_ESTATES += ');'

    CREATE_EHPARAMS =  'CREATE TABLE hparams_eval ('
    CREATE_EHPARAMS += 'zid INT, name TEXT, model TEXT,epoch TEXT, value REAL,'
    CREATE_EHPARAMS += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_EHPARAMS += ');'

    CREATE_EHPFUNCS =  'CREATE TABLE hpfuncs_eval ('
    CREATE_EHPFUNCS += 'zid INT, name TEXT, model TEXT, epoch TEXT,value REAL,'
    CREATE_EHPFUNCS += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_EHPFUNCS += ');'

    CREATE_EOUT =  'CREATE TABLE out_eval ('
    CREATE_EOUT += 'zid INT, name TEXT, model TEXT, epoch TEXT, value REAL, '
    CREATE_EOUT += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_EOUT += ');'

    def _add_eval(self, icmd, zid, epoch, qname, qmodel, value, discrete=False):
        addstr = '({}, \'{}\', \'{}\', \'{}\', {}),'
        if value is None:
            return icmd
        
        if isinstance(value, float):
            icmd += addstr.format(
                zid, qname, qmodel, epoch.isoformat(), value
            )
        elif isinstance(value, np.ndarray):
            if not isinstance(value[0], str):   # there are also arrays containing strings > should not be written to db
                if value.dtype == 'bool':
                    value = value.astype('float')
                add = [addstr.format(
                    zid, qname, qmodel, epoch.isoformat(), val
                ) for val in value]
                icmd += ''.join(add)
        return icmd


class EvalDB_Quantiles(EvaluationDB):
    CREATE_EOBS =  'CREATE TABLE obs_eval ('
    CREATE_EOBS += 'zid INT, name TEXT, model TEXT, epoch TEXT, value TEXT, '
    CREATE_EOBS += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_EOBS += ');'

    CREATE_ESTATES =  'CREATE TABLE states_eval ('
    CREATE_ESTATES += 'zid INT, name TEXT, model TEXT, epoch TEXT, value TEXT,'
    CREATE_ESTATES += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_ESTATES += ');'

    CREATE_EHPARAMS =  'CREATE TABLE hparams_eval ('
    CREATE_EHPARAMS += 'zid INT, name TEXT, model TEXT,epoch TEXT, value TEXT,'
    CREATE_EHPARAMS += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_EHPARAMS += ');'

    CREATE_EHPFUNCS =  'CREATE TABLE hpfuncs_eval ('
    CREATE_EHPFUNCS += 'zid INT, name TEXT, model TEXT, epoch TEXT,value TEXT,'
    CREATE_EHPFUNCS += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_EHPFUNCS += ');'

    CREATE_EOUT =  'CREATE TABLE out_eval ('
    CREATE_EOUT += 'zid INT, name TEXT, model TEXT, epoch TEXT, value TEXT, '
    CREATE_EOUT += 'FOREIGN KEY (zid) REFERENCES zones(zid)'
    CREATE_EOUT += ');'

    def __init__(self, directory, dbname):
        super().__init__(directory, dbname)
        self.n_quantiles = 20

    def _add_eval(self, icmd, zid, epoch, qname, qmodel, value, discrete=False):
        if not isinstance(value, np.ndarray) or isinstance(value[0], str):
            # quantities may be set to None in the models or may be also arrays 
            # containing strings > no write to database in these cases
            return icmd
        if self.n_quantiles > 100:
            msg = 'More than 100 quantiles for the distribution approximation '
            msg += 'not supported!'
            raise ValueError(msg)
        if value.shape[0] < self.n_quantiles + 1:
            msg = 'Value which should be saved for ' + qname
            msg += ' has to be a numpy.ndarray length being at least {}!'.format(
                self.n_quantiles
            )
            raise ValueError(msg)

        if discrete:
            # discrete distribution > count values
            uvls, cnts = np.unique(value, return_counts=True)
            vals = ''.join([
                str(uv) + '>' + str(cn) + ',' for uv, cn in zip(uvls, cnts)
            ])
            vstr = 'discrete:' + vals[:-1]
        else:
            # will be treated as continuous distribution > compute quantiles
            dq = round(1. / self.n_quantiles, 2)
            iqrs = np.quantile(value, np.arange(0.0, 1.0 + dq, dq))
            vals = ''.join([str(val) + ',' for val in iqrs])
            vstr = 'continuous:' + vals[:-1]

        addstr = '({}, \'{}\', \'{}\', \'{}\', \'{}\'),'
        icmd += addstr.format(zid, qname, qmodel, epoch.isoformat(), vstr)
        return icmd
