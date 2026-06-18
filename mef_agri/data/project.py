import os
import shutil
from geopandas import GeoDataFrame
from copy import deepcopy
from inspect import isclass
from importlib import import_module
from datetime import date, timedelta
from multiprocessing import cpu_count

from ..utils.gpkg import Geopackage, SQLTable, ForeignKey, GeometryType
from .utils import (
    daterange_consider_existing_dates, merge_dateranges, timerange_from_epochs
)
from .interface import Interface


################################################################################
# SCHEMA OF PROJECT DATABASE
################################################################################
class DB:
    class TBL_DSRCS:
        NAME = 'data_sources'
        COL_DID = 'did'
        COL_INAME = 'intf_name'
        COL_IMODULE = 'intf_module'
        Q_INTFS = f'SELECT * FROM {NAME};'
    class TBL_FIELDS:
        NAME = 'fields'
        COL_FIELDNAME = 'field_name'
        COL_GEOM = SQLTable.GEOM_COL_NAME
        COL_FID = SQLTable.ID_COL_NAME
    class TBL_DAVLBL:
        NAME = 'data_available'
        COL_DID = 'did'
        COL_FIELD = 'field'
        COL_TSTART = 'tstart'
        COL_TSTOP = 'tstop'
        Q_DATERANGES = f'SELECT {COL_TSTART}, {COL_TSTOP} FROM {NAME} WHERE '
        Q_DATERANGES += f'{COL_DID}=' + '\'{did}\' AND ' + f'{COL_FIELD}='
        Q_DATERANGES += '\'{fld}\';'
        INSERT_START = f'INSERT INTO {NAME} '
        INSERT_START += f'({COL_DID}, {COL_FIELD}, {COL_TSTART}, {COL_TSTOP}) '
        INSERT_START += 'VALUES '
        INSERT_TUPLE = '(\'{did}\', \'{fld}\', \'{tstart}\', \'{tstop}\')'
        DEL_BY_DID_FLD = f'DELETE FROM {NAME} WHERE {COL_DID}=' + '\'{did}\' '
        DEL_BY_DID_FLD += f'AND {COL_FIELD}=' + '\'{fld}\';'
    class TBL_DEPOCHS:
        NAME = 'data_epochs'
        COL_DID = 'did'
        COL_FIELD = 'field'
        COL_EPOCH = 'epoch'
        INSERT_START = f'INSERT INTO {NAME} '
        INSERT_START += f'({COL_DID}, {COL_FIELD}, {COL_EPOCH}) VALUES '
        INSERT_TUPLE = '(\'{did}\', \'{fld}\', \'{epoch}\')'
        Q_EPOCHS = f'SELECT {COL_EPOCH} FROM {NAME} WHERE '
        Q_EPOCHS += f'{COL_DID}=' + '\'{did}\' AND '
        Q_EPOCHS += f'{COL_FIELD}=' + '\'{fld}\' AND '
        Q_EPOCHS += f'date({COL_EPOCH}) >= ' + 'date(\'{tstart}\') AND '
        Q_EPOCHS += f'date({COL_EPOCH}) <= ' + 'date(\'{tstop}\');'


_tds = SQLTable()
_tds.name = DB.TBL_DSRCS.NAME
_tds.columns = {
    DB.TBL_DSRCS.COL_DID: 'TEXT',
    DB.TBL_DSRCS.COL_INAME: 'TEXT',
    DB.TBL_DSRCS.COL_IMODULE: 'TEXT'
}
_tds.primary_key = [DB.TBL_DSRCS.COL_DID]

_tfld = SQLTable()
_tfld.name = DB.TBL_FIELDS.NAME
_tfld.columns = {
    DB.TBL_FIELDS.COL_FIELDNAME: 'TEXT'
}
_tfld.geom = GeometryType.POLYGON
_tfld.unique_cols = [DB.TBL_FIELDS.COL_FIELDNAME]

_tda = SQLTable()
_tda.name = DB.TBL_DAVLBL.NAME
_tda.columns = {
    DB.TBL_DAVLBL.COL_DID: 'TEXT',
    DB.TBL_DAVLBL.COL_FIELD: 'TEXT',
    DB.TBL_DAVLBL.COL_TSTART: 'CHAR(12)',
    DB.TBL_DAVLBL.COL_TSTOP: 'CHAR(12)'
}
_tda.primary_key = [
    DB.TBL_DAVLBL.COL_DID, DB.TBL_DAVLBL.COL_FIELD, DB.TBL_DAVLBL.COL_TSTART
]
_tda_tds_fk = ForeignKey()
_tda_tds_fk.fkey_columns = [DB.TBL_DAVLBL.COL_DID]
_tda_tds_fk.ftable = _tds.name
_tda_tds_fk.ftable_columns = _tds.primary_key
_tda_tds_fk.on_delete = 'CASCADE'
_tda_tds_fk.on_update = 'CASCADE'
_tda_fld_fk = ForeignKey()
_tda_fld_fk.fkey_columns = [DB.TBL_DAVLBL.COL_FIELD]
_tda_fld_fk.ftable = _tfld.name
_tda_fld_fk.ftable_columns = _tfld.unique_cols
_tda_fld_fk.on_delete = 'CASCADE'
_tda_fld_fk.on_update = 'CASCADE'
_tda.foreign_keys = [_tda_tds_fk, _tda_fld_fk]

_tde = SQLTable()
_tde.name = DB.TBL_DEPOCHS.NAME
_tde.columns = {
    DB.TBL_DEPOCHS.COL_DID: 'TEXT',
    DB.TBL_DEPOCHS.COL_FIELD: 'TEXT',
    DB.TBL_DEPOCHS.COL_EPOCH: 'CHAR(12)'
}
_tde.primary_key = list(_tde.columns.keys())
_tde_tds_fk = ForeignKey()
_tde_tds_fk.fkey_columns = [DB.TBL_DEPOCHS.COL_DID]
_tde_tds_fk.ftable = _tds.name
_tde_tds_fk.ftable_columns = _tds.primary_key
_tde_tds_fk.on_delete = 'CASCADE'
_tde_tds_fk.on_update = 'CASCADE'
_tde_fld_fk = ForeignKey()
_tde_fld_fk.fkey_columns = [DB.TBL_DEPOCHS.COL_FIELD]
_tde_fld_fk.ftable = _tfld.name
_tde_fld_fk.ftable_columns = _tfld.unique_cols
_tde_fld_fk.on_delete = 'CASCADE'
_tde_fld_fk.on_update = 'CASCADE'
_tde.foreign_keys = [_tde_tds_fk, _tde_fld_fk]

################################################################################
# PROJECT CLASS
################################################################################
class ProjectData(Geopackage):
    """
    This class represents the data (together with its administration and 
    structure) which is necessary to evaluate crop and soil models on 
    available fields (see :func:`fields`).
    Data interfaces :class:`mef_agri.data.interface.Interface`, have to be 
    registered with :func:`add_data_interface`, such that they can be used 
    in :func:`add_data` and :func:`get_data`.
    """
    DATA_DIRECTORY = 'data'

    def __init__(self, project_dir, gpkg_name):
        ddir = os.path.join(project_dir, self.DATA_DIRECTORY)
        if not os.path.exists(ddir):
            os.mkdir(ddir)

        super().__init__(project_dir, gpkg_name)
        self._dis:dict = None  # dictionary containing data-interface instances
        self._pdir:str = project_dir  # project directory
        self._flds:GeoDataFrame = None
        self._nprcs:int = max(cpu_count() // 2, 2)  # number of cores to use for tasks
        self.tables = {
            _tds.name: _tds,
            _tfld.name: _tfld,
            _tda.name: _tda,
            _tde.name: _tde
        }

    @property
    def directory(self) -> str:
        """
        :return: project directory where the .gpkg file is located (absolute path)
        :rtype: str
        """
        return self._pdir

    @property
    def interfaces(self) -> dict:
        """
        :return: data interface instances (see :class:`mef_agri.data.interface.Interface`)
        :rtype: dict
        """
        if self._dis is None:
            self._dis = {}
            dis = self.query(DB.TBL_DSRCS.Q_INTFS)
            for tpl in dis.itertuples():
                di = getattr(import_module(tpl.intf_module), tpl.intf_name)()
                if self.n_processes is not None:
                    di.n_processes = self.n_processes
                di.project_directory = self.directory
                di.project_ref = self
                self._dis[tpl.did] = di
        return self._dis
    
    @property
    def fields(self) -> GeoDataFrame:
        """
        :return: fields available in .gpkg file of the project
        :rtype: geopandas.GeoDataFrame
        """
        if self._flds is None:
            self._flds = self.query_gdf(DB.TBL_FIELDS.NAME)
        return self._flds
    
    @property
    def n_processes(self) -> int:
        """
        Settable

        Default value: half the number of ``multiprocessing.cpu_count()`` but at least 2

        :return: number of processes which should be used to execute functions decorated with ``Interface.add_data_task``
        :rtype: int
        """
        return self._nprcs

    @n_processes.setter
    def n_processes(self, value):
        self._nprcs = value
        self._pids = None

    def add_data_interface(self, data_intf:Interface) -> None:
        """
        Method to add a :class:`mef_agri.data.interface.Interface` to the 
        project.

        :param data_intf: the desired :class:`mef_agri.data.interface.Interface`-subclass
        :type data_intf: class or instance
        """
        # intialize data interface if necessary and set project and save paths
        if isclass(data_intf):
            data_intf = data_intf()

        # check if data interface already has been added
        if data_intf.data_source_id in list(self.interfaces.keys()):
            return
        
        # create folder within the data folder for the 
        dpath = os.path.join(
            self._pdir, self.DATA_DIRECTORY, data_intf.data_source_id
        )
        if not os.path.exists(dpath):
            os.mkdir(dpath)

        # add data interface to the dict containing all interfaces of the prj
        self._dis[data_intf.data_source_id] = data_intf

        # insert data interface information to the project database
        self.execute(self.tables[DB.TBL_DSRCS.NAME].sql_insert(
            did=data_intf.data_source_id,
            intf_name=data_intf.__class__.__name__,
            intf_module=data_intf.__class__.__module__
        ))

    def add_data(self, tstart:date, tstop:date, dids=None, fields=None) -> None:
        """
        Method which requests/downloads data from corresponding data-sources and 
        saves them locally as GeoTIFF files in the project folder structure. 

        If ``dids`` is not provided, all data-sources will be considered.

        If ``fields`` is not provided, all fields will be considered.

        :param tstart: start epoch of the requested data
        :type tstart: datetime.date
        :param tstop: last epoch of the requested data
        :type tstop: datetime.date
        :param dids: data-source-ids available in the corresponding data-interface, defaults to None
        :type dids: list[str] or str, optional
        :param fields: name of the fields, defaults to None
        :type fields: list[str] or str, optional
        """
        # prepare dids and fields
        dids = self._prepare_list(dids, list(self.interfaces.keys()))
        fields = self._prepare_list(
            fields, self.fields[DB.TBL_FIELDS.COL_FIELDNAME].values.tolist()
        )
        
        # get data from interfaces
        for did in dids:
            di = self._dis[did]
            dspath = os.path.join(
                self._pdir, self.DATA_DIRECTORY, di.data_source_id
            )
            for field in fields:
                print(did + ' -> ' + field)  # TODO think on how to change this to interact with GUI

                # prepare paths
                fpath = os.path.join(dspath, field)
                if not os.path.exists(os.path.join(self._pdir, fpath)):
                    os.mkdir(os.path.join(self._pdir, fpath))

                # prepare geometry stuff
                aoi = self.fields[
                    self.fields[DB.TBL_FIELDS.COL_FIELDNAME] == field
                ]

                ret = self.query(
                    DB.TBL_DAVLBL.Q_DATERANGES.format(did=did, fld=field)
                )
                # process static data
                if di.static_data:
                    if len(ret) > 0:
                        return
                    di.prj_add_data(fpath, aoi)
                    insda = f'INSERT INTO {DB.TBL_DAVLBL.NAME} '
                    insda += f'({DB.TBL_DAVLBL.COL_DID}, '
                    insda += f'{DB.TBL_DAVLBL.COL_FIELD}) VALUES ({did}, '
                    insda += f'{field});'
                    insde = DB.TBL_DEPOCHS.INSERT_START
                    insde += DB.TBL_DEPOCHS.INSERT_TUPLE.format(
                        did=did, fld=field, epoch=date.today().isoformat()
                    )
                    self.execute(insda)
                    self.execute(insde)
                    return

                # prepare timeranges for requesting dynamic data
                trngs = []
                for tpl in ret.itertuples():
                    trngs.append([
                        date.fromisoformat(tpl.tstart), 
                        date.fromisoformat(tpl.tstop)
                    ])
                trngs_requ = daterange_consider_existing_dates(
                    [tstart, tstop], trngs
                )

                # process data interfaces
                epochs_new = []
                for trng in trngs_requ:
                    try:
                        epochs_new += di.prj_add_data(fpath, aoi, trng)
                    except Exception as exc:
                        print(exc)
                        break

                # check if saved data and epochs_new are consistent
                # create sql-commands for .gpkg
                oneday = timedelta(days=1)
                insdeps, do_ins_deps = DB.TBL_DEPOCHS.INSERT_START, False
                trngs_new = []
                for trng in trngs_requ:
                    day = trng[0]
                    epochs = []
                    while day <= trng[1]:
                        epdir = os.path.join(fpath, day.isoformat())
                        c1 = os.path.isdir(epdir)
                        c2 = day in epochs_new
                        if c1:
                            if c2:
                                do_ins_deps = True
                                insdeps += DB.TBL_DEPOCHS.INSERT_TUPLE.format(
                                    did=did, fld=field, epoch=day.isoformat()
                                ) + ', '
                                epochs.append(day)
                            else:
                                shutil.rmtree(epdir)
                        day += oneday
                    if epochs:
                        trngs_new.append([epochs[0], epochs[-1]])

                # write epochs with available data to .gpkg
                if do_ins_deps:
                    self.execute(insdeps[:-2] + ';')

                # update timeranges and write them to .gpkg
                trngs_new = merge_dateranges(trngs + trngs_new)
                sql_ins = DB.TBL_DAVLBL.INSERT_START
                for trng in trngs_new:
                    sql_ins += DB.TBL_DAVLBL.INSERT_TUPLE.format(
                        did=did, fld=field, tstart=trng[0].isoformat(),
                        tstop=trng[1].isoformat()
                    ) + ', '
                self.execute(DB.TBL_DAVLBL.DEL_BY_DID_FLD.format(
                    did=did, fld=field
                ))
                self.execute(sql_ins[:-2] + ';')

    def get_data(self, tstart:date, tstop:date=None, dids=None, fields=None):
        """
        Get data from project. 
        The returned dictionary exhibits the same structure as the 
        folders in ``Project.DATA_DIRECTORY``.
        Available data is loaded by calling the classmethod 
        :func:`mef_agri.utils.raster.GeoRaster.from_directory` from the class 
        specified with 
        :func:`mef_agri.data.interface.Interface.georaster_class`.

        :param tstart: first epoch for which data should be provided
        :type tstart: datetime.date
        :param tstop: last epoch for which data should be provided, defaults to None
        :type tstop: datetime.date, optional
        :param dids: defining data interfaces for which data should be provided (:func:`mef_agri.data.interface.Interface.data_source_id`), defaults to None
        :type dids: list[str] | str, optional
        :param fields: defining fields for which data should be provided, defaults to None
        :type fields: list[str] | str, optional
        :return: dictionary with georasters
        :rtype: dict
        """
        dids = self._prepare_list(dids, list(self.interfaces.keys()))
        fields = self._prepare_list(
            fields, self.fields[DB.TBL_FIELDS.COL_FIELDNAME].values.tolist()
        )
        if tstop is None:
            tstop = deepcopy(tstart)
        ret = {}
        for did in dids:
            di:Interface = self._dis[did]
            ret[did] = {}
            for field in fields:
                ret[did][field] = {}
                fdir = os.path.join(
                    self._pdir, self.DATA_DIRECTORY, di.data_source_id, field
                )
                if di.static_data:
                    if di.data_types is None:
                        ret[did][field] = di.georaster_class.from_directory(
                            fdir
                        )
                    else:
                        ret[did][field] = {}
                        for dtype in di.data_types:
                            ret[did][field][dtype] = \
                                di.georaster_class.from_directory(
                                    os.path.join(fdir, dtype)
                                )
                    continue

                epochs = self.query(DB.TBL_DEPOCHS.Q_EPOCHS.format(
                    did=did, fld=field, tstart=tstart, tstop=tstop
                ))
                for ep in epochs[DB.TBL_DEPOCHS.COL_EPOCH].values.tolist():
                    ddir = os.path.join(fdir, ep)
                    if di.data_types is None:
                        ret[did][field][ep] = di.georaster_class.from_directory(
                            ddir
                        )
                    else:
                        ret[did][field][ep] = {}
                        for dtype in di.data_types:
                            ret[did][field][ep][dtype] = \
                                di.georaster_class.from_directory(
                                    os.path.join(ddir, dtype)
                                )
        return ret
                            
    @staticmethod
    def _prepare_list(provided:list[str] | str | None, all_values:list[str]):
            if provided is None:
                return all_values
            elif isinstance(provided, str):
                return [provided]
            else:
                return provided
