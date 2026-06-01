import os
import pandas as pd
from inspect import isclass
from importlib import import_module
from datetime import date
from numpy import nan

from ..utils.gpkg import Geopackage, SQLTable, ForeignKey, GeometryType


################################################################################
# SCHEMA OF PROJECT DATABASE
################################################################################
_tds = SQLTable()
_tds.name = 'data_sources'
_tds.columns = {
    'did': 'TEXT',
    'intf_name': 'TEXT',
    'intf_module': 'TEXT'
}
_tds.primary_key = ['did']

_tfld = SQLTable()
_tfld.name = 'fields'
_tfld.columns = {
    'field_name': 'TEXT'
}
_tfld.geom = GeometryType.POLYGON
_tfld.unique_cols = ['field_name']

_tda = SQLTable()
_tda.name = 'data_available'
_tda.columns = {
    'did': 'TEXT',
    'field': 'TEXT',
    'tstart': 'CHAR(12)',
    'tstop': 'CHAR(12)'
}
_tda.primary_key = ['did', 'field']
_tda_tds_fk = ForeignKey()
_tda_tds_fk.fkey_columns = ['did']
_tda_tds_fk.ftable = _tds.name
_tda_tds_fk.ftable_columns = ['did']
_tda_tds_fk.on_delete = 'CASCADE'
_tda_tds_fk.on_update = 'CASCADE'
_tda_fld_fk = ForeignKey()
_tda_fld_fk.fkey_columns = ['field']
_tda_fld_fk.ftable = 'fields'
_tda_fld_fk.ftable_columns = ['field_name']
_tda_fld_fk.on_delete = 'CASCADE'
_tda_fld_fk.on_update = 'CASCADE'
_tda.foreign_keys = [_tda_tds_fk, _tda_fld_fk]


################################################################################
# PROJECT CLASS
################################################################################
class Project(Geopackage):
    DATA_DIRECTORY = 'data'

    ERR_PRJPATH = 'Provided path does not exist!'
    ERR_GPKGFILE = 'No .gpkg file in provided project path!'
    ERR_FLDNOTAVLBL = 'Provided field-name not available in the GeoPackage!'
    ERR_DIDNOTAVLBL = 'Provided data-source-id not available or provided yet!'
    
    WRN_ADDDATA = 'Error when adding data from data source `{dsrc}` '
    WRN_ADDDATA += 'at field `{fld}` - no entry inserted in `data_available`'
    WRN_ADDDATA += '-table for this case!'

    QUERY_INTERFACES = 'SELECT * FROM ' + _tds.name + ';'

    def __init__(self, project_dir, gpkg_name):
        super().__init__(project_dir, gpkg_name)
        self._dis:dict = None
        self._pdir:str = project_dir
        self.tables = {
            _tds.name: _tds,
            _tfld.name: _tfld,
            _tda.name: _tda
        }

    @property
    def directory(self) -> str:
        return self._pdir

    @property
    def interfaces(self) -> dict:
        if self._dis is None:
            dis = self.query(self.QUERY_INTERFACES)
            for tpl in dis.itertuples():
                di = getattr(import_module(tpl.intf_module), tpl.intf_name)()
                di.project_directory = self.directory
                di.project_ref = self
                self._dis[tpl.did] = di
        return self._dis
    
    @property
    def fields(self) -> pd.DataFrame:
        pass

    def add_data_interface(self, data_intf) -> None:
        """
        Method to add a :class:`mef_agri.data.interface.DataInterface` to the 
        project.

        :param data_intf: the desired :class:`mef_agri.data.interface.DataInterface`-subclass
        :type data_intf: class or instance
        """
        # check if data interface already has been added
        if data_intf.DATA_SOURCE_ID in list(self.interfaces.keys()):
            return
        
        # create folder within the data folder for the 
        dpath = os.path.join(
            self._pp, self.DATA_DIRECTORY, data_intf.DATA_SOURCE_ID
        )
        if not os.path.exists(dpath):
            os.mkdir(dpath)

        # intialize data interface if necessary and set project and save paths
        if isclass(data_intf):
            data_intf = data_intf()
        data_intf.project_directory = self.directory
        data_intf.project_ref = self
        # add data interface to the dict containing all interfaces of the prj
        self._dis[data_intf.DATA_SOURCE_ID] = data_intf

        # insert data interface information to the project database
        self.execute(self.tables['data_sources'].sql_insert(
            did=data_intf.DATA_SOURCE_ID,
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
        :type tstart: date
        :param tstop: last epoch of the requested data
        :type tstop: date
        :param dids: data-source-ids available in the corresponding data-interface, defaults to None
        :type dids: list[str] or str, optional
        :param fields: name of the fields, defaults to None
        :type fields: list[str] or str, optional
        """
        # prepare dids and fields
        def prepare_list(provided:list[str] | str | None, all_values:list[str]):
            if provided is None:
                return all_values
            elif isinstance(provided, str):
                return [provided]
            else:
                return provided
        dids = prepare_list(dids, list(self.interfaces.keys()))
        fields = prepare_list(fields, self.fields['field_name'].values.tolist())
        
        # get data from interfaces
        for did in dids:
            di = self._dis[did]
            # prepare paths
            dspath = os.path.join(self.DATA_DIRECTORY, di.DATA_SOURCE_ID)
            for field in fields:
                # prepare paths
                fpath = os.path.join(dspath, field)
                if not os.path.exists(os.path.join(self._pp, fpath)):
                    os.mkdir(os.path.join(self._pp, fpath))
                di.save_directory = fpath

                # prepare geometry stuff
                aoi = self.fields['field_name' == field]
                di.current_field = field
                
                # TODO work further from here on
                print(did + ' -> ' + field)  # TODO think on how to change this to interact with GUI

                # prepare timeranges input and output
                trngs_requ, trngs = self._db.get_date_ranges(
                    did, field, tstart, tstop
                )
                trngs_requ_upd = []

                # get data from data interfaces
                prc_ok = True
                for trng in trngs_requ:
                    # when there is an error, nothing is inserted into the .gpkg 
                    # user has to manually delete the corresponding folders and 
                    # files or manually update data_available-information in .gpkg
                    try:
                        trng_upd = di.add_prj_data(aoi, trng[0], trng[1])
                        if not nan in trng_upd:
                            trngs_requ_upd.append(trng_upd)
                        if di.add_prj_data_error:
                            prc_ok = False
                            print(self.WRN_ADDDATA.format(dsrc=did, fld=field))
                            break    
                    except Exception as exc:
                        prc_ok = False
                        print(exc)
                        print(self.WRN_ADDDATA.format(dsrc=did, fld=field))
                        break

                if prc_ok:
                    self._db.update_date_ranges(
                        did, field, trngs_requ_upd, trngs
                    )
                    self._db._conn.commit()
