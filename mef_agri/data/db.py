import sqlite3
import pandas as pd
from datetime import date

from .utils import (
    daterange_consider_existing_dates, merge_dateranges
)


class ProjectDatabase(object):
    """
    This class represents the project database which is created by extending 
    the db-schema of the geopackage in the project folder (see ``project_path``
    argument of the constructor of :class:`mef_agri.data.project.Project`). 

    The following tables are added

    * ``data_sources`` - table with columns 
    
        * ``did`` - id of the data source (class variable ``DATA_SOURCE_ID`` of an class inheriting from :class:`mef_agri.data.interface.DataInterface`) - PK
        * ``intf_name`` - name of the class which inherits from :class:`mef_agri.data.interface.DataInterface`
        * ``intf_module`` - python module containing ``intf_name``

    * ``data_available`` - table with columns

        * ``did`` - id of the data source (class variable ``DATA_SOURCE_ID`` of an class inheriting from :class:`mef_agri.data.interface.DataInterface`) - FK to ``data_sources``
        * ``field`` - name of the field - FK to table specified fwith ``field_table`` and column ``field_name_column``
        * ``tstart`` - first date/epoch where data is available
        * ``tstop`` - last date/epoch where data is available

    * ``project_infos`` - table with columns

        * ``setting_id`` - name of a specific project setting
        * ``setting_val`` - value of a specific project setting

    Comment 1
    
    ``field_table`` and ``field_name_column`` have to be provided only at the 
    first time when this class is initialized in a project folder. Afterwards 
    these information is available in the ``project_infos`` table.

    Comment 2

    Rows in ``data_available`` represent a time span where data is available 
    at every day within ``tstart``- and ``tstop``-column. Thus, if there is 
    missing data, the time interval will be split into separate rows in this 
    table.
    

    :param dbpath: absolute path of the database
    :type dbpath: str
    :param field_table: name of the table which contains the field geometries, defaults to None
    :type field_table: str, optional
    :param field_name_column: name of the column of the field-table containing the field names, defaults to None
    :type field_name_column: str, optional
    """
    ERR_NO_FIELD_INFO = 'Name of field-table and name of column containing '
    ERR_NO_FIELD_INFO += 'field-names have to be both available when '
    ERR_NO_FIELD_INFO += 'initializing project-related tables in the gpkg!'

    CREATE_DATA_SRCS = 'CREATE TABLE data_sources ('
    CREATE_DATA_SRCS += 'did TEXT PRIMARY KEY, intf_name TEXT, intf_module TEXT'
    CREATE_DATA_SRCS += ');'

    INSERT_DATA_SRCS = 'INSERT INTO data_sources (did, intf_name, intf_module) '
    INSERT_DATA_SRCS += 'VALUES (\'{did}\', \'{iname}\', \'{imodule}\');'

    CREATE_DATA_AVLBL = 'CREATE TABLE data_available ('
    CREATE_DATA_AVLBL += 'did TEXT, field TEXT, tstart CHAR(12), tstop CHAR(12), '
    CREATE_DATA_AVLBL += 'FOREIGN KEY (did) REFERENCES data_sources(did), '
    CREATE_DATA_AVLBL += 'FOREIGN KEY (field) REFERENCES {ftbl}({fncol})'
    CREATE_DATA_AVLBL += ');'

    INSERT_DATA_AVLBL = 'INSERT INTO data_available (did, tstart, tstop) '
    INSERT_DATA_AVLBL += 'VALUES (\'{did}\', \'{tstart}\', \'{tstop}\');'
    
    CREATE_PRJ_INFO = 'CREATE TABLE project_infos ('
    CREATE_PRJ_INFO += 'setting_id TEXT PRIMARY KEY, setting_val TEXT'
    CREATE_PRJ_INFO += ');'

    INSERT_PRJ_INFO = 'INSERT INTO project_infos (setting_id, setting_val) '
    INSERT_PRJ_INFO += 'VALUES (\'{sid}\', \'{sval}\');'

    GET_PRJ_INFO = 'SELECT * FROM project_infos;'

    def __init__(
            self, dbpath:str, field_table:str=None, field_name_column:str=None
        ):
        self._dbp:str = dbpath
        self._conn = sqlite3.connect(dbpath)
        self._curs = self._conn.cursor()

        # check if project part is initiailized in the geopackage/database
        # if table `project_misc` does not exist, an exception is raised
        try:
            self._curs.execute(self.GET_PRJ_INFO)
        except:
            if (field_table is None) and (field_name_column is None):
                raise ValueError(self.ERR_NO_FIELD_INFO)
            self._curs.execute(self.CREATE_DATA_SRCS)
            self._curs.execute(self.CREATE_DATA_AVLBL.format(
                ftbl=field_table, fncol=field_name_column
            ))
            self._curs.execute(self.CREATE_PRJ_INFO)
            self._curs.execute(self.INSERT_PRJ_INFO.format(
                sid='field_table', sval=field_table
            ))
            self._curs.execute(self.INSERT_PRJ_INFO.format(
                sid='field_name_column', sval=field_name_column
            ))

        self._prj_infos = self.get_data_frame(self.GET_PRJ_INFO)
        self._fld_tbl:str = self._prj_infos[
            self._prj_infos['setting_id'] == 'field_table'
        ]['setting_val'].values[0]
        self._fld_nc:str = self._prj_infos[
            self._prj_infos['setting_id'] == 'field_name_column'
        ]['setting_val'].values[0]
        self._last_exc:str = 'No exception ocurred.'

    @property
    def field_table_name(self) -> str:
        """
        :return: name of table containing field geometries and names
        :rtype: str
        """
        return self._fld_tbl
    
    @property
    def field_name_column(self) -> str:
        """
        :return: name of the column in :func:`field_table_name` containing field names
        :rtype: str
        """
        return self._fld_nc
    
    @property
    def file_path(self) -> str:
        """
        :return: absolute path of geopackage db-file
        :rtype: str
        """
        return self._dbp
    
    def check_data_source(self, did:str) -> bool:
        """
        Check if provided data-source-is is already available in the 
        ``data_sources``-table.

        :param did: ``DATA_SOURCE_ID`` from :class:`mef_agri.data.interface.DataInterface`
        :type did: str
        :return: flag if available
        :rtype: bool
        """
        cmd = 'SELECT * FROM data_sources WHERE did==\'{}\';'.format(did)
        return bool(self.execute_sql_command(cmd))
    
    def insert_data_source(
            self, did:str, intf_name:str, intf_module:str
        ) -> bool:
        """
        Insert a data-source (i.e. a class inheriting from 
        :class:`mef_agri.data.interface.DataInterface`) in the the 
        ``data_sources``-table

        :param did: ``DATA_SOURCE_ID`` from :class:`mef_agri.data.interface.DataInterface`
        :type did: str
        :param intf_name: name of the class
        :type intf_name: str
        :param intf_module: module containing the class
        :type intf_module: str
        :return: indicator if corresponding sql-command has been successfully executed
        :rtype: bool
        """
        ret = self.execute_sql_command(self.INSERT_DATA_SRCS.format(
            did=did, iname=intf_name, imodule=intf_module
        ))
        return False if ret is None else True
    
    def get_data_sources(self) -> pd.DataFrame:
        """
        :return: DataFrame representing the ``data_sources``-table
        :rtype: pandas.DataFrame
        """
        return self.get_data_frame('SELECT * FROM data_sources;')
    
    def get_date_ranges(
            self, did:str, field:str, tstart:date, tstop:date
        ) -> tuple[list[list[date, date]], list[list[date, date]]]:
        """
        Compares the provided date-range (``tstart`` and ``tstop``) with the 
        ones available in the database and returns only date-ranges which do not 
        overlap with the available ones.

        The first list within the returned tuple contains the output of 
        :func:`mef_agri.data.utils.daterange_consider_existing_dates`. The 
        second list contains the existing date-ranges in ``data_avalaile``-table.

        :param did: data-source-id available in the corresponding data-interface
        :type did: str
        :param field: name of the field
        :type field: str
        :param tstart: start-date of the considered date-range
        :type tstart: datetime.date
        :param tstop: last date of the considered date-range
        :type tstop: datetime.date
        :return: non overlapping date-ranges and existing date-ranges in ``data_available``-table
        :rtype: tuple[list[list[datetime.date, datetime.date]], list[list[datetime.date, datetime.date]]]
        """
        cmd = 'SELECT tstart, tstop FROM data_available WHERE '
        cmd += 'did=\'{}\' AND field=\'{}\';'
        trngs = self.execute_sql_command(cmd.format(did, field))
        trngs = [
            [date.fromisoformat(t1), date.fromisoformat(t2)] for t1, t2 in trngs
        ]
        trngs_requ = daterange_consider_existing_dates([tstart, tstop], trngs)
        return trngs_requ, trngs

    def update_date_ranges(
            self, did:str, field:str, trngs_requ:list[list[date]],
            trngs:list[list[date]]
        ) -> None:
        """
        Update date ranges in ``data_available``-table by merging existing 
        date-ranges and requested date-range such that there is no redundant 
        information (i.e. ensure non-overlapping dateranges)

        :param did: data-source-id available in the corresponding data-interface
        :type did: str
        :param field: name of the field
        :type field: str
        :param trngs_requ: requested date-ranges
        :type trngs_requ: list[list[datetime.date, datetime.date]]
        :param trngs: existing date-ranges
        :type trngs: list[list[datetime.date, datetime.date]]
        """
        trngs_new = merge_dateranges(trngs + trngs_requ)
        cmd = 'DELETE FROM data_available WHERE did=\'{}\' AND field=\'{}\''
        self.execute_sql_command(cmd.format(did, field))
        cmd = 'INSERT INTO data_available (did, field, tstart, tstop) VALUES '
        cmd += '(\'{}\', \'{}\', \'{}\', \'{}\');'
        for trng in trngs_new:
            self.execute_sql_command(cmd.format(
                did, field, trng[0].isoformat(), trng[1].isoformat()
            ))

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
        :func:`self._curs.execute`) to a pandas.DataFrame

        :param cmd: sql command
        :type cmd: str
        :return: DataFrame with result from sql-query
        :rtype: pandas.DataFrame
        """
        res = self.execute_sql_command(cmd)
        if res is None:
            return None
        else:
            descr = self._curs.description
            return pd.DataFrame(res, columns=[col[0] for col in descr])
        
    def close(self) -> None:
        """
        Commit changes/writes to the database (otherwise changes are not 
        persistent).
        """
        self._conn.commit()
