import sqlite3
from sqlite3 import Connection, Cursor
import os
import functools
import pyproj
import pandas as pd
import geopandas as gpd
from copy import deepcopy
from shapely import to_wkt
from numpy import nan


class GeometryType:
    GEOMETRY = 'GEOMETRY'
    POINT = 'POINT'
    LINESTRING = 'LINESTRING'
    POLYGON = 'POLYGON'
    MULTIPOINT = 'MULTIPOINT'
    MULTILINESTRING = 'MULTILINESTRING'
    MULTIPOLYGON = 'MULTIPOLYGON'
    GEOMETRYCOLLECTION = 'GEOMETRYCOLLECTION'


class ERRORS:
    NONSTR_ELEMENTS = 'Elements in provided iterable are not of type `str`!'
    INVALID_DTYPE = 'Provided datatype is non-valid, has to be `str`, '
    INVALID_DTYPE += '`list[str]` or tuple[str]!'
    NONSTR_VALUE = 'Provided value has to be of type `str`!'
    NONINT_VALUE = 'Provided value has to be of type `int`!'
    NONDICT_VALUE = 'Provided value has to be of type `dict`!'
    GPKG_PKEY_SET = 'There is already a primary key specified, cannot represent'
    GPKG_PKEY_SET += ' the current table as vector feature!'
    GPKG_GEOM_SET = 'This table represents a vector feature (i.e. it contains '
    GPKG_GEOM_SET += 'a geometry-column and the default `id` column being used '
    GPKG_GEOM_SET += 'as primary key)!'
    SQL_COL_NOT_AVLBL = 'Provided column name not existing in table schema!'
    GDF_NO_UCOLS = 'Provided GeoDataFrame does not contain all columns which '
    GDF_NO_UCOLS += 'are set as unique in the corresponding table!'


class DB:
    """
    Decorators for methods in :class:`Geopackage`.
    """
    NULLSTRINGS = ('', 'nan', 'NaN', 'None')

    @staticmethod
    def connect(func):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            if obj._conn is None:
                obj._conn = sqlite3.connect(os.path.join(obj._sd, obj._fn))
                obj._conn.enable_load_extension(True)
                obj._conn.load_extension('mod_spatialite')
                obj._curs = obj._conn.cursor()
                obj._curs.execute('PRAGMA foreign_keys=ON;')
                obj._curs.execute('PRAGMA journal_mode=WAL')
                obj._curs.execute('PRAGMA busy_timeout=5000')
            return func(obj, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def commit_and_close(func):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            try:
                ret = func(obj, *args, **kwargs)
                obj._conn.commit()
                return ret
            except Exception:
                obj._conn.rollback()
                raise
            finally:
                obj._conn.close()
                obj._conn = None
        return wrapper

    @staticmethod
    def close(func):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            try:
                return func(obj, *args, **kwargs)
            finally:
                obj._conn.close()
                obj._conn = None
        return wrapper
    
    @staticmethod
    def insert_text(text:str) -> str:
        """
        This method adds the single apostrophes around the provided input 
        string which is necessary for the SQL INSERT-command to indicate a 
        CHAR/TEXT value.
        Additionally the provided string is checked for single apostrophes, 
        which would "destroy" the INSERT-command. The solution is to replace  
        them with CHAR(39) (ASCII for single apostrophe) and concatenate this 
        with the || operator. SQLite/Geopackage interprets this correctly 
        afterwards.

        Example:

        input string: there's an apostrophe (sign is this ') in this string
        output text in the insert-command: 'there' || CHAR(39) || 's an apostrophe (sign is this ' || CHAR(39) || ') in this string'

        :param text: python string representing the text
        :type text: str
        :return: text ready to be used as insert-value in an sql insert-command
        :rtype: str
        """
        if (text is None) or (text in DB.NULLSTRINGS):
            return 'null'
        return '\'' + text.replace(
            '\'', 
            '\' || CHAR(39) || \''
        ) + '\''


def _check_iterable(val):
        if isinstance(val, str):
            return True, [val]
        elif isinstance(val, (list, tuple)):
            for el in val:
                if not isinstance(el, str):
                    return False, ERRORS.NONSTR_ELEMENTS
            if isinstance(val, tuple):
                return True, list(val)
            else:
                return True, val
        return False, ERRORS.INVALID_DTYPE


class ForeignKey(object):
    """
    Class which represents a foreign key being used in the creation of a 
    database or table.

    Important Note: the order of :func:`fkey_columns` and :func:`ftable_columns`
    has to be consistent!
    """
    CONSTRAINT_ACTIONS = [
        'NO ACTION', 'RESTRICT', 'SET NULL', 'CASCADE'
    ]

    def __init__(self) -> None:
        self._fkcols:list[str] = []
        self._ft:str = None
        self._ftcols:list[str] = []
        self._ondel = None
        self._onupd = None

    @property
    def ftable(self) -> str:
        """
        :return: name of the referenced table (settable)
        :rtype: str
        """
        return self._ft
    
    @ftable.setter
    def ftable(self, val):
        if isinstance(val, str):
            self._ft = val
        else:
            raise ValueError(ERRORS.NONSTR_VALUE)

    @property
    def fkey_columns(self) -> list[str]:
        """
        :return: name of the column(s) representing the foreign key (settable)
        :rtype: list[str]
        """
        return self._fkcols
    
    @fkey_columns.setter
    def fkey_columns(self, val):
        check, ret = _check_iterable(val)
        if check:
            self._fkcols = ret
        else:
            raise ValueError(ret)
        
    @property
    def ftable_columns(self) -> list[str]:
        """
        :return: name of the primary key column(s) (or unique) of the referenced table (settable)
        :rtype: list[str]
        """
        return self._ftcols
    
    @ftable_columns.setter
    def ftable_columns(self, val):
        check, ret = _check_iterable(val)
        if check:
            self._ftcols = ret
        else:
            raise ValueError(ret)
        
    @property
    def on_delete(self) -> str:
        """
        :return: ON DELETE constraint action - defaults to ``None`` (i.e. constraint is not inserted in the CREATE TABLE command)
        :rtype: str
        """
        return self._ondel
    
    @on_delete.setter
    def on_delete(self, val):
        if not (val in self.CONSTRAINT_ACTIONS):
            msg = 'Provided value for ON DELETE action is not supported - '
            msg += 'see `ForeignKey.CONSTRAINT_ACTIONS`!'
            raise ValueError(msg)
        self._ondel = val

    @property
    def on_update(self) -> str:
        """
        :return: ON UPDATE constraint action - defaults to ``None`` (i.e. constraint is not inserted in the CREATE TABLE command)
        :rtype: str
        """
        return self._onupd
    
    @on_update.setter
    def on_update(self, val):
        if not (val in self.CONSTRAINT_ACTIONS):
            msg = 'Provided value for ON UPDATE action is not supported - '
            msg += 'see `ForeignKey.CONSTRAINT_ACTIONS`!'
            raise ValueError(msg)
        self._onupd = val
        
        
class SQLTable(object):
    """
    Wrapper-class for gpgk-tables.
    """

    GEOM_COL_NAME = 'geometry'
    ID_COL_NAME = 'fid'

    def __init__(self) -> None:
        self._tn:str = None
        self._cols:dict = {}
        self._pkey:list[str] = []
        self._u:list[str] = []
        self._fkeys:list[ForeignKey] = []
        self._g:str = None
        self._gz:int = 0
        self._gm:int = 0

    @property
    def geom(self) -> str:
        """
        :return: type of geometry if table represents a vector feature, if ``None`` it is a standard sql-table, see ``GeometryType``
        :rtype: str
        """
        return self._g
    
    @geom.setter
    def geom(self, val):
        if self.primary_key:
            raise ValueError(ERRORS.GPKG_PKEY_SET)
        if not isinstance(val, str):
            raise ValueError(ERRORS.NONSTR_VALUE)
        self._g = val
        
    @property
    def geom_zval(self) -> int:
        """
        Specifies if the geometry has z/height coordinate.

        * 0 - z values prohibited
        * 1 - z values mandatory 
        * 2 - z values optional

        :return: flag for geomtry z-value
        :rtype: int
        """
        return self._gz
    
    @geom_zval.setter
    def geom_zval(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError(ERRORS.NONINT_VALUE)
        self._gz = int(val)
    
    @property
    def geom_mval(self) -> int:
        """
        Specifies if the geometry has a domain specific measurements (e.g. 
        specified velocity for line geometries)

        * 0 - m values prohibited
        * 1 - m values mandatory 
        * 2 - m values optional

        :return: flag for geometry m-value
        :rtype: int
        """
        return self._gm
    
    @geom_mval.setter
    def geom_mval(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError(ERRORS.NONINT_VALUE)
        self._gm = val

    @property
    def name(self) -> str:
        """
        :return: name of the table (settable)
        :rtype: str
        """
        return self._tn

    @name.setter
    def name(self, val):
        if not isinstance(val, str):
            raise ValueError(ERRORS.NONSTR_VALUE)
        self._tn = val
        
    @property
    def columns(self) -> dict:
        """
        The keys of this dict are the column names and the values represent the 
        data type, i.e. the values have to be of type `str` which correspond to 
        valid SQL data types.

        :return: column definitions
        :rtype: dict
        """
        return self._cols

    @columns.setter
    def columns(self, val):
        check, ret = _check_iterable(list(val.values()))
        if check:
            self._cols = val
        else:
            raise ValueError(ret)
        
    @property
    def primary_key(self) -> list[str]:
        """
        :return: column names which represent the primary key
        :rtype: list[str]
        """
        return self._pkey
    
    @primary_key.setter
    def primary_key(self, val):
        if self.geom:
            raise ValueError(ERRORS.GPKG_GEOM_SET)
        check, ret = _check_iterable(val)
        if check:
            self._pkey = ret
        else:
            raise ValueError(ret)
        
    @property
    def unique_cols(self) -> list[str]:
        """
        :return: unique columns
        :rtype: list[str]
        """
        return self._u
    
    @unique_cols.setter
    def unique_cols(self, val):
        check, ret = _check_iterable(val)
        if check:
            self._u = ret
        else:
            raise ValueError(ret)
        
    @property
    def foreign_keys(self) -> list[ForeignKey]:
        """
        :return: foreign key definitions - see ``field_trials.experiment.gpkg.ForeignKey``
        :rtype: list[ForeignKey]
        """
        return self._fkeys
    
    @foreign_keys.setter
    def foreign_keys(self, val):
        self._fkeys = val

    def sql_create(self) -> str:
        """
        :return: SQL-create statement for the table
        :rtype: str
        """
        ret = 'CREATE TABLE ' + self.name + '('
        if self.geom is not None:
            ret += self.ID_COL_NAME + ' INTEGER PRIMARY KEY AUTOINCREMENT, '
            ret += self.GEOM_COL_NAME + ' ' + self.geom + ', '
            # remove user defined primary key columns if it is a vector feature table
            # https://www.geopackage.org/spec130/#example_feature_table_sql
            self._pkey = []

        for key, val in self.columns.items():
            ret += key + ' ' + val + ', '
        if self.primary_key:
            ret += 'PRIMARY KEY ('
            for pkey in self.primary_key:
                ret += pkey + ', '
            ret = ret[:-2] + '), '
        if self.unique_cols:
            ret += 'UNIQUE ('
            for ucol in self.unique_cols:
                ret += ucol + ', '
            ret = ret[:-2] + '), '
        if self.foreign_keys:
            for fkey in self.foreign_keys:
                ret += 'FOREIGN KEY ('
                for col in fkey.fkey_columns:
                    ret += col + ', '
                ret = ret[:-2] + ') REFERENCES ' + fkey.ftable + '('
                for col in fkey.ftable_columns:
                    ret += col + ', '
                ret = ret[:-2] + '), '
                if fkey.on_delete is not None:
                    ret = ret[:-2] + ' ON DELETE ' + fkey.on_delete + ', '
                if fkey.on_update is not None:
                    ret = ret[:-2] + ' ON UPDATE ' + fkey.on_update + ', '
        ret = ret[:-2] + ');'
        return ret
    
    def sql_insert(self, **kwargs) -> str:
        """
        Creates the SQL INSERT command from the provided values.
        How to use ``kwargs``

        * keys - correspond to the columns of the table for which values should be inserted
        * values

            * data which should be inserted into the table
            * can be scalar variables for each column or iterables (list or tuples)
            * if they are iterables, their length has to be the same for each ``kwargs`` value
            * type conversion to ``str`` has to be possible for all values

        :return: sql-insert command
        :rtype: str
        """
        sql = 'INSERT INTO {} ('.format(self.name)
        tpls = None
        for attr, val in kwargs.items():
            # check if provided attribute/column exists in table schema
            if attr not in list(self.columns.keys()):
                raise ValueError(ERRORS.SQL_COL_NOT_AVLBL)
            sql += attr + ', '

            # convert the provided values to iterables (if not done by user)
            if not isinstance(val, (list, tuple)):
                vals = [val]
            else:
                vals = deepcopy(val)
            n = len(vals)

            # initialize tuple-strings for insert-command
            if not tpls:
                tpls = ['(' for i in range(n)]

            # extend tuple-strings for insert-command
            if self.columns[attr] == 'TEXT':
                tpls = \
            [tpls[i] + DB.insert_text(str(vals[i])) + ', ' for i in range(n)]
            else:
                tpls = [tpls[i] + str(vals[i]) + ', ' for i in range(n)]
        
        # create final insert command
        tpls = [tpls[i][:-2] + '), ' for i in range(len(tpls))]
        sql = sql[:-2] + ') VALUES ' + ''.join(tpls)
        sql = sql[:-2] + ';'
        return sql
    
    def sql_select_from(self, *args) -> str:
        """
        Creates a basic SELECT statement. ``args`` have to correspond to 
        column names otherwise an Exception is raised.
        If no ``args`` are provided, all columns will be used for selection.

        :return: SELECT [arg for arg in args] FROM table.name;
        :rtype: str
        """
        sql = 'SELECT '
        if not args:
            sql += '* '
        else:
            for arg in args:
                if not arg in self.columns.keys():
                    raise ValueError(ERRORS.SQL_COL_NOT_AVLBL)
                sql += arg + ', '
            sql = sql[:-2]
        return sql + ' FROM {};'.format(self.name)


class Geopackage(object):
    CRS_EPSG = 25833
    PANDAS_NONE_VALUES = ('None', 'NaN', 'nan', nan)

    # create commands for mandatory gpkg tables
    # https://www.geopackage.org/spec130/#gpkg_contents_sql
    _SQL_SRS = 'CREATE TABLE gpkg_spatial_ref_sys ('
    _SQL_SRS += 'srs_name TEXT NOT NULL, '
    _SQL_SRS += 'srs_id INTEGER PRIMARY KEY, '
    _SQL_SRS += 'organization TEXT NOT NULL, '
    _SQL_SRS += 'organization_coordsys_id INTEGER NOT NULL, '
    _SQL_SRS += 'definition TEXT NOT NULL, '
    _SQL_SRS += 'description TEXT'
    _SQL_SRS += ');'
    _SQL_CONT = 'CREATE TABLE gpkg_contents ('
    _SQL_CONT += 'table_name TEXT PRIMARY KEY, '
    _SQL_CONT += 'data_type TEXT NOT NULL, '
    _SQL_CONT += 'identifier TEXT UNIQUE, '
    _SQL_CONT += 'description TEXT DEFAULT \'\', '
    _SQL_CONT += 'last_change DATETIME NOT NULL DEFAULT (strftime('
    _SQL_CONT += '\'%Y-%m-%' + 'dT%H:%M:%' + 'fZ\', \'now\')), '
    _SQL_CONT += 'min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, '
    _SQL_CONT += 'srs_id INTEGER, '
    _SQL_CONT += 'CONSTRAINT fk_gc_r_srs_id FOREIGN KEY '
    _SQL_CONT += '(srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)'
    _SQL_CONT += ');'
    _SQL_GEOM = 'CREATE TABLE gpkg_geometry_columns ('
    _SQL_GEOM += 'table_name TEXT NOT NULL, '
    _SQL_GEOM += 'column_name TEXT NOT NULL, '
    _SQL_GEOM += 'geometry_type_name TEXT NOT NULL, '
    _SQL_GEOM += 'srs_id INTEGER NOT NULL, '
    _SQL_GEOM += 'z TINYINT NOT NULL, '  # 0 - z values prohibited, 1 - z values mandatory, 2 - z values optional
    _SQL_GEOM += 'm TINYINT NOT NULL, '  # 0 - m values prohibited, 1 - m values mandatory, 2 - m values optional
    _SQL_GEOM += 'CONSTRAINT pk_geom_cols '
    _SQL_GEOM += 'PRIMARY KEY (table_name, column_name), '
    _SQL_GEOM += 'CONSTRAINT uk_gc_table_name UNIQUE (table_name), '
    _SQL_GEOM += 'CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) '
    _SQL_GEOM += 'REFERENCES gpkg_contents(table_name), '
    _SQL_GEOM += 'CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) '
    _SQL_GEOM += 'REFERENCES gpkg_spatial_ref_sys(srs_id)'
    _SQL_GEOM += ');'
    # insert commands for required tables
    _INS_SRS = 'INSERT INTO gpkg_spatial_ref_sys ('
    _INS_SRS += 'srs_id, srs_name, organization, organization_coordsys_id, '
    _INS_SRS += 'definition, description'
    _INS_SRS += ') VALUES ('
    _INS_SRS += '{}, {}, {}, {}, {}, {}'
    _INS_SRS += ');'
    _INS_CONT = 'INSERT INTO gpkg_contents ('
    _INS_CONT += 'table_name, data_type, identifier, srs_id'
    _INS_CONT += ') VALUES ('
    _INS_CONT += '{}, {}, {}, {}'
    _INS_CONT += ');'
    _INS_GEOM = 'INSERT INTO gpkg_geometry_columns ('
    _INS_GEOM += 'table_name, column_name, geometry_type_name, srs_id, z, m'
    _INS_GEOM += ') VALUES ('
    _INS_GEOM += '{}, {}, {}, {}, {}, {}'
    _INS_GEOM += ');'


    def __init__(self, save_dir:str, file_name:str) -> None:
        self._sd = save_dir  # absolute path to .gpkg file
        self._fn = file_name  # name of the .gpkg file
        if not self._fn[-5:] == '.gpkg':
            self._fn += '.gpkg'        
        self._tbls:dict = {}
        self._conn:Connection = None
        self._curs:Cursor = None

        # check if .gpkg exists and is initialized
        self._init:bool = False
        if os.path.isfile(os.path.join(self._sd, self._fn)):
            self._init = True

    @property
    def connection(self) -> Connection:
        """
        :return: connection instance to gpkg
        :rtype: Connection
        """
        return self._conn
    
    @property
    def is_initialized(self) -> bool:
        """
        :return: flag if the specified gpkg-file exists and is initialized
        :rtype: bool
        """
        return self._init
    
    @property
    def absolute_path(self) -> str:
        """
        :return: absolute path of the geopackage file (including the file name)
        :rtype: str
        """
        return os.path.join(self._sd, self._fn)

    @property
    def tables(self) -> dict:
        """
        :return: tables in the geopackage (keys of the dict correspond to the table-names)
        :rtype: dict
        """
        return self._tbls
    
    @tables.setter
    def tables(self, val):
        if not isinstance(val, dict):
            raise ValueError(ERRORS.NONDICT_VALUE)
        self._tbls = val

    @DB.connect
    @DB.commit_and_close
    def initialize(self) -> None:
        """
        Initialize the geopackage file with mandatory information according to 
        the geopackage standard

        * mandatory tables => https://www.geopackage.org/spec130/#gpkg_contents_sql
        * mandatory spatial-reference-system data => https://opengeospatial.github.io/e-learning/geopackage/text/contents.html

        """
        print('---------------------------------------------------------------')
        # create mandatory tables
        self._curs.execute(self._SQL_SRS)
        self._curs.execute(self._SQL_CONT)
        self._curs.execute(self._SQL_GEOM)
        print(self._curs.fetchall())
        print('\nCreated mandatory tables\n')
        
        print('-----------------------------------------------------------')
        # mandatory srs data
        # https://opengeospatial.github.io/e-learning/geopackage/text/contents.html
        self._curs.execute(self._INS_SRS.format(
            0, '\'undefined_geographic\'', '\'pyproj\'', 0, 
            '\'' + pyproj.crs.GeographicCRS().to_wkt() + '\'', 
            '\'undefined geographic coordinate reference systems\''
        ))
        undefc = pyproj.crs.ProjectedCRS(
            pyproj.crs.coordinate_operation.MercatorAConversion()
        )
        self._curs.execute(self._INS_SRS.format(
            -1, '\'undefined_cartesian\'', '\'pyproj\'', -1, 
            '\'' + undefc.to_wkt() + '\'', 
            '\'undefined cartesian coordinate reference system\''
        ))
        self._curs.execute(self._INS_SRS.format(
            4326, '\'WGS 84\'', '\'EPSG\'', 4326, 
            '\'' + pyproj.CRS.from_epsg(4326).to_wkt() + '\'',
            '\'WGS 84 - World Geodetic Datum 1984, used in GPS\''
        ))
        print(self._curs.fetchall())
        print('\nInserted mandatory coordinate reference systems\n')
        
        print('-----------------------------------------------------------')

    @DB.connect
    @DB.commit_and_close
    def create_tables(self) -> None:
        """
        Create tables specfied in ``tables``-property
        """
        print('-----------------------------------------------------------')
        qsrs = 'SELECT srs_id FROM gpkg_spatial_ref_sys WHERE srs_id='
        qsrs += str(self.CRS_EPSG) + ';'
        self._curs.execute(qsrs)
        qres = self._curs.fetchall()
        if not qres:
            crs = pyproj.CRS.from_epsg(self.CRS_EPSG)
            sql = self._INS_SRS.format(
                self.CRS_EPSG, 
                '\'' + crs.name + '\'', 
                '\'EPSG\'', 
                self.CRS_EPSG, 
                DB.insert_text(crs.to_wkt()),
                '\'---\''
            )
            self._curs.execute(sql)
            print(self._curs.fetchall())
            print('\ninserted srs {}\n'.format(self.CRS_EPSG))
            print('-----------------------------------------------------------')

        for table in self.tables.values():
            idf = '\'attributes\'' if table.geom is None else '\'features\''
            tbln = '\'' + table.name + '\''
            self._curs.execute(self._INS_CONT.format(tbln, idf, tbln, self.CRS_EPSG))
            if table.geom is not None:
                self._curs.execute(self._INS_GEOM.format(
                    '\'' + table.name + '\'', '\'' + table.GEOM_COL_NAME + '\'',
                    '\'' + table.geom + '\'', self.CRS_EPSG, 
                    table.geom_zval, table.geom_mval
                ))
            print(table.sql_create())
            self._curs.execute(table.sql_create())
            print(self._curs.fetchall())
            print('\ncreated table {} and its dependencies\n'.format(table.name))
            print('-----------------------------------------------------------')

    @DB.connect
    @DB.commit_and_close
    def insert_data(self, table:str, **kwargs) -> None:
        """
        Method to insert data into a table. See :func:`SQLTable.sql_insert` on 
        how to structure ``kwargs``.
        This method can only be used for non-geometry columns to insert data.
        Otherwise use :func:`insert_from_gdf`.

        :param table: name of the table in the gpkg
        :type table: str
        """
        sql = self.tables[table].sql_insert(**kwargs)
        self._curs.execute(sql)
        print(self._curs.fetchall())
        print('\ninserted data into {}'.format(table))
        print('-----------------------------------------------------------')

    @DB.connect
    @DB.commit_and_close
    def execute(self, sql:str) -> None:
        """
        Executes sql commands which should be written to the geoapackage, i.e. 

        * DDL: CREATE, DROP, ALTER
        * DML: INSERT, DELETE, UPDATE

        :param sql: sql command
        :type sql: str
        """
        self._curs.execute(sql)


    @DB.connect
    @DB.close
    def query(self, sql:str) -> pd.DataFrame:
        """
        Performs a query, i.e. a sql SELECT command.

        :param sql: sql command
        :type sql: str
        :return: query result as DataFrame
        :rtype: pandas.DataFrame
        """
        self._curs.execute(sql)
        cols = [descr[0] for descr in self._curs.description]
        df = pd.DataFrame(columns=cols)
        i = 0
        for row in self._curs.fetchall():
            df.loc[i] = row
            i += 1
        return df
    
    def query_gdf(self, table:str, where:str=None) -> gpd.GeoDataFrame:
        """
        This method queries the ``table`` of the geopackage and returns the 
        result as ``geopandas.GeoDataFrame``.
        ``where`` offers the possibility to filter the query result and has to 
        contain the whole WHERE-clause of SQL, e.g. 
        ``WHERE column1='text-value' AND column2<=numeric-value``.
        There is no special index set in the returned GeoDataFrame, thus, the 
        feature-id-column (i.e. ``'fid'``) appears as standard column.

        :param table: table name
        :type table: str
        :param where: WHERE-clause of SQL, defaults to None
        :type where: str, optional
        :raises ValueError: if provided ``table`` does not represent a vector feature
        :return: query result
        :rtype: gpd.GeoDataFrame
        """
        # check if provided table is a vector feature
        gtype = self.get_table_geometry_type(table)
        if gtype is None:
            msg = 'Provided `table` does not represent a vector feature!'
            raise ValueError(msg)
        # get column names of table
        cols = list(self.tables[table].columns.keys())
        sql = 'SELECT {}, ST_AsText({}) AS {}, '.format(
            SQLTable.ID_COL_NAME, SQLTable.GEOM_COL_NAME, SQLTable.GEOM_COL_NAME
        )
        sql += ', '.join(cols) + ' FROM ' + table
        if where is not None:
            sql += ' ' + where
        sql += ';'
        ret = self.query(sql)
        ret[SQLTable.GEOM_COL_NAME] = gpd.GeoSeries.from_wkt(
            ret[SQLTable.GEOM_COL_NAME]
        )
        srs_id = self.get_table_srs_id(table)
        return gpd.GeoDataFrame(
            ret, geometry=SQLTable.GEOM_COL_NAME, crs=srs_id
        )

    def insert_from_gdf(
            self, table:str, gdf:gpd.GeoDataFrame, _gdf_checked:bool=False
        ):
        """
        This method inserts data into the geopackage from the provided ``gdf``.
        ``gdf`` is screened for duplicates - within ``gdf`` and also with tuples 
        already available in the geopackage - which will not be inserted into 
        the corresponding ``table``.
        It is recommended, that each table in the geopackage has a primary key 
        or unique constraint because this is the most secure way to detect 
        duplicates!
        Otherwise all columns from ``table`` except fid- and geometry-columns 
        are used to detect duplicates.
        In this case it is possible that duplicates are not correctly detected,
        e.g. due to changed values in non-identifying columns or different 
        specifications of pandas-nan-values.

        :param table: table name
        :type table: str
        :param gdf: GeoDataFrame which should be inserted into ``table``
        :type gdf: gpd.GeoDataFrame
        :param _gdf_checked: for internal use if ``gdf`` has already been passed into ``Geopackage._check_none_values()``, defaults to False
        :type _gdf_checked: bool, optional
        :raises ValueError: if provided ``gdf`` contains multiple geometry types
        :raises ValueError: if provided ``gdf`` does not represent a vector feature from the geopackage
        :raises ValueError: if provided ``gdf`` is not of the same geometry type as the specified ``table`` in the geopackage
        :raises ValueError: if provided ``gdf`` does not contain all columns from the unique-specification of the ``table`` in the geopackage
        """
        # check for None values if not already done within the class
        if not _gdf_checked:
            gdf = self._check_none_values(gdf)
        # check if geometry type of ``gdf`` equals the one in the gpkg
        geoms = gdf.geom_type.drop_duplicates()
        if len(geoms) > 1:
            msg = 'Provided `gdf` exhibits multiple geometry types, which is not '
            msg += 'supported in one table of a geopackage!'
            raise ValueError(msg)
        gt_db = self.get_table_geometry_type(table)
        if gt_db is None:
            msg = 'Provided `table` does not represent a vector feature!'
            raise ValueError(msg)
        if geoms.values[0].lower() != gt_db.lower():
            msg = 'Provided `gdf` does not exhibit the same geometry type as the '
            msg += 'corresponding table in the geopackage!'
            raise ValueError(msg)
        
        # child classes can do a check on the gdf if necessary before processing
        gdf = self.check_gdf_before_process(table, gdf)

        # check epsg/srs_id of provided ``gdf``
        srs_id = self.get_table_srs_id(table)
        if gdf.crs.to_epsg() != srs_id:
            ngdf = gdf.to_crs(srs_id)
        else:
            ngdf = deepcopy(gdf)

        # data which is already present in the table
        egdf = self.query_gdf(table)
        # ignore columns of provided gdf which are not available in the table
        for col in ngdf.columns:
            if not col in egdf.columns:
                ngdf.drop(columns=col, inplace=True)
        # add feature-id column name and geometry column name to the list of columns
        # such that they also appear in the in the sql-insert command
        ncols = ngdf.columns.to_list()
        if not SQLTable.ID_COL_NAME in ncols:
            ncols.append(SQLTable.ID_COL_NAME)
        if not SQLTable.GEOM_COL_NAME in ncols:
            ncols.append(SQLTable.GEOM_COL_NAME)

        # check if new gdf still contains unique columns from the table
        ucols = self.tables[table].unique_cols
        if not ucols:
            # if no unique columns are specified for the table, use the columns
            # of the incoming gdf to search for duplicates
            ucols = deepcopy(ncols)
            # exclude feature id and geometry column from the search for 
            # duplicates as they are dependent on how the provided gdf has been 
            # loaded
            del(ucols[ucols.index(SQLTable.ID_COL_NAME)])
            del(ucols[ucols.index(SQLTable.GEOM_COL_NAME)])
        elif not set(ucols).issubset(set(list(ngdf.columns))):
            raise ValueError(ERRORS.GDF_NO_UCOLS)

        # get column sql data type definitions
        cts = self.query(
            'SELECT cname, dtype FROM definitions WHERE tname=\'{}\';'.format(table)
        ).set_index('cname').to_dict(orient='index')

        # processing
        next_fid = len(egdf)
        ins = 'INSERT INTO {} ('.format(table)
        ins += ', '.join(ncols) + ') VALUES '
        insert_data = False
        for tpl in ngdf.itertuples(index=False):
            # check if current tuple already exists in geopackage
            dfi = gpd.GeoDataFrame(
                [tpl], geometry=SQLTable.GEOM_COL_NAME, crs=srs_id
            )
            dupl_check = pd.concat(
                [egdf[ucols], dfi[ucols]], ignore_index=True
            ).duplicated().values.tolist()
            if True in dupl_check:
                continue

            # prepare sql-insert command
            insert_data = True
            di = []
            for col in ncols:
                if col == SQLTable.ID_COL_NAME:
                    vi = str(next_fid)
                elif col == SQLTable.GEOM_COL_NAME:
                    wkt = DB.insert_text(to_wkt(getattr(tpl, col)))
                    vi = 'ST_GeomFromText({}, {})'.format(wkt, srs_id)
                else:
                    ct = cts[col]['dtype']
                    if ct == 'TEXT':
                        vi = DB.insert_text(str(getattr(tpl, col)))
                    else:
                        vi = str(getattr(tpl, col))
                di.append(vi)
            ins += '(' + ', '.join(di) + '), '
            next_fid += 1
        
        if insert_data:
            self.execute(ins[:-2] + ';')

    def get_table_srs_id(self, table:str) -> int:
        """
        Query the epsg (i.e. ``srs_id``) of the provided ``table`` from 
        ``gpkg_contents`` (mandatory table of geopackage-format).
        If ``table`` does not represent a geometry feature, ``None`` will be 
        returned.

        :param table: table name
        :type table: str
        :return: epsg code (i.e. ``srs_id``)
        :rtype: int
        """
        sql = 'SELECT data_type, srs_id FROM gpkg_contents '
        sql += 'WHERE table_name=\'{}\';'.format(table)
        ret = self.query(sql)
        if ret['data_type'].values[0] == 'attributes':
            return None
        return ret['srs_id'].values[0]
    
    def get_table_geometry_type(self, table:str) -> str:
        """
        Query the geometry type of the provided ``table``.
        If ``table`` does not represent a geometry feature, ``None`` will be 
        returned.

        :param table: table name
        :type table: str
        :return: geometry type
        :rtype: str
        """
        sql = 'SELECT geometry_type_name FROM gpkg_geometry_columns '
        sql += 'WHERE table_name=\'{}\';'.format(table)
        ret = self.query(sql)
        if len(ret) == 0:
            return None
        return ret['geometry_type_name'].values[0]
            
    def _check_none_values(self, df:pd.DataFrame) -> pd.DataFrame:
        ndf = deepcopy(df)
        for none_value in self.PANDAS_NONE_VALUES:
            ndf.replace({none_value: None}, inplace=True)
        return ndf
