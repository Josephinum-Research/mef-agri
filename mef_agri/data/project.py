import os
import geopandas as gpd
from datetime import date
from inspect import isclass
from importlib import import_module

from .db import ProjectDatabase


class Project(object):
    """
    This class does the setup of a project. 
    
    A geopackage has to be located in 
    the provided folder ``project_path`` which at least contains a table with 
    field geometries and field names. This geopackage will be extended to 
    contain information on which data is available on which dates (see
    :class:`mef_agri.data.db.ProjectDatabase`). 

    If data-interfaces/-sources are added with :func:`add_data_interface`, the 
    folder structure will be extended (or initialized in the first call of this 
    method). The top-level data-folder will be located directly in the 
    project-folder and its name is specified via the class variable 
    ``DATA_DIRECTORY`` (default is ``data``). 
    
    An example structure is as follows

    * project.DATA_DIRECTORY

        * data_interface_1.DATA_SOURCE_ID (e.g. dynamic data)

            * field_1

                * date_1

                    * TIFF file(s)

                * date_2

                    * TIFF file(s)

                * ...

            * field_2
            
                * date_1

                    * TIFF file(s)

                * date_2

                    * TIFF file(s)

                * ...

            * ...

        * data_interface_2.DATA_SOURCE_ID (e.g. static data)

            * field_1

                * TIFF file(s)

            * field_2

                * TIFF file(s)

            * ...

    :param project_path: absolute path of folder where project is located in
    :type project_path: str
    :param gpkg_field_table: name of the table within the .gpkg in the project folder which contains the field geometries, defaults to None
    :type gpkg_field_table: str, optional
    :param gpkg_field_name_column: name of column within the field-table which contains the field names, defaults to None
    :type gpkg_field_name_column: str, optional
    """
    DATA_DIRECTORY = 'data'

    ERR_PRJPATH = 'Provided path does not exist!'
    ERR_GPKGFILE = 'No .gpkg file in provided project path!'
    ERR_FLDNOTAVLBL = 'Provided field-name not available in the GeoPackage!'
    ERR_DIDNOTAVLBL = 'Provided data-source-id not available or provided yet!'
    
    WRN_ADDDATA = 'Error when adding data from data source `{dsrc}` '
    WRN_ADDDATA += 'at field `{fld}` - no entry inserted in `data_available`'
    WRN_ADDDATA += '-table for this case!'

    def __init__(
            self, project_path:str, gpkg_field_table:str=None, 
            gpkg_field_name_column:str=None
        ):
        if not os.path.exists(project_path):
            raise ValueError(self.ERR_PRJPATH)
        self._pp:str = project_path
        
        # database setup
        self._db:ProjectDatabase = None
        self._flds:gpd.GeoDataFrame = None
        for file in os.listdir(project_path):
            if '.gpkg' in file and not file.split('.gpkg')[-1]:
                fgpkg = os.path.join(project_path, file)
                self._db = ProjectDatabase(
                    fgpkg, field_table=gpkg_field_table, 
                    field_name_column=gpkg_field_name_column
                )
                self._flds = gpd.read_file(
                    fgpkg, layer=self._db.field_table_name
                )
        if self._db is None:
            raise ValueError(self.ERR_GPKGFILE)
        
        # check if data path exists
        dpath = os.path.join(self._pp, self.DATA_DIRECTORY)
        if not os.path.exists(dpath):
            os.mkdir(dpath)

        # load previously added interfaces
        self._dis:dict = {}
        dis = self._db.get_data_sources()
        for tpl in dis.itertuples():
            di = getattr(import_module(tpl.intf_module), tpl.intf_name)()
            di.project_directory = self._pp
            self._dis[di.DATA_SOURCE_ID] = di

    @property
    def database(self) -> ProjectDatabase:
        """
        :return: prject database
        :rtype: mef_agri.data.db.ProjectDatabase
        """
        return self._db
    
    @property
    def fields(self) -> gpd.GeoDataFrame:
        """
        :return: table from geopackage containing field geometries
        :rtype: geopandas.GeoDataFrame
        """
        return self._flds
    
    @property
    def interfaces(self) -> dict:
        """
        :return: dictionaray containing references to interface-objects (keys are the data-source-idas)
        :rtype: dict
        """
        return self._dis
    
    @property
    def project_path(self) -> str:
        """
        :return: absolute path to the project folder
        :rtype: str
        """
        return self._pp
    
    def add_data_interface(self, data_intf) -> None:
        """
        Method to add a :class:`mef_agri.data.interface.DataInterface` to the 
        project.

        :param data_intf: the desired :class:`mef_agri.data.interface.DataInterface`-subclass
        :type data_intf: class or instance
        """
        # if data interface already has been added, the check in the project 
        # database should yield True and nothing is left to do
        if self._db.check_data_source(data_intf.DATA_SOURCE_ID):
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
        data_intf.project_directory = self._pp
        # add data interface to the dict containing all interfaces of the prj
        self._dis[data_intf.DATA_SOURCE_ID] = data_intf
        # insert data interface information to the project database
        self._db.insert_data_source(
            data_intf.DATA_SOURCE_ID, data_intf.__class__.__name__,
            data_intf.__class__.__module__
        )
        # commit changes to gpkg/sqlite database
        self._db._conn.commit()

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
        dids, fields = self._get_dis_flds(dids, fields)
        for did in dids:
            di = self._dis[did]
            dspath = os.path.join(self.DATA_DIRECTORY, di.DATA_SOURCE_ID)
            for field in fields:
                fpath = os.path.join(dspath, field)
                if not os.path.exists(os.path.join(self._pp, fpath)):
                    os.mkdir(os.path.join(self._pp, fpath))
                
                print(did + ' -> ' + field)

                # add data into project directory structure
                aoi = self._flds[
                    self._flds[self._db.field_name_column] == field
                ]
                di.save_directory = fpath
                trngs_requ, trngs = self._db.get_date_ranges(
                    did, field, tstart, tstop
                )
                prc_ok = True
                trngs_requ_upd = []
                for trng in trngs_requ:
                    # when there is an error, nothing is inserted into the .gpkg 
                    # user has to manually delete the corresponding folders and 
                    # files or manually update data_available-information in .gpkg
                    try:
                        trng_upd = di.add_prj_data(aoi, trng[0], trng[1])
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

    def get_data(self, epoch:date, dids=None, fields=None) -> dict:
        """
        Method to load raster data from GeoTIFFs of different data sources in 
        the project folder structure.
        The first level of keys of the resulting dictionary corresponds to the 
        field names specified in `fields` and the second level of keys 
        corresponds to the data-source-ids specified in `dids`. 
        The values of the second level are again dictionaries containing the 
        :class:`mef_agri.utils.raster.GeoRaster` instances or ``None`` if there
        is nothing available for the provided arguments.

        :param epoch: date of the data
        :type epoch: datetime.date
        :param dids: data-source-ids available in the corresponding data-interface, defaults to None
        :type dids: list[str] or str, optional
        :param fields: name of the fields, defaults to None
        :type fields: list[str] or str, optional
        :return: dictionary with ``GeoRaster`` instances
        :rtype: dict
        """
        dids, fields = self._get_dis_flds(dids, fields)

        ret = {}
        for field in fields:
            ret[field] = {}
            for did in dids:
                di = self._dis[did]
                di.save_directory = os.path.join(
                    self.DATA_DIRECTORY, di.DATA_SOURCE_ID, field
                )
                ret[field][di.DATA_SOURCE_ID] = di.get_prj_data(epoch)
                
        return ret
    
    def get_field_geodata(self, field_name:str) -> tuple:
        """
        Get the information about georeference of a field as 
        `shapely.geometry.Polygon` and EPSG code of the coordinate reference 
        system.

        :param field_name: name of the field
        :type field_name: str
        :return: polygon and crs
        :rtype: tuple
        """
        crs = self._flds.crs.to_epsg()
        ix = self._flds[self._db.field_name_column] == field_name
        plgn = self._flds[ix]['geometry'].values[0]
        return plgn, crs
    
    def _get_dis_flds(self, dids, fields) -> tuple[list, list]:
        # if no specific data source is specified, data from all data sources 
        # will be added for specified fields
        all_dids = list(self._dis.keys())
        if dids is None:
            dids = all_dids
        elif isinstance(dids, str):
            dids = [dids]
        for did in dids:
            if not did in all_dids:
                # commit db-changes made within this loop before erroneous data
                # source id appears
                self.quit_project()
                raise ValueError(self.ERR_DIDNOTAVLBL)

        # if no field is specified, the data will be added for all fields
        all_fields = self._flds[self._db.field_name_column].values.tolist()
        if fields is None:
            fields = all_fields
        elif isinstance(fields, str):
            fields = [fields]
        for field in fields:
            if not field in all_fields:
                # commit db-changes made within this loop before erroneous field
                # name definition appears
                self.quit_project()
                raise ValueError(self.ERR_FLDNOTAVLBL)

        return dids, fields
    
    def quit_project(self) -> None:
        """
        Method which should be always called in the end of script to make all 
        changes in :class:`mef_agri.data.db.ProjectDatabase` persistent.
        """
        self._db.close()

    @classmethod
    def default_prj(
        cls, project_path:str, gpkg_field_table:str=None, 
        gpkg_field_name_column:str=None
    ):
        """
        classmethod which returns an initialized project containing the 
        following data-sources.

        * TODO planetary_computer - sentinel2
        * TODO geosphere_austria - inca
        * TODO ebod_austria - soil
        * TODO management

        :param project_path: absolute path of folder where project is located in
        :type project_path: str
        :param gpkg_field_table: name of the table within the .gpkg in the project folder which contains the field geometries, defaults to None
        :type gpkg_field_table: str, optional
        :param gpkg_field_name_column: name of column within the field-table which contains the field names, defaults to None
        :type gpkg_field_name_column: str, optional
        """
        from .planetary_computer.sentinel2.interface import Sentinel2Interface
        from .geosphere_austria.inca.interface import INCAInterface
        from .ebod_austria.interface import EbodInterface

        prj = cls(project_path, gpkg_field_table, gpkg_field_name_column)
        prj.add_data_interface(Sentinel2Interface)
        prj.add_data_interface(INCAInterface)
        prj.add_data_interface(EbodInterface)
        # TODO management

        prj.quit_project()

        return cls(project_path)
