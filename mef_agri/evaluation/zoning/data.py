import os
import datetime as dt
from shutil import rmtree

from . import Zoning
from ...utils.misc import check_date


class ZoningData(object):
    """
    Class which handles saving and getting zoning-georasters.
    """
    ZONING_DIRECTORY = 'zoning'

    def __init__(self, prjdir:str):
        self._zdir:str = os.path.join(prjdir, self.ZONING_DIRECTORY)
        if not os.path.exists(self._zdir):
            os.mkdir(self._zdir)

    def _get_field_dir(self, fname:str) -> str:
        fdir = os.path.join(self._zidr, fname)
        if not os.path.exists(fdir):
            os.mkdir(fdir)
        return fdir

    def create_zoning_id(self, field_name:str, zoning:Zoning) -> str:
        """
        Create a unique zoning-id for the provided ``field_name`` based on the 
        content of the zoning-folder within a project.
        The zoning id is composed as 
        ``z_<zoning.valid_from>_<zoning.valid_until>_<numeric-id>``

        :param field_name: name of the field
        :type field_name: str
        :param zoning: zoning georaster
        :type zoning: mef_agri.evaluation.zoning.Zoning
        :return: unique (with respect to the field) zoning id
        :rtype: str
        """
        fdir = os.path.join(self._zidr, field_name)
        zi = 1
        zid_prefix = 'z_' + zoning.valid_from.isoformat() + '_'
        zid_prefix = zoning.valid_until.isoformat() + '_'
        while True:
            if zi < 10:
                zid = zid_prefix + '0' + str(zi)
            else:
                zid = zid_prefix + str(zi)
            zfdir = os.path.join(fdir, zid)
            if os.path.exists(zfdir):
                zi += 1
            else:
                os.mkdir(zfdir)
                break
        return zid

    def save_zoning(self, field_name:str, zoning:Zoning, zid:str|None=None):
        """
        Save zoning-georaster to project folder structure.

        :param field_name: name of the field
        :type field_name: str
        :param zoning: zoning georaster
        :type zoning: Zoning
        :param zid: zoning-id which should be used to save zoning georaster, defaults to None
        :type zid: str | None, optional
        """
        if zid is None:
            zid = self.create_zoning_id(field_name, zoning)
        zdir = os.path.join(self._zdir, field_name, zid)
        zoning.save_geotiff(zdir, compress=True, filename='zoning')

    def get_zoning_by_id(self, field_name:str, zid:str) -> Zoning:
        """
        Get zoning georaster by provided zoning-id (i.e. name of folder 
        containing the zoning georaster).

        :param field_name: name of the field
        :type field_name: str
        :param zid: zoning-id
        :type zid: str
        :return: zoning georaster
        :rtype: Zoning
        """
        fdir = self._get_field_dir(field_name)
        zdir = os.path.join(fdir, zid)
        return Zoning.from_directory(zdir)
    
    def get_zonings(self, field_name:str, epoch:dt.date|str) -> list[Zoning]:
        """
        Get zoning georasters which are valid for the provided ``epoch``.

        :param field_name: name of the field
        :type field_name: str
        :param epoch: epoch for which valid zonings should be returned
        :type epoch: dt.date | str
        :return: list of zoning georasters
        :rtype: list[Zoning]
        """
        fdir = self._get_field_dir(field_name)
        ret = []
        epoch = check_date(epoch)
        for zid in os.listdir(fdir):
            try:
                zid_parts = zid.split('_')
                e1 = dt.date.fromisoformat(zid_parts[1])
                e2 = dt.date.fromisoformat(zid_parts[2])
                if e1 <= epoch <= e2:
                    ret.append(self.get_zoning_by_id(field_name, zid))
            except:
                continue
        return ret
    
    def delete_zoning(self, field_name:str, zid:str):
        """
        Delete zoning georaster by provided zoning-id

        :param field_name: name of the field
        :type field_name: str
        :param zid: zoning-id
        :type zid: str
        """
        fdir = self._get_field_dir(field_name)
        zdir = os.path.join(fdir, zid)
        rmtree(zdir)
