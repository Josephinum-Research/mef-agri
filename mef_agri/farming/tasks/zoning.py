from datetime import date

from .task import Task


class Zoning(Task):
    """
    Inherits from :class:`mef_agri.farming.tasks.task.Task`.

    This class is used when zoning information within a field is manually 
    provided. The :func:`raster` attribute hast to contain a layer for each 
    zone, where pixels within a zone are indicated with ones and the remaining 
    pixels contain zeros (float values). Thus, each layer can be used as mask 
    in the following way ``arr[np.where(zoning_task['layer_id'])] = value``.

    :func:`date_begin` indicates the start of the validity of the zoning and 
    :func:`valid_until` indicates its end.
    """
    def __init__(self):
        super().__init__()
        self._meta['valid_until'] = None

    @property
    def valid_until(self) -> date:
        """
        :return: epoch until when the zoning is valid
        :rtype: datetime.date
        """
        return date.fromisoformat(self._meta['valid_until'])
    
    @valid_until.setter
    def valid_until(self, val):
        self._meta['valid_until'] = self._check_date(val).isoformat()
