import numpy as np
from datetime import date

from ..eval_def import EvaluationDefinitions
from ...utils.raster import GeoRaster


class EvalInterface(object):
    def __init__(self):
        self._did:str = None
        self._tind:bool = False

    @property
    def eval_defs(self) -> EvaluationDefinitions:
        """
        :return: reference to the instance of `EvaluationDefinitions` which should be processed by the interface
        :rtype: mef_agri.evaluation.eval_def.EvaluationDefinitions
        """
        return self._edefs

    @property
    def data_source_id(self) -> str:
        """
        class variable ``DATA_SOURCE_ID`` of a data interface within the 
        ``mef_agri.data`` module which inherits from 
        :class:`mef_agri.data.interface.DataInterface`.

        :return: id of data source which data will be used within this interface
        :rtype: str
        """
        return self._did
    
    @data_source_id.setter
    def data_source_id(self, val):
        if not isinstance(val, str):
            msg = 'Provided value for `data_source_id` has to be a string ('
            msg += 'class variable ``DATA_SOURCE_ID`` of a data interface from '
            msg += '`mef_agri.data`)!'
            raise ValueError(msg)
        self._did = val
    
    @property
    def time_independent(self) -> bool:
        """
        :return: boolean flag if data is time independent or not (will be used in :func:`mef_agri.evaluation.interfaces.prepare.EvaluationPreparator.prepare`)
        :rtype: bool
        """
        return self._tind
    
    @time_independent.setter
    def time_independent(self, val):
        self._tind = bool(val)

    def process_data(
            self, edefs:EvaluationDefinitions, rasters:dict[GeoRaster], 
            gcs:np.ndarray, epoch:date, zid:str
        ) -> EvaluationDefinitions:
        """
        Within this method, representative values for a zone should be derived 
        from the raster data in the local project folder. 
        ``kwargs`` may contain specific quantities for interfaces which inherit 
        from this base-class.

        :param rasters: dictionary which corresponds to the third level of keys of the resulting dictionary from :func:`ssc_data.project.Project.get_data`
        :type rasters: dict[GeoRaster]
        :param gcs: geo-coordinates representing all raster elements within a zone (see :func:`ssc_csm.zoning.field.Field.zones`)
        :type gcs: numpy.ndarray
        :param epoch: current evaluation epoch (looping all considered epochs is done e.g. in :class:`ssc_csm.data.prepare.OfflinePreparator`)
        :type epoch: datetime.date
        :param zid: zone id (looping of the zones is done e.g. in :class:`ssc_csm.data.prepare.OfflinePreparator`)
        :type zid: str
        :return: updated instance of :class:`ssc_csm.data.eval_def.EvaluationDefinitions`
        :rtype: ssc_csm.data.eval_def.EvaluationDefinitions
        """
        # derive representative values from the GeoRasters for the zone and 
        # set them in the model definitions
        pass

    