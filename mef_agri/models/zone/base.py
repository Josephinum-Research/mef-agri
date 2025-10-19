import numpy as np

from ..base import Model
from ..crop_rotation import CropRotation


class Zone(Model):
    """
    Base class for zones.
    The zone model represents the top-level-model in a model-tree
    """
    def __init__(self) -> None:
        super().__init__(model_name='zone')

        self._gcs:np.ndarray = None
        self._lat:float = None
        self._height:float = None
        self._cropr:CropRotation = CropRotation(self)

        # logging
        self._log = ''

        # flag if crop is present/sown/planted
        self._crop:bool = False

    @property
    def crop_rotation(self):
        """
        :return: crop rotation object
        :rtype: mef_agri.models.crop_rotation.CropRotation
        """
        return self._cropr
    
    @property
    def gcs(self) -> np.ndarray:
        """
        :return: geo-coordinates of elements from which zone is composed of (e.g. squares/pixels)
        :rtype: numpy.ndarray
        """
        return self._gcs
    
    @gcs.setter
    def gcs(self, values:np.ndarray):
        self._gcs = values
    
    @property
    def latitude(self) -> float:
        """
        :return: mean latitude of the zone [rad]
        :rtype: float
        """
        return self._lat
    
    @latitude.setter
    def latitude(self, value:float):
        self._lat = value
    
    @property
    def height(self) -> float:
        """
        :return: mean height/altitude of the zone [m]
        :rtype: float
        """
        return self._height
    
    @height.setter
    def height(self, value:float):
        self._height = value

    def update(self, epoch):
        super().update(epoch)
        self._cropr.update(epoch)
