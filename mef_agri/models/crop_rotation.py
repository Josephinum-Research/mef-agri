import pandas as pd
from datetime import date
from importlib import import_module


class CropRotation(object):
    """
    This class is responsible for 
    
    * the initialization of the crop models and the connection of its model tree with the model tree of the top-level zone model (at the day of sowing)
    * the disconnnection of the model trees at the day of harvest

    The case of parallel crops (e.g. mixture of catch crops or nurse crops) are 
    not supported for now. There can only be one crop at a time wtihin a zone.
    Otherwise exceptions will be raised.

    """
    def __init__(self, zone):
        from .zone.base import Zone
        from .base import Model

        self._zmdl:Zone = zone
        self._cmdl:Model = None
        self._data:pd.DataFrame = None
        self._crpprs:bool = False
        self._crpsow:bool = False
        self._sowd:date = None

    @property
    def crop_present(self) -> bool:
        """
        :return: indicator if crop is present in the currently considedered zone of a field
        :rtype: bool
        """
        return self._crpprs
    
    @property
    def crop_sown(self) -> bool:
        """
        :return: indicator if crop is sown at current day (i.e. only ``True`` on the date of sowing)
        :rtype: bool
        """
        return self._crpsow
    
    @property
    def current_crop(self):
        """
        :return: crop model of currently present crop
        :rtype: mef_agri.models.base.Model
        """
        return self._cmdl

    def add_data(self, data:pd.DataFrame) -> None:
        """
        add a crop/vegetation period to the crop rotation. The provided 
        dataframe has to contain the following columns

        * ``cmodel`` - name of the class representing the crop model (``str``)
        * ``cmodel_module`` - module containing ``cmodel`` (``str``)
        * ``epoch_start`` - day of sowing (``datetime.date``)
        * ``epoch_end`` - day of harvest (``datetime.date``)

        :func:`mef_agri.evaluation.db.EvaluationDB.get_crop_rotation` returns 
        the appropriate ``pandas.DataFrame``.

        :param data: dataframe with columns ``cmodel``, ``cmodel_module``, ``epoch_start``, ``epoch_end``
        :type data: pandas.DataFrame
        """
        if self._data is None:
            self._data = data
        else:
            self._data = pd.concat([self._data, data], ignore_index=True)

    def update(self, epoch:date) -> None:
        """
        Is called in the update method of the top-level zone model of a 
        model-tree.

        :param epoch: current evaluation epoch
        :type epoch: datetime.date
        """
        self._crpsow = False

        episo = epoch.isoformat()
        if episo in self._data['epoch_start'].values:
            if (self._sowd is not None) or (self._crpprs):
                msg = 'There can only be one crop at a time within a zone!'
                raise ValueError(msg)
            self._crpsow = True
            self._sowd = date.fromisoformat(episo)

            epdata = self._data[self._data['epoch_start'] == episo]
            # import corresponding crop model and initialize it
            self._cmdl = getattr(
                import_module(epdata['cmodel_module'].values[0]), 
                epdata['cmodel'].values[0]
            )()
            self._cmdl.model_tree.n_particles = \
                self._zmdl.model_tree.n_particles
            self._zmdl.model_tree.connnect(self._cmdl.model_tree)
        elif self._sowd is not None:
            # imply that crop is present at day after sowing 
            # to correctly trigger updates in the crop model
            self._crpprs = True
            self._sowd = None
        elif episo in self._data['epoch_end'].values:
            self._zmdl.model_tree.disconnect(self._cmdl.model_tree)
            self._crpprs = False
