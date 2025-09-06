import pandas as pd
from datetime import date
from importlib import import_module


class CropRotation(object):
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
        return self._crpprs
    
    @property
    def crop_sown(self) -> bool:
        return self._crpsow
    
    @property
    def current_crop(self):
        return self._cmdl

    def add_data(self, data:pd.DataFrame) -> None:
        if self._data is None:
            self._data = data
        else:
            # TODO stack dataframes
            # TODO check incoming dataframe for redundancies
            raise NotImplementedError()

    def update(self, epoch:date) -> None:
        self._crpsow = False

        episo = epoch.isoformat()
        if episo in self._data['epoch_start'].values:
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
