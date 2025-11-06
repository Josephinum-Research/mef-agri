import numpy as np
from datetime import date

from ..base import EvalInterface
from ....data.jr_management.interface import ManagementInterface
from ....farming.tasks.sowing import Sowing
from ....farming.tasks.harvest import Harvest
from ....farming.tasks.fertilization import MineralFertilization
from ....models.crop.model_epic import Crop_Simple
from ....models.base import Quantities as Q
from ...stats_utils import DISTRIBUTIONS
from ....farming import fertilizers


class Management_JRV01(EvalInterface):
    CROP_MODEL = Crop_Simple
    DEFAULT_CNRATIO = 60.  # approximate valid for wheat/barley, maize, soybean

    def __init__(self):
        super().__init__()
        self.data_source_id = ManagementInterface.DATA_SOURCE_ID
        self._crbeg, self._cult, self._crop = None, None, None

    def process_data(self, edefs, rasters, gcs, epoch, zid):
        if rasters is None:
            return edefs
        for tname, task in rasters.items():
            rows, cols = task.get_raster_coordinates_array(*gcs)

            ###########################   SOWING   #############################
            if tname == Sowing.__name__:
                # crop rotation stuff
                self._crbeg = epoch
                self._cult = task.cultivar
                self._crop = task.crop

                # management model stuff
                qinfo = {
                    'epoch': epoch,
                    'value': np.nanmean(
                        task['sowing_amount'][:, rows, cols], axis=1
                    )[0],
                    'distr': {
                        'distr_id': DISTRIBUTIONS.GAMMA_1D,
                        'std': 20.
                    }
                }
                edefs.set_zone_qinfos(
                    zid, Q.OBS, 'zone.management.sowing', 'sowing_amount', qinfo
                )

            ########################   HARVEST   ###############################
            elif tname == Harvest.__name__:
                # crop rotation stuff
                if self._crbeg is None:
                    msg = self.__class__.__name__ + ' - No crop has been sown '
                    msg += 'yet!'
                    raise ValueError(msg)
                cm = self.CROP_MODEL()
                edefs.add_crop_rotation(cm, self._crbeg, epoch)

                # crop model stuff
                if self._cult == 'generic':
                    zipped = zip(cm.model_tree.model_ids, cm.model_tree.models)
                    for smid, submdl in zipped:
                        if hasattr(submdl, 'DEFAULT_PARAM_VALUES'):
                            defpvals = getattr(submdl, 'DEFAULT_PARAM_VALUES')
                            for pname, pinfo in defpvals[self._crop].items():
                                pinfo['epoch'] = self._crbeg
                                edefs.set_crop_qinfos(
                                    cm.__class__.__name__, self._crbeg, Q.PARAM,
                                    smid, pname, pinfo
                                )
                        if hasattr(submdl, 'INITIAL_STATE_VALUES'):
                            initvals = getattr(submdl, 'INITIAL_STATE_VALUES')
                            for pname, pinfo in initvals.items():
                                pinfo['epoch'] = self._crbeg
                                edefs.set_crop_qinfos(
                                    cm.__class__.__name__, self._crbeg, Q.STATE,
                                    smid, pname, pinfo
                                )
                else:
                    raise NotImplementedError()

                # management model stuff
                qrr = {
                    'epoch': epoch, 
                    'value': task.residues_removed,
                    'distr': {
                        'distr_id': DISTRIBUTIONS.TRUNCNORM_1D,
                        'std': 0.01,
                        'lb': 0.0,
                        'ub': 1.0
                    }
                }
                edefs.set_zone_qinfos(
                    zid, Q.OBS, 'zone.management.harvest', 'residues_removed', 
                    qrr
                )
                qcn = {
                    'epoch': epoch,
                    'value': self.DEFAULT_CNRATIO,
                    'distr': {
                        'distr_id': DISTRIBUTIONS.GAMMA_1D,
                        'std': 10.
                    }
                }
                edefs.set_zone_qinfos(
                    zid, Q.OBS, 'zone.management.harvest', 'cn_ratio_res', qcn
                )
                yfm = np.nanmean(task['yield_fm'][:, rows, cols], axis=1)[0]
                moist = np.nanmean(task['moisture'][:, rows, cols], axis=1)[0]
                qy = {
                    'epoch': epoch,
                    'value': yfm * (1. - moist),
                    'distr': {
                        'distr_id': DISTRIBUTIONS.GAMMA_1D,
                        'std': 100.
                    }
                }
                edefs.set_zone_qinfos(
                    zid, Q.OBS, 'zone.management.harvest', 'cyield', qy
                )

            ###############   MINERAL FERTILIZATION   ##########################
            elif tname == MineralFertilization.__name__:
                fert:fertilizers.Fertilizer = getattr(fertilizers, task.fertilizer)
                amnt = np.nanmean(
                    task['fertilizer_amount'][:, rows, cols], axis=1
                )[0]
                qno3 = {
                    'epoch': epoch,
                    'value': fert.NO3 * amnt,
                    'distr': {
                        'distr_id': DISTRIBUTIONS.GAMMA_1D,
                        'std': 10.0
                    }
                }
                edefs.set_zone_qinfos(
                    zid, Q.OBS, 'zone.management.fertilization', 'NO3_applied', 
                    qno3
                )
                qnh4 = {
                    'epoch': epoch,
                    'value': fert.NH4 * amnt,
                    'distr': {
                        'distr_id': DISTRIBUTIONS.GAMMA_1D,
                        'std': 10.0
                    }
                }
                edefs.set_zone_qinfos(
                    zid, Q.OBS, 'zone.management.fertilization', 'NH4_applied',
                    qnh4
                )

        return edefs
