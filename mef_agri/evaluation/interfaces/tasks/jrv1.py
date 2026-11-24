from copy import deepcopy

from . import TaskInterface
from ...stats_utils import DISTRIBUTIONS as D
from ....models.base import Model, Quantities as Q
from ....models.utils import Units as U
from ....farming.crops import Crops
from ....farming.fertilizers import Fertilizer
from ....farming.tasks.sowing import Sowing, SowingAmount, SowingDensity
from ....farming.tasks.harvest import Harvest, CYield
from ....farming.tasks.fertilization import (
    MineralFertilization, FertilizerAmount
)


class SowingInterface(TaskInterface):
    def __init__(self):
        super().__init__()
        self.data_source_id = Sowing.__name__
        self.add_quantity(
            'sowing_amount', Q.OBS, U.kg_ha, 
            model_hints=['zone', 'management', 'sowing']
        )

    def _process_appl(self, appl, meta, epoch):
        if isinstance(appl, SowingAmount):
            samnt = U.convert(appl.amount.value, appl.amount.unit, U.kg_ha)
        elif isinstance(appl, SowingDensity):
            dens = U.convert(appl.density.value, appl.density.unit, U.n_ha)
            tgw = U.convert(appl.tgw.value, appl.tgw.unit, U.kg)
            samnt = dens * tgw
        sa = self.get_qdef('sowing_amount', Q.OBS)
        sa.values.append(samnt)
        sa.epochs.append(epoch)
        if not sa.distributions:
            sa.distributions = {'distr_id': D.GAMMA_1D, 'std': 5.}
        self.set_qdef('sowing_amount', Q.OBS, sa)

        cmodel:Model = meta[self.META_KEY_CMODEL]
        cmid = self.crop_model_id(
            appl.crop.value, appl.cultivar.value, epoch, add=True
        )
        for mid in cmodel.model_tree.model_ids_intern:
            mdl = cmodel.model_tree.get_model(mid)
            if hasattr(mdl, 'DEFAULT_PARAM_VALUES'):
                self._loop_mqs(
                    mdl, cmid, mid, epoch, 
                    mdl.DEFAULT_PARAM_VALUES[appl.crop.value]
                )
            if hasattr(mdl, 'INITIAL_STATE_VALUES'):
                self._loop_mqs(
                    mdl, cmid, mid, epoch,
                    mdl.INITIAL_STATE_VALUES[appl.crop.value]
                )

    def _loop_mqs(self, model:Model, cmid:str, mid:str, epoch:str, mqdict:dict):
        for pname, dpdef in mqdict.items():
            self.add_quantity(
                pname, 
                Q.PARAM, 
                model.model_tree.get_qinfos(pname, mid
                )['unit'],
                model_id=mid,
                cmid=cmid
            )
            pdef = self.get_qdef(pname, Q.PARAM, cmid=cmid)
            pdef.values.append(dpdef['value'])
            pdef.epochs.append(epoch)
            pdef.distributions = dpdef['distr']
            self.set_qdef(pname, Q.PARAM, pdef, cmid=cmid)


class HarvestInterface(TaskInterface):
    def __init__(self):
        super().__init__()
        self.data_source_id = Harvest.__name__
        self.add_quantity(
            'cyield', Q.OBS, U.kg_ha, model_hints=['management', 'harvest']
        )
        self.add_quantity(
            'residues_removed', Q.OBS, U.frac, 
            model_hints=['management', 'harvest']
        )

    def _process_appl(self, appl:CYield, meta, epoch):
        cy = self.get_qdef('cyield', Q.OBS)
        cy.values.append(
            U.convert(appl.crop_yield.value, appl.crop_yield.unit, U.kg_ha)
        )
        cy.epochs.append(epoch)
        ystd = None
        if meta[self.META_KEY_CROP] == Crops.Maize.NAME:
            ystd = 150.
        elif meta[self.META_KEY_CROP] == Crops.Soybean.NAME:
            ystd = 30.
        elif meta[self.META_KEY_CROP] == Crops.WinterWheat.NAME:
            ystd = 50.
        elif meta[self.META_KEY_CROP] == Crops.WinterBarley.NAME:
            ystd = 50.
        cy.distributions.append({'distr_id': D.GAMMA_1D, 'std': ystd})
        self.set_qdef('cyield', Q.OBS, cy)

        rr = self.get_qdef('residues_removed', Q.OBS)
        rr.values.append(
            U.convert(
                appl.residues_removed.value, appl.residues_removed.unit, U.frac
            )
        )
        rr.epochs.append(epoch)
        rr.distributions.append({
            'distr_id': D.TRUNCNORM_1D, 'std': 0.001, 'lb': 0., 'ub': 1.
        })
        self.set_qdef('residues_removed', Q.OBS, rr)


class MinFertInterface(TaskInterface):
    def __init__(self):
        super().__init__()
        self.data_source_id = MineralFertilization.__name__
        self.add_quantity(
            'NO3_applied', Q.OBS, U.kg_ha, 
            model_hints=['management', 'fertilization']
        )
        self.add_quantity(
            'NH4_applied', Q.OBS, U.kg_ha,
            model_hints=['management', 'fertilization']
        )

    def _process_appl(self, appl:FertilizerAmount, meta, epoch):
        fert = Fertilizer.from_json(appl.fertilizer_params)
        fert.name = appl.fertilizer_name
        no3 = self.get_qdef('NO3_applied', Q.OBS)
        no3.epochs.append(epoch)
        no3.values.append(
            U.convert(appl.amount.value * fert.NO3, appl.amount.unit, U.kg_ha)
        )
        no3.distributions.append({'distr_id': D.GAMMA_1D, 'std': 10.})
        self.set_qdef('NO3_applied', Q.OBS, no3)
        nh4 = self.get_qdef('NH4_applied', Q.OBS)
        nh4.epochs.append(epoch)
        nh4.values.append(
            U.convert(appl.amount.value * fert.NH4, appl.amount.unit, U.kg_ha)
        )
        nh4.distributions.append({'distr_id': D.GAMMA_1D, 'std': 10.})
        self.set_qdef('NH4_applied', Q.OBS, nh4)
