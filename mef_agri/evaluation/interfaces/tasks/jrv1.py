from copy import deepcopy

from . import TaskInterface
from ...stats_utils import DISTRIBUTIONS as D
from ....models.base import Quantities as Q
from ....models.utils import Units as U
from ....models.crop.model_epic import Crop_Extended
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


    @TaskInterface.model_quantity
    def sowing_amount(self) -> TaskInterface.MQDefinition:
        """
        :return: biomass of seeds per area
        :rtype: TaskInterface.MQDefinition
        """

    def process_data(self, data, meta):
        self.sowing_amount.qtype = Q.OBS
        self.sowing_amount.model = 'zone.management.sowing'
        self.sowing_amount.unit = U.kg_ha

        epochs, samnt = [], []
        for epoch, zvals in data.items():
            appls = self._get_appls(zvals, meta[epoch])
            for appl in appls:
                epochs.append(epoch)
                if isinstance(appl, SowingAmount):
                    samnt.append(
                        U.convert(
                            appl.amount.value,
                            appl.amount.unit,
                            self.sowing_amount.unit
                        )
                    )
                elif isinstance(appl, SowingDensity):
                    dens = U.convert(
                        appl.density.value, appl.density.unit, U.n_ha
                    )
                    tgw = U.convert(
                        appl.tgw.value, appl.tgw.unit, U.kg
                    )
                    samnt.append(dens * tgw * 1e-3)

                # TODO get default parameters and initial state values 
                # TODO from Crop_Extended and child models by using 
                # TODO appl.crop.value
                # TODO set sowing amount as initial value for biomass and root-biomass

        self.sowing_amount.values = samnt
        self.sowing_amount.epochs = epochs
        self.sowing_amount.distributions = {'distr_id': D.GAMMA_1D, 'std': 5.}


class HarvestInterface(TaskInterface):
    def __init__(self):
        super().__init__()
        self.data_source_id = Harvest.__name__

    @TaskInterface.model_quantity
    def cyield(self) -> TaskInterface.MQDefinition:
        """
        :return: crop yield
        :rtype: TaskInterface.MQDefinition
        """

    @TaskInterface.model_quantity
    def residues_removed(self) -> TaskInterface.MQDefinition:
        """
        :return: fraction of aboveground-biomass minus yield which is removed from the field after harvest
        :rtype: TaskInterface.MQDefinition
        """

    def process_data(self, data, meta):
        self.cyield.qtype = Q.OBS
        self.cyield.model = 'zone.management.harvest'
        self.cyield.unit = U.kg_ha
        self.residues_removed.qtype = Q.OBS
        self.residues_removed.model = 'zone.management.harvest'
        self.residues_removed.unit = U.frac

        epochs, cyield, resrem = [], [], []
        for epoch, zvals in data.items():
            appls = self._get_appls(zvals, meta[epoch])
            for appl in appls:
                epochs.append(epoch)
                cy:CYield = appl
                cyield.append(
                    U.convert(
                        cy.crop_yield.value, 
                        cy.crop_yield.unit, 
                        self.cyield.unit
                    )
                )
                resrem.append(
                    U.convert(
                        cy.residues_removed.value,
                        cy.residues_removed.unit,
                        self.residues_removed.unit
                    )
                )
        self.cyield.values = cyield
        self.cyield.epochs = deepcopy(epochs)
        self.cyield.distributions = {'distr_id': D.GAMMA_1D, 'std': 100.}
        self.residues_removed.values = resrem
        self.residues_removed.epochs = deepcopy(epochs)
        self.residues_removed.distributions = {
            'distr_id': D.TRUNCNORM_1D,
            'std': 0.001,
            'lb': 0.,
            'ub': 1.
        }


class MinFertInterface(TaskInterface):
    def __init__(self):
        super().__init__()
        self.data_source_id = MineralFertilization.__name__

    @TaskInterface.model_quantity
    def NO3_applied(self) -> TaskInterface.MQDefinition:
        """
        :return: applied NO3
        :rtype: TaskInterface.MQDefinition
        """

    @TaskInterface.model_quantity
    def NH4_applied(self) -> TaskInterface.MQDefinition:
        """
        :return: applied NH4
        :rtype: TaskInterface.MQDefinition
        """

    def process_data(self, data, meta):
        self.NO3_applied.qtype = Q.OBS
        self.NO3_applied.model = 'zone.management.fertilization'
        self.NO3_applied.unit = U.kg_ha
        self.NH4_applied.qtype = Q.OBS
        self.NH4_applied.model = 'zone.management.fertilization'
        self.NH4_applied.unit = U.kg_ha

        epochs, no3appl, nh4appl = [], [], []
        for epoch, zvals in data.items():
            appls = self._get_appls(zvals, meta[epoch])
            for appl in appls:
                epochs.append(epoch)
                fa:FertilizerAmount = appl
                fert = Fertilizer.from_json(fa.fertilizer_params)
                fert.name = fa.fertilizer_name
                no3appl.append(
                    U.convert(
                        fa.amount.value * fert.NO3, 
                        fa.amount.unit,
                        self.NO3_applied.unit
                    )
                )
                nh4appl.append(
                    U.convert(
                        fa.amount.value * fert.NH4,
                        fa.amount.unit,
                        self.NH4_applied.unit
                    )
                )

        self.NO3_applied.values = no3appl
        self.NO3_applied.epochs = deepcopy(epochs)
        self.NO3_applied.distributions = {'distr_id': D.GAMMA_1D, 'std': 10.}
        self.NH4_applied.values = nh4appl
        self.NH4_applied.epochs = deepcopy(epochs)
        self.NH4_applied.distributions = {'distr_id': D.GAMMA_1D, 'std': 10.}
