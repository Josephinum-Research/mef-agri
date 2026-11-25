from .. import Interface
from ....data.ebod_austria.ebod import EBOD_DATA
from ....data.ebod_austria.interface import EbodInterface
from ....models.base import Quantities as Q
from ....models.utils import Units as U
from ...zoning.utils import ZONE_VALUE_METHOD
from ...stats_utils import DISTRIBUTIONS as D
from .ptfs import (
    OENORM_L1050, HYDRCOND_TOTH2015, ROOTDEPTH_NESTROY11, BULKDENS_HOLLIS12
)


class EBod2SWAT_V1(Interface):
    def __init__(self):
        super().__init__()
        self.data_source_id = EbodInterface.data_source_id
        self.zone_value_method = ZONE_VALUE_METHOD.MODE
        self.add_quantity('clay_content', Q.PARAM, U.frac, model_hints='soil')
        self.add_quantity('sand_content', Q.PARAM, U.frac, model_hints='soil')
        self.add_quantity(
            'hydraulic_conductivity_sat', Q.PARAM, U.mm_day, model_hints='soil'
        )
        self.add_quantity('rooting_depth_max', Q.PARAM, U.m, model_hints='soil')
        self.add_quantity('bulk_density', Q.PARAM, U.g_cm3, model_hints='soil')
        self.add_quantity(
            'decomposition_res_opt', Q.PARAM, U.per_day, model_hints='soil'
        )
        self.add_quantity(
            'mineralization_opt', Q.PARAM, U.per_day, model_hints='soil'
        )

        # set values for decomposition_res_opt and mineralization_opt
        odec = self.get_qdef('decomposition_res_opt', Q.PARAM)
        odec.values.append(0.05)
        odec.distributions = {
            'distr_id': D.TRUNCNORM_1D, 'std': 0.003, 'lb': 0.0, 'ub': 1.0
        }
        self.set_qdef('decomposition_res_opt', Q.PARAM, odec)

        omin = self.get_qdef('mineralization_opt', Q.PARAM)
        omin.values.append(1e-4)
        omin.distributions = {
            'distr_id': D.TRUNCNORM_1D, 'std': 3e-5, 'lb':0.0, 'ub': 1.0
        }
        self.set_qdef('mineralization_opt', Q.PARAM, omin)

    def process_data(self, data):
        zvals = list(data.values())[0]
        sv = zvals[EBOD_DATA.SOIL_VAR]
        sb = zvals[EBOD_DATA.SOIL_BASE]
        hv = zvals[EBOD_DATA.HUMUS_VAL]
        
        cc = self.get_qdef('clay_content', Q.PARAM)
        cc.values.append(OENORM_L1050.get_clay_content(sv))
        cc.distributions = {
            'distr_id': D.TRUNCNORM_1D, 'lb': 0., 'ub': 1.,
            'std': OENORM_L1050.get_clay_content_std(sv)
        }
        self.set_qdef('clay_content', Q.PARAM, cc)

        sc = self.get_qdef('sand_content', Q.PARAM)
        sc.values.append(OENORM_L1050.get_sand_content(sv))
        sc.distributions = {
            'distr_id': D.TRUNCNORM_1D, 'lb': 0., 'ub': 1.,
            'std': OENORM_L1050.get_sand_content_std(sv)
        }
        self.set_qdef('sand_content', Q.PARAM, sc)

        hc = self.get_qdef('hydraulic_conductivity_sat', Q.PARAM)
        hc.values.append(HYDRCOND_TOTH2015.get_hydrcond_satsoil(sv))
        hc.distributions = {
            'distr_id': D.GAMMA_1D, 
            'std': HYDRCOND_TOTH2015.get_hydrcond_satsoil_std(sv)
        }
        self.set_qdef('hydraulic_conductivity_sat', Q.PARAM, hc)

        rd = self.get_qdef('rooting_depth_max', Q.PARAM)
        rd.values.append(ROOTDEPTH_NESTROY11.get_rootdepth(sb))
        rd.distributions = {
            'distr_id': D.GAMMA_1D, 
            'std': ROOTDEPTH_NESTROY11.get_rootdepth_std(sb)
        }
        self.set_qdef('rooting_depth_max', Q.PARAM, rd)

        bd = self.get_qdef('bulk_density', Q.PARAM)
        bd.values.append(BULKDENS_HOLLIS12.get_bulk_density(sv, hv))
        bd.distributions = {
            'distr_id': D.GAMMA_1D, 
            'std': BULKDENS_HOLLIS12.get_bulk_density_std()
        }
        self.set_qdef('bulk_density', Q.PARAM, bd)
