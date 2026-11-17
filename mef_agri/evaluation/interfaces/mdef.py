import numpy as np
from inspect import isclass

from . import Interface
from ..db import EvalDB
from ...models.base import Model, Quantities as Q
from ...models.utils import Units


class ModelDefinition(object):
    def __init__(self, zone_model:Model):
        self._qdefs:dict = {}
        self._mtree:dict = {}
        if isclass(zone_model):
            zone_model = zone_model()
        self._mtree['zone'] = self._create_model_def(
            zone_model, self._mtree['zone']
        )

    @property
    def model_quantities(self) -> dict:
        """
        dictionary with following structure

        .. code-block::

            {
                'zone': {
                    'model1': {
                        'state': {
                            'state1': {
                                'values': [...],
                                'epochs': [...],
                                'distr': [{...}, ...],
                                'unit': ....
                            },
                            ...
                        }
                        'parameter': {
                            'param1': {
                                'values': [...],
                                'epochs': [...],
                                'distr': [{...}, ...],
                                'unit': ....
                            },
                            ...
                        },
                        'observation': {
                            'obs1': {
                                'values': [...],
                                'epochs': [...],
                                'distr': [{...}, ...],
                                'unit': ....
                            }
                        }
                    },
                    'model1.model2': {...},
                }
                'crop-1': {
                    'model1': {...},
                    ...
                },
                'crop-2': {...},
                ...
            }

        :return: dictionary containing the quantity definitions for each model
        :rtype: dict
        """
        return self._qdefs

    @property
    def model_tree(self) -> dict:
        """
        dictionary with the following structure

        .. code-block::

            {
                'zone': {
                    'model1': {
                        'model1.model2': {
                            'model1.model2.model4': {},
                            'model1.model2.model5': {
                                'model1.model2.model5.model6: {...},
                                'mdoel1.model2.model5.model7: {}
                            }
                        },
                        'model1.model3': {}
                    },
                    'model10': {...}
                },
                'crop-1': {
                    'model1': {...}
                },
                'crop-2': {...},
                ...
            }

        i.e. empty dictionaries at the leaves of the model-tree (no child 
        models present).

        :return: dictionary representing the model-tree/hierarchy
        :rtype: dict
        """
        return self._mtree

    def add_crop_model(self, crop_model:Model, mid:str):
        if isclass(crop_model):
            crop_model = crop_model()
        self._mtree[mid] = self._create_model_def(crop_model, self._mtree[mid])

    def set_from_interfaces(self, intfs:list[Interface]):
        for intf in intfs:
            for qname in intf.model_quantities:
                iqdef:Interface.MQDefinition = getattr(intf, qname)
                mqdef = self._qdefs[iqdef.model][iqdef.qtype][qname]
                mqdef['values'] = Units.convert(np.array(iqdef.values)).tolist()
                mqdef['epochs'] = iqdef.epochs
                mqdef['distr'] = iqdef.distributions
                self._qdefs[iqdef.model][iqdef.qtype][qname] = mqdef

    def get_insert_commands(self, edb:EvalDB, eid:int) -> list[str]:
        cmds = []
        for mid, mqs in self._qdefs.items():
            for qtype, qs in mqs.items():
                for qname, qdef in qs.items():
                    if None in qdef.values():
                        continue

                    nvals = len(qdef['values'])
                    ins_data = dict(
                        eid=[eid for i in range(nvals)],
                        name=[qname for i in range(nvals)],
                        model=[mid for i in range(nvals)],
                        epoch=qdef['epochs'],
                        value=qdef['values'],
                        distr=qdef['distr']
                    )
                    if qtype == Q.STATE:
                        get_sql = edb.schema.table_states_def.sql_insert
                    elif qtype == Q.PARAM:
                        get_sql = edb.schema.table_params_def.sql_insert
                    elif qtype == Q.OBS:
                        get_sql = edb.schema.table_obs_def.sql_insert
                    cmds.append(get_sql(**ins_data))
        return cmds

    def _create_model_def(self, model:Model, mtree_parent_dict:dict):
        self._qdefs[model.model_id] = self._get_model_qdefs(model)
        mtree_parent_dict[model.model_id] = {}
        for mid in model.model_tree.tree_dict[model.model_id]['child_ids']:
            mtree_parent_dict[model.model_id] = self._create_model_def(
                model.model_tree.tree_dict[mid]['ref'], 
                mtree_parent_dict[model.model_id]
            )
        return mtree_parent_dict

    def _get_model_qdefs(self, model:Model) -> dict:
        ret = {}
        for qname, qcont in model.qdefs.items():
            if qcont['qtype'] in (Q.ROUT, Q.DOUT):
                continue
            if not qcont['qtype'] in ret.keys():
                ret[qcont['qtype']] = {}
            ret[qcont['qtype']][qname] = {
                'values': None,
                'epochs': None,
                'distr': None,
                'unit': qcont['unit'],
            }
        return ret
