import os
import json
import numpy as np
from inspect import isclass
from datetime import date
from copy import deepcopy
from importlib import import_module

from ..models.base import Model, Quantities as Q
from ..models.tree import ModelTree
from .zoning.field import Field


def dict_from_model(model:Model) -> dict:
    """
    Create a dictionary which contains all definitions of the model quantities 
    (states, observations, parameters, parameter-functions). 
    This method loops the whole model-tree.

    :param model: model instance
    :type model: Model
    :return: dictionary with model quantity informations
    :rtype: dict
    """
    ret = {Q.STATE: {}, Q.PARAM: {}, Q.OBS: {}, Q.PFUNC: {}}
    mtree:ModelTree = model.model_tree
    for mid in mtree.model_ids:
        mdl:Model = mtree.tree_dict[mid]['ref']
        
        # process states
        for st in mdl.state_names:
            if not mid in ret[Q.STATE].keys():
                ret[Q.STATE][mid] = {}
            ret[Q.STATE][mid][st] = {'epoch': None, 'value': None, 
                'distr': {'distr_id': None, 'std': None}}

        # process hyper parameters
        for mp in mdl.parameter_names:
            if not mid in ret[Q.PARAM].keys():
                ret[Q.PARAM][mid] = {}
            ret[Q.PARAM][mid][mp] = {'epoch': None, 'value': None, 
                'distr': {'distr_id': None, 'std': None, 'sample': False}}

        # process observations
        for obs in mdl.observation_names:
            if not mid in ret[Q.OBS].keys():
                ret[Q.OBS][mid] = {}
            ret[Q.OBS][mid][obs] = {'epochs': [], 'values': [],
                'distr': {'distr_id': None, 'std': None, 'sample': False}}

        # process pfunctions
        for pf in mdl.pfunction_names:
            if not mid in ret[Q.PFUNC].keys():
                ret[Q.PFUNC][mid] = {}
            ret[Q.PFUNC][mid][pf] = {
                'epoch': None, 'fdef': {
                    'ftype': None,
                    'values-x': None, 'values-y': None, 
                    'distr-x': {'distr_id': None, 'std': None}, 
                    'distr-y': {'distr_id': None, 'std': None}, 
                    'sample': False
                }
            }

    return ret


class ObsValueList(object):
    """
    Auxiliary class to encode a list with observation values as a line in 
    .json file (i.e. no line breaks after each list element).
    """
    def __init__(self, obs:list[float]) -> None:
        self._l = obs

    def get_list_repr(self) -> str:
        return ''.join(str(val) + ', ' for val in self._l)[:-2]
    
    @staticmethod
    def list_from_repr(lrepr:str) -> list[float]:
        return [float(val) for val in lrepr.split(', ')]
    

class ObsEpochList(object):
    """
    Auxiliary class to encode a list with observation epochs as a line in 
    .json file (i.e. no line breaks after each list element).
    """
    def __init__(self, epochs:list[str]):
        self._l = epochs

    def get_list_repr(self) -> str:
        return ''.join(val + ', ' for val in self._l)[:-2]
    
    @staticmethod
    def list_from_repr(lrepr) -> list[str]:
        return [val for val in lrepr.split(', ')]
    

class GCSCoords(object):
    """
    Auxiliary class to encode a list with x- or y-geo-coordinates as a line in 
    .json file (i.e. no line breaks after each list element).
    """
    def __init__(self, gcs:list):
        self._x = gcs[0]
        self._y = gcs[1]

    def get_list_repr(self) -> dict:
        return {
            'x': ''.join(str(val) + ', ' for val in self._x)[:-2],
            'y': ''.join(str(val) + ', ' for val in self._y)[:-2]
        }
    
    @staticmethod
    def gcs_from_repr(lrepr:dict) -> list:
        return [
            [float(val) for val in lrepr['x'].split(', ')],
            [float(val) for val in lrepr['y'].split(', ')]
        ]
    

class ExtendedJSONEncoder(json.JSONEncoder):
    """
    Extension of JSONEncoder which considers :class:`ObsValueList`, 
    :class:`ObsEpochList` and :class:`GCSCoords` in the json-encoding.
    """
    def default(self, o):
        if isinstance(o, (ObsValueList, ObsEpochList, GCSCoords)):
            return o.get_list_repr()
        return super().default(o)


class EvaluationDefinitions(object):
    """
    Class which is used to set up the 
    :class:`mef_agri.evaluation.db.EvaluationDB` for model evaluation. 
    The information about model quantities and crop rotation 
    definitions is stored in a dictionary which will be saved to a .json file.

    In the zone-model entry, there will be the same dict for each zone in 
    the provided field-instance. 
        
    The assumption is, that the crop model definitions are equal for all 
    zones (i.e. there is no key-level which corresponds to the zone-ids).
        
    Each of the dicts containing the infos about the model definitions is 
    separated according to the four model quantities states, observations, 
    parameters and parameter functions.

    The first level of keys in :func:`defs` is also accessible by directly 
    use square brackets to instances of this class.

    :param zone_model: top-level zone-model which should be evaluated
    :type zone_model: class or instance
    :param field: field-object which holds zoning information, defaults to None
    :type field: sitespecificcultivation.evaluation.zoning.field.Field, optional  # TODO
    """
    EVAL_FOLDER_NAME = 'eval'

    def __init__(self, zone_model):
        if isclass(zone_model):
            zone_model = zone_model()
        
        self._zmdld = dict_from_model(zone_model)
        self._zmdln = zone_model.__class__.__name__
        self._cmdls = []
        self._edefs = {
            'eval_info': None,
            'epoch_start': None,
            'epoch_end': None,
            'field': None,
            'add_info': {
                'crs': None,
                'height': None,
                'zones': {}
            },
            'zmodel_name': self._zmdln,
            'zmodel_module': zone_model.__class__.__module__, 
            'zone_models': {},
            'crop_rotation': [],
        }

    @property
    def defs(self) -> dict:
        """
        :return: dictionary containing all definitions and specifications for evaluation
        :rtype: dict
        """
        return self._edefs
    
    def add_crop_rotation(
            self, crop_model, epoch_start:date, epoch_end:date
        ) -> None:
        """
        Append a dictionary with vegetation period information in the 
        ``'crop_rotation'`` entry of :func:`defs`.

        :param crop_model: crop model
        :type crop_model: class or instance
        :param epoch_start: start of vegetation period
        :type epoch_start: datetime.date
        :param epoch_end: end of vegetation period
        :type epoch_end: datetime.date
        """
        if isclass(crop_model):
            mmodule = crop_model.__module__
            mname = crop_model.__name__
            crop_model = crop_model()
        else:
            mmodule = crop_model.__class__.__module__
            mname = crop_model.__class__.__name__

        self._edefs['crop_rotation'].append({
            'cmodel_name': mname,
            'cmodel_module': mmodule,
            'epoch_start': epoch_start.isoformat(),
            'epoch_end': epoch_end.isoformat(),
            'crop_model': dict_from_model(crop_model)
        })
    
    def set_epoch_start_end(self, epoch_start:date, epoch_end:date) -> None:
        """
        Set start an end epoch in the evaluation definitions dict.

        :param epoch_start: first evaluation epoch
        :type epoch_start: date
        :param epoch_end: last evaluation epoch
        :type epoch_end: date
        """
        self._edefs['epoch_start'] = epoch_start.isoformat()
        self._edefs['epoch_end'] = epoch_end.isoformat()

    def set_zone_states_init_epoch(self, epoch:date) -> None:
        """
        :param epoch: epoch wich corresponds to the intial states
        :type epoch: datetime.date
        """
        if not self._edefs['zone_models']:
            msg = 'No zone-model information available yet - a prior call of '
            msg += '`provide_field_info()` is necessary!'
            raise ValueError(msg)
        for zd in self._edefs['zone_models'].values():
            for md in zd['state'].values():
                for sd in md.values():
                    sd['epoch'] = epoch.isoformat()

    def set_zone_params_init_epoch(self, epoch:date) -> None:
        """
        :param epoch: epoch which corresponds to the hyper-parameters
        :type epoch: datetime.date
        """
        if not self._edefs['zone_models']:
            msg = 'No zone-model information available yet - a prior call of '
            msg += '`provide_field_info()` is necessary!'
            raise ValueError(msg)
        for zd in self._edefs['zone_models'].values():
            for md in zd['parameter'].values():
                for pd in md.values():
                    if pd['epoch'] is None:
                        pd['epoch'] = epoch.isoformat()
    
    def provide_field_info(
            self, fname:str, fcrs:int, fheight:float, zones:dict
        ) -> None:
        """
        Set field informations in the :func:`defs` dictionary. 

        The values in ``zones`` (i.e. again dictionaries) have to contain the 
        keys ``'lat'`` and ``'gcs'``. 

        :param fname: name of the field
        :type fname: str
        :param fcrs: epsg-code in which the field is defined
        :type fcrs: int
        :param fheight: approximate/mean height of the field
        :type fheight: float
        :param zones: dictionary containing zone information (keys have to match the zone ids/names)
        :type zones: dict
        """
        self._edefs['field'] = fname
        self._edefs['add_info']['crs'] = fcrs
        self._edefs['add_info']['height'] = fheight
        for zid, zinfo in zones.items():
            self._edefs['zone_models'][zid] = deepcopy(self._zmdld)
            self._edefs['add_info']['zones'][zid] = {
                'latitude': zinfo['lat'],
                'gcs': zinfo['gcs'].tolist()
            }

    def get_qinfos_from_zone_model(
            self, zone:str, qtype:str, mname:str, qname:str
        ) -> dict:
        """
        Get information about a specific model quantity

        :param zone: zone-id
        :type zone: str
        :param qtype: quantity type
        :type qtype: str
        :param mname: model name
        :type mname: str
        :param qname: quantity name
        :type qname: str
        :return: dictionary containing quantity information
        :rtype: dict
        """
        return self._edefs['zone_models'][zone][qtype][mname][qname]

    def set_zone_qinfos(
            self, zid:str, qtype:str, mname:str, qname:str, qinfo:dict
        ) -> None:
        """
        Set information about model quantities in the :func:`defs` 
        dictionary for the zone models. 
        The assumption is, that the states, parameters and parameter-functions
        are only defined initially (i.e. once at the beginning 
        of an evaluation). Thus, if providing information about the same 
        quantity a second time, it will be overwritten in :func:`defs`.
        Observations are treated differently. The values are gathered in a list 
        (as well as the epochs) and the distribution informations are assumed to 
        be the same for all epochs.

        The ``qinfo`` dictionary should contain the following keys in the case 
        of ``qtype`` being state, observation or parameter

        * `epoch`: datetime.date
        * `value`: float or int
        * `distr`: dict

        The `distr` dictionary should exhibit the structure required for the 
        :class:`mef_agri.evaluation.stats_utils.RVSampler` class.

        In the case of ``qtype`` being a pfunction, the ``qinfo`` dictionary 
        should contain the following keys

        * `epoch`: datetime.date
        * `fdef`: dict

        the `fdef` dictionary should exhibit the structure required for the 
        `mef_agri.models.utils.PFunction` class.

        :param zid: zone id from the field-instance
        :type zid: str
        :param qtype: one attribute out of `sitespecificcultivaiton.evaluation.models.base.QUANTITIES` (except `DETERMINISTIC_OUTPUT` and `RANDOM_OUTPUT`)
        :type qtype: str
        :param mname: name of the model in the model-tree
        :type mname: str
        :param qname: name of the quantity within the model
        :type qname: str
        :param qinfo: dictionary containing necessary information about model quantity
        :type qinfo: dict
        """
        if not self._edefs['zone_models']:
            msg = 'No zone models are available yet!'
            raise ValueError(msg)
        
        mdldefs = self._edefs['zone_models'][zid][qtype]
        self._check_model_defs(mdldefs, mname, qname)
        self._edefs['zone_models'][zid][qtype][mname][qname] = self._set_qinfos(
            mdldefs[mname][qname], qinfo, qtype
        )

    def set_crop_qinfos(
            self, cmname:str, sowing_date:date | str, qtype:str, mname:str, 
            qname:str, qinfo:dict
    ) -> None:
        """
        Set the model quantity information for a specific vegetation period. If 
        ``cmname`` and ``sowing_date`` do not match any entry in the 
        ``crop_rotation`` entry of :func:`defs`, nothing will be updated.

        :param cmname: class-name of the crop model
        :type cmname: str
        :param sowing_date: date of sowing (if ``str``, it has to be in the iso-format)
        :type sowing_date: datetime.date | str
        :param qtype: see :class:`mef_agri.models.base.__QS__`
        :type qtype: str
        :param mname: name of the model in the model-tree
        :type mname: str
        :param qname: name of the quantity within the model
        :type qname: str
        :param qinfo: dictionary containing necessary information about the model quantity
        :type qinfo: dict
        """
        if isinstance(sowing_date, date):
            sowing_date = sowing_date.isoformat()
        
        for i in range(len(self._edefs['crop_rotation'])):
            ccmn = self._edefs['crop_rotation'][i]['cmodel_name'] == cmname
            csd = self._edefs['crop_rotation'][i]['epoch_start'] == sowing_date
            if ccmn and csd:
                mdldefs = self._edefs['crop_rotation'][i]['crop_model'][qtype]
                self._check_model_defs(mdldefs, mname, qname)
                mdldefs[mname][qname] = self._set_qinfos(
                    mdldefs[mname][qname], qinfo, qtype
                )
                self._edefs['crop_rotation'][i]['crop_model'][qtype] = mdldefs


    def _check_model_defs(self, mdldefs:dict, mname:str, qname:str) -> None:
        if not mname in mdldefs.keys():
            raise ValueError('Provided model name not available!')
        if not qname in mdldefs[mname].keys():
            raise ValueError('Provided quantity name not available!')

    def _set_qinfos(self, exqi:dict, qinfo:dict, qtype:str) -> dict:
        if qtype == Q.PFUNC:
            for fkey, fval in qinfo['fdef'].items():
                exqi['fdef'][fkey] = fval
            exqi['epoch'] = qinfo['epoch'].isoformat()
            return exqi
        
        if qtype == Q.OBS:
            exqi['values'].append(qinfo['value'])
            exqi['epochs'].append(qinfo['epoch'].isoformat())
        else:
            exqi['value'] = qinfo['value']
            exqi['epoch'] = qinfo['epoch'].isoformat()
        for dkey, dval in qinfo['distr'].items():
            exqi['distr'][dkey] = dval
        return exqi

    def save(self, save_dir:str, file_name:str) -> None: 
        """
        Save :func:`defs` to the provided location.

        :param save_dir: directory (absolute path)
        :type save_dir: str
        :param file_name: name of the saved file
        :type file_name: str
        """
        def loop_obs(mdict:dict) -> dict:
            for mdldef in mdict[Q.OBS].values():
                for qinfo in mdldef.values():
                    qinfo['values'] = ObsValueList(qinfo['values'])
                    qinfo['epochs'] = ObsEpochList(qinfo['epochs'])
            return mdict

        sd = deepcopy(self._edefs)
        # prepare the zones and zone models to be saved to json
        for zid in sd['zone_models'].keys():
            sd['zone_models'][zid] = loop_obs(sd['zone_models'][zid])
            sd['add_info']['zones'][zid]['gcs'] = GCSCoords(
                sd['add_info']['zones'][zid]['gcs']
            )

        # prepare crop models to be saved to json
        for i in range(len(sd['crop_rotation'])):
            sd['crop_rotation'][i]['crop_model'] = loop_obs(
                sd['crop_rotation'][i]['crop_model']
            )

        # save to json
        if not (file_name[-5:] == '.json'):
            file_name += '.json'
        fio = open(os.path.join(save_dir, file_name), 'w')
        json.dump(sd, fio, indent=2, cls=ExtendedJSONEncoder)
        fio.close()

    def sample_all_params(self) -> None:
        """
        Call this method to indicate, that all parameters should be 
        sampled in the evaluation.
        """
        for zdict in self._edefs['zone_models'].values():
            zdict[Q.PARAM] = self._sample_q(zdict[Q.PARAM])
        for i in range(len(self._edefs['crop_rotation'])):
            self._edefs['crop_rotation'][i]['crop_models'][Q.PARAM] = \
                self._sample_q(
                    self._edefs['crop_rotation'][i]['crop_models'][Q.PARAM]
                )

    def sample_all_obs(self) -> None:
        """
        Call this method to indicate, that all observations should be sampled 
        in the evaluation.
        """
        for zdict in self._edefs['zone_models'].values():
            zdict[Q.OBS] = self._sample_q(zdict[Q.OBS])
        for i in range(len(self._edefs['crop_rotation'])):
            self._edefs['crop_rotation'][i]['crop_models'][Q.OBS] = \
                self._sample_q(
                    self._edefs['crop_rotation'][i]['crop_models'][Q.OBS]
                )

    def sample_all_pfuncs(self) -> None:
        """
        Call this method to indicate, that all pfunctions should be sampled 
        in the evaluation.
        """
        for zdict in self._edefs['zone_models'].values():
            zdict[Q.PFUNC] = self._sample_q(zdict[Q.PFUNC])
        for i in range(len(self._edefs['crop_rotation'])):
            self._edefs['crop_rotation'][i]['crop_models'][Q.PFUNC] = \
                self._sample_q(
                    self._edefs['crop_rotation'][i]['crop_models'][Q.PFUNC]
                )

    def sample_zone_params(self, mid:str, child_models_too:bool=True) -> None:
        """
        Method to indicate, that all parameters within one model in the 
        zone model-tree should be sampled. This is applied to all zones.

        :param mid: id of the model in the model-tree
        :type mid: str
        :param child_models_too: indicate if all parameters of child models should be sampled too, defaults to True
        :type child_models_too: bool, optional
        """
        for zid in self._edefs['zone_models'].keys():
            dzp = self._edefs['zone_models'][zid][Q.PARAM]
            if mid in dzp.keys():
                dzp[mid] = self._sample_q_in_model(dzp[mid])
                self._edefs['zone_models'][zid][Q.PARAM] = dzp
            
            if not child_models_too:
                return
            
            for mdlid, mdldef in dzp.items():
                if mid + '.' in mdlid:
                    dzp[mdlid] = self._sample_q_in_model(mdldef)
            self._edefs['zone_models'][zid][Q.PARAM] = dzp
        

    def sample_crop_params(
            self, mid:str, sowing_date:date | str=None, 
            child_models_too:bool=True
        ) -> None:
        """
        Method to indicate, that all parameters within one model in the 
        crop model-tree should be sampled. If ``sowing_date`` is not specified, 
        this will be applied to all sown crops (as long as ``mid`` appears in 
        the corresponding crop model-tree).

        :param mid: id of the model in the model-tree
        :type mid: str
        :param sowing_date: date of sowing (if it is a string, it should be formatted like ``date.isoformat()``), defaults to None
        :type sowing_date: date | str, optional
        :param child_models_too: indicate if all parameters of child models should be sampled too, defaults to True
        :type child_models_too: bool, optional
        """
        def process_crop_model(cmd:dict):
            dcp = cmd[Q.PARAM]
            if mid in dcp.keys():
                dcp[mid] = self._sample_q_in_model(dcp[mid])
                cmd[Q.PARAM] = dcp
            
            if not child_models_too:
                return
                            
            for mdlid, mdldef in dcp.items():
                if mid + '.' in mdlid:
                    dcp[mdlid] = self._sample_q_in_model(mdldef)
            cmd[Q.PARAM] = dcp
            return cmd

        if (sowing_date is not None) and isinstance(sowing_date, date):
            sowing_date = sowing_date.isoformat()

        for i in range(len(self._edefs['crop_rotation'])):
            vp = self._edefs['crop_rotation'][i]
            if (sowing_date is not None):
                if vp['epoch_start'] == sowing_date:
                    vp['crop_model'] = process_crop_model(vp['crop_model'])
                    self._edefs['crop_rotation'][i] = vp
                    break
            else:
                vp['crop_model'] = process_crop_model(vp['crop_model'])
                self._edefs['crop_rotation'][i] = vp

    def add_eval_info(self, addinfo:str) -> None:
        """
        Additional information about the evaluation will be set in the 
        :func:`defs` dictionary

        :param addinfo: additional evaluation information
        :type addinfo: str
        """
        if self._edefs['eval_info'] is None:
            self._edefs['eval_info'] = ''
        self._edefs['eval_info'] += ' - ' + addinfo

    def _sample_q(self, d:dict) -> dict:
        """
        Auxiliary method which iterates over the model definitions in 
        ``d`` and marks all quantities to be sampled.
        Thus, ``d`` is a dictionary accessible via the quantity-group 
        keys (i.e. ``'state'``, ``'hyper_parameter'``, ``'observation'``, ...) 
        within the ``self._edefs`` dictionary.

        :param d: dictionary containing model definitions
        :type d: dict
        :return: dictionary with updated information
        :rtype: dict
        """
        for mid, md in d.items():
            d[mid] = self._sample_q_in_model(md)
        return d
    
    @staticmethod
    def _sample_q_in_model(md:dict) -> dict:
        """
        Auxiliary method which iterates all quantity definitions within the 
        model-definition dictionary and marks all quantities to be sampled.

        :param md: dictionare for one model which contains the quantity defintions
        :type md: dict
        :return: updated dictionary
        :rtype: dict
        """
        for qk, qd in md.items():
            if 'fdef' in qd.keys():
                md[qk]['fdef']['sample'] = True
            else:
                md[qk]['distr']['sample'] = True
        return md

    def __getitem__(self, key):
        return self._edefs[key]
    
    def __setitem__(self, key, val):
        self._edefs[key] = val
    
    @classmethod
    def from_json(cls, json_path:str):
        """
        Build :class:`EvaluationDefinitions` from a provided json-file

        :param json_path: absolute path of the json-file
        :type json_path: str
        """
        def loop_zone_dict(
                d:dict, edefs:EvaluationDefinitions, zid:str
            ) -> EvaluationDefinitions:
            for qtype, qinfos in d.items():
                for qmodel, qs in qinfos.items():
                    for qname, qdef in qs.items():
                        if 'epochs' in qdef.keys():
                            epochs = ObsEpochList.list_from_repr(
                                qdef['epochs']
                            )
                            values = ObsValueList.list_from_repr(
                                qdef['values']
                            )
                            for ep, val in zip(epochs, values):
                                edefs.set_zone_qinfos(
                                    zid, qtype, qmodel, qname, {
                                        'epoch': date.fromisoformat(ep), 
                                        'value': val, 
                                        'distr': qdef['distr']
                                    }
                                )
                        else:
                            qdef['epoch'] = date.fromisoformat(qdef['epoch'])
                            edefs.set_zone_qinfos(
                                zid, qtype, qmodel, qname, qdef
                            )
            return edefs
        
        def loop_crop_dict(
                d:dict, edefs:EvaluationDefinitions, cmname:str, sdate:str
            ) -> EvaluationDefinitions:
            for qtype, qinfos in d.items():
                for qmodel, qs in qinfos.items():
                    for qname, qdef in qs.items():
                        if 'epochs' in qdef.keys():
                            epochs = ObsEpochList.list_from_repr(
                                qdef['epochs']
                            )
                            values = ObsValueList.list_from_repr(
                                qdef['values']
                            )
                            for ep, val in zip(epochs, values):
                                edefs.set_crop_qinfos(
                                    cmname, sdate, qtype, qmodel, qname, {
                                        'epoch': date.fromisoformat(ep), 
                                        'value': val, 
                                        'distr': qdef['distr']
                                    }
                                )
                        else:
                            qdef['epoch'] = date.fromisoformat(qdef['epoch'])
                            edefs.set_crop_qinfos(
                                cmname, sdate, qtype, qmodel, qname, qdef
                            )
            return edefs

        fio = open(json_path, 'r')
        defs = json.load(fio)
        fio.close()

        zmodel = getattr(
            import_module(defs['zmodel_module']), defs['zmodel_name']
        )
        obj = cls(zmodel)
        obj['eval_info'] = defs['eval_info']
        obj['epoch_start'] = defs['epoch_start']
        obj['epoch_end'] = defs['epoch_end']
        # process zone infos
        zids, zinfos = [], []
        for zid, zinfo in defs['add_info']['zones'].items():
            zids.append(zid)
            zinfos.append(
                {'lat': zinfo['latitude'], 
                 'gcs': np.array(GCSCoords.gcs_from_repr(zinfo['gcs']))}
            )
        obj.provide_field_info(
            defs['field'], defs['add_info']['crs'], defs['add_info']['height'],
            zids, zinfos
        )

        # process zone model
        for zid, zdict in defs['zone_models'].items():
            obj = loop_zone_dict(zdict, obj, zid)
        # process crop rotation
        for vp in defs['crop_rotation']:
            cmodel = getattr(
                import_module(vp['cmodel_module']), vp['cmodel_name']
            )
            dstart = date.fromisoformat(vp['epoch_start'])
            dstop = date.fromisoformat(vp['epoch_end'])
            obj.add_crop_rotation(cmodel, dstart, dstop)
            obj = loop_crop_dict(
                vp['crop_model'], obj, vp['cmodel_name'], vp['epoch_start']
            )

        return obj
    