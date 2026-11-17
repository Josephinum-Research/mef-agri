from importlib import import_module

from .. import Interface
from ....farming.tasks import Task, Application


class TaskInterface(Interface):
    META_KEY_CMODEL = 'crop_model_obj'
    META_KEY_CROP = 'sown_crop'
    META_KEY_CULT = 'sown_cultivar'

    def process_data(self, data:dict, meta:dict):
        """
        The ``data`` dictionary has to have the following structure
        
        * the first level of keys has to contain the epochs (i.e. iso-formatted date-strings)
        * depending on :func:`mef_agri.data.interface.Interface.data_types`, there are only two level of keys or a third one
        
            * if it is  ``None``, the second level of keys represents the layer-ids of the georaster
            * otherwise, the second level of keys represents the data-types and the third level contains the layer-ids
        
        ``meta``has to contain the task-metadata.
        The keys are the same epochs as in ``data`` and the values are the 
        dictionaries from :func:`mef_agri.farming.tasks.Task.metadata`.
        
        :param data: dictionary containing zoned values (i.e. representative value for all pixels in one zone)
        :type data: dict
        :param meta: task metadata
        :type meta: dict
        """
        pass

    @staticmethod
    def _get_appls(zoned_values:dict, metadata:dict) -> list[Application]:
        """
        Get application instances containing all necessary data to process the 
        interface from ``zoned_values`` and ``metadata``.
        ``zoned_values`` and ``metadata`` have to represent one task at one 
        epoch.

        :param zoned_values: representative values for pixels in a zone for all layer-ids of the task-georaster (i.e. key > layer-id, value > float)
        :type zoned_values: dict
        :param metadata: metadata-dictionary from task-georaster
        :type metadata: dict
        :return: application instances for one task and epoch
        :rtype: list[Application]
        """
        appl_names, appls = [], []
        for task_lid in zoned_values.keys():
            aname = Task.appl_name_from_layer_id(task_lid)
            if aname in appl_names:
                continue
            
            appl_names.append(aname)
            appl_def = metadata[Task.META_APPL_KEY][aname]
            # create application instance
            appl:Application = getattr(
                import_module(appl_def[Task.META_APPL_MODULE]), 
                appl_def[Task.META_APPL_NAME]
            )()
            # set up numeric values
            for numval in appl.numeric_values:
                nvdef = appl_def[Task.META_APPL_NVALS][numval.name]
                numval.value = zoned_values[nvdef[Task.META_APPL_NVALS_LID]]
                numval.unit = nvdef[Task.META_APPL_NVALS_UNIT]
            # set up descriptive values
            for descrval in appl.descriptive_values:
                descrval.value = appl_def[Task.META_APPL_DVALS][descrval.name]
            appls.append(appl)

        return appls
