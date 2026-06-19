from ...models.base import Model

class Interface(object):
    def __init__(self, prjdata):
        self._ds:str = None
        self._qs:dict = {}

    @property
    def data_source(self) -> str:
        """
        :return: id of the data-interface (:class:`mef_agri.data.interface.Interface`) which provides data
        :rtype: str
        """
        return self._ds
    
    def _add_provided_quantity(self, qname:str, unit:str, qtype:str) -> None:
        """
        To be used in child class to indicate which model-quantities are 
        provided by the interface

        :param qname: name of the quantity (methods of models in ``mef_agri.models`` decorated with :func:`mef_agri.models.base.Model.is_quantity`)
        :type qname: str
        :param unit: unit of the quantity (see :class:`mef_agri.models.utils.__UNITS__`)
        :type unit: str
        :param qtype: quantity type (see :class:`mef_agri.models.base.__QS__`)
        :type qtype: str
        """
        self._qs[qname] = {'unit': unit, 'type': qtype}

    def process(self):
        # TODO fix zoning approach first
        pass
