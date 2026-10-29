from .. import Interface


class INCA2SWAT_V1(Interface):
    def __init__(self):
        super().__init__()

    @Interface.model_quantity
    def temperature_min(self) -> Interface.MQDefinition:
        """
        TODO

        :return: _description_
        :rtype: Interface.MQDefinition
        """

