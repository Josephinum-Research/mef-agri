from mef_agri.evaluation.interfaces import Interface


class TestIface(Interface):
    def __init__(self):
        super().__init__()

    @Interface.model_quantity
    def obs1(self) -> Interface.MQDefinition:
        """
        :return: first observation
        :rtype: Interface.MQDefinition
        """


if __name__ == '__main__':
    intf = TestIface()
    print(intf.model_quantities)
    intf.obs1.values = 10.
    print(intf.obs1.values)
