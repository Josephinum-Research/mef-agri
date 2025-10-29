from ..interface import DataInterface


# TODO develop interface for jr field trial excel sheets to extract management information
class ManagementInterface(DataInterface):
    def __init__(self, obj_res=10):
        super().__init__(obj_res)

    def add_prj_data(self, aoi, tstart, tstop):
        pass

    def get_prj_data(self, epoch):
        pass
