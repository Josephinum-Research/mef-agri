import os
from multiprocessing import cpu_count
from datetime import date

from mef_agri.data.project import ProjectData
from mef_agri.data.geosphere_austria.inca.interface import INCAInterface


def init_prj(prj:ProjectData):
    prj.initialize()
    prj.create_tables()
    prj.add_data_interface(INCAInterface)


if __name__ == '__main__':
    develdir = os.path.join('/', 'home', 'aet', 'devel')
    wdir = os.path.join(develdir, 'projects', 'test_prj')
    prj = ProjectData(wdir, 'fields.gpkg')
    prj.n_processes = 8
    print(prj.query('SELECT * FROM data_available;'))
    print(prj.query('SELECT * FROM data_epochs;'))
    #prj.add_data(date(2025, 5, 1), date(2025, 6, 1), fields='oberaschbach')
