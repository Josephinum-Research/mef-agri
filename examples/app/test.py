import os
from multiprocessing import cpu_count
from datetime import date

from mef_agri.data.project import Project
from mef_agri.data.geosphere_austria.inca.interface import INCAInterface


if __name__ == '__main__':
    wdir = os.path.join(
        '/', 'home', 'andreas', 'development', 'data', 'test_prj'
    )
    prj = Project(wdir, 'fields.gpkg')
    prj.n_processes = 16
    #prj.initialize()
    #prj.create_tables()
    #prj.add_data_interface(INCAInterface)
    prj.add_data(date(2026, 1, 1), date(2026, 6, 1), fields='oberaschbach')
