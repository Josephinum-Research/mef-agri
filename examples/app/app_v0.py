import os

from mef_agri.app import run_app
from mef_agri.data.planetary_computer.sentinel2.interface import (
    Sentinel2Interface
)
from mef_agri.data.geosphere_austria.inca.interface import INCAInterface
from mef_agri.data.ebod_austria.interface import EbodInterface


if __name__ == '__main__':
    #pdir = os.path.join(
    #    '/', 'home', 'andreas', 'development', 'projects', 'ettlinger'
    #)
    pdir = os.path.join('/', 'home', 'aet', 'devel', 'projects', 'test')
    run_app(
        project_path=pdir,
        data_interfaces=[Sentinel2Interface, INCAInterface, EbodInterface]
    )
