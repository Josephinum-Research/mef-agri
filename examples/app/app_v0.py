import os

from mef_agri.app import run_app


if __name__ == '__main__':
    pdir = os.path.join(
        '/', 'home', 'andreas', 'development', 'data', 'test_prj'
    )
    #pdir = os.path.join('/', 'home', 'aet', 'devel', 'projects', 'test')
    run_app(project_path=pdir)
