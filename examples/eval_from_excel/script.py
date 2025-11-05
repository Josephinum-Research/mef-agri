import os
import datetime as dt
import matplotlib.pyplot as plt

from mef_agri.data.project import get_project_jrv01, Project
from mef_agri.models.base import Quantities as Q
from mef_agri.evaluation.zoning.jr import ZoningJR_V01
from mef_agri.evaluation.db import EvalDB_Quantiles
from mef_agri.evaluation.eval_def import EvaluationDefinitions
from mef_agri.evaluation.estimation.pf import ParticleFilter as ModelPropagation
from mef_agri.evaluation.estimation.bpf import BPF_EPIC_Obs_LAI
from mef_agri.evaluation.analysis.visualization import Visualization

dbname = 'ex.db'


def add_prj_data(d1:dt.date, d2:dt.date):
    prj.add_data(d1, d2)  # adding data to project data folder


def prepare_eval(
        prj:Project, fname:str, zoning_date:dt.date, d1:dt.date, d2:dt.date
    ):
    # PREPARE DATA FOR EVALUATION
    z = ZoningJR_V01(prj)
    z.zoning_date = zoning_date
    # INITIALIZE MODEL QUANTITIES FROM DATA INTERFACES
    ed = z.prepare_data(d1, d2, fname)
    # INITIALIZE FURTHER MODEL QUANTITIES
    ed = z.init_soil_moisture(ed, 0.5, 0.01)
    ed = z.init_temp_values(ed, 1.0)
    ed = z.init_wudf(ed, 2.0, 0.3)
    ed = z.init_cropres_amounts(ed, 500.0, 30., 17., 2.0)
    ed = z.init_n_amounts(ed, 25., 3., 5., 1.)
    ed = z.init_org_amounts(ed, 0.02, 0.05, 0.01, 0.01)
    ed = z.set_decomposition_rates(ed)
    z.save_edefs(ed, fname)


def set_up_db(field_name:str, prjp:str, edname:str):
    dbp = os.path.join(prjp, EvaluationDefinitions.EVAL_FOLDER_NAME)
    edp = os.path.join(dbp, field_name)
    ed = EvaluationDefinitions.from_json(os.path.join(edp, edname + '.json'))
    EvalDB_Quantiles.from_eval_def(dbp, dbname, ed)
    # from_eval_def is called a second time to have two evaluation ids in the 
    # database (one for model propagation and one for bootstrap particle filter)
    EvalDB_Quantiles.from_eval_def(dbp, dbname, ed)


def evaluation(prjp:str, nps:int):
    dbp = os.path.join(prjp, EvaluationDefinitions.EVAL_FOLDER_NAME)
    db = EvalDB_Quantiles(dbp, dbname)
    db.evaluation_id = 1
    mp = ModelPropagation(nps, db)
    mp.process(wdir)

    db.evaluation_id = 2
    pf = BPF_EPIC_Obs_LAI(nps, db)
    pf.process(wdir)


def plot_lai(vis:Visualization, t1:dt.date, t2:dt.date, zid):
    if zid == 1:
        zmp, zpf = 1, 3
    elif zid == 2:
        zmp, zpf = 2, 4
    f, a = plt.subplots(1, 1)
    vis.add_axes_obj(a, 'lai', t1, t2)
    vis.time_series(
        'lai', 'lai', Q.STATE, 'crop.leaves', zmp, label='mpr', 
        color_theme=vis.COLOR_THEMES.GREEN
    )
    vis.time_series(
        'lai', 'lai', Q.STATE, 'crop.leaves', zpf, label='bpf', 
        color_theme=vis.COLOR_THEMES.RED, distr_xshift=0.1
    )
    vis.time_series_non_regular(
        'lai', 'lai', Q.ROUT, 'zone.sentinel2_lai', zpf, label='obs',
        color_theme=vis.COLOR_THEMES.BLUE
    )


if __name__ == '__main__':
    wdir = ...  # TODO provide working/project directory
    d1, d2 = dt.date(2024, 10, 1), dt.date(2025, 8, 30)
    fname = 'Mayerland'

    prj = get_project_jrv01(wdir, 'fields', 'fname')

    #prepare_eval(prj, fname, dt.date(2024, 10, 1), d1, d2)
    #set_up_db(fname, prj.project_path, 'eid_')
    #evaluation(prj.project_path, 1000)

    dbdir = os.path.join(wdir, EvaluationDefinitions.EVAL_FOLDER_NAME)
    db = EvalDB_Quantiles(dbdir, dbname)
    vis = Visualization(db)
    plot_lai(vis, d1, d2, 2)

    prj.quit_project()
    plt.show()
