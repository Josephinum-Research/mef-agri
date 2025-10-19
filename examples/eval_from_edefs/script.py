import os
from datetime import date
import matplotlib.pyplot as plt

from mef_agri.models.base import Quantities as Q
from mef_agri.evaluation.eval_def import EvaluationDefinitions
from mef_agri.evaluation.db import EvalDB_Quantiles
from mef_agri.evaluation.estimation.pf import ParticleFilter as ModelPropagation
from mef_agri.evaluation.estimation.bpf import BPF_EPIC_Obs_LAI
from mef_agri.evaluation.analysis.visualization import Visualization


wdir = os.path.split(__file__)[0]
dbname = 'ex.db'


def set_up_db():
    ed = EvaluationDefinitions.from_json(os.path.join(wdir, 'exeds.json'))
    EvalDB_Quantiles.from_eval_def(wdir, dbname, ed)
    # from_eval_def is called a second time to have two evaluation ids in the 
    # database (one for model propagation and one for bootstrap particle filter)
    EvalDB_Quantiles.from_eval_def(wdir, dbname, ed)


def evaluation(nps:int):
    db = EvalDB_Quantiles(wdir, dbname)
    db.evaluation_id = 1
    mp = ModelPropagation(nps, db)
    mp.process(wdir)

    db.evaluation_id = 2
    pf = BPF_EPIC_Obs_LAI(nps, db)
    pf.process(wdir)


def plot_lai(vis:Visualization, t1:date, t2:date):
    f, a = plt.subplots(1, 1)
    vis.add_axes_obj(a, 'lai', t1, t2)
    vis.time_series(
        'lai', 'lai', Q.STATE, 'crop.leaves', 1, label='mpr', 
        color_theme=vis.COLOR_THEMES.GREEN
    )
    vis.time_series(
        'lai', 'lai', Q.STATE, 'crop.leaves', 2, label='bpf', 
        color_theme=vis.COLOR_THEMES.RED, distr_xshift=0.1
    )
    vis.time_series_non_regular(
        'lai', 'lai', Q.ROUT, 'zone.sentinel2_lai', 2, label='obs',
        color_theme=vis.COLOR_THEMES.BLUE
    )


if __name__ == '__main__':
    #set_up_db()
    #evaluation(10000)

    db = EvalDB_Quantiles(wdir, dbname)
    vis = Visualization(db)
    epoch_start, epoch_end =  date(2024, 10, 1),  date(2025, 8, 25)
    plot_lai(vis, epoch_start, epoch_end)

    plt.show()
