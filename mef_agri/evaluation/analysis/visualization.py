import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from datetime import date
from pandas import date_range

from ..db import EvalDB_Quantiles, EvalDB_AllParticles, EvaluationDB
from ...models.base import Quantities, DISTRIBUTION_TYPE
from ...models.utils import day_of_year


def mode_from_quantiles(qs:np.ndarray) -> float:
    dqs = np.diff(qs, axis=1)
    ixm = np.argmin(dqs, axis=1)
    rows = np.arange(qs.shape[0])
    return qs[rows, ixm] + 0.5 * dqs[rows, ixm]


class Visualization(object):
    class COLOR_THEMES:
        BLUE = 'blue'
        GREEN = 'green'
        RED = 'red'
        YELLOW = 'yellow'
        VIOLET = 'violet'
        BROWN = 'brown'

    _QMAP_EVAL = {
        Quantities.STATE: 'states_eval',
        Quantities.PARAM: 'params_eval',
        Quantities.OBS: 'obs_eval',
        Quantities.ROUT: 'out_eval'
    }
    _CTHEMES = {
        COLOR_THEMES.BLUE: {
            'distr': 'lightskyblue', 'mode': 'navy', 'mean': 'dodgerblue'
        },
        COLOR_THEMES.GREEN: {
            'distr': 'lightgreen', 'mode': 'darkgreen', 'mean': 'limegreen'
        },
        COLOR_THEMES.RED: {
            'distr': 'lightsalmon', 'mode': 'darkred', 'mean': 'orangered'
        },
        COLOR_THEMES.YELLOW: {
            'distr': 'khaki', 'mode': 'gold', 'mean': 'orange'
        },
        COLOR_THEMES.VIOLET: {
            'distr': 'thistle', 'mode': 'indigo', 'mean': 'fuchsia'
        },
        COLOR_THEMES.BROWN: {
            'distr': 'wheat', 'mode': 'saddlebrown', 'mean': 'chocolate'
        }
    }

    def __init__(self, db:EvaluationDB):
        if issubclass(db.__class__, EvalDB_Quantiles):
            self._dbtype = 'dbq'
        elif issubclass(db.__class__, EvalDB_AllParticles):
            self._dbtype = 'dba'
        else:
            msg = '`db` has to be an instance of {} or {}!'.format(
                EvalDB_AllParticles.__name__, EvalDB_Quantiles.__name__
            )
            raise ValueError(msg)
        self._db:EvaluationDB = db
        self._ezinfo:str = None
        self._axs:dict = {}

    @property
    def db(self) -> EvaluationDB:
        return self._db

    @property
    def evaluations_and_zones(self) -> str:
        if self._ezinfo is None:
            sql = 'SELECT z.zname, z.zid, z.eid, e.eval_info FROM zones AS z '
            sql += 'JOIN evaluations AS e ON e.eid=z.eid;'
            self._ezinfo = str(self._db.get_data_frame(sql))
        return self._ezinfo
    
    def add_axes_obj(self, axs:Axes, axsid:str, xstart:date, xstop:date) -> None:
        epochs = []
        for day in date_range(xstart, xstop):
            epochs.append(day.date().isoformat())
        xvals = self._compute_xvals(epochs, epochs[0])
        neps = len(epochs)

        axs.set_xbound(lower=xvals[0], upper=xvals[-1])
        ixs = np.arange(0, neps + 7, 7)
        axs.set_xticks(xvals[0] + ixs)
        ixs[-1] = neps - 1
        axs.set_xticklabels(np.array(epochs)[ixs], rotation=90)

        self._axs[axsid] = {'obj': axs, 'x0': xstart.isoformat()}

    def _compute_xvals(self, epochs:list[str], epoch0:str) -> np.ndarray:
        y0 = int(epoch0.split('-')[0])
        y0eps = int(epochs[0].split('-')[0])
        offs = 0
        for i in range(y0eps - y0):
            offs += 366 if ((y0 + i) % 4) == 0 else 365

        doys = []
        for epoch in epochs:
            doys.append(day_of_year(date.fromisoformat(epoch)) + offs)
        xvals = np.array(doys)
        diffs = np.diff(xvals)
        i1, i2 = 0, 1
        for di in diffs:
            if di < 0:
                epi = epochs[i1]
                mod4 = (int(epi.split('-')[0]) % 4)
                add = 366 if mod4 == 0 else 365
                xvals[i1 + 1:] += add * i2
                i2 += 1
            i1 += 1
        return xvals
    
    def time_series_def(
            self, axsid:str, qname:str, qtype:str, qmodel:str, zid:int, **kwargs
        ) -> None:
        data = self._db.get_quantity_def(qname, qtype, qmodel, zid=zid)
        if data.size == 0:
            return
        xvals = self._compute_xvals(
            data['epoch'].values.tolist(), self._axs[axsid]['x0']
        )
        self._axs[axsid]['obj'].plot(xvals, data['value'], **kwargs)

    def time_series(
            self, axsid:str, qname:str, qtype:str, qmodel:str, 
            zid:int, distr:bool=True, mean:bool=False, median:bool=False,
            conf:int=None,
            color_theme:str=COLOR_THEMES.GREEN, distr_xshift:float=0.0,
            label:str=None
        ) -> None:
        distc = self._CTHEMES[color_theme]['distr']
        modec = self._CTHEMES[color_theme]['mode']
        meanc = self._CTHEMES[color_theme]['mean']
        axs = self._axs[axsid]['obj']

        data = self._db.get_quantity_eval(qname, qtype, qmodel, zid=zid)
        if data.size == 0:
            return
        xvals = self._compute_xvals(
            data['epoch'].values.tolist(), self._axs[axsid]['x0']
        )
        values = self._data_values(data)
        for val, xval in zip(values, xvals):
            if distr:
                self._distr_points(val, axs, xval, distc, xshift=distr_xshift)
        if conf is not None:
            q = (1.0 - conf) / 2.
            lq = np.quantile(values, q, axis=1)
            uq = np.quantile(values, 1. - q, axis=1)
            axs.fill_between(xvals, lq, uq, color=distc, alpha=0.5)
            #axs.plot(xvals, lq, color=modec, linestyle=':')
            #axs.plot(xvals, uq, color=modec, linestyle=':')
        axs.plot(xvals, mode_from_quantiles(values), color=modec, label=label)
        if mean:
            axs.plot(xvals, np.mean(values, axis=1), color=meanc)

    def time_series_non_regular(
            self, axsid:str, qname:str, qtype:str, qmodel:str, zid:int,
            color_theme:str=COLOR_THEMES.GREEN, label:str=None, **kwargs
        ) -> None:
        data = self._db.get_quantity_eval(qname, qtype, qmodel, zid=zid)
        if data.size == 0:
            return
        
        xvals = self._compute_xvals(
            data['epoch'].values.tolist(), self._axs[axsid]['x0']
        )
        values = self._data_values(data)
        vmed = np.nanmedian(values, axis=1)
        xvals = xvals[~np.isnan(vmed)]
        vmed = vmed[~np.isnan(vmed)]

        axs = self._axs[axsid]['obj']
        markercol = self._CTHEMES[color_theme]['mean']
        axs.plot(
            xvals, vmed, color=markercol, marker='x', linestyle='', label=label,
            **kwargs
        )

    def get_eval_data_values(
            self, qname:str, qtype:str, qmodel:str, zid:int
        ) -> np.ndarray:
        df = self._db.get_quantity_eval(qname, qtype, qmodel, zid=zid)
        ret = df.drop(columns='value', inplace=False)
        values = self._data_values(df)
        ret['values'] = values.tolist()
        return ret
    
    def _data_values(self, data:pd.DataFrame) -> np.ndarray:
        values = None
        for tpl in data.itertuples():
            distrtype, valstr = tpl.value.split(':')
            val = None
            if distrtype == DISTRIBUTION_TYPE.CONTINUOUS:
                val = np.array([float(vstr) for vstr in valstr.split(',')])
                
                if values is None:
                    values = np.atleast_2d(val).copy()
                else:
                    values = np.vstack((values, np.atleast_2d(val)))

            elif distrtype == DISTRIBUTION_TYPE.DISCRETE:
                # TODO
                pass

        return values

    def get_xvalues(self, axsid:str, dates:list[date]) -> np.ndarray:
        epochs = []
        for di in dates:
            epochs.append(di.isoformat())
        return self._compute_xvals(epochs, self._axs[axsid]['x0'])

    @staticmethod
    def _distr_points(val, axs, xpos, mcolor, xshift=0.0):
        if np.isin(True, np.isnan(val)):
            return
        dv = np.diff(val)
        ypos = None
        for i in range(dv.shape[0]):
            dy = dv[i] / 3.
            if dy < 1e-8:
                yi = np.array([val[i]])
            else:
                yi = np.arange(val[i], val[i + 1], dy)
            if ypos is None:
                ypos = yi.copy()
            ypos = np.concatenate((ypos, yi), axis=0)
        ypos = np.append(ypos, val[-1])

        axs.plot(
            (xpos + xshift) * np.ones((ypos.shape[0],)), ypos, 
            marker='.', markerfacecolor=mcolor, markersize=6,
            markeredgecolor='none', linestyle='none'
        )
        return axs
