import numpy as np
from matplotlib.axes import Axes


def plot_raster(
        axs:Axes, gcs:np.ndarray, ores:float, rcolor:str, lcolor:str,
        lw:float=1., ms:float=5.
    ) -> Axes:
    urs = gcs.T + np.array([[ores, 0.]])
    lls = gcs.T + np.array([[0., -ores]])
    lrs = gcs.T + np.array([[ores, -ores]])
    for ul, ur, ll, lr in zip(gcs.T, urs, lls, lrs):
        axs.plot(
            [ul[0], ur[0]], [ul[1], ur[1]], color=lcolor,
            linestyle='-', linewidth=lw
        )
        axs.plot(
            [ul[0], ll[0]], [ul[1], ll[1]], color=lcolor,
            linestyle='-', linewidth=lw
        )
        axs.plot(
            [ll[0], lr[0]], [ll[1], lr[1]], color=lcolor,
            linestyle='-', linewidth=lw
        )
        axs.plot(
            [ur[0], lr[0]], [ur[1], lr[1]], color=lcolor,
            linestyle='-', linewidth=lw
        )
    axs.plot(*gcs, color=rcolor, marker='.', linestyle='', markersize=ms)
    return axs