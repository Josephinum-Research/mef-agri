import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from mef_agri.utils.raster import GeoRaster
from mef_agri.utils.rv_manipulation import RasterVectorIntersection
from mef_agri.data.jr_management.interface import *


def check_gpkg():
    objres = 10.
    flds = gpd.read_file(os.path.join(wdir, 'testdb.gpkg'), layer='fields')
    zns = gpd.read_file(os.path.join(wdir, 'testdb.gpkg'), layer='parcels')
    fx, fy = flds['geometry'].values[0].exterior.coords.xy
    z1x, z1y = zns[zns['pid'] == 1]['geometry'].values[0].exterior.coords.xy
    z2x, z2y = zns[zns['pid'] == 2]['geometry'].values[0].exterior.coords.xy

    frstr = GeoRaster.from_gdf_and_objres(flds, objres)
    frstr.raster = np.random.normal(size=frstr.raster_shape)

    rvi1 = RasterVectorIntersection(zns[zns['pid'] == 1], frstr)
    rvi1.fraction = 1e-4
    rvi1.compute()
    rvi2 = RasterVectorIntersection(zns[zns['pid'] == 2], frstr)
    rvi2.fraction = 1e-4
    rvi2.compute()
    rvi = RasterVectorIntersection(zns, frstr)
    rvi.fraction = 1e-4
    rvi.compute()

    tr = np.zeros(frstr.raster_shape[1:], dtype='int')
    tr[np.where(rvi.assignment[0, :, :])] = 99
    print(tr)
    
    f1, a1 = plt.subplots(1, 1)
    a1.imshow(rvi.processed_raster[0, :, :], extent=frstr.extent)
    #a1.imshow(rvi2.processed_raster[0, :, :], extent=frstr.extent)
    a1.plot(fx, fy, color='red')
    a1.plot(z1x, z1y, color='gold')
    a1.plot(z2x, z2y, color='gold')
    a1.axis('equal')

    

if __name__ == '__main__':
    wdir = os.path.split(__file__)[0]

    check_gpkg()

    fpath = os.path.join(wdir, 'management', 'testdata.xlsx')
    df1 = read_sheet(fpath, sheet_fielddata, columns_fielddata)
    df2 = read_sheet(fpath, sheet_fertilization, columns_fertilization)
    df3 = read_sheet(fpath, sheet_harvest, columns_harvest)
    for tpl in df1.itertuples():
        print(getattr(tpl, 'pid'))

    plt.show()
