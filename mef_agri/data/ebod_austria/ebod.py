import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from ...utils.gis import bbox_from_gdf
from ...utils.raster import GeoRaster
from ...utils.misc import PixelUnits


class EBOD_DATA:
    ID = 'id_short'
    ID_BOFO = 'id_bofo'
    SOIL_TYPE = 'soil_type'
    SOIL_TYPESUB = 'soil_type_sub'
    SOIL_BASE = 'soil_base'
    SOIL_REACT = 'soil_react'
    SOIL_COMPR = 'soil_pot_compr'
    SOIL_VAR = 'soil_var'
    SOIL_MAT = 'soil_ori'
    WATER_COND = 'water_cond'
    WATER_POR = 'water_por'
    WATER_FLDCAP = 'water_fldcap'
    HUMUS_TYPE = 'humus_type'
    HUMUS_VAL = 'humus_val'
    HUMUS_BAL = 'humus_bal'
    LIME_AMOUNT = 'lime_amnt'
    FIELD_VAL = 'fld_val'
    GREENL_VAL = 'grn_val'
    NITRATE_STOR = 'nitr_store'


EBOD_PROPERTIES = {
    'bofo_id': EBOD_DATA.ID_BOFO,
    'kurzbezeichnung': EBOD_DATA.ID, 
    'bodentyp': EBOD_DATA.SOIL_TYPESUB, 
    'typengruppe': EBOD_DATA.SOIL_TYPE, 
    'gruendigkeit': EBOD_DATA.SOIL_BASE,
    'wasserverhaeltnisse': EBOD_DATA.WATER_COND, 
    'durchlaessigkeit': EBOD_DATA.WATER_POR, 
    'humusart': EBOD_DATA.HUMUS_TYPE, 
    'humuswert': EBOD_DATA.HUMUS_VAL, 
    'kalkgehalt': EBOD_DATA.LIME_AMOUNT, 
    'bodenreaktion': EBOD_DATA.SOIL_REACT, 
    'ackerwert': EBOD_DATA.FIELD_VAL, 
    'gruenlandwert': EBOD_DATA.GREENL_VAL,
    'nutzbare_feldkapazitaet': EBOD_DATA.WATER_FLDCAP, 
    'nitratrueckhalt': EBOD_DATA.NITRATE_STOR, 
    'humusbilanz': EBOD_DATA.HUMUS_BAL, 
    'potentielle_verdichtungsempfindlichkeit': EBOD_DATA.SOIL_COMPR, 
    'bodenart': EBOD_DATA.SOIL_VAR, 
    'ausgangsmaterial': EBOD_DATA.SOIL_MAT
}


EBOD_MAPPINGS = {
    EBOD_DATA.SOIL_TYPE: {
        'B': 1,  # Braunerde
        'K': 2,  # Bodenformkomplex
        'G': 3,  # Gley
        'T': 4,  # Reliktboden
        'M': 5,  # Moor
        'A': 6,  # Auboden
        'N': 7,  # Anmoor
        'P': 8,  # Pseudogley
        'R': 9,  # Rendsina+Ranker
        'U': 10,  # untypischer Boden
        'S': 11,  # Schwarzerde
        'Z': 12,  # Salzboden
        'O': 13,  # Podsol
        'C': 14,  # Rohboden
        'X': 15,  # nicht identifizierbar
    },
    EBOD_DATA.SOIL_TYPESUB: {
        'M': 1,  # Moor
        'HM': 2,  # Hochmoor
        'UM': 3,  # Uebergangsmoor
        'NM': 4,  # Niedermoor
        'N': 5,  # Anmoor
        'A': 6,  # Auboden
        'RA': 7,  # Rohauboden
        'GA': 8,  # Grauer Auboden
        'BA': 9,  # Brauner Auboden
        'SA': 10,  # Schwemmboden
        'G': 11,  # Gley
        'TG': 12, # Typischer Gley
        'EG': 13,  # Extremer Gley
        'HG': 14,  # Hanggley
        'KZ': 15,  # Solontschak/Salzboden
        'SZ': 16,  # Solontschak-Solonetz
        'ZZ': 17,  # Solonetz
        'GC': 18,  # Gesteinsrohboden
        'LC': 19,  # Lockersediment-Rohboden
        'R': 20,  # Rendsina
        'ER': 21, # Eurendsina
        'PR': 22,  # Pararendsina
        'RR': 23,  # Ranker
        'GS': 24,  # Gebirgsschwarzerde
        'FS': 25,  # Feuchtschwarzerde
        'TS': 26,  # Tschernosem
        'BS': 27,  # Brauner Tschernosem
        'PS': 28,  # Paratschernosem
        'B': 29,  # Braunerde
        'FB': 30,  # Felsbraunerde
        'LB': 31,  # Lockersediment-Braunerde
        'PB': 32,  # Parabraunerde
        'SO': 33,  # Semipodsol
        'P': 34,  # Pseudogley
        'TP': 35,  # Typischer Pseudogley
        'EP': 36,  # Extremer Pseudogley
        'SP': 37,  # Stagnogley
        'HP': 38,  # Hangpseudogley
        'T': 39,  # Reliktboden
        'BT': 40,  # Braunlehm
        'RT': 41,  # Rotlehm
        'ET': 42,  # Roterde
        'GT': 43,  # Reliktpseudogley
        'U': 44,  # untypischer Boden
        'OU': 45,  # Ortsboden
        'FU': 46,  # Farb-Ortsboden
        'TU': 47,  # Textur-Ortsboden
        'SU': 48,  # Struktur-Ortsboden
        'RU': 49,  # Restboden
        'KU': 50,  # Kulturrohboden
        'IU': 51,  # Rigolboden
        'LU': 52,  # Kolluvium
        'GU': 53,  # Gartenboden
        'HU': 54,  # Haldenboden
        'PU': 55,  # Planieboden
        'K': 56,  # Bodenformkomplex
        'X': 57,  # nicht identifizierbar
    },
    # according to OENORM L 1050 (2016)
    EBOD_DATA.SOIL_VAR: {
        'S': 1,  # Sand
        'zS': 2,  # schluffiger Sand
        'lS': 3,  # lehmiger Sand
        'tS': 4,  # toniger Sand
        'sZ': 5,  # sandiger Schluff
        'Z': 6,  # Schluff
        'lZ': 7,  # lehmiger Schluff
        'sL': 8,  # sandiger Lehm
        'L': 9,  # Lehm
        'zL': 10,  # schluffiger Lehm
        'sT': 11,  # sandiger Ton
        'lT': 12,  # lehmiger Ton
        'T': 13  # Ton
    }
}


class EbodRaster(GeoRaster):
    def __init__(self):
        """
        `GeoRaster` representing a window of the Ebod-soil-map.
        """
        super().__init__()

    @classmethod
    def from_vector_tile_geojson(cls, gjpath:str):
        return EbodRaster.from_geodataframe(gpd.read_file(gjpath))

    @classmethod
    def from_vector_tile_geodataframe(
        cls, vtgdf:gpd.GeoDataFrame, aoi:gpd.GeoDataFrame, obj_res:float, 
        excl_ebod_props:list[str]=None
    ):
        """
        Creates an `EbodRaster` instance from the provided gdf. The gdf should 
        contain the information from the vector tiles which can be requested 
        from the Ebod soil-map.

        :param vtgdf: data from vector tiles requested from ebod
        :type vtgdf: gpd.GeoDataFrame
        :param aoi: area of interest
        :type aoi: geopandas.GeoDataFrame
        :param obj_res: desired object resolution of the `EbodRaster`
        :type obj_res: float
        :param excl_ebod_props: ebod properties which should be omitted in the creation of the `EbodRaster`, defaults to None
        :type excl_ebod_props: list[str], optional
        """
        tepsg = aoi.crs.to_epsg()
        if tepsg in (4326,):
            msg = 'EbodRaster cannot be computed from an `aoi` with crs in '
            msg += '[4326,] - only metric projections are supported yet!'
            raise ValueError(msg)
        if vtgdf.crs.to_epsg() != tepsg:
            vtgdf.to_crs(tepsg, inplace=True)

        # compute raster from provided aoi and obj-resolution
        ebod = cls()
        ebod.crs = tepsg
        ebod.bounds = bbox_from_gdf(aoi)
        ebod.units = PixelUnits.INT32
        ebod.nodata_value = -1
        ebod.layer_index = 0

        eprps, sprps = EBOD_PROPERTIES.keys(), EBOD_PROPERTIES.values()
        if excl_ebod_props is None:
            excl_ebod_props = []
        # compute transformation matrix and raster shape from provided bounding 
        # box and object resolution
        # number of layers/channels is determined from the available ebod 
        # properties "minus" the properties excluded by the user
        ebod.transformation = np.array([
            [obj_res, 0.0, ebod.bounds[0]],
            [0.0, -obj_res, ebod.bounds[3]],
            [0.0, 0.0, 1.0]
        ])
        ebod.raster_shape = (
            len(eprps) - len(excl_ebod_props),
            int(((ebod.bounds[3] - ebod.bounds[1]) // obj_res) + 1),
            int(((ebod.bounds[2] - ebod.bounds[0]) // obj_res) + 1)
        )

        # initialize the raster/numpy.ndarray with the provided 
        # geotiff-nodata-values
        ebod.raster = np.zeros(ebod.raster_shape, dtype=ebod.units)
        ebod.raster[:] = ebod.nodata_value
        lids_done = False
        # fill numpy.ndarray with values
        for i in np.arange(ebod.raster_shape[1]):
            for j in np.arange(ebod.raster_shape[2]):
                # compute point in object space from image coordinates and 
                # transformation matrix
                pic = (ebod.transformation @ np.array([[j, i, 1.0]]).T)[:2, 0]
                pic += np.array([0.5 * obj_res, -0.5 * obj_res])
                pi = Point(*pic)
                # search for polygons containing point
                # can be more than one polygon due to the overlaps of vector 
                # tiles or also zero (streets, ...)
                di = vtgdf[vtgdf.contains(pi)]
                if len(di) == 0:
                    continue
                di = di[di.index == di.index[0]]

                # iterate over the properties in the DataFrame containing the 
                # polygons where point is located in
                # the value from the first polygon in the DataFrame will be used 
                # (the values are the same for all polygons in the DataFrame)
                ci = 0
                for eprp, sprp in zip(eprps, sprps):
                    if eprp in excl_ebod_props:
                        continue
                    if not lids_done:
                        if not ebod.layer_ids:
                            ebod.layer_ids = []
                        ebod.layer_ids.append(sprp)

                    if sprp in EBOD_MAPPINGS.keys():
                        val = EBOD_MAPPINGS[sprp][di[sprp].values[0]]
                    else:
                        val = di[sprp].values[0]
                    if np.isnan(val):
                        val = ebod.nodata_value
                    ebod.raster[ci, i, j] = val

                    ci += 1
                lids_done = True

        return ebod
