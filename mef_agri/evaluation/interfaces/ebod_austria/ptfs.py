from numpy import exp

from ....data.ebod_austria.ebod import EBOD_DATA
from ...stats_utils import DISTRIBUTIONS


r"""

[1]
bibtex -> techreport
Standard (bibtex -> type)
Austrian Standards International (bibtex -> institution)
Vienna, Austria (bibtex -> address)
Boden als Pflanzenstandort - Begriffe und Untersuchungsverfahren (bibtex -> title)
\"Onorm L 1050 (bibtex -> key)
2016 (bibtex -> year)

[2]
Schwarz, S. and Aust, G. and Englisch, M. and Herzberger, E. and Kessler, D. and Reiter, R.
Bodenart und Bodenschwere - Hintergrundinformationen
Mitteilungen der \"OBG, vol. 86
2022
https://www.ages.at/download/sdl-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2MDk0NTkyMDAsImV4cCI6NDA3MDkwODgwMCwidXNlciI6MCwiZ3JvdXBzIjpbMCwtMV0sImZpbGUiOiJmaWxlYWRtaW4vQUdFU18yMDIyLzVfVU1XRUxUL0JvZGVuL1dpc3Nlbl91bmRfQmlsZHVuZy9Cb2RlbmFydF91bmRfQm9kZW5zY2h3ZXJlX0hpbnRlcmdydW5kaW5mb3JtYXRpb24ucGRmIiwicGFnZSI6MjI0N30.tSfhSSlDuwOXHycwifR451E0t78Nhrwm4dPEXA07w-o/Bodenart_und_Bodenschwere_Hintergrundinformation.pdf

[3]
Toth, B. and Weynants, M. and Nemes, A. and Mako, A. and Bilas, G. and Toth, G.
New generation of hydraulic pedotransfer functions for Europe
European Journal of Soil Science, vol. 66, pp. 226-238
January 2015
https://doi.org/doi:10.1111/ejss.12192

[4]
Nestroy, O. and Aust, G. and Blum, W.E.H. and Englisch, M. and Hager, H. and Herzberger, E. and Kilian, W. and Nelhiebel, P. and Ortner, G. and Pecina, E.a nd Pehamberger, A. and Schneider, W. and Wagner, J.
Systematische Gliederung der B\"oden \"Osterreichs - \"Osterreichische Bodensystematik 2000 in der revidierten Fassung von 2011
\"Osterreichische Bodenkundliche Gesellschaft, Vienna, Austria
ISSN 0029-893-X
2011
https://www.bodensystematik.de/OEBG-Systematik.pdf

[5]
Hollis, J.M. and Hannam, J. and Bellamy, P.H.
Empirically-derived pedotransfer functions for predicting bulk density in european soils
European Journal of Soil Science, vol. 63, pp. 96-109
2012
https://doi.org/10.1111/j.1365-2389.2011.01412.x

[6]
Zeri, M. and dos Santos Alvala, R.C. and Carneiro, R. and Zeri, G.C. and da Costa, J.M.N. and Spatafora, L.R. and Urbano, D. and Vall-Ilossera, M. and Marengo, J.A.
Tools for communicating agricultural drougth over the brazilian semiarid using the soil moisture index
Water, vol. 10, no. 10
2018
http://dx.doi.org/10.3390/w10101421

"""


# Sand content of soil varieties according to [1] and [2] table 2
SAND_OENORM_L1050 = {
    'mean': {
        '1': 0.825,  # Sand
        '2': 0.55,  # schluffiger Sand
        '3': 0.55,  # lehmiger Sand
        '4': 0.775,  # toniger Sand
        '5': 0.275,  # sandiger Schluff
        '6': 0.125,  # Schluff
        '7': 0.15,  # lehmiger Schluff
        '8': 0.475,  # sandiger Lehm
        '9': 0.35,  # Lehm
        '10': 0.1,  # schluffiger Lehm
        '11': 0.625,  # sandiger Ton
        '12': 0.3,  # lehmiger Ton
        '13': 0.25 # Ton
    },
    'std': {
        '1': 0.06,  # Sand
        '2': 0.05,  # schluffiger Sand
        '3': 0.08,  # lehmiger Sand
        '4': 0.04,  # toniger Sand
        '5': 0.06,  # sandiger Schluff
        '6': 0.04,  # Schluff
        '7': 0.05,  # lehmiger Schluff
        '8': 0.09,  # sandiger Lehm
        '9': 0.1,  # Lehm
        '10': 0.03,  # schluffiger Lehm
        '11': 0.04,  # sandiger Ton
        '12': 0.1,  # lehmiger Ton
        '13': 0.08  # Ton
    }
}


# Clay content of soil varieties according to [1] and [2] table 2
CLAY_OENORM_L1050 = {
    'mean': {
        '1': 0.05,  # Sand
        '2': 0.025,  # schluffiger Sand
        '3': 0.1,  # lehmiger Sand
        '4': 0.175,  # toniger Sand
        '5': 0.075,  # sandiger Schluff
        '6': 0.125,  # Schluff
        '7': 0.2,  # lehmiger Schluff
        '8': 0.2,  # sandiger Lehm
        '9': 0.325,  # Lehm
        '10': 0.35,  # schluffiger Lehm
        '11': 0.35,  # sandiger Ton
        '12': 0.45,  # lehmiger Ton
        '13': 0.75  # Ton
    },
    'std': {
        '1': 0.015,  # Sand
        '2': 0.01,  # schluffiger Sand
        '3': 0.015,  # lehmiger Sand
        '4': 0.025,  # toniger Sand
        '5': 0.025,  # sandiger Schluff
        '6': 0.04,  # Schluff
        '7': 0.015,  # lehmiger Schluff
        '8': 0.015,  # sandiger Lehm
        '9': 0.025,  # Lehm
        '10': 0.03,  # schluffiger Lehm
        '11': 0.025,  # sandiger Ton
        '12': 0.015,  # lehmiger Ton
        '13': 0.08  # Ton
    }
}

# saturated hydraulic conductivity - [3] detail document table S1 (Method 19 - 
# top-soils); unit mm/day
HYDRCONDSAT_TOTH2015 = {
    'mean': {
        '1': 83.3,  # Sand > sand
        '2': 89.5,  # schluffiger Sand > loamy sand
        '3': 89.5,  # lehmiger Sand > loamy sand
        '4': 89.5,  # toniger Sand > loamy sand
        '5': 436.3,  # sandiger Schluff > sandy clay loam
        '6': 13.8,  # Schluff > silt
        '7': 13.8,  # lehmiger schluff > silty clay loam
        '8': 448.8,  # sandiger Lehm > sandy loam
        '9': 141.7,  # Lehm > loam
        '10': 11.7,  # schluffiger Lehm > silt loam
        '11': 438.0,  # sandiger Ton > sandy clay
        '12': 0.1,  # lehmiger Ton > silty clay
        '13': 170.7  # Ton > clay
    },
    'std': {
        '1': 5.0,
        '2': 5.0,
        '3': 5.0,
        '4': 5.0,
        '5': 20.0,
        '6': 1.0,
        '7': 1.0,
        '8': 20.0,
        '9': 7.0,
        '10': 1.0,
        '11': 10.0,
        '12': 0.1,
        '13': 10.0
    }
}

# rooting depth from soil depth ("Gruendigkeit" in ebod)
# mapping according to [4] table 1
ROOTDEPTH_NESTROY11 = {
    'mean': {
        '0': 0.5,  # stark schwankend
        '1': 0.2,  # seichtgruendig
        '2': 0.35,  # seicht- bis mittelgruendig
        '3': 0.5,  # mittelgruendig
        '4': 0.65,  # mittel- bis tiefgruendig
        '5': 0.8  # tiefgruendig
    },
    'std': {
        '0': 0.15,
        '1': 0.075,
        '2': 0.075,
        '3': 0.075,
        '4': 0.075,
        '5': 0.075
    }
}


# according to https://bodenkarte.at > "Karteninhalte - Humusgehalt"
OM_EBOD = {
    '0': 0.3,  # mehr als 30% org. Substanz
    '1': 0.01,  # schwach humos
    '2': 0.02,  # schwach bis mittelhumos
    '3': 0.03,  # mittelhumos
    '4': 0.04,  # mittel- bis stark humos
    '5': 0.05,  # stark humos
    '15': 0.03  # stark schwankend
}


def pedotransfer_ebod_swat_jrv1(data:dict) -> dict:
    ret = {}

    ##### SAND AND CLAY FROM EBOD ##############################################
    ret['clay_content'] = {
        'qname': 'clay_content',
        'qmodel': 'zone.soil',
        'value': CLAY_OENORM_L1050['mean'][str(data[EBOD_DATA.SOIL_VAR])],
        'distr': {
            'distr_id': DISTRIBUTIONS.GAMMA_1D,
            'std': CLAY_OENORM_L1050['std'][str(data[EBOD_DATA.SOIL_VAR])]
        }
    }
    ret['sand_content'] = {
    'qname': 'sand_content',
    'qmodel': 'zone.soil',
    'value': SAND_OENORM_L1050['mean'][str(data[EBOD_DATA.SOIL_VAR])],
    'distr': {
        'distr_id': DISTRIBUTIONS.GAMMA_1D,
        'std': SAND_OENORM_L1050['std'][str(data[EBOD_DATA.SOIL_VAR])]
        }
    }

    ##### HYDRAULIC CONDUCTIVITY ###############################################
    ret['hydraulic_conductivity_sat'] = {
        'qname': 'hydraulic_conductivity_sat',
        'qmodel': 'zone.soil',
        'value': HYDRCONDSAT_TOTH2015['mean'][str(data[EBOD_DATA.SOIL_VAR])],
        'distr': {
            'distr_id': DISTRIBUTIONS.GAMMA_1D,
            'std': HYDRCONDSAT_TOTH2015['mean'][str(data[EBOD_DATA.SOIL_VAR])]
        }
    }

    ##### ROOTING DEPTH MAX. ###################################################
    ret['rooting_depth_max'] = {
        'qname': 'rooting_depth_max',
        'qmodel': 'zone.soil',
        'value': ROOTDEPTH_NESTROY11['mean'][str(data[EBOD_DATA.SOIL_BASE])],
        'distr': {
            'distr_id': DISTRIBUTIONS.GAMMA_1D,
            'std': ROOTDEPTH_NESTROY11['std'][str(data[EBOD_DATA.SOIL_BASE])]
        }
    }

    ##### BULK DENSITY #########################################################
    # conversion of humus/organic matter into organic carbon
    # the value 1.724 is mentioned at https://bodenkarte.at as well as in 
    # [5] paragraph below equ. 4
    oc = OM_EBOD[str(data[EBOD_DATA.HUMUS_VAL])] / 1.724
    # regression equation to get bulk density [g/cm3]
    # [5] table 5 (row "cultivated top soils")
    bd = 0.80806 + 0.823844 * exp(-0.27993 * oc) 
    bd += 0.0014065 * ret['sand_content']['value']
    bd -= 0.0010299 * ret['clay_content']['value']
    ret['bulk_density'] = {
        'qname': 'bulk_density',
        'qmodel': 'zone.soil',
        'value': bd,
        'distr': {
            'distr_id': DISTRIBUTIONS.GAMMA_1D,
            'std': 0.05  # typical values for bulk density 1.35 - 1.65 - [6] table 1
        }
    }

    return ret
