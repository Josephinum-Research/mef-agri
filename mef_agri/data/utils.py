import datetime
import numpy as np


DELTADAY = datetime.timedelta(days=1)
EPOCH0 = datetime.date(1970, 1, 1)

def merge_dateranges(
    dranges:list[list[datetime.date, datetime.date]]
) -> list[list[datetime.date, datetime.date]]:
    """
    Merge dateranges if they "touch" each other 
    (i.e. the timedelta is <= a day).
    Dateranges in `dranges` have to be mutually exclusive!

    :param dranges: list of dateranges
    :type dranges: list[list[datetime.date, datetime.date]]
    :return: list of merged dateranges
    :rtype: list[list[datetime.date, datetime.date]]
    """
    # sort dateranges
    tstarts = [(drange[0] - EPOCH0).total_seconds() for drange in dranges]
    t0 = [
        EPOCH0 + datetime.timedelta(seconds=tstart) 
        for tstart in np.sort(tstarts)
    ]
    tstops = [(drange[1] - EPOCH0).total_seconds() for drange in dranges]
    t1 = [
        EPOCH0 + datetime.timedelta(seconds=tstop) 
        for tstop in np.sort(tstops)
    ]
    drngs = [[tstart, tstop] for tstart, tstop in zip(t0, t1)]
    
    # merge dateranges if timedelta is smaller than or equal one day
    drngs_out = [drngs[0]]
    for drng in drngs[1:]:
        if (drng[0] - drngs_out[-1][1]) <= DELTADAY:
            drngs_out[-1] = [drngs_out[-1][0], drng[1]]
        else:
            drngs_out.append(drng)
    return drngs_out

def daterange_consider_existing_dates(
    drange:list[datetime.date, datetime.date], 
    dranges:list[list[datetime.date, datetime.date]]
) -> list[list[datetime.date, datetime.date]]:
    """
    Adjust ``drange`` if there are overlaps with the date-ranges within 
    ``dranges``.

    * Example 1
    
        * ``drange = [2024-01-01, 2024-01-31]``
        * ``dranges = [ [2024-01-15, 2024-01-31] ]``
        * returns ``[ [2024-01-01, 2024-01-14] ]``

    * Example 2

        * ``drange = [2024-01-01, 2024-06-30]``
        * ``dranges = [ [2023-12-15, 2024-01-31], [2024-03-01, 2024-03-31] ]``
        * returns ``[ [2024-02-01, 2024-02-29] , [2024-04-01, 2024-06-30] ]``

    :param drange: date-range which should be adjusted
    :type drange: list[datetime.date, datetime.date]
    :param dranges: dateranges which should be considered
    :type dranges: list[list[datetime.date, datetime.date]]
    :return: adjusted dateranges
    :rtype: list[list[datetime.date, datetime.date]]
    """
    
    trng_out, outflag = [], True
    for i in range(len(dranges)):
        tstart_inside = drange[0] >= dranges[i][0] and drange[0] <= dranges[i][1]
        tstop_inside = drange[1] >= dranges[i][0] and drange[1] <= dranges[i][1]
        if tstart_inside and tstop_inside:
            outflag = False
            break
        elif not tstart_inside and not tstop_inside:
            if (drange[1] < dranges[i][0]) or (drange[0] > dranges[i][1]):
                continue
            else:
                trng_out += daterange_consider_existing_dates(
                    [drange[0], dranges[i][0] - DELTADAY], dranges[i:]
                )
                trng_out += daterange_consider_existing_dates(
                    [dranges[i][1] + DELTADAY, drange[1]], dranges[i:]
                )
                outflag = False
                break
        else:
            if tstart_inside:
                drange[0] = dranges[i][1] + DELTADAY
            else:
                drange[1] = dranges[i][0] - DELTADAY

    if outflag:
        trng_out.append(drange)
    return trng_out