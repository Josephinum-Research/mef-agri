import numpy as np
from scipy import stats


def effective_sample_size_choice(wi:np.ndarray, nthresh:float=None) -> bool:
    if nthresh is None:
        # default threshold wil be half the number of particles
        nthresh = 0.5 * wi.shape[0]
    neff = 1. / np.sum(np.power(wi, 2.))
    if neff <= nthresh:
        return True
    else:
        return False


def resampling_multinomial(wi:np.ndarray) -> np.ndarray:
    csw = np.cumsum(wi / np.sum(wi))
    pi = stats.uniform.rvs(0, 1, size=wi.shape[0])
    return np.searchsorted(csw, pi)


def resampling_systematic(wi:np.ndarray) -> np.ndarray:
    csw = np.cumsum(wi / np.sum(wi))
    ni = wi.shape[0]
    iv = 1. / ni
    pi = iv * np.arange(ni) + stats.uniform.rvs(0, iv, size=1)
    return np.searchsorted(csw, pi)


def resampling_stratified(wi:np.ndarray) -> np.ndarray:
    csw = np.cumsum(wi / np.sum(wi))
    ni = wi.shape[0]
    iv = 1. / ni
    pi = iv * np.arange(ni) + stats.uniform.rvs(0, iv, size=ni)
    return np.searchsorted(csw, pi)
