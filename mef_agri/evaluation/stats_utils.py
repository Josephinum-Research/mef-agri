import numpy as np


class DISTRIBUTION_TYPE:
    """
    Helper class which holds string values (class variables) to specify if a 
    distribution is continuous or discrete:

    * *DISCRETE* = ``'discrete'``
    * *CONTINUOUS* = ``'continuous'``

    """
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'


class DISTRIBUTIONS:
    """
    Helper class which holds string values (class variables) to define the 
    distribution type:

    * *NORMAL_1D* = ``'normal'``
    * *GAMMA_1D* = ``'gamma'``
    * *BETA_1D* = ``'beta'``
    * *CATEGORICAL_1D* = ``'categorical'``
    * *TRUNCNORM_1D* = ``'truncnorm'``

    """
    NORMAL_1D = 'normal'
    GAMMA_1D = 'gamma'
    BETA_1D = 'beta'
    CATEGORICAL_1D = 'categorical'
    TRUNCNORM_1D = 'truncnorm'


def get_gamma_params(std:float, mean:float=None, mode:float=None) -> tuple:
    """
    Compute the shape and rate parameter for the gamma-distribution from the 
    provided standard deviation as well as mean or mode.

    :param std: standard deviation of the gamma-distribution
    :type std: float
    :param mean: mean or expectation value of the gamma-distribution, defaults to None
    :type mean: float, optional
    :param mode: mode of the gamma-distribution, defaults to None
    :type mode: float, optional
    :return: shape and rate parameter of the gamma distribution
    :rtype: tuple[float, float]
    """
    if mode is not None:
        # mode of resulting gamma distribution is located at the given mean
        aux1 = 2.0 * std ** 2
        aux2 = np.sqrt(mode ** 2 + 2.0 * aux1)
        a = (mode ** 2 + mode * aux2) / aux1 + 1.0
        b = (mode + aux2) / aux1
    else:
        # mean of resulting gamma distribution is located at the given mean
        a = mean ** 2 / std ** 2
        b = mean / std ** 2
    return a, b


def get_beta_params(std:float, mean:float=None, mode:float=None) -> tuple:
    """
    Compute the shape parameters for the beta-distribution from the provided 
    standard deviation as well as mean or mode. 

    Important Notes:
    
    If ``mode`` is specified, it will be treated as mean value.
    The reason is, that the computation of the shape parameters with the mode 
    equation of the beta-distribution cannot be solved analytically. 
    Attempts to solve this numerically (e.g. scipy.optimize.fsolve) do not yield 
    satisfying results.

    Edge cases (``mean`` values close to 0.0 or 1.0 with "low" 
    ``std``) lead to non-intuitive resulting distribution shapes or even to 
    negative shape-parameters (i.e. beta-distribution is not defined for 
    the provided combination of ``mean`` and ``std``).

    :param std: standard deviation of the beta-distribution
    :type std: float
    :param mean: mean or expectation value of the beta-distribution, defaults to None
    :type mean: float, optional
    :param mode: mode of the beta-distribution, defaults to None
    :type mode: float, optional
    :return: shape paramaters of the beta-distribution
    :rtype: tuple[float, float]
    """
    if mean is None:
        mean = mode
    a = mean * ((1. - mean) / (std * std) - 1.)
    b = a * ((1.0 - mean) / mean)
    return a, b


def get_truncnorm_params(
        std:float, mean:float, lower_thresh:float, upper_thresh:float
    ) -> tuple:
    """
    Compute the `a` and `b` parameter which are used in
    ``scipy.stats.truncnorm.rvs()`` from the given standard deviation and mean 
    value as well as from the lower and upper thresholds of the resulting 
    truncated normal distribution.

    :param std: standard deviation of non-truncated normal distribution
    :type std: float
    :param mean: mean value of non-truncated normal distribution, which will be the mode of the resulting truncated normal distr.
    :type mean: float
    :param lower_thresh: lower threshold of resulting truncated normal distr.
    :type lower_thresh: float
    :param upper_thresh: upper threshold of resulting truncated normal distr.
    :type upper_thresh: float
    :return: ``a`` and ``b`` parameters used by ``scipy.stats.truncnorm.rvs()``
    :rtype: tuple[float, float]
    """
    return (lower_thresh - mean) / std, (upper_thresh - mean) / std


def get_values_probs(value:int, std:int, lb:int, ub:int) -> tuple:
    """
    Creates a categorical distribution (details in :func:`RVSampler.get_sampled_values`)

    :param value: most probable value (i.e. mode)
    :type value: int
    :param std: number of discrete values on "one" side of ``value``
    :type std: int
    :param lb: lower bound
    :type lb: int
    :param ub: upper bound
    :type ub: int
    :return: two arrays - first one contains the values and the second one the corresponding probabilities
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    adds = np.arange(std + 1)
    values = np.concatenate((value - np.flip(adds), (value + adds)[1:]))
    mpr = std * 2
    rprobs = [mpr - i - 1 for i in range(std)]
    probs = np.array(np.flip(rprobs).tolist() + [mpr] + rprobs)
    ixs = np.where((values >= lb) & (values <= ub))
    probs = probs[ixs] / np.sum(probs[ixs])
    return values[ixs], probs


class RVSampler(object):
    """
    Class which is used to sample model quantities. It is especially 
    important in the case of multiprocessing as the random state of the 
    `scipy.stats` distributions is initialized in each instanciation of this 
    class (i.e. in each core used for evaluation). Otherwise, the samples 
    within each process/core would be equal.
    """
    
    def __init__(self) -> None:
        gen = np.random.default_rng()
        
        from scipy.stats import norm, gamma, beta, rv_discrete, truncnorm
        self.normal = norm
        self.normal.random_state = gen
        self.gamma = gamma
        self.gamma.random_state = gen
        self.beta = beta
        self.beta.random_state = gen
        self.cat = rv_discrete
        self.cat.random_state = gen
        self.truncnorm = truncnorm
        self.truncnorm.random_state = gen

    def get_sampled_values(
            self, value:float, distr:dict, size:int
        ) -> np.ndarray:
        """
        Get a sample from the distribution specified in the `distr` dictionary. 
        Thus each `dinfo` dict has to contain at least a key `distr_id` to 
        specify the distribution to be used for sampling.
        Depending on the chosen distribution, the provided `value` will be 
        used accordingly.

        **Normal distribution**

        ``value`` will be used as mean value and ``distr`` should 
        contain the following keys

        - ``'std'`` (*float*) - standard deviation

        **Gamma distribution**

        ``value`` will be used as mode and the ``distr`` should contain the 
        following keys

        - ``'std'`` (*float*) - standard deviation

        **Beta distribution**

        ``value`` will be used as mean value and the ``distr`` should 
        contain the following keys

        - ``'std'`` (*float*) - standard deviation

        **Categorical distribution**

        ``value`` will be used as category/integer/discrete value with the 
        highest probability and ``distr`` should contain the following keys 

        - ``'std'`` (*float*) - will be used as number of possible categories/values on "one side" of ``value``
        - ``'lb'`` (*float*) - lower bound of the possible values (truncates if ``std`` would exceed this value)
        - ``'ub'`` (*float*) - upper bound of the possible values (truncates if ``std`` would exceed this value)

        **truncated Normal distribution**

        ``value`` will be the mean/mode of the resulting truncated normal 
        distribution and ``distr`` should contain the following keys

        - ``'std'`` (*float*): standard deviation of original normal distribution
        - ``'lb'`` (*float*) - lower bound of the truncated normal distribution
        - ``'ub'`` (*float*) - upper bound of the truncated normal dsitribution

        :param value: reference value for sampled distribution - usage depends on the sampling distribution
        :type value: float
        :param dinfo: dictionary containing the the values which are necessary to compute the paramters used in ``scipy.stats`` which is necessary to draw samples
        :type dinfo: dict
        :param size: number of samples drawn from the distribution
        :type size: int
        :return: array containing sampled values (length according to ``size``)
        :rtype: numpy.ndarray
        """
        if distr['distr_id'] == DISTRIBUTIONS.NORMAL_1D:
            return self.normal.rvs(
                loc=value, scale=distr['std'], size=size
            )
        elif distr['distr_id'] == DISTRIBUTIONS.GAMMA_1D:
            a, b = get_gamma_params(distr['std'], mode=value)
            return self.gamma.rvs(a=a, loc=0.0, scale=1.0 / b, size=size)
        
        elif distr['distr_id'] == DISTRIBUTIONS.BETA_1D:
            a, b = get_beta_params(distr['std'], mean=value)
            return self.beta.rvs(a, b, size=size)
        
        elif distr['distr_id'] == DISTRIBUTIONS.CATEGORICAL_1D:
            distr = self.cat(values=get_values_probs(
                value, distr['std'], distr['lb'], distr['ub'])
            )
            return distr.rvs(size=size)
        elif distr['distr_id'] == DISTRIBUTIONS.TRUNCNORM_1D:
            a, b = get_truncnorm_params(
                distr['std'], value, distr['lb'], distr['ub']
            )
            return self.truncnorm.rvs(
                a=a, b=b, loc=value, scale=distr['std'], size=size
            )
        else:
            raise ValueError('Provided distribution not supported')
