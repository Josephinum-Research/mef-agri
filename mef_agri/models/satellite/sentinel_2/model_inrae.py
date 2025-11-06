import numpy as np

from ...base import Quantities as Q
from ...utils import Units as U
from ...requ import Requirement
from ...nns import Layer, NeuralNetwork


class Layer1(Layer):
    def transfer_function(self, inp):
        outp = (2. / (1. + np.exp(-2. * inp))) - 1.
        return outp
    

class Layer2(Layer):
    def transfer_function(self, inp):
        return inp.copy()


class Sentinel2_LAI10m(NeuralNetwork):
    """
    Neural network introduced in [R7]_ which only uses 10 m bands (NNET10). It 
    computes the leaf area index from the bands B03, B04 and B08 as well as from 
    the sun and view zenith and azimuth angles.

    The input values are defined as observations and are derived from the 
    measurements (i.e. bands and angles) from Sentinel-2. The neural network 
    affine transforms and biases are defined as parameters and the 
    normalization, denormalizationa and min/max values of the cosines of angles 
    are set as fixed float values within this class.

    The output value of lai from the neural network is not checked. Note, that 
    the network has been trained only with values in the range of [0.0, 8.0].

    The model has been tested with the test values provided at
    https://github.com/senbox-org/s2tbx/tree/master/s2tbx-biophysical/src/main/resources/auxdata/3_0/S2A_10m
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._b03_nmin, self._b03_nmax = 0.0, 0.248293766099
        self._b04_nmin, self._b04_nmax = 0.0, 0.305676110468
        self._b08_nmin, self._b08_nmax = 0.00856081085077, 0.760810831896
        self._vzc_nmin, self._vzc_nmax = 0.979624800125, 0.999999999969
        self._szc_nmin, self._szc_nmax = 0.342108564072, 0.927484749175
        self._rac_nmin, self._rac_nmax = -0.999999998674, 0.999999999887
        self._lai_nmin, self._lai_nmax = 0.000233773908827, 13.834592547

    @NeuralNetwork.is_layer(Layer1(5, 6), 1)
    def layer1(self):
        """
        First layer of the NNET10-LAI model of INRAE [R7]_ consisting of five 
        tansig-nodes with six inputs

        * Band B03
        * Band B04
        * Band B08
        * cosine of view zenith angle
        * cosine of sun zenith angle
        * cosine of relative azimuth (sun azimuth minus view azimuth)

        """

    @NeuralNetwork.is_layer(Layer2(1, 5), 2)
    def layer2(self):
        """
        Second/output layer of the NNET10-LAI model of INRAE [R7]_ consisting 
        of one linear/equality node with the five inputs coming from 
        :func:`layer1`.
        """

    @NeuralNetwork.is_quantity(Q.OBS, U.frac)
    def b03(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`o_{\textrm{S2-b3},k}\ [\ ]`

        :return: reflectance value of band 03 of Sentinel-2, i.e. green
        :rtype: numpy.ndarray
        """

    @NeuralNetwork.is_quantity(Q.OBS, U.frac)
    def b04(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`o_{\textrm{S2-b4},k}\ [\ ]`

        :return: reflectance value of band 04 of Sentinel-2, i.e. red
        :rtype: numpy.ndarray
        """

    @NeuralNetwork.is_quantity(Q.OBS, U.frac)
    def b08(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`o_{\textrm{S2-b8},k}\ [\ ]`

        :return: reflectance value of band 08 of Sentinel-2, i.e. nir
        :rtype: numpy.ndarray
        """

    @NeuralNetwork.is_quantity(Q.OBS, U.deg)
    def sun_azimuth(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`o_{\textrm{S2-sa},k}\ [^\circ]`

        :return: azimuth angle of the sun at current site
        :rtype: numpy.ndarray
        """

    @NeuralNetwork.is_quantity(Q.OBS, U.deg)
    def sun_zenith(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`o_{\textrm{S2-sz},k}\ [^\circ]`

        :return: sun zenith angle at current site
        :rtype: numpy.ndarray
        """

    @NeuralNetwork.is_quantity(Q.OBS, U.deg)
    def view_azimuth(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`o_{\textrm{S2-va},k}\ [^\circ]`

        :return: azimuth angle of the satellite at current site
        :rtype: numpy.ndarray
        """

    @NeuralNetwork.is_quantity(Q.OBS, U.deg)
    def view_zenith(self) -> np.ndarray:
        r"""
        MQ - Observation

        :math:`o_{\textrm{S2-vz},k}\ [^\circ]`

        :return: zenith angle of the satellite at current site
        :rtype: numpy.ndarray
        """

    @NeuralNetwork.is_quantity(Q.ROUT, U.none)
    def lai(self) -> np.ndarray:
        r"""
        MQ - Random Output

        :math:`o_{\textrm{S2-lai},k}\ [\ ]`

        :return: leaf area index computed with neural network
        :rtype: numpy.ndarray
        """

    @NeuralNetwork.is_required('lai_max', 'crop.leaves', U.none)
    def lai_max(self) -> Requirement:
        r"""
        RQ - ``'lai_max'`` from model with id ``'crop.leaves'``

        :math:`c_{\textrm{L-lmx},0}\ [\ ]` - [R2]_ (equ. 8, table 2)

        :return: max. possible value of lai 
        :rtype: mef_agri.models.requ.Requirement
        """

    def initialize(self, epoch):
        super().initialize(epoch)

    def update(self, epoch):
        super().update(epoch)

        cp = self.model_tree.get_model('zone').crop_rotation.crop_present
        if not cp:
            self.reset_quantities(self.random_output_names, force=True)
            self.reset_quantities(self.observation_names, force=True)
            return
        
        if self.get_obs_epoch('b03') != epoch:
            # no new obserations are available
            self.reset_quantities(self.random_output_names, force=True)
            self.reset_quantities(self.observation_names, force=True)
            return

        # cosines of the required angles
        vzcos = np.cos(np.deg2rad(self.view_zenith))
        szcos = np.cos(np.deg2rad(self.sun_zenith))
        racos = np.cos(np.deg2rad(self.sun_azimuth - self.view_azimuth))
        self.cos_vz = vzcos.copy()
        self.cos_sz = szcos.copy()
        self.cos_ra = racos.copy()

        # normalization of inputs
        inp = np.zeros((self.model_tree.n_particles, 6, 1))
        inp[:, 0, 0] = self._normalize(self.b03, self._b03_nmin, self._b03_nmax)
        inp[:, 1, 0] = self._normalize(self.b04, self._b04_nmin, self._b04_nmax)
        inp[:, 2, 0] = self._normalize(1.5 * self.b08, self._b08_nmin, self._b08_nmax)
        inp[:, 3, 0] = self._normalize(vzcos, self._vzc_nmin, self._vzc_nmax)
        inp[:, 4, 0] = self._normalize(szcos, self._szc_nmin, self._szc_nmax)
        inp[:, 5, 0] = self._normalize(racos, self._rac_nmin, self._rac_nmax)

        # computation of neural network and denormalization of output
        res = self.compute(inp)
        self.lai_nonnorm = res[:, 0, 0].copy()
        self.lai = self._denormalize(
            self.lai_nonnorm, self._lai_nmin, self._lai_nmax
        )
        lainoise = np.random.normal(0.0, 0.02, size=len(self.lai))
        laimax = np.mean(self.lai_max.value) * 1.1
        self.lai = np.where(self.lai <= laimax, self.lai, laimax + lainoise)

    def _normalize(self, arr:np.ndarray, vmin:float, vmax:float) -> np.ndarray:
        aux =  2. * (arr - vmin) / (vmax - vmin) - 1.
        return aux
    
    def _denormalize(
            self, arr:np.ndarray, vmin:float, vmax:float
        ) -> np.ndarray:
        return 0.5 * (arr + 1.) * (vmax - vmin) + vmin

    def _cos_min_max(
            self, cosvals:np.ndarray, cosmin:float, cosmax:float
        ) -> np.ndarray:
        ccs = [
            cosvals < cosmin,
            cosmin <= cosvals <= cosmax,
            cosmax < cosvals
        ]
        cosvals_constr = np.select(ccs, cosmin, cosvals, cosmax)
        return cosvals_constr
