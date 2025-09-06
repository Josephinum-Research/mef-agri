import functools
import inspect
import numpy as np

from .base import Model, Quantities as Q, Units as U, MODEL_DECORATORS
from ..evaluation.stats_utils import DISTRIBUTION_TYPE


NNMODEL_DECORATORS = MODEL_DECORATORS + [
    '@NeuralNetwork.is_layer', '@NeuralNetwork.is_quantity', 
    '@NeuralNetwork.is_child_model', '@NeuralNetwork.is_required'
]


class Layer(object):
    r"""
    The layer object represents a layer of a neural network consisting of nodes, 
    where the number of nodes equals :math:`n_n`.
    Each node represents a transfer function (the same for all nodes in the 
    layer!) which input results from an affine 
    transformation of the original input vector with an additional shift or 
    bias. The transfer function processes each input separately, i.e. it is a 
    scalar function and does not change the shape of the transformed original 
    input vector.
    The general equation is 
    :math:`\mathbf{o} = f(\mathbf{W}\cdot\mathbf{i} + \mathbf{b})`, where 

    * :math:`\mathbf{o}\in\mathbb{R}^{n_n \times 1}` is the output vector of the layer 
    * :math:`\mathbf{i}\in\mathbb{R}^{n_i \times 1}` is the original input vector, which goes into the layer
    * :math:`\mathbf{W}\in\mathbb{R}^{n_n \times n_i}` is the affine transformation matrix
    * :math:`\mathbf{b}\in\mathbb{R}^{n_n \times 1}` is the bias vector

    This class considers multiple realizations of the affine transformation 
    matrix, the bias vector and also the original input vector. The 
    implementation is done with numpy and multiple realizations means, that the 
    arrays have three dimensions. The first dimension corresponds to the number 
    of realizations or particles and the second and third dimension are the 
    common vector/matrix dimensions.
    """

    ERRMSG_REALIZATIONS = 'If considering multiple realizations of the {}, '
    ERRMSG_REALIZATIONS += 'the first dimension has to match the number of '
    ERRMSG_REALIZATIONS += 'particles set in the model-tree, i.e. {}!'
    ERRMSG_DIMENSIONS = 'The shape of the {} has to be ({}, {})!'
    ERRMSG_ADD = 'In the case of multiple realizations, this shape corresponds '
    ERRMSG_ADD += 'to the second and third dimension of the numpy.ndarray.'
    ERRMSG_LENSHP = 'Shape of the {} has to be either '
    ERRMSG_LENSHP += '(n_realizations, n_nodes, {}) or (n_nodes, {})!'

    def __init__(self, n_nodes:int, n_input:int):
        self._nnds:int = n_nodes
        self._ninp:int = n_input
        self._w:np.ndarray = None
        self._b:np.ndarray = None
        self._nn = None  # parent neural network instance

    @property
    def n_nodes(self) -> int:
        return self._nnds
    
    @property
    def n_inputs(self) -> int:
        return self._ninp

    @property
    def affine_transform(self) -> np.ndarray:
        return self._w
    
    @affine_transform.setter
    def affine_transform(self, value):
        if len(value.shape) == 3:
            if value.shape[0] != self._nn.model_tree.n_particles:
                raise ValueError(self.ERRMSG_REALIZATIONS.format(
                    'affine transformation matrix', 
                    self._nn.model_tree.n_particles))
            if not value.shape[1:] == (self._nnds, self._ninp):
                raise ValueError(self.ERRMSG_DIMENSIONS.format(
                    'affine transformation matrix', self._nnds, self._ninp
                    ) + self.ERRMSG_ADD)
            self._w = value.copy()
        elif len(value.shape) == 2:
            if not value.shape == (self._nnds, self._ninp):
                raise ValueError(self.ERRMSG_DIMENSIONS.format(
                    'affine transformation matrix', self._nnds, self._ninp))
            self._w = np.zeros((1, self._nnds, self._ninp))
            self._w[0, :, :] = value.copy()
        else:
            raise ValueError(self.ERRMSG_LENSHP.format(
                'affine transformation matrix', 'n_inputs', 'n_inputs'
            ))

    @property
    def bias(self) -> np.ndarray:
        return self._b
    
    @bias.setter
    def bias(self, value):
        if len(value.shape) == 3:
            if value.shape[0] != self._nn.model_tree.n_particles:
                raise ValueError(self.ERRMSG_REALIZATIONS.format(
                    'bias vector', self._nn.model_tree.n_particles
                ))
            if not value.shape[1:] == (self._nnds, 1):
                raise ValueError(self.ERRMSG_DIMENSIONS.format(
                    'bias vector', self._nnds, 1) + self.ERRMSG_ADD
                )
            self._b = value.copy()
        elif len(value.shape) == 2:
            if not value.shape == (self._nnds, 1):
                raise ValueError(self.ERRMSG_DIMENSIONS.format(
                    'bias vector', self._nnds, 1
                ))
            self._b = np.zeros((1, self._nnds, 1))
            self._b[0, :, :] = value.copy()
        elif len(value.shape) == 1:
            if not value.shape[0] == self._nnds:
                raise ValueError(self.ERRMSG_DIMENSIONS.format(
                    'bias vector', self._nnds, ''
                ))
            self._b = np.zeros((1, self._nnds, 1))
            self._b[0, :, 0] = value.copy()
        else:
            raise ValueError(self.ERRMSG_LENSHP.format('bias vector', 1, 1))

    @property
    def parent_nn(self):
        return self._nn
    
    @parent_nn.setter
    def parent_nn(self, value):
        self._nn = value

    def compute(self, inp:np.ndarray) -> np.ndarray:
        return self.transfer_function(self._w @ inp + self._b)
    
    def transfer_function(self, inp:np.ndarray) -> np.ndarray:
        msg = 'transfer_function not implemented in Layer-child-class!'
        raise NotImplementedError(msg)
    

class NeuralNetwork(Model):
    def __init__(self, **kwargs):
        self.decorators = NNMODEL_DECORATORS
        self._layers:list = []
        super().__init__(**kwargs)

    def _get_decorated_methods(self) -> str:
        """
        Overwrite this method from the parent class. Only difference is, that 
        ``NeuralNetwork`` class itself should not be screened for decorators.

        :return: source code as string of child classes of ``NeuralNetwork``
        :rtype: str
        """
        def loop(cls, ret):
            ret += inspect.getsource(cls)
            for supcls in cls.__bases__:
                if issubclass(supcls, Model):
                    if supcls in (Model, NeuralNetwork):  # only difference is here
                        pass
                    else:
                        ret = loop(supcls, ret)
            return ret

        sc = ''
        sc = loop(self.__class__, sc)
        decos = []
        for dec  in self.decorators:
            for part in sc.split(dec)[1:]:
                decos.append(part.split('def ')[1].split('(self')[0].strip())
        return decos

    def initialize(self, epoch):
        super().initialize(epoch)

        # initialize the weights and biases of the network with provided values
        for layer, li in zip(self._layers, range(len(self._layers))):
            b = np.zeros((self.model_tree.n_particles, layer.n_nodes, 1))
            atm = np.zeros((
                self.model_tree.n_particles, layer.n_nodes, layer.n_inputs
            ))
            for ni in range(layer.n_nodes):
                bname = 'l{}_b{}'.format(li + 1, ni + 1)
                b[:, ni, 0] = getattr(self, bname)
                for ii in range(layer.n_inputs):
                    wname = 'l{}_w{}_{}'.format(li + 1, ni + 1, ii + 1)
                    atm[:, ni, ii] = getattr(self, wname)
            layer.bias = b
            layer.affine_transform = atm

    def compute(self, inp:np.ndarray) -> np.ndarray:
        li = inp.copy()
        for layer in self._layers:
            li = layer.compute(li)
        return li

    @staticmethod
    def is_layer(layer:Layer, order:int):
        """
        Decorator to define a layer in the neural network.

        :param n_nodes: number of nodes in the layer
        :type n_nodes: int
        :param n_input: number of input quantities to the layer
        :type n_input: int
        :param order: order of the layer within the neural network ( !!! 1-based !!! )
        :type order: int
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(obj):
                # set the layer object as attribute of the neural network with 
                # provided name
                setattr(obj, func.__name__, layer)
                getattr(obj, '_layers').append(layer)
                # provide a reference to the parent neural network in the layer 
                # object
                layer.parent_nn = obj

                # set the layer parameters (weights and biases) as hyper 
                # parameters in the neural network
                lname = 'l{}_'.format(order)
                for i in range(layer.n_nodes):
                    # define the biases
                    bname = lname + 'b{}'.format(i + 1)
                    obj._qs[bname] = {
                        'qtype': Q.HPARAM, 'unit': U.none, 
                        'distr_type': DISTRIBUTION_TYPE.CONTINUOUS
                    }
                    obj._qnames[Q.HPARAM].append(bname)
                    setattr(obj, bname, None)

                    # define the weights
                    for j in range(layer.n_inputs):
                        wname = lname + 'w{}_{}'.format(i + 1, j + 1)
                        obj._qs[wname] = {
                            'qtype': Q.HPARAM, 'unit': U.none,
                            'distr_type': DISTRIBUTION_TYPE.CONTINUOUS
                        }
                        obj._qnames[Q.HPARAM].append(wname)
                        setattr(obj, wname, None)
            return wrapper
        return decorator
