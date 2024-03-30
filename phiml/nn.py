"""
Unified neural network library.
Includes

* Flexible NN creation of popular architectures
* Optimizer creation
* Training functionality
* Parameter access
* Saving and loading networks and optimizer states.
"""
import warnings
from typing import Callable, Union, Sequence, Dict, TypeVar

from .backend import default_backend, Backend, BACKENDS
from .backend._backend import init_backend
from .math import Tensor, use as _use


use = _use


def _native_lib():
    if default_backend().supports(Backend.nn_library):
        return default_backend().nn_library()
    ml_backends = [b for b in BACKENDS if b.supports(Backend.nn_library)]
    if not ml_backends:
        if not init_backend('all-imported'):
            raise RuntimeError(f"No ML library available. Please import either jax, torch, or tensorflow.")
        ml_backends = [b for b in BACKENDS if b.supports(Backend.nn_library)]
    if len(ml_backends) > 1:
        warnings.warn(f"Multiple ML libraries loaded {tuple(ml_backends)} but none set as default. Defaulting to {ml_backends[0]}.", RuntimeWarning, stacklevel=3)
    return ml_backends[0].nn_library()


Network = TypeVar('Network')
Optimizer = TypeVar('Optimizer')


def parameter_count(net: Network) -> int:
    """
    Counts the number of parameters in a model.

    See Also:
        `get_parameters()`.

    Args:
        net: PyTorch model

    Returns:
        Total parameter count as `int`.
    """
    return sum([value.shape.volume for name, value in get_parameters(net).items()])


def get_parameters(net: Network) -> Dict[str, Tensor]:
    """
    Returns all parameters of a neural network.

    Args:
        net: Neural network.

    Returns:
        `dict` mapping parameter names to `phiml.math.Tensor`s.
    """
    return _native_lib().get_parameters(net)


def save_state(obj: Union[Network, Optimizer], path: str):
    """
    Write the state of a module or optimizer to a file.

    See Also:
        `load_state()`

    Args:
        obj: `torch.Network or torch.optim.Optimizer`
        path: File path as `str`.
    """
    return _native_lib().save_state(**locals())


def load_state(obj: Union[Network, Optimizer], path: str):
    """
    Read the state of a module or optimizer from a file.

    See Also:
        `save_state()`

    Args:
        obj: `torch.Network or torch.optim.Optimizer`
        path: File path as `str`.
    """
    return _native_lib().load_state(**locals())


def update_weights(net: Network, optimizer: Optimizer, loss_function: Callable, *loss_args, **loss_kwargs):
    """
    Computes the gradients of `loss_function` w.r.t. the parameters of `net` and updates its weights using `optimizer`.

    This is the PyTorch version. Analogue functions exist for other learning frameworks.

    Args:
        net: Learning model.
        optimizer: Optimizer.
        loss_function: Loss function, called as `loss_function(*loss_args, **loss_kwargs)`.
        *loss_args: Arguments given to `loss_function`.
        **loss_kwargs: Keyword arguments given to `loss_function`.

    Returns:
        Output of `loss_function`.
    """
    return _native_lib().update_weights(net, optimizer, loss_function, *loss_args, **loss_kwargs)


def adam(net: Network, learning_rate: float = 1e-3, betas=(0.9, 0.999), epsilon=1e-07):
    """
    Creates an Adam optimizer for `net`, alias for [`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).
    Analogue functions exist for other learning frameworks.
    """
    return _native_lib().adam(**locals())


def sgd(net: Network, learning_rate: float = 1e-3, momentum=0., dampening=0., weight_decay=0., nesterov=False):
    """
    Creates an SGD optimizer for 'net', alias for ['torch.optim.SGD'](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
    Analogue functions exist for other learning frameworks.
    """
    return _native_lib().sgd(**locals())


def adagrad(net: Network, learning_rate: float = 1e-3, lr_decay=0., weight_decay=0., initial_accumulator_value=0., eps=1e-10):
    """
    Creates an Adagrad optimizer for 'net', alias for ['torch.optim.Adagrad'](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)
    Analogue functions exist for other learning frameworks.
    """
    return _native_lib().adagrad(**locals())


def rmsprop(net: Network, learning_rate: float = 1e-3, alpha=0.99, eps=1e-08, weight_decay=0., momentum=0., centered=False):
    """
    Creates an RMSProp optimizer for 'net', alias for ['torch.optim.RMSprop'](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
    Analogue functions exist for other learning frameworks.
    """
    return _native_lib().rmsprop(**locals())


def mlp(in_channels: int,
              out_channels: int,
              layers: Sequence[int],
              batch_norm=False,
              activation: Union[str, Callable] = 'ReLU',
              softmax=False) -> Network:
    """
    Fully-connected neural networks are available in Φ-ML via mlp().

    Args:
        in_channels: size of input layer, int
        out_channels = size of output layer, int
        layers: tuple of linear layers between input and output neurons, list or tuple
        activation: activation function used within the layers, string
        batch_norm: use of batch norm after each linear layer, bool

    Returns:
        Dense net model as specified by input arguments
    """
    return _native_lib().mlp(**locals())


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: Union[int, Sequence] = 16,
          batch_norm: bool = True,
          activation: Union[str, type] = 'ReLU',
          in_spatial: Union[tuple, int] = 2,
          periodic=False,
          use_res_blocks: bool = False,
          down_kernel_size=3,
          up_kernel_size=3) -> Network:
    """
    Built-in U-net architecture, classically popular for Semantic Segmentation in Computer Vision, composed of downsampling and upsampling layers.

    Args:
        in_channels: input channels of the feature map, dtype: int
        out_channels: output channels of the feature map, dtype: int
        levels: number of levels of down-sampling and upsampling, dtype: int
        filters: filter sizes at each down/up sampling convolutional layer, if the input is integer all conv layers have the same filter size,
        activation: activation function used within the layers, dtype: string
        batch_norm: use of batchnorm after each conv layer, dtype: bool
        in_spatial: spatial dimensions of the input feature map, dtype: int
        use_res_blocks: use convolutional blocks with skip connections instead of regular convolutional blocks, dtype: bool
        down_kernel_size: Kernel size for convolutions on the down-sampling (first half) side of the U-Net.
        up_kernel_size: Kernel size for convolutions on the up-sampling (second half) of the U-Net.

    Returns:
        U-net model as specified by input arguments.
    """
    return _native_lib().u_net(**locals())


def conv_net(in_channels: int,
             out_channels: int,
             layers: Sequence[int],
             batch_norm: bool = False,
             activation: Union[str, type] = 'ReLU',
             in_spatial: Union[int, tuple] = 2,
             periodic=False) -> Network:
    """
    Built in Conv-Nets are also provided. Contrary to the classical convolutional neural networks, the feature map spatial size remains the same throughout the layers. Each layer of the network is essentially a convolutional block comprising of two conv layers. A filter size of 3 is used in the convolutional layers.

    Args:
        in_channels: input channels of the feature map, dtype: int
        out_channels: output channels of the feature map, dtype: int
        layers: list or tuple of output channels for each intermediate layer between the input and final output channels, dtype: list or tuple
        activation: activation function used within the layers, dtype: string
        batch_norm: use of batchnorm after each conv layer, dtype: bool
        in_spatial: spatial dimensions of the input feature map, dtype: int

    Returns:
        Conv-net model as specified by input arguments
    """
    return _native_lib().conv_net(**locals())


def res_net(in_channels: int,
            out_channels: int,
            layers: Sequence[int],
            batch_norm: bool = False,
            activation: Union[str, type] = 'ReLU',
            in_spatial: Union[int, tuple] = 2,
            periodic=False) -> Network:
    """
    Similar to the conv-net, the feature map spatial size remains the same throughout the layers.
    These networks use residual blocks composed of two conv layers with a skip connection added from the input to the output feature map.
    A default filter size of 3 is used in the convolutional layers.

    Args:
        in_channels: input channels of the feature map, dtype: int
        out_channels: output channels of the feature map, dtype: int
        layers: list or tuple of output channels for each intermediate layer between the input and final output channels, dtype: list or tuple
        activation: activation function used within the layers, dtype: string
        batch_norm: use of batchnorm after each conv layer, dtype: bool
        in_spatial: spatial dimensions of the input feature map, dtype: int

    Returns:
        Res-net model as specified by input arguments
    """
    return _native_lib().res_net(**locals())


def conv_classifier(in_features: int,
                    in_spatial: Union[tuple, list],
                    num_classes: int,
                    blocks=(64, 128, 256, 256, 512, 512),
                    block_sizes=(2, 2, 3, 3, 3),
                    dense_layers=(4096, 4096, 100),
                    batch_norm=True,
                    activation='ReLU',
                    softmax=True,
                    periodic=False):
    """
    Based on VGG16.
    """
    return _native_lib().conv_classifier(**locals())


def invertible_net(num_blocks: int = 3,
                   construct_net: Union[str, Callable] = 'u_net',
                   **construct_kwargs):
    """
    Invertible NNs are capable of inverting the output tensor back to the input tensor initially passed.
    These networks have far-reaching applications in predicting input parameters of a problem given its observations.
    Invertible nets are composed of multiple concatenated coupling blocks wherein each such block consists of arbitrary neural networks.

    Currently, these arbitrary neural networks could be set to u_net(default), conv_net, res_net or mlp blocks with in_channels = out_channels.
    The architecture used is popularized by ["Real NVP"](https://arxiv.org/abs/1605.08803).

    Invertible nets are only implemented for PyTorch and TensorFlow.

    Args:
        num_blocks: number of coupling blocks inside the invertible net, dtype: int
        construct_net: Function to construct one part of the neural network.
            This network must have the same number of inputs and outputs.
            Can be a `lambda` function or one of the following strings: `mlp, u_net, res_net, conv_net`
        construct_kwargs: Keyword arguments passed to `construct_net`.

    Returns:
        Invertible neural network model
    """
    return _native_lib().invertible_net(num_blocks, construct_net, **construct_kwargs)


# def fno(in_channels: int,
#         out_channels: int,
#         mid_channels: int,
#         modes: Sequence[int],
#         activation: Union[str, type] = 'ReLU',
#         batch_norm: bool = False,
#         in_spatial: int = 2):
#     """
#     ["Fourier Neural Operator"](https://github.com/zongyi-li/fourier_neural_operator) network contains 4 layers of the Fourier layer.
#     1. Lift the input to the desire channel dimension by self.fc0 .
#     2. 4 layers of the integral operators u' = (W + K)(u). W defined by self.w; K defined by self.conv .
#     3. Project from the channel space to the output space by self.fc1 and self.fc2.
#
#     Args:
#         in_channels: input channels of the feature map, dtype: int
#         out_channels: output channels of the feature map, dtype: int
#         mid_channels: channels used in Spectral Convolution Layers, dtype: int
#         modes: Fourier modes for each spatial channel, dtype: List[int] or int (in case all number modes are to be the same for each spatial channel)
#         activation: activation function used within the layers, dtype: string
#         batch_norm: use of batchnorm after each conv layer, dtype: bool
#         in_spatial: spatial dimensions of the input feature map, dtype: int
#
#     Returns:
#         Fourier Neural Operator model as specified by input arguments.
#     """
#     return _native_lib().fno(**locals())
