"""
Unified neural network library.
Includes

* Flexible NN creation of popular architectures
* Optimizer creation
* Training functionality
* Parameter access
* Saving and loading networks and optimizer states.
"""
import numpy as np
import os
import time
import warnings
from dataclasses import dataclass
from itertools import count
from typing import Callable, Union, Sequence, Dict, TypeVar, Any, Optional

from .backend import default_backend, Backend, BACKENDS, ComputeDevice
from .backend._backend import init_backend
from . import math, non_channel
from .math import Tensor, Shape, use as _use, batch, layout, load, shape, convert, to_device, pack_dims, EMPTY_SHAPE, DimFilter
from .math.perm import random_permutation

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


def save_state(obj: Union[Network, Optimizer], path: str) -> str:
    """
    Write the state of a module or optimizer to a file.

    See Also:
        `load_state()`

    Args:
        obj: `torch.Network or torch.optim.Optimizer`
        path: File path as `str`.

    Returns:
        Path to the saved file.
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


def set_learning_rate(optimizer: Optimizer, learning_rate: Union[float, Tensor]):
    """
    Sets the global learning rate for the given optimizer.

    Args:
        optimizer (optim.Optimizer): The optimizer whose learning rate needs to be updated.
        learning_rate (float): The new learning rate to set.
    """
    _native_lib().set_learning_rate(optimizer, float(learning_rate))


def get_learning_rate(optimizer: Optimizer) -> float:
    """
    Returns the global learning rate of the given optimizer.

    Args:
        optimizer (optim.Optimizer): The optimizer whose learning rate needs to be retrieved.

    Returns:
        float: The learning rate of the optimizer.
    """
    return _native_lib().get_learning_rate(optimizer)


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


@dataclass(frozen=True)
class TrainingState:
    name: str
    model: Network
    optimizer: Optimizer
    learning_rate: float
    epoch: int
    max_epochs: Optional[int]
    iter: int
    max_iter: Optional[int]
    is_epoch_end: bool
    epoch_loss: Tensor
    batch_loss: Optional[Tensor]
    additional_batch_output: Optional[tuple]
    indices: Tensor
    termination_reason: Optional[str]
    peak_memory: Optional[int]

    @property
    def current(self) -> int:
        return self.epoch if self.is_epoch_end else self.iter

    @property
    def max(self) -> int:
        return self.max_epochs if self.is_epoch_end else self.max_iter

    @property
    def mean_loss(self) -> float:
        return float(self.epoch_loss) if self.is_epoch_end else float(math.mean(self.batch_loss, 'dset_linear'))


def train(name: Optional[str], model, optimizer, loss_fn: Callable,
          *files_or_data: Union[str, Tensor],
          max_epochs: int = None, max_iter: int = None, max_hours: float = None, stop_on_loss: float = None,
          batch_size: int = 1, file_shape: Shape = EMPTY_SHAPE, dataset_dims: DimFilter = batch, device: ComputeDevice = None, drop_last=False, loss_kwargs=None,
          lr_schedule_iter=None, checkpoint_frequency=None, loader=math.load,
          on_iter_end: Callable = None, on_epoch_end: Callable = None,
          measure_peak_memory: bool = True) -> TrainingState:
    """
    Call `update_weights()` for each batch in a loop for each epoch.

    Args:
        name: Name of the model. This is used as a name to save the model and optimizer states.
            If not specified, no model or optimizer states are saved.
        model: PyTorch module or Keras model (TensorFlow).
        optimizer: PyTorch or TensorFlow/Keras optimizer.
        loss_fn: Loss function for training. This function should take the data as input, run the model and return the loss. It may return additional outputs, but the loss must be the first value.
        *files_or_data: Training data or file names containing training data or a mixture of both. Files are loaded using `loader`.
        max_epochs: Epoch limit.
        max_iter: Iteration limit. The number of iterations depends on the batch size and the number of files.
        max_hours: Training time limit in hours (`float`).
        stop_on_loss: Stop training if the mean epoch loss falls below this value.
        batch_size: Batch size for training. The batch size is limited by the number of data points in the dataset.
        file_shape: Shape of data stored in each file.
        dataset_dims: Which dims of the training data list training examples, as opposed to features of data points.
        device: Device to use for training. If `None`, the default device is used.
        drop_last: If `True`, drop the last batch if it is smaller than `batch_size`.
        loss_kwargs: Keyword arguments passed to `loss_fn`.
        lr_schedule_iter: Function `(i: int) -> float` that returns the learning rate for iteration `i`. If `None`, the learning rate of the `optimizer` is used as is.
        checkpoint_frequency: If not `None`, save the model and optimizer state every `checkpoint_frequency` epochs.
        loader: Function `(file: str) -> data: Tensor` to load data from files. Defaults to `phiml.math.load()`.
        on_iter_end: Function `(i: int, max_iter: int, name: str, model, optimizer, learning_rate, loss, *additional_output) -> None` called after each iteration. The function is called with the current iteration number `i` starting at 0, the maximum number of iterations `max_iter`, the name of the model `name`, the model `model`, the optimizer `optimizer`, the learning rate `learning_rate`, the loss value `loss` and any additional output from `loss_fn`.
        on_epoch_end: Function `(epoch: int, max_epochs: int, name: str, model, optimizer, learning_rate, epoch_loss) -> None` called after each epoch. The function is called with the current epoch number `epoch` starting at 0, the maximum number of epochs `max_epochs`, the name of the model `name`, the model `model`, the optimizer `optimizer`, the learning rate `learning_rate` and the average loss for the epoch `epoch_loss`.
        measure_peak_memory: If `True`, measure the peak memory usage during training and store it in the returned `TrainingState`. This is only supported by some backends.

    Returns:
        `TrainingResult` containing the termination reason, last epoch and last iteration.
    """
    files_or_data = [layout(fs) if isinstance(fs, str) else fs for fs in files_or_data]
    data_shape = shape(files_or_data) & file_shape
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs
    device = device if device is not None else default_backend().get_default_device()
    if measure_peak_memory:
        default_backend().reset_peak_memory(device)
    default_backend().set_default_device('CPU')
    data = [convert(math.map(lambda f: loader(f), fs, map_name="Loading data")) if isinstance(fs, str) or (isinstance(fs, Tensor) and fs.dtype.kind == object) else fs for fs in files_or_data]
    default_backend().set_default_device(device)
    data_shape = shape(data)
    dataset_dims = data_shape.only(dataset_dims)
    batch_size = min(batch_size, dataset_dims.volume)
    batch_count = dataset_dims.volume // batch_size if drop_last else (dataset_dims.volume + batch_size - 1) // batch_size
    name and os.makedirs(name, exist_ok=True)
    learning_rate = None if lr_schedule_iter is not None else get_learning_rate(optimizer)
    if max_epochs is None and max_iter is not None:
        max_epochs = int(np.ceil(max_iter / batch_count))
    elif max_epochs is not None and max_iter is not None:
        max_iter = max_epochs * batch_count
    termination_reason = None
    niter = 0
    epoch = 0
    t0 = time.perf_counter()
    for epoch in range(max_epochs) if max_epochs is not None else count():
        default_backend().set_default_device('CPU')
        indices = pack_dims(random_permutation(dataset_dims, dims=dataset_dims), non_channel, batch('dset_linear'))
        default_backend().set_default_device(device)
        epoch_loss = 0
        for i in range(batch_count):
            if lr_schedule_iter is not None:
                learning_rate = lr_schedule_iter(niter)
                set_learning_rate(optimizer, learning_rate)
            batch_idx = indices.dset_linear[i * batch_size:(i + 1) * batch_size]
            data_batch = to_device(math.slice(data, batch_idx), device)
            output = update_weights(model, optimizer, loss_fn, *data_batch, **loss_kwargs)
            loss, *additional_output = output if isinstance(output, (tuple, list)) else (output,)
            epoch_loss += math.sum(loss, 'dset_linear')
            if on_iter_end is not None:
                try:
                    on_iter_end(TrainingState(name, model, optimizer, learning_rate, epoch, max_epochs, niter, max_iter, False, epoch_loss / ((i+1)*batch_count), loss, additional_output, batch_idx, None, None))
                except StopTraining as stop:
                    termination_reason = stop.reason
                    break
            niter += 1
            if max_iter is not None and niter >= max_iter:
                termination_reason = 'max_iter'
                break
            if max_hours is not None and time.perf_counter() - t0 > max_hours * 3600:
                termination_reason = 'max_hours'
                break
        if termination_reason is not None:
            break
        if name and checkpoint_frequency is not None and (epoch + 1) % checkpoint_frequency == 0:
            save_state(model, f"{name}/model_{epoch + 1}")
            save_state(optimizer, f"{name}/optimizer_{epoch + 1}")
        epoch_loss /= indices.dset_linear.size
        if on_epoch_end is not None:
            try:
                on_epoch_end(TrainingState(name, model, optimizer, learning_rate, epoch, max_epochs, niter, max_iter, True, epoch_loss, None, None, indices, None, None))
            except StopTraining as stop:
                termination_reason = stop.reason
                break
        if stop_on_loss is not None and (epoch_loss.mean < stop_on_loss):
            termination_reason = 'stop_on_loss'
            break
    peak_memory = default_backend().get_peak_memory(device) if measure_peak_memory else None
    return TrainingState(name, model, optimizer, learning_rate, epoch, max_epochs, niter, max_iter, True, None, None, None, None, termination_reason, peak_memory)


class StopTraining(Exception):
    """ This exception is raised by the `on_epoch_end` or `on_iter_end` callbacks to stop training. """
    def __init__(self, reason: str = 'stop'):
        super().__init__(reason)
        self.reason = reason
