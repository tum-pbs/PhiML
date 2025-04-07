"""
PyTorch implementation of the unified machine learning API.
Equivalent functions also exist for the other frameworks.

For API documentation, see `phiml.nn`.
"""
from typing import Callable, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from . import TORCH
from ._torch_backend import register_module_call
from ... import math


def get_parameters(net: nn.Module, wrap=True) -> dict:
    if not wrap:
        return {name: param for name, param in net.named_parameters()}
    result = {}
    for name, param in net.named_parameters():
        if name.endswith('.weight'):
            order = [
                None,
                'output',
                'input,output',
                'x,input,output',
                'x,y,input,output',
                'x,y,z,input,output'
            ][param.ndim]
            uml_tensor = math.wrap(param, math.channel(order))
        elif name.endswith('.bias'):
            uml_tensor = math.wrap(param, math.channel('output'))
        else:
            uml_tensor = math.wrap(param, math.channel(",".join([f"d{i}" for i in range(param.ndim)])))
        result[name] = uml_tensor
    return result


def save_state(obj: Union[nn.Module, optim.Optimizer], path: str):
    if not path.endswith('.pth'):
        path += '.pth'
    torch.save(obj.state_dict(), path)
    return path


def load_state(obj: Union[nn.Module, optim.Optimizer], path: str):
    if not path.endswith('.pth'):
        path += '.pth'
    obj.load_state_dict(torch.load(path))


def update_weights(net: Union[nn.Module, Sequence[nn.Module]], optimizer: optim.Optimizer, loss_function: Callable, *loss_args, check_nan=False, **loss_kwargs):
    optimizer.zero_grad()
    output = loss_function(*loss_args, **loss_kwargs)
    loss = output[0] if isinstance(output, tuple) else output
    loss = loss if isinstance(loss, torch.Tensor) else loss.sum
    loss.backward()
    if isinstance(optimizer, optim.LBFGS):
        def closure():
            result = loss_function(*loss_args, **loss_kwargs)
            loss_val = result[0] if isinstance(result, tuple) else result
            return loss_val.sum
        optimizer.step(closure=closure)
    else:
        if check_nan:
            nets = [net] if isinstance(net, nn.Module) else net
            for subnet in nets:
                for p in subnet.parameters():
                    if not torch.all(torch.isfinite(p.grad)):
                        raise RuntimeError(f"NaN in network gradient detected. Parameter: {p}")
        optimizer.step()
    return output


def _as_module(net: Union[nn.Module, Sequence[nn.Module]]) -> nn.Module:
    if isinstance(net, nn.Module):
        return net
    elif isinstance(net, (list, tuple)):
        class ModuleContainer(nn.Module):
            def __init__(self, modules):
                super(ModuleContainer, self).__init__()
                for i, module in enumerate(modules):
                    self.add_module(f'module{i}', module)

            def forward(self, x):
                raise AssertionError("Container module must not be called.")
        return ModuleContainer(net)
    else:
        raise ValueError(f"Invalid network type {type(net)}.")


def adam(net: Union[nn.Module, Sequence[nn.Module]], learning_rate: float = 1e-3, betas=(0.9, 0.999), epsilon=1e-07):
    return optim.Adam(_as_module(net).parameters(), learning_rate, betas, epsilon)


def sgd(net: Union[nn.Module, Sequence[nn.Module]], learning_rate: float = 1e-3, momentum=0., dampening=0., weight_decay=0., nesterov=False):
    return optim.SGD(_as_module(net).parameters(), learning_rate, momentum, dampening, weight_decay, nesterov)


def adagrad(net: Union[nn.Module, Sequence[nn.Module]], learning_rate: float = 1e-3, lr_decay=0., weight_decay=0., initial_accumulator_value=0., eps=1e-10):
    return optim.Adagrad(_as_module(net).parameters(), learning_rate, lr_decay, weight_decay, initial_accumulator_value, eps)


def rmsprop(net: Union[nn.Module, Sequence[nn.Module]], learning_rate: float = 1e-3, alpha=0.99, eps=1e-08, weight_decay=0., momentum=0., centered=False):
    return optim.RMSprop(_as_module(net).parameters(), learning_rate, alpha, eps, weight_decay, momentum, centered)


def set_learning_rate(optimizer: optim.Optimizer, learning_rate: float):
    """
    Sets the global learning rate for the given optimizer.

    Args:
        optimizer (optim.Optimizer): The optimizer whose learning rate needs to be updated.
        learning_rate (float): The new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def get_learning_rate(optimizer: optim.Optimizer):
    """
    Returns the current learning rate of the optimizer.

    Args:
        optimizer (optim.Optimizer): The optimizer whose learning rate needs to be retrieved.

    Returns:
        float: The current learning rate of the optimizer.
    """
    return optimizer.param_groups[0]['lr']


def _bias0(conv):
    def initialize(*args, **kwargs):
        module = conv(*args, **kwargs)
        module.bias.data.fill_(0)
        return module
    return initialize


CONV = [None, _bias0(nn.Conv1d), _bias0(nn.Conv2d), _bias0(nn.Conv3d)]
NORM = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
ACTIVATIONS = {'ReLU': nn.ReLU, 'Sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'SiLU': nn.SiLU, 'GeLU': nn.GELU}


def mlp(in_channels: int,
              out_channels: int,
              layers: Sequence[int],
              batch_norm=False,
              activation: Union[str, Callable] = 'ReLU',
              softmax=False) -> nn.Module:
    layers = [in_channels, *layers, out_channels]
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    net = DenseNet(layers, activation, batch_norm, softmax)
    return net.to(TORCH.get_default_device().ref)


class DenseNet(nn.Module):

    def __init__(self,
                 layers: list,
                 activation: type,
                 batch_norm: bool,
                 use_softmax: bool):
        super(DenseNet, self).__init__()
        self._layers = layers
        self._activation = activation
        self._batch_norm = batch_norm
        for i, (s1, s2) in enumerate(zip(layers[:-2], layers[1:-1])):
            self.add_module(f'linear{i}', _bias0(nn.Linear)(s1, s2, bias=True))
            if batch_norm:
                self.add_module(f'norm{i}', nn.BatchNorm1d(s2))
        self.add_module(f'linear_out', _bias0(nn.Linear)(layers[-2], layers[-1], bias=True))
        self.softmax = nn.Softmax() if use_softmax else None

    def forward(self, x):
        register_module_call(self)
        x = TORCH.as_tensor(x)
        for i in range(len(self._layers) - 2):
            x = self._activation()(getattr(self, f'linear{i}')(x))
            if self._batch_norm:
                x = getattr(self, f'norm{i}')(x)
        x = getattr(self, f'linear_out')(x)
        if self.softmax:
            x = self.softmax(x)
        return x


class DenseResNetBlock(nn.Module):

    def __init__(self, layers, batch_norm, activation):
        super(DenseResNetBlock, self).__init__()
        self._layers = layers
        self._activation = activation
        self._batch_norm = batch_norm
        for i, (s1, s2) in enumerate(zip(layers[:-1], layers[1:])):
            self.add_module(f'linear{i}', _bias0(nn.Linear)(s1, s2, bias=True))
            if batch_norm:
                self.add_module(f'norm{i}', nn.BatchNorm1d(s2))

    def forward(self, x):
        x0 = x
        for i in range(len(self._layers) - 1):
            x = self._activation()(getattr(self, f'linear{i}')(x))
            if self._batch_norm:
                x = getattr(self, f'norm{i}')(x)
        return x + x0


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
          up_kernel_size=3) -> nn.Module:
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    net = UNet(d, in_channels, out_channels, filters, batch_norm, activation, periodic, use_res_blocks, down_kernel_size, up_kernel_size)
    return net.to(TORCH.get_default_device().ref)


class UNet(nn.Module):

    def __init__(self, d: int, in_channels: int, out_channels: int, filters: tuple, batch_norm: bool, activation: type, periodic: bool, use_res_blocks: bool, down_kernel_size: int, up_kernel_size: int):
        super(UNet, self).__init__()
        self._levels = len(filters)
        self._spatial_rank = d
        if use_res_blocks:
            self.inc = ResNetBlock(d, in_channels, filters[0], batch_norm, activation, periodic, down_kernel_size)
        else:
            self.inc = DoubleConv(d, in_channels, filters[0], filters[0], batch_norm, activation, periodic, down_kernel_size)
        for i in range(1, self._levels):
            self.add_module(f'down{i}', Down(d, filters[i - 1], filters[i], batch_norm, activation, periodic, use_res_blocks, down_kernel_size))
            self.add_module(f'up{i}', Up(d, filters[-i] + filters[-i - 1], filters[-i - 1], batch_norm, activation, periodic, use_res_blocks, up_kernel_size))
        self.add_module('outc', CONV[d](filters[0], out_channels, kernel_size=1))

    def forward(self, x):
        register_module_call(self)
        x = TORCH.as_tensor(x)
        x = self.inc(x)
        xs = [x]
        for i in range(1, self._levels):
            x = getattr(self, f'down{i}')(x)
            xs.insert(0, x)
        for i in range(1, self._levels):
            x = getattr(self, f'up{i}')(x, xs[i])
        x = self.outc(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, d: int, in_channels: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: type, periodic: bool, kernel_size=3):
        super().__init__()
        self.add_module('double_conv', nn.Sequential(
            CONV[d](in_channels, mid_channels, kernel_size=kernel_size, padding=1, padding_mode='circular' if periodic else 'zeros'),
            NORM[d](mid_channels) if batch_norm else nn.Identity(),
            activation(),
            CONV[d](mid_channels, out_channels, kernel_size=kernel_size, padding=1, padding_mode='circular' if periodic else 'zeros'),
            NORM[d](out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        ))

    def forward(self, x):
        return self.double_conv(x)


MAX_POOL = [None, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]


class Down(nn.Module):
    """Downscaling with maxpool then double conv or resnet_block"""

    def __init__(self, d: int, in_channels: int, out_channels: int, batch_norm: bool, activation: Union[str, type], use_res_blocks: bool, periodic, kernel_size: int):
        super().__init__()
        self.add_module('maxpool', MAX_POOL[d](2))
        if use_res_blocks:
            self.add_module('conv', ResNetBlock(d, in_channels, out_channels, batch_norm, activation, periodic, kernel_size))
        else:
            self.add_module('conv', DoubleConv(d, in_channels, out_channels, out_channels, batch_norm, activation, periodic, kernel_size))

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    _MODES = [None, 'linear', 'bilinear', 'trilinear']

    def __init__(self, d: int, in_channels: int, out_channels: int, batch_norm: bool, activation: type, periodic: bool, use_res_blocks: bool, kernel_size: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=Up._MODES[d])
        if use_res_blocks:
            self.conv = ResNetBlock(d, in_channels, out_channels, batch_norm, activation, periodic, kernel_size)
        else:
            self.conv = DoubleConv(d, in_channels, out_channels, in_channels // 2, batch_norm, activation, periodic, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diff = [x2.size()[i] - x1.size()[i] for i in range(2, len(x1.shape))]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvNet(nn.Module):

    def __init__(self, in_spatial, in_channels, out_channels, layers, batch_norm, activation, periodic: bool):
        super(ConvNet, self).__init__()
        activation = ACTIVATIONS[activation]
        if len(layers) < 1:
            layers.append(out_channels)
        self.layers = layers
        self.add_module(f'Conv_in', nn.Sequential(
            CONV[in_spatial](in_channels, layers[0], kernel_size=3, padding=1, padding_mode='circular' if periodic else 'zeros'),
            NORM[in_spatial](layers[0]) if batch_norm else nn.Identity(),
            activation()))
        for i in range(1, len(layers)):
            self.add_module(f'Conv{i}', nn.Sequential(
                CONV[in_spatial](layers[i - 1], layers[i], kernel_size=3, padding=1, padding_mode='circular' if periodic else 'zeros'),
                NORM[in_spatial](layers[i]) if batch_norm else nn.Identity(),
                activation()))
        self.add_module(f'Conv_out', CONV[in_spatial](layers[len(layers) - 1], out_channels, kernel_size=1))

    def forward(self, x):
        x = getattr(self, f'Conv_in')(x)
        for i in range(1, len(self.layers)):
            x = getattr(self, f'Conv{i}')(x)
        x = getattr(self, f'Conv_out')(x)
        return x


def conv_net(in_channels: int,
             out_channels: int,
             layers: Sequence[int],
             batch_norm: bool = False,
             activation: Union[str, type] = 'ReLU',
             in_spatial: Union[int, tuple] = 2,
             periodic=False) -> nn.Module:
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    net = ConvNet(d, in_channels, out_channels, layers, batch_norm, activation, periodic)
    net = net.to(TORCH.get_default_device().ref)
    return net


class ResNetBlock(nn.Module):

    def __init__(self, in_spatial, in_channels, out_channels, batch_norm, activation, periodic: bool, kernel_size=3):
        # Since in_channels and out_channels might be different, we need a sampling layer for up/down sampling input in order to add it as a skip connection
        super(ResNetBlock, self).__init__()
        if in_channels != out_channels:
            self.sample_input = CONV[in_spatial](in_channels, out_channels, kernel_size=1, padding=0)
            self.bn_sample = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        else:
            self.sample_input = nn.Identity()
            self.bn_sample = nn.Identity()
        self.activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
        self.bn1 = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        self.conv1 = CONV[in_spatial](in_channels, out_channels, kernel_size=kernel_size, padding=1, padding_mode='circular' if periodic else 'zeros')
        self.bn2 = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        self.conv2 = CONV[in_spatial](out_channels, out_channels, kernel_size=kernel_size, padding=1, padding_mode='circular' if periodic else 'zeros')

    def forward(self, x):
        x = TORCH.as_tensor(x)
        out = self.activation()(self.bn1(self.conv1(x)))
        out = self.activation()(self.bn2(self.conv2(out)))
        out = (out + self.bn_sample(self.sample_input(x)))
        return out


def get_mask(inputs, reverse_mask, data_format='NHWC'):
    """ Compute mask for slicing input feature map for Invertible Nets """
    shape = inputs.shape
    if len(shape) == 2:
        N = shape[-1]
        range_n = torch.arange(0, N)
        even_ind = range_n % 2
        checker = torch.reshape(even_ind, (-1, N))
    elif len(shape) == 4:
        H = shape[2] if data_format == 'NCHW' else shape[1]
        W = shape[3] if data_format == 'NCHW' else shape[2]
        range_h = torch.arange(0, H)
        range_w = torch.arange(0, W)
        even_ind_h = range_h % 2
        even_ind_w = range_w % 2
        ind_h = even_ind_h.unsqueeze(-1).repeat(1, W)
        ind_w = even_ind_w.unsqueeze(0).repeat(H, 1)
        checker = torch.logical_xor(ind_h, ind_w)
        checker = checker.reshape(1, 1, H, W) if data_format == 'NCHW' else checker.reshape(1, H, W, 1)
        checker = checker.long()
    else:
        raise ValueError('Invalid tensor shape. Dimension of the tensor shape must be 2 (NxD) or 4 (NxCxHxW or NxHxWxC), got {}.'.format(inputs.get_shape().as_list()))
    if reverse_mask:
        checker = 1 - checker
    return checker.to(TORCH.get_default_device().ref)


class ResNet(nn.Module):

    def __init__(self, in_spatial, in_channels, out_channels, layers, batch_norm, activation, periodic: bool):
        super(ResNet, self).__init__()
        self.layers = layers
        if len(self.layers) < 1:
            layers.append(out_channels)
        self.add_module('Res_in', ResNetBlock(in_spatial, in_channels, layers[0], batch_norm, activation, periodic))
        for i in range(1, len(layers)):
            self.add_module(f'Res{i}', ResNetBlock(in_spatial, layers[i - 1], layers[i], batch_norm, activation, periodic))
        self.add_module('Res_out', CONV[in_spatial](layers[len(layers) - 1], out_channels, kernel_size=1))

    def forward(self, x):
        x = TORCH.as_tensor(x)
        x = getattr(self, 'Res_in')(x)
        for i in range(1, len(self.layers)):
            x = getattr(self, f'Res{i}')(x)
        x = getattr(self, 'Res_out')(x)
        return x


def res_net(in_channels: int,
            out_channels: int,
            layers: Sequence[int],
            batch_norm: bool = False,
            activation: Union[str, type] = 'ReLU',
            in_spatial: Union[int, tuple] = 2,
            periodic=False) -> nn.Module:
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    net = ResNet(d, in_channels, out_channels, layers, batch_norm, activation, periodic)
    net = net.to(TORCH.get_default_device().ref)
    return net


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
    assert isinstance(in_spatial, (tuple, list))
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    net = ConvClassifier(in_features, in_spatial, num_classes, batch_norm, softmax, blocks, block_sizes, dense_layers, periodic, activation)
    return net.to(TORCH.get_default_device().ref)


class ConvClassifier(nn.Module):

    def __init__(self, in_features, in_spatial: list, num_classes: int, batch_norm: bool, use_softmax: bool, blocks: tuple, block_sizes: tuple, dense_layers: tuple, periodic: bool, activation):
        super(ConvClassifier, self).__init__()
        d = len(in_spatial)
        self.in_spatial = in_spatial
        self._blocks = blocks
        self.add_module('maxpool', MAX_POOL[d](2))
        for i, (prev, next) in enumerate(zip((in_features,) + tuple(blocks[:-1]), blocks)):
            block_size = block_sizes[i]
            layers = []
            for j in range(block_size):
                layers.append(CONV[d](prev if j == 0 else next, next, kernel_size=3, padding=1, padding_mode='circular' if periodic else 'zeros'))
                layers.append(NORM[d](next) if batch_norm else nn.Identity())
                layers.append(activation())
            self.add_module(f'conv{i+1}', nn.Sequential(*layers))
        flat_size = int(np.prod(in_spatial) * blocks[-1] / (2**d) ** len(blocks))
        self.mlp = mlp(flat_size, num_classes, dense_layers, batch_norm, activation, use_softmax)
        self.flatten = nn.Flatten()

    def forward(self, x):
        for i in range(len(self._blocks)):
            x = getattr(self, f'conv{i+1}')(x)
            x = self.maxpool(x)
        xf = self.flatten(x)
        y = self.mlp(xf)
        return y


NET = {'u_net': u_net, 'res_net': res_net, 'conv_net': conv_net}


class CouplingLayer(nn.Module):

    def __init__(self, construct_net: Callable, construction_kwargs: dict, reverse_mask):
        super(CouplingLayer, self).__init__()
        self.reverse_mask = reverse_mask
        self.s1 = construct_net(**construction_kwargs)
        self.t1 = construct_net(**construction_kwargs)
        self.s2 = construct_net(**construction_kwargs)
        self.t2 = construct_net(**construction_kwargs)

    def forward(self, x, invert=False):
        x = TORCH.as_tensor(x)
        mask = get_mask(x, self.reverse_mask, 'NCHW')
        if invert:
            v1 = x * mask
            v2 = x * (1 - mask)
            u2 = (1 - mask) * (v2 - self.t1(v1)) * torch.exp(-self.s1(v1))
            u1 = mask * (v1 - self.t2(u2)) * torch.exp(-self.s2(u2))
            return u1 + u2
        else:
            u1 = x * mask
            u2 = x * (1 - mask)
            v1 = mask * (u1 * torch.exp(self.s2(u2)) + self.t2(u2))
            v2 = (1 - mask) * (u2 * torch.exp(self.s1(v1)) + self.t1(v1))
            return v1 + v2


class InvertibleNet(nn.Module):
    def __init__(self, num_blocks: int, construct_net, construction_kwargs: dict):
        super(InvertibleNet, self).__init__()
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.add_module(f'coupling_block{i + 1}', CouplingLayer(construct_net, construction_kwargs, (i % 2 == 0)))

    def forward(self, x, backward=False):
        if backward:
            for i in range(self.num_blocks, 0, -1):
                x = getattr(self, f'coupling_block{i}')(x, backward)
        else:
            for i in range(1, self.num_blocks + 1):
                x = getattr(self, f'coupling_block{i}')(x, backward)
        return x


def invertible_net(num_blocks: int,
                   construct_net: Union[str, Callable],
                   **construct_kwargs):  # mlp, u_net, res_net, conv_net
    if construct_net == 'mlp':
        def construct_net(in_channels: int, layers: Sequence[int], batch_norm=False, activation='ReLU', softmax=False, out_channels: int = None):
            assert not softmax, "Softmax not supported inside invertible net"
            assert out_channels is None or out_channels == in_channels, "out_channels must match in_channels or be unspecified"
            activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
            layers = [in_channels, *layers, in_channels]
            return DenseResNetBlock(layers, batch_norm=batch_norm, activation=activation)
    if isinstance(construct_net, str):
        construct_net = globals()[construct_net]
    if 'in_channels' in construct_kwargs and 'out_channels' not in construct_kwargs:
        construct_kwargs['out_channels'] = construct_kwargs['in_channels']
    return InvertibleNet(num_blocks, construct_net, construct_kwargs).to(TORCH.get_default_device().ref)


def coupling_layer(in_channels: int,
                   activation: Union[str, type] = 'ReLU',
                   batch_norm=False,
                   reverse_mask=False,
                   in_spatial: Union[tuple, int] = 2):
    if isinstance(in_spatial, tuple):
        in_spatial = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    net = CouplingLayer(in_channels, activation, batch_norm, in_spatial, reverse_mask)
    net = net.to(TORCH.get_default_device().ref)
    return net


# class SpectralConv(nn.Module):
#
#     def __init__(self, in_channels, out_channels, modes, in_spatial):
#         super(SpectralConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.in_spatial = in_spatial
#         assert in_spatial >= 1 and in_spatial <= 3
#         if isinstance(modes, int):
#             mode = modes
#             modes = [mode for i in range(in_spatial)]
#         self.scale = 1 / (in_channels * out_channels)
#         self.modes = {i + 1: modes[i] for i in range(len(modes))}
#         self.weights = {}
#         rand_shape = [in_channels, out_channels]
#         rand_shape += [self.modes[i] for i in range(1, in_spatial + 1)]
#         for i in range(2 ** (in_spatial - 1)):
#             self.weights[f'w{i + 1}'] = nn.Parameter(self.scale * torch.randn(rand_shape, dtype=torch.cfloat))
#
#     def complex_mul(self, input, weights):
#         if self.in_spatial == 1:
#             return torch.einsum("bix,iox->box", input, weights)
#         elif self.in_spatial == 2:
#             return torch.einsum("bixy,ioxy->boxy", input, weights)
#         elif self.in_spatial == 3:
#             return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         # --- Convert to Fourier space ---
#         dims = [-i for i in range(self.in_spatial, 0, -1)]
#         x_ft = torch.fft.rfftn(x, dim=dims)
#         outft_dims = [batch_size, self.out_channels] + [x.size(-i) for i in range(self.in_spatial, 1, -1)] + [x.size(-1) // 2 + 1]
#         out_ft = torch.zeros(outft_dims, dtype=torch.cfloat, device=x.device)
#         # --- Multiply relevant fourier modes ---
#         if self.in_spatial == 1:
#             out_ft[:, :, :self.modes[1]] = self.complex_mul(x_ft[:, :, :self.modes[1]], self.weights['w1'].to(x_ft.device))
#         elif self.in_spatial == 2:
#             out_ft[:, :, :self.modes[1], :self.modes[2]] = self.complex_mul(x_ft[:, :, :self.modes[1], :self.modes[2]], self.weights['w1'].to(x_ft.device))
#             out_ft[:, :, -self.modes[1]:, :self.modes[2]] = self.complex_mul(x_ft[:, :, -self.modes[1]:, :self.modes[2]], self.weights['w2'].to(x_ft.device))
#         elif self.in_spatial == 3:
#             out_ft[:, :, :self.modes[1], :self.modes[2], :self.modes[3]] = self.complex_mul(x_ft[:, :, :self.modes[1], :self.modes[2], :self.modes[3]], self.weights['w1'].to(x_ft.device))
#             out_ft[:, :, -self.modes[1]:, :self.modes[2], :self.modes[3]] = self.complex_mul(x_ft[:, :, -self.modes[1]:, :self.modes[2], :self.modes[3]], self.weights['w2'].to(x_ft.device))
#             out_ft[:, :, :self.modes[1], -self.modes[2]:, :self.modes[3]] = self.complex_mul(x_ft[:, :, :self.modes[1], -self.modes[2]:, :self.modes[3]], self.weights['w3'].to(x_ft.device))
#             out_ft[:, :, -self.modes[1]:, -self.modes[2]:, :self.modes[3]] = self.complex_mul(x_ft[:, :, -self.modes[1]:, -self.modes[2]:, :self.modes[3]], self.weights['w4'].to(x_ft.device))
#         # --- Return to Physical Space ---
#         x = torch.fft.irfftn(out_ft, s=[x.size(-i) for i in range(self.in_spatial, 0, -1)])
#         return x
#
#
# class FNO(nn.Module):
#     """
#     Fourier Neural Operators
#     source: https://github.com/zongyi-li/fourier_neural_operator
#
#     The overall network contains 4 layers of the ["Fourier layer"](https://github.com/zongyi-li/fourier_neural_operator).
#     1. Lift the input to the desire channel dimension by self.fc0 .
#     2. 4 layers of the integral operators u' = (W + K)(u).
#         W defined by self.w; K defined by self.conv .
#     3. Project from the channel space to the output space by self.fc1 and self.fc2.
#
#     input shape and output shape: (batchsize b, channels c, *spatial)
#     """
#
#     def __init__(self, in_channels, out_channels, width, modes, activation, batch_norm, in_spatial):
#         super(FNO, self).__init__()
#         self.activation = activation
#         self.width = width
#         self.in_spatial = in_spatial
#         self.fc0 = _bias0(nn.Linear)(in_channels + in_spatial, self.width)
#         for i in range(4):
#             self.add_module(f'conv{i}', SpectralConv(self.width, self.width, modes, in_spatial))
#             self.add_module(f'w{i}', CONV[in_spatial](self.width, self.width, kernel_size=1))
#             self.add_module(f'bn{i}', NORM[in_spatial](self.width) if batch_norm else nn.Identity())
#         self.fc1 = _bias0(nn.Linear)(self.width, 128)
#         self.fc2 = _bias0(nn.Linear)(128, out_channels)
#
#     # Adding extra spatial channels eg. x, y, z, .... to input x
#     def get_grid(self, shape, device):
#         batch_size = shape[0]
#         grid_channel_sizes = shape[2:]  # shape =  (batch_size, channels, *spatial)
#         self.grid_channels = {}
#         for i in range(self.in_spatial):
#             self.grid_channels[f'dim{i}'] = torch.tensor(torch.linspace(0, 1, grid_channel_sizes[i]), dtype=torch.float)
#             reshape_dim_tuple = [1, 1] + [1 if i != j else grid_channel_sizes[j] for j in range(self.in_spatial)]
#             repeat_dim_tuple = [batch_size, 1] + [1 if i == j else grid_channel_sizes[j] for j in range(self.in_spatial)]
#             self.grid_channels[f'dim{i}'] = self.grid_channels[f'dim{i}'].reshape(reshape_dim_tuple).repeat(repeat_dim_tuple)
#         return torch.cat([self.grid_channels[f'dim{i}'] for i in range(self.in_spatial)], dim=1).to(device)
#
#     def forward(self, x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat([x, grid], dim=1)
#         permute_tuple = [0] + [2 + i for i in range(self.in_spatial)] + [1]
#         permute_tuple_reverse = [0] + [self.in_spatial + 1] + [i + 1 for i in range(self.in_spatial)]
#         # Transpose x such that channels shape lies at the end to pass it through linear layers
#         x = x.permute(permute_tuple)
#         x = self.fc0(x)
#         # Transpose x back to its original shape to pass it through convolutional layers
#         x = x.permute(permute_tuple_reverse)
#         for i in range(4):
#             x1 = getattr(self, f'w{i}')(x)
#             x2 = getattr(self, f'conv{i}')(x)
#             x = getattr(self, f'bn{i}')(x1) + getattr(self, f'bn{i}')(x2)
#             x = self.activation()(x)
#         x = x.permute(permute_tuple)
#         x = self.activation()(self.fc1(x))
#         x = self.fc2(x)
#         x = x.permute(permute_tuple_reverse)
#         return x
#
#
# def fno(in_channels: int,
#         out_channels: int,
#         mid_channels: int,
#         modes: Sequence[int],
#         activation: Union[str, type] = 'ReLU',
#         batch_norm: bool = False,
#         in_spatial: int = 2):
#     activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
#     net = FNO(in_channels, out_channels, mid_channels, modes, activation, batch_norm, in_spatial)
#     net = net.to(TORCH.get_default_device().ref)
#     return net
