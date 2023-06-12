# UnifyML

![Build Status](https://github.com/holl-/UnifyML/actions/workflows/unit-tests.yml/badge.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/unifyml.svg)](https://pypi.org/project/unifyml/)
[![PyPI license](https://img.shields.io/pypi/l/unifyml.svg)](https://pypi.org/project/unifyml/)
[![Code Coverage](https://codecov.io/gh/holl-/UnifyML/branch/develop/graph/badge.svg)](https://codecov.io/gh/holl-/UnifyML/branch/develop/)
[![Google Collab Book](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/holl-/UnifyML/blob/main/docs/Introduction.ipynb)

UnifyML is a math and neural network library built on top of either [Jax](https://github.com/google/jax#installation), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/install) or [NumPy](https://numpy.org/), depending on your preferences.
It lets you write complex code that [runs on any of these backends](https://holl-.github.io/UnifyML/Introduction.html).

```python
from jax import numpy as jnp
import torch
import tensorflow as tf
import numpy as np

from unifyml import math

math.sin(1.)
math.sin(jnp.asarray([1.]))
math.sin(torch.tensor([1.]))
math.sin(tf.constant([1.]))
math.sin(np.asarray([1.]))
```

## Installation

Installation with [pip](https://pypi.org/project/pip/) on [Python 3.6](https://www.python.org/downloads/) and later:
```bash
$ pip install unifyml
```
Install [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/install) or [Jax](https://github.com/google/jax#installation) to enable machine learning capabilities and GPU execution.
For optimal GPU performance, you may compile the custom CUDA operators, see the [detailed installation instructions](https://holl-.github.io/UnifyML/Installation_Instructions.html).


You can verify your installation by running
```bash
$ python3 -c "import unifyml; unifyml.verify()"
```
This will check for compatible PyTorch, Jax and TensorFlow installations as well.

## Why should I use UnifyML?

Apart from the obvious benefit that your code will work with PyTorch, Jax, TensorFlow and NumPy, UnifyML brings a number of additional features to the table.

**General advantages**

* *No more data type troubles*: Set the [FP precision globally or by context](https://holl-.github.io/UnifyML/Data_Types.html)!
* *No more reshaping troubles*: UnifyML performs [reshaping under-the-hood.](https://holl-.github.io/UnifyML/Shapes.html)
* *Is `neighbor_idx.at[jnp.reshape(idx, (-1,))].set(jnp.reshape(cell_idx, (-1,) + cell_idx.shape[-2:]))` correct?*: UnifyML provides a custom Tensor class that lets you write [easy-to-read, more concise, more explicit, less error-prone code](https://holl-.github.io/UnifyML/Tensors.html).

**Unique features**

* **n-dimensional operations**: With UnifyML, you can write code that [automatically works in 1D, 2D and 3D](https://holl-.github.io/UnifyML/N_Dimensional.html), choosing the corresponding operations based on the input dimensions.
* **Preconditioned linear solves**: UnifyML can [build sparse matrices from your Python functions](https://holl-.github.io/UnifyML/Matrices.html) and run linear solvers [with preconditioners](https://holl-.github.io/UnifyML/Linear_Solves.html).
* **Flexible neural network architectures**: [UnifyML provides various configurable neural network architectures, from MLPs to U-Nets.](https://holl-.github.io/UnifyML/Networks.html)
* **Non-uniform tensors**: UnifyML allows you to [stack tensors of different sizes and keeps track of the resulting shapes](https://holl-.github.io/UnifyML/Non_Uniform.html).


## What parts of my code can I unify?

With UnifyML, you can write a [full neural network training script](https://holl-.github.io/UnifyML/Examples.html) that can run with Jax, PyTorch and TensorFlow.
In particular, UnifyML provides abstractions for the following functionality:

* Neural network creation and optimization
* Math functions
* Tensor operations like `gather`, `scatter`, `pad`, etc.
* Sparse tensors
* Just-in-time (JIT) compilation
* Computing gradients of functions via automatic differentiation

However, UnifyML does not currently abstract the following use cases:

* Custom or non-standard network architectures or optimizers require backend-specific code.
* UnifyML abstracts compute devices but does not currently allow mapping operations onto multiple GPUs.
* UnifyML has no data loading module. However, it can convert data, once loaded, to any other backend.
* Some less-used math functions have not been wrapped yet. If you come across one you need, feel free to open an issue.
* Higher-order derivatives are not supported for all backends.


## UnifyML's `Tensor` class

Many of UnifyML's functions can be called on native tensors, i.e. Jax/PyTorch/TensorFlow tensors and NumPy arrays.
In these cases, the function maps to the corresponding one from the matching backend.

However, we have noticed that code written this way is often hard-to-read, verbose and error-prone.
One main reason for that is that dimensions are typically referred to by index and the meaning of that dimension might not be obvious (for examples, see [here](https://github.com/tumaer/JAXFLUIDS/blob/477e28813f07e3836588bd8a50cd0149fbbea94f/src/jaxfluids/stencils/derivative/deriv_second_order_face.py#L49), [here](https://github.com/jax-md/jax-md/blob/23dba354ec29c8b0c53f61a85d10bb64ed7a0058/jax_md/partition.py#L798) or [here](https://github.com/pyg-team/pytorch_geometric/blob/05490776e576addd4727e0a4bcd18e7cc0a16f3c/torch_geometric/transforms/grid_sampling.py#L39)).

UnifyML includes a `Tensor` class with the goal to [remedy these shortcomings](https://holl-.github.io/UnifyML/Tensors.html).
A UnifyML `Tensor` wraps one of the native tensors, such as `ndarray`, `torch.Tensor` or `tf.Tensor`, but extends them by two features:

1. **Names**: All dimensions are named. Referring to a specific dimension can be done as `tensor.<dimension name>`. Elements along dimensions can also be named.
2. **Types**: Every dimension is assigned a type flag, such as *channel*, *batch* or *spatial*.

For a full explanation of why these changes make your code not only easier to read but also shorter, see [here](https://holl-.github.io/UnifyML/Tensors.html).
Here's the gist:

* With dimension names, the dimension order becomes irrelevant and you don't need to worry about it.
* Missing dimensions are automatically added when and where needed.
* Tensors are automatically transposed to match.
* Slicing by name is a lot more readable, e.g. `image.channels['red']` vs `image[:, :, :, 0]`.
* Functions will automatically use the right dimensions, e.g. convolutions and FFTs act on spatial dimensions by default.
* You can have arbitrarily many batch dimensions (or none) and your code will work the same.
* The number of spatial dimensions control the dimensionality of not only your data but also your code. [Your 2D code also runs in 3D](https://holl-.github.io/UnifyML/N_Dimensional.html)!


## Examples

Click [here](https://holl-.github.io/UnifyML/Examples.html) for more examples.

### Training an MLP

```python
...
```


### Simulation/Math Demo

```python
...
```


## Further Documentation

[📖 **Overview**](https://holl-.github.io/UnifyML/unifyml/)
&nbsp; • &nbsp; [🔗 **API**](https://holl-.github.io/UnifyML/unifyml/)
&nbsp; • &nbsp; [**▶ YouTube Tutorials**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/UnifyML/blob/main/docs/Introduction.ipynb) [**Introduction**](https://holl-.github.io/UnifyML/Introduction.html)
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/UnifyML/blob/main/docs/Examples.ipynb) [**Examples**](https://holl-.github.io/UnifyML/Examples.html)


## Contributions

Contributions are welcome!

If you find a bug, feel free to open a GitHub issue or get in touch with the developers.
If you have changes to be merged, check out our [style guide](https://github.com/holl-/UnifyML/blob/main/CONTRIBUTING.md) before opening a pull request.


## 📄 Citation

We are in the process of submitting a paper. Updates will follow.


## Projects using UnifyML

UnifyML is used by the simulation framework [Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow) to integrate differentiable simulations with machine learning.
