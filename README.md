![ML4Science](docs/images/Banner.png)


# ML 4 Science

![Build Status](https://github.com/holl-/ML4Science/actions/workflows/unit-tests.yml/badge.svg)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ml4s.svg)](https://pypi.org/project/ml4s/)
[![PyPI license](https://img.shields.io/pypi/l/ml4s.svg)](https://pypi.org/project/ml4s/)
[![Code Coverage](https://codecov.io/gh/holl-/ML4Science/branch/develop/graph/badge.svg)](https://codecov.io/gh/holl-/ML4Science/branch/develop/)
[![Google Collab Book](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/holl-/ML4Science/blob/main/docs/Introduction.ipynb)

ML4Science is a math and neural network library built on top of either [Jax](https://github.com/google/jax#installation), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/install) or [NumPy](https://numpy.org/), depending on your preferences.
It lets you write complex code that [runs on any of these backends](https://holl-.github.io/ML4Science/Introduction.html).

```python
from jax import numpy as jnp
import torch
import tensorflow as tf
import numpy as np

from ml4s import math

math.sin(1.)
math.sin(jnp.asarray([1.]))
math.sin(torch.tensor([1.]))
math.sin(tf.constant([1.]))
math.sin(np.asarray([1.]))
```

[📖 **Documentation**](https://holl-.github.io/ML4Science/)
&nbsp; • &nbsp; [🔗 **API**](https://holl-.github.io/ML4Science/ml4s/)
&nbsp; • &nbsp; [**▶ Videos**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/ML4Science/blob/main/docs/Introduction.ipynb) [**Introduction**](https://holl-.github.io/ML4Science/Introduction.html)
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/ML4Science/blob/main/docs/Examples.ipynb) [**Examples**](https://holl-.github.io/ML4Science/Examples.html)

## Installation

Installation with [pip](https://pypi.org/project/pip/) on [Python 3.6](https://www.python.org/downloads/) and later:
```bash
$ pip install ml4s
```
Install [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/install) or [Jax](https://github.com/google/jax#installation) to enable machine learning capabilities and GPU execution.
For optimal GPU performance, you may compile the custom CUDA operators, see the [detailed installation instructions](https://holl-.github.io/ML4Science/Installation_Instructions.html).


You can verify your installation by running
```bash
$ python3 -c "import ml4s; ml4s.verify()"
```
This will check for compatible PyTorch, Jax and TensorFlow installations as well.

## Why should I use ML4Science?

**Compatibility**

* Writing code that works with PyTorch, Jax, and TensorFlow makes it easier to share code with other people and collaborate.
* Your published research code will reach a broader audience.
* When you run into a bug / roadblock with one library, you can simply switch to another.
* ML4Science can efficiently [convert tensors between ML libraries](https://holl-.github.io/ML4Science/Convert.html) on-the-fly, so you can even mix the different ecosystems.


**Fewer mistakes**

* *No more data type troubles*: ML4Science [automatically converts data types](https://holl-.github.io/ML4Science/Data_Types.html) where needed and lets you specify the [FP precision globally or by context](https://holl-.github.io/ML4Science/Data_Types.html#Precision)!
* *No more reshaping troubles*: ML4Science performs [reshaping under-the-hood.](https://holl-.github.io/ML4Science/Shapes.html)
* *Is `neighbor_idx.at[jnp.reshape(idx, (-1,))].set(jnp.reshape(cell_idx, (-1,) + cell_idx.shape[-2:]))` correct?*: ML4Science provides a custom Tensor class that lets you write [easy-to-read, more concise, more explicit, less error-prone code](https://holl-.github.io/ML4Science/Tensors.html).

**Unique features**

* **n-dimensional operations**: With ML4Science, you can write code that [automatically works in 1D, 2D and 3D](https://holl-.github.io/ML4Science/N_Dimensional.html), choosing the corresponding operations based on the input dimensions.
* **Preconditioned linear solves**: ML4Science can [build sparse matrices from your Python functions](https://holl-.github.io/ML4Science/Matrices.html) and run linear solvers [with preconditioners](https://holl-.github.io/ML4Science/Linear_Solves.html).
* **Flexible neural network architectures**: [ML4Science provides various configurable neural network architectures, from MLPs to U-Nets.](https://holl-.github.io/ML4Science/Networks.html)
* **Non-uniform tensors**: ML4Science allows you to [stack tensors of different sizes and keeps track of the resulting shapes](https://holl-.github.io/ML4Science/Non_Uniform.html).


## What parts of my code can I unify?

With ML4Science, you can write a [full neural network training script](https://holl-.github.io/ML4Science/Examples.html) that can run with Jax, PyTorch and TensorFlow.
In particular, ML4Science provides abstractions for the following functionality:

* [Neural network creation and optimization](https://holl-.github.io/ML4Science/Networks.html)
* [Math functions and tensor operations](https://holl-.github.io/ML4Science/ml4s/math)
* [Sparse tensors / matrices](https://holl-.github.io/ML4Science/Matrices.html)
* [Just-in-time (JIT) compilation](https://holl-.github.io/ML4Science/JIT.html)
* [Computing gradients of functions via automatic differentiation](https://holl-.github.io/ML4Science/Autodiff.html)

However, ML4Science does not currently abstract the following use cases:

* Custom or non-standard network architectures or optimizers require backend-specific code.
* ML4Science [abstracts compute devices](https://holl-.github.io/ML4Science/Devices.html) but does not currently allow mapping operations onto multiple GPUs.
* ML4Science has no data loading module. However, it can [convert data](https://holl-.github.io/ML4Science/Convert.html), once loaded, to any other backend.
* Some less-used math functions have not been wrapped yet. If you come across one you need, feel free to open an issue.
* Higher-order derivatives are not supported for all backends.


## ML4Science's `Tensor` class

Many of ML4Science's functions can be called on native tensors, i.e. Jax/PyTorch/TensorFlow tensors and NumPy arrays.
In these cases, the function maps to the corresponding one from the matching backend.

However, we have noticed that code written this way is often hard-to-read, verbose and error-prone.
One main reason is that dimensions are typically referred to by index and the meaning of that dimension might not be obvious (for examples, see [here](https://github.com/tumaer/JAXFLUIDS/blob/477e28813f07e3836588bd8a50cd0149fbbea94f/src/jaxfluids/stencils/derivative/deriv_second_order_face.py#L49), [here](https://github.com/jax-md/jax-md/blob/23dba354ec29c8b0c53f61a85d10bb64ed7a0058/jax_md/partition.py#L798) or [here](https://github.com/locuslab/deq/blob/1fb7059d6d89bb26d16da80ab9489dcc73fc5472/lib/solvers.py#L207)).

ML4Science includes a `Tensor` class with the goal to [remedy these shortcomings](https://holl-.github.io/ML4Science/Tensors.html).
A ML4Science `Tensor` wraps one of the native tensors, such as `ndarray`, `torch.Tensor` or `tf.Tensor`, but extends them by two features:

1. **Names**: All dimensions are named. Referring to a specific dimension can be done as `tensor.<dimension name>`. Elements along dimensions can also be named.
2. **Types**: Every dimension is assigned a type flag, such as *channel*, *batch* or *spatial*.

For a full explanation of why these changes make your code not only easier to read but also shorter, see [here](https://holl-.github.io/ML4Science/Tensors.html).
Here's the gist:

* With dimension names, the dimension order becomes irrelevant and you don't need to worry about it.
* Missing dimensions are automatically added when and where needed.
* Tensors are automatically transposed to match.
* Slicing by name is a lot more readable, e.g. `image.channels['red']` vs `image[:, :, :, 0]`.
* Functions will automatically use the right dimensions, e.g. convolutions and FFTs act on spatial dimensions by default.
* You can have arbitrarily many batch dimensions (or none) and your code will work the same.
* The number of spatial dimensions control the dimensionality of not only your data but also your code. [Your 2D code also runs in 3D](https://holl-.github.io/ML4Science/N_Dimensional.html)!


## Examples

The following three examples are taken from the [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/ML4Science/blob/main/docs/Examples.ipynb) [examples notebook](https://holl-.github.io/ML4Science/Examples.html) where you can also find examples on automatic differentiation, JIT compilation, and more.
You can change the `math.use(...)` statements to any of the supported ML libraries.

### Training an MLP

The following script trains an [MLP](https://holl-.github.io/ML4Science/ml4s/nn#ml4s.nn.mlp) with three hidden layers to learn a noisy 1D sine function in the range [-2, 2].

```python
from ml4s import math, nn
math.use('torch')

net = nn.mlp(1, 1, layers=[128, 128, 128], activation='ReLU')
optimizer = nn.adam(net, learning_rate=1e-3)

data_x = math.random_uniform(math.batch(batch=128), low=-2, high=2)
data_y = math.sin(data_x) + math.random_normal(math.batch(batch=128)) * .2

def loss_function(x, y):
    return math.l2_loss(y - math.native_call(net, x))

for i in range(100):
    loss = nn.update_weights(net, optimizer, loss_function, data_x, data_y)
    print(loss)
```

We didn't even have to import `torch` in this example since all calls were routed through ML4Science.


### Solving a sparse linear system with preconditioners

ML4Science supports [solving dense as well as sparse linear systems](https://holl-.github.io/ML4Science/Linear_Solves.html) and can [build an explicit matrix representation from linear Python functions](https://holl-.github.io/ML4Science/Matrices.html) in order to compute preconditioners.
We recommend using ML4Science's tensors, but you can pass native tensors to [`solve_linear()`](https://holl-.github.io/ML4Science/ml4s/math#ml4s.math.solve_linear) as well.
The following example solves the 1D Poisson problem ∇x = b with b=1 with incomplete LU decomposition.

```python
from ml4s import math
import numpy as np

def laplace_1d(x):
    return math.pad(x[1:], (0, 1)) + math.pad(x[:-1], (1, 0)) - 2 * x

b = np.ones((6,))
solve = math.Solve('scipy-CG', rel_tol=1e-5, x0=0*b, preconditioner='ilu')
sol = math.solve_linear(math.jit_compile_linear(laplace_1d), b, solve)
```

Decorating the linear function with [`math.jit_compile_linear`](https://holl-.github.io/ML4Science/ml4s/math#ml4s.math.jit_compile_linear) lets ML4Science compute the sparse matrix inside [`solve_linear()`](https://holl-.github.io/ML4Science/ml4s/math#ml4s.math.solve_linear). In this example, the matrix is a tridiagonal band matrix.
Note that if you JIT-compile the [`math.solve_linear()`](https://holl-.github.io/ML4Science/ml4s/math#ml4s.math.solve_linear) call, the sparsity pattern and incomplete LU preconditioner are [computed at JIT time](https://holl-.github.io/ML4Science/NumPy_Constants.html).
The L and U matrices then enter the computational graph as constants and are not recomputed every time the function is called.


## Contributions

Contributions are welcome!

If you find a bug, feel free to open a GitHub issue or get in touch with the developers.
If you have changes to be merged, check out our [style guide](https://github.com/holl-/ML4Science/blob/main/CONTRIBUTING.md) before opening a pull request.


## 📄 Citation

We are in the process of submitting a paper. Updates will follow.


## Projects using ML4Science

ML4Science is used by the simulation framework [Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow) to integrate differentiable simulations with machine learning.
