![Φ<sub>ML</sub>](docs/images/Banner.png)


# Φ<sub>ML</sub>

![Build Status](https://github.com/tum-pbs/PhiML/actions/workflows/unit-tests.yml/badge.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06171/status.svg)](https://doi.org/10.21105/joss.06171)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/phiml.svg)](https://pypi.org/project/phiml/)
[![PyPI license](https://img.shields.io/pypi/l/phiml.svg)](https://pypi.org/project/phiml/)
[![Code Coverage](https://codecov.io/gh/tum-pbs/PhiML/branch/main/graph/badge.svg)](https://codecov.io/gh/tum-pbs/PhiML/branch/main/)
[![Google Collab Book](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tum-pbs/PhiML/blob/main/docs/Introduction.ipynb)

Φ<sub>ML</sub> is a math and neural network library designed for science applications.
It enables you to quickly evaluate many network architectures on your data sets, perform linear and non-linear optimization, and write differentiable simulations.
Φ<sub>ML</sub> is compatible with [Jax](https://github.com/google/jax#installation), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/install) and [NumPy](https://numpy.org/) and your code can be [executed on all of these backends](https://tum-pbs.github.io/PhiML/Introduction.html).

[📖 **Documentation**](https://tum-pbs.github.io/PhiML/)
&nbsp; • &nbsp; [🔗 **API**](https://tum-pbs.github.io/PhiML/phiml/)
&nbsp; • &nbsp; [**▶ Videos**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/tum-pbs/PhiML/blob/main/docs/Introduction.ipynb) [**Introduction**](https://tum-pbs.github.io/PhiML/Introduction.html)
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/tum-pbs/PhiML/blob/main/docs/Examples.ipynb) [**Examples**](https://tum-pbs.github.io/PhiML/Examples.html)

## Installation

Installation with [pip](https://pypi.org/project/pip/) on [Python 3.6](https://www.python.org/downloads/) and later:
```bash
$ pip install phiml
```
Install [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/install) or [Jax](https://github.com/google/jax#installation) to enable machine learning capabilities and GPU execution.
For optimal GPU performance, you may compile the custom CUDA operators, see the [detailed installation instructions](https://tum-pbs.github.io/PhiML/Installation_Instructions.html).


You can verify your installation by running
```bash
$ python3 -c "import phiml; phiml.verify()"
```
This will check for compatible PyTorch, Jax and TensorFlow installations as well.

## Why should I use Φ<sub>ML</sub>?

**Unique features**

* **Preconditioned (sparse) linear solves**: Φ<sub>ML</sub> can [build sparse matrices from your Python functions](https://tum-pbs.github.io/PhiML/Matrices.html) and run [linear solvers with preconditioners](https://tum-pbs.github.io/PhiML/Linear_Solves.html).
* **n-dimensional operations**: With Φ<sub>ML</sub>, you can write code that [automatically works in 1D, 2D and 3D](https://tum-pbs.github.io/PhiML/N_Dimensional.html), choosing the corresponding operations based on the input dimensions.
* **Flexible neural network architectures**: [Φ<sub>ML</sub> provides various configurable neural network architectures, from MLPs to U-Nets.](https://tum-pbs.github.io/PhiML/Networks.html)
* **Non-uniform tensors**: Φ<sub>ML</sub> allows you to [stack tensors of different sizes and keeps track of the resulting shapes](https://tum-pbs.github.io/PhiML/Non_Uniform.html).

**Compatibility**

* Writing code that works with PyTorch, Jax, and TensorFlow makes it easier to share code with other people and collaborate.
* Your published research code will reach a broader audience.
* When you run into a bug / roadblock with one library, you can simply switch to another.
* Φ<sub>ML</sub> can efficiently [convert tensors between ML libraries](https://tum-pbs.github.io/PhiML/Convert.html) on-the-fly, so you can even mix the different ecosystems.


**Fewer mistakes**

* *No more data type troubles*: Φ<sub>ML</sub> [automatically converts data types](https://tum-pbs.github.io/PhiML/Advantages_Data_Types.html) where needed and lets you specify the [FP precision globally or by context](https://tum-pbs.github.io/PhiML/Data_Types.html#Precision)!
* *No more reshaping troubles*: Φ<sub>ML</sub> performs [reshaping under-the-hood.](https://tum-pbs.github.io/PhiML/Dimension_Names_Types.html)
* *Is `neighbor_idx.at[jnp.reshape(idx, (-1,))].set(jnp.reshape(cell_idx, (-1,) + cell_idx.shape[-2:]))` correct?*: Φ<sub>ML</sub> provides a custom Tensor class that lets you write [easy-to-read, more concise, more explicit, less error-prone code](https://tum-pbs.github.io/PhiML/Tensors.html).


## What parts of my code are library-agnostic?

With Φ<sub>ML</sub>, you can write a [full neural network training script](https://tum-pbs.github.io/PhiML/Examples.html) that can run with Jax, PyTorch and TensorFlow.
In particular, Φ<sub>ML</sub> provides abstractions for the following functionality:

* [Neural network creation and optimization](https://tum-pbs.github.io/PhiML/Networks.html)
* [Math functions and tensor operations](https://tum-pbs.github.io/PhiML/phiml/math)
* [Sparse tensors / matrices](https://tum-pbs.github.io/PhiML/Matrices.html)
* [Just-in-time (JIT) compilation](https://tum-pbs.github.io/PhiML/JIT.html)
* [Computing gradients of functions via automatic differentiation](https://tum-pbs.github.io/PhiML/Autodiff.html)

However, Φ<sub>ML</sub> does not currently abstract the following use cases:

* Custom or non-standard network architectures or optimizers require backend-specific code.
* Φ<sub>ML</sub> [abstracts compute devices](https://tum-pbs.github.io/PhiML/Devices.html) but does not currently allow mapping operations onto multiple GPUs.
* Φ<sub>ML</sub> has no data loading module. However, it can [convert data](https://tum-pbs.github.io/PhiML/Convert.html), once loaded, to any other backend.
* Some less-used math functions have not been wrapped yet. If you come across one you need, feel free to open an issue.
* Higher-order derivatives are not supported for all backends.


## Φ<sub>ML</sub>'s `Tensor` class

Many of Φ<sub>ML</sub>'s functions can be called on native tensors, i.e. Jax/PyTorch/TensorFlow tensors and NumPy arrays.
In these cases, the function maps to the corresponding one from the matching backend.

However, we have noticed that code written this way is often hard-to-read, verbose and error-prone.
One main reason is that dimensions are typically referred to by index and the meaning of that dimension might not be obvious (for examples, see [here](https://github.com/tumaer/JAXFLUIDS/blob/477e28813f07e3836588bd8a50cd0149fbbea94f/src/jaxfluids/stencils/derivative/deriv_second_order_face.py#L49), [here](https://github.com/jax-md/jax-md/blob/23dba354ec29c8b0c53f61a85d10bb64ed7a0058/jax_md/partition.py#L798) or [here](https://github.com/locuslab/deq/blob/1fb7059d6d89bb26d16da80ab9489dcc73fc5472/lib/solvers.py#L207)).

Φ<sub>ML</sub> includes a `Tensor` class with the goal to [remedy these shortcomings](https://tum-pbs.github.io/PhiML/Tensors.html).
A Φ<sub>ML</sub> `Tensor` wraps one of the native tensors, such as `ndarray`, `torch.Tensor` or `tf.Tensor`, but extends them by two features:

1. **Names**: All dimensions are named. Referring to a specific dimension can be done as `tensor.<dimension name>`. Elements along dimensions can also be named.
2. **Types**: Every dimension is assigned a type flag, such as *channel*, *batch* or *spatial*.

For a full explanation of why these changes make your code not only easier to read but also shorter, see [here](https://tum-pbs.github.io/PhiML/Tensors.html).
Here's the gist:

* With dimension names, the dimension order becomes irrelevant and you don't need to worry about it.
* Missing dimensions are automatically added when and where needed.
* Tensors are automatically transposed to match.
* Slicing by name is a lot more readable, e.g. `image.channels['red']` vs `image[:, :, :, 0]`.
* Functions will automatically use the right dimensions, e.g. convolutions and FFTs act on spatial dimensions by default.
* You can have arbitrarily many batch dimensions (or none) and your code will work the same.
* The number of spatial dimensions control the dimensionality of not only your data but also your code. [Your 2D code also runs in 3D](https://tum-pbs.github.io/PhiML/N_Dimensional.html)!


## Examples

The following three examples are taken from the [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/tum-pbs/PhiML/blob/main/docs/Examples.ipynb) [examples notebook](https://tum-pbs.github.io/PhiML/Examples.html) where you can also find examples on automatic differentiation, JIT compilation, and more.
You can change the `math.use(...)` statements to any of the supported ML libraries.

### Training an MLP

The following script trains an [MLP](https://tum-pbs.github.io/PhiML/phiml/nn#phiml.nn.mlp) with three hidden layers to learn a noisy 1D sine function in the range [-2, 2].

```python
from phiml import math, nn
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

We didn't even have to import `torch` in this example since all calls were routed through Φ<sub>ML</sub>.


### Solving a sparse linear system with preconditioners

Φ<sub>ML</sub> supports [solving dense as well as sparse linear systems](https://tum-pbs.github.io/PhiML/Linear_Solves.html) and can [build an explicit matrix representation from linear Python functions](https://tum-pbs.github.io/PhiML/Matrices.html) in order to compute preconditioners.
We recommend using Φ<sub>ML</sub>'s tensors, but you can pass native tensors to [`solve_linear()`](https://tum-pbs.github.io/PhiML/phiml/math#phiml.math.solve_linear) as well.
The following example solves the 1D Poisson problem ∇x = b with b=1 with incomplete LU decomposition.

```python
from phiml import math
import numpy as np

def laplace_1d(x):
    return math.pad(x[1:], (0, 1)) + math.pad(x[:-1], (1, 0)) - 2 * x

b = np.ones((6,))
solve = math.Solve('scipy-CG', rel_tol=1e-5, x0=0*b, preconditioner='ilu')
sol = math.solve_linear(math.jit_compile_linear(laplace_1d), b, solve)
```

Decorating the linear function with [`math.jit_compile_linear`](https://tum-pbs.github.io/PhiML/phiml/math#phiml.math.jit_compile_linear) lets Φ<sub>ML</sub> compute the sparse matrix inside [`solve_linear()`](https://tum-pbs.github.io/PhiML/phiml/math#phiml.math.solve_linear). In this example, the matrix is a tridiagonal band matrix.
Note that if you JIT-compile the [`math.solve_linear()`](https://tum-pbs.github.io/PhiML/phiml/math#phiml.math.solve_linear) call, the sparsity pattern and incomplete LU preconditioner are [computed at JIT time](https://tum-pbs.github.io/PhiML/NumPy_Constants.html).
The L and U matrices then enter the computational graph as constants and are not recomputed every time the function is called.


## Contributions

Contributions are welcome!

If you find a bug, feel free to open a GitHub issue or get in touch with the developers.
If you have changes to be merged, check out our [style guide](https://github.com/tum-pbs/PhiML/blob/main/CONTRIBUTING.md) before opening a pull request.


## 📄 Citation

Please use the following citation:

```
@article{Holl2024,
    doi = {10.21105/joss.06171},
    url = {https://doi.org/10.21105/joss.06171},
    year = {2024},
    publisher = {The Open Journal},
    volume = {9},
    number = {95},
    pages = {6171},
    author = {Philipp Holl and Nils Thuerey},
    title = {Φ-ML: Intuitive Scientific Computing with Dimension Types for Jax, PyTorch, TensorFlow & NumPy},
    journal = {Journal of Open Source Software}
}
```
Also see the corresponding [journal article](https://doi.org/10.21105/joss.06171) and [software archive of version 1.4.0](https://figshare.com/articles/software/_-ML_1_4_0/25282300).


## Projects using Φ<sub>ML</sub>

Φ<sub>ML</sub> is used by the simulation framework [Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow) to integrate differentiable simulations with machine learning.
