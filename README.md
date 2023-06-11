# UnifyML

(Badges)

UnifyML is a math and neural network library built on top of either [Jax](), [PyTorch](), [TensorFlow]() or [NumPy](), depending on your needs.
It lets you write complex code that runs on any of these backends.

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


```bash
$ pip install unifyml
```

## Why should I use UnifyML?

Apart from the obvious benefit that your code will work with PyTorch, Jax, TensorFlow and NumPy, UnifyML brings a number of additional features to the table.

**General advantages**

* *No more data type troubles*: Set the [FP precision globally or by context]()!
* *No more reshaping troubles*: UnifyML performs [reshaping under-the-hood.]()
* *Is `neighbor_idx.at[jnp.reshape(idx, (-1,))].set(jnp.reshape(cell_idx, (-1,) + cell_idx.shape[-2:]))` correct?*: UnifyML provides a custom Tensor class that lets you write [easy-to-read, more concise, more explicit, less error-prone code]().

**Unique features**

* *n-dimensional operations*: With UnifyML, you can write code that [automatically works in 1D, 2D and 3D](), choosing the corresponding operations based on the input dimensions.
* *Preconditioned linear solves*: UnifyML can [build sparse matrices from your Python functions]() and run linear solvers [with preconditioners]().
* *Flexible neural network architectures*: [UnifyML provides various configurable neural network architectures, from MLPs to U-Nets.]()


## What parts of my code can I unify?

With UnifyML, you can write a [full neural network training script]() that can run with Jax, PyTorch and TensorFlow.
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

Most of UnifyML's functions can be called on native tensors, i.e. Jax/PyTorch/TensorFlow tensors and NumPy arrays.


## Examples


### Training an MLP


### Simulation/Integration/...


## Further documentation

[API](), [Tutorial Videos]()

### How to

* [Linear solves]()
* [Matrix multiplication]()


## Projects using UnifyML

Φ<sub>Flow</sub>


## Contributions


## 📄 Citation

Please cite the following [paper]() if you use UnifyML.

```
@article{rauber2020eagerpy,
    title=...
}
```
