# ML4Science

[🌐 **Homepage**](https://github.com/holl-/ML4Science)
&nbsp;&nbsp;&nbsp; [📖 **Documentation**](https://holl-.github.io/ML4Science/)
&nbsp;&nbsp;&nbsp; [🔗 **API**](https://holl-.github.io/ML4Science/ml4s)
&nbsp; • &nbsp; [**▶ Videos**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/ML4Science/blob/main/docs/Examples.ipynb) [**Examples**](https://holl-.github.io/ML4Science/Examples.html)

ML4Science provides a unified math and neural network API for Jax, PyTorch, TensorFlow and NumPy.

See the [installation Instructions](https://holl-.github.io/ML4Science/Installation_Instructions.html) on how to compile the optional custom CUDA operations.

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
