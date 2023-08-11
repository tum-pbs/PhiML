# Φ<sub>ML</sub>

[🌐 **Homepage**](https://github.com/tum-pbs/PhiML)
&nbsp; • &nbsp; [📖 **Documentation**](https://tum-pbs.github.io/PhiML/)
&nbsp; • &nbsp; [🔗 **API**](https://tum-pbs.github.io/PhiML/phiml)
&nbsp; • &nbsp; [**▶ Videos**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/tum-pbs/PhiML/blob/main/docs/Examples.ipynb) [**Examples**](https://tum-pbs.github.io/PhiML/Examples.html)

Φ<sub>ML</sub> provides a unified math and neural network API for Jax, PyTorch, TensorFlow and NumPy.

See the [installation Instructions](https://tum-pbs.github.io/PhiML/Installation_Instructions.html) on how to compile the optional custom CUDA operations.

```python
from jax import numpy as jnp
import torch
import tensorflow as tf
import numpy as np

from phiml import math

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
* Φ<sub>ML</sub> can efficiently [convert tensors between ML libraries](https://tum-pbs.github.io/PhiML/Convert.html) on-the-fly, so you can even mix the different ecosystems.


**Fewer mistakes**

* *No more data type troubles*: Φ<sub>ML</sub> [automatically converts data types](https://tum-pbs.github.io/PhiML/Data_Types.html) where needed and lets you specify the [FP precision globally or by context](https://tum-pbs.github.io/PhiML/Data_Types.html#Precision)!
* *No more reshaping troubles*: Φ<sub>ML</sub> performs [reshaping under-the-hood.](https://tum-pbs.github.io/PhiML/Shapes.html)
* *Is `neighbor_idx.at[jnp.reshape(idx, (-1,))].set(jnp.reshape(cell_idx, (-1,) + cell_idx.shape[-2:]))` correct?*: Φ<sub>ML</sub> provides a custom Tensor class that lets you write [easy-to-read, more concise, more explicit, less error-prone code](https://tum-pbs.github.io/PhiML/Tensors.html).

**Unique features**

* **n-dimensional operations**: With Φ<sub>ML</sub>, you can write code that [automatically works in 1D, 2D and 3D](https://tum-pbs.github.io/PhiML/N_Dimensional.html), choosing the corresponding operations based on the input dimensions.
* **Preconditioned linear solves**: Φ<sub>ML</sub> can [build sparse matrices from your Python functions](https://tum-pbs.github.io/PhiML/Matrices.html) and run linear solvers [with preconditioners](https://tum-pbs.github.io/PhiML/Linear_Solves.html).
* **Flexible neural network architectures**: [Φ<sub>ML</sub> provides various configurable neural network architectures, from MLPs to U-Nets.](https://tum-pbs.github.io/PhiML/Networks.html)
* **Non-uniform tensors**: Φ<sub>ML</sub> allows you to [stack tensors of different sizes and keeps track of the resulting shapes](https://tum-pbs.github.io/PhiML/Non_Uniform.html).
