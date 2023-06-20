# UnifyML

[🌐 **Homepage**](https://github.com/holl-/UnifyML)
&nbsp;&nbsp;&nbsp; [📖 **Documentation**](https://holl-.github.io/UnifyML/)
&nbsp;&nbsp;&nbsp; [🔗 **API**](https://holl-.github.io/UnifyML/unifyml)
&nbsp; • &nbsp; [**▶ Videos**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/UnifyML/blob/main/docs/Examples.ipynb) [**Examples**](https://holl-.github.io/UnifyML/Examples.html)

UnifyML provides a unified math and neural network API for Jax, PyTorch, TensorFlow and NumPy.

See the [installation Instructions](https://holl-.github.io/UnifyML/Installation_Instructions.html) on how to compile the optional custom CUDA operations.

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



**Compatibility**

* Writing code that works with PyTorch, Jax, and TensorFlow makes it easier to share code with other people and collaborate.
* Your published research code will reach a broader audience.
* When you run into a bug / roadblock with one library, you can simply switch to another.
* UnifyML can efficiently [convert tensors between ML libraries](https://holl-.github.io/UnifyML/Convert.html) on-the-fly, so you can even mix the different ecosystems.


**Fewer mistakes**

* *No more data type troubles*: UnifyML [automatically converts data types](https://holl-.github.io/UnifyML/Data_Types.html) where needed and lets you specify the [FP precision globally or by context](https://holl-.github.io/UnifyML/Data_Types.html#Precision)!
* *No more reshaping troubles*: UnifyML performs [reshaping under-the-hood.](https://holl-.github.io/UnifyML/Shapes.html)
* *Is `neighbor_idx.at[jnp.reshape(idx, (-1,))].set(jnp.reshape(cell_idx, (-1,) + cell_idx.shape[-2:]))` correct?*: UnifyML provides a custom Tensor class that lets you write [easy-to-read, more concise, more explicit, less error-prone code](https://holl-.github.io/UnifyML/Tensors.html).

**Unique features**

* **n-dimensional operations**: With UnifyML, you can write code that [automatically works in 1D, 2D and 3D](https://holl-.github.io/UnifyML/N_Dimensional.html), choosing the corresponding operations based on the input dimensions.
* **Preconditioned linear solves**: UnifyML can [build sparse matrices from your Python functions](https://holl-.github.io/UnifyML/Matrices.html) and run linear solvers [with preconditioners](https://holl-.github.io/UnifyML/Linear_Solves.html).
* **Flexible neural network architectures**: [UnifyML provides various configurable neural network architectures, from MLPs to U-Nets.](https://holl-.github.io/UnifyML/Networks.html)
* **Non-uniform tensors**: UnifyML allows you to [stack tensors of different sizes and keeps track of the resulting shapes](https://holl-.github.io/UnifyML/Non_Uniform.html).
