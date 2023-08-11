# ![Φ<sub>ML</sub>](images/Banner.png)

[🌐 **GitHub**](https://github.com/tum-pbs/PhiML)
&nbsp;&nbsp;&nbsp; [🔗 **API**](phiml)
&nbsp;&nbsp;&nbsp; [**▶ Videos**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/tum-pbs/PhiML/blob/main/docs/Introduction.ipynb) [**Introduction**](https://tum-pbs.github.io/PhiML/Introduction.html)
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/tum-pbs/PhiML/blob/main/docs/Examples.ipynb) [**Examples**](https://tum-pbs.github.io/PhiML/Examples.html)


We recommend starting with the [introduction notebook](Introduction.html) which walks you through the installation and introduces the general structure of Φ<sub>ML</sub>.
Check out the corresponding introductory [▶ tutorial video]() as well!

## Tensors & Math

**Getting started**

* [Shapes](Shapes.html)
* [Φ<sub>ML</sub>'s `Tensor`](Tensors.html)
* [Data types](Data_Types.html)


**Advanced topics**

* [Writing *n*-dimensional code](N_Dimensional.html)
* [Matrices and automatic matrix generation](Matrices.html)
* [Linear solves](Linear_Solves.html)
* [Non-uniform tensors](Non_Uniform.html)
* [NumPy for constants](NumPy_Constants.html)
* [Selecting compute devices](Devices.html)
* [Automatic differentiation](Autodiff.html)
* [Just-in-time (JIT) compilation](JIT.html)
* [Converting between Jax, PyTorch, TensorFlow, NumPy](Convert.html)
* [What to avoid in Φ<sub>ML</sub>](Limitations.md)

## Neural Networks

* [Training neural networks](Networks.html)
* [Neural Network API](phiml/nn.html)



## API Documentation

The [🔗 API documentation](phiml) is generated using [pdoc3](https://pdoc3.github.io/pdoc/).
To manually generate the documentation, add the Φ<sub>ML</sub> directory to your Python path and run
```bash
$ pdoc --html --output-dir docs --force phiml
```
This requires PyTorch, TensorFlow and Jax to be installed, in addition to the standard Φ<sub>ML</sub> requirements.


## Contributing to Φ<sub>ML</sub>

Contributions are welcome!

If you find a bug, feel free to open a GitHub issue or get in touch with the developers.
If you have changes to be merged, check out our [style guide](https://github.com/tum-pbs/PhiML/blob/main/CONTRIBUTING.md) before opening a pull request.
