# ![ML4Science](images/Banner.png)

[🌐 **GitHub**](https://github.com/holl-/ML4Science)
&nbsp;&nbsp;&nbsp; [🔗 **API**](ml4s)
&nbsp;&nbsp;&nbsp; [**▶ Videos**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/ML4Science/blob/main/docs/Introduction.ipynb) [**Introduction**](https://holl-.github.io/ML4Science/Introduction.html)
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/ML4Science/blob/main/docs/Examples.ipynb) [**Examples**](https://holl-.github.io/ML4Science/Examples.html)


We recommend starting with the [introduction notebook](Introduction.html) which walks you through the installation and introduces the general structure of ML4Science.
Check out the corresponding introductory [▶ tutorial video]() as well!

## Tensors & Math

**Getting started**

* [Shapes](Shapes.html)
* [ML4Science's `Tensor`](Tensors.html)
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
* [What to avoid in ML4Science](Limitations.md)

## Neural Networks

* [Training neural networks](Networks.html)
* [Neural Network API](ml4s/nn.html)



## API Documentation

The [🔗 API documentation](ml4s) is generated using [pdoc3](https://pdoc3.github.io/pdoc/).
To manually generate the documentation, add the PhiFlow directory to your Python path and run
```bash
$ pdoc --html --output-dir docs --force ml4s
```
This requires PyTorch, TensorFlow and Jax to be installed, in addition to the standard ML4Science requirements.


## Contributing to ML4Science

Contributions are welcome!

If you find a bug, feel free to open a GitHub issue or get in touch with the developers.
If you have changes to be merged, check out our [style guide](https://github.com/holl-/ML4Science/blob/main/CONTRIBUTING.md) before opening a pull request.
