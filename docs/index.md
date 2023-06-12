# ![UnifyML](figures/Logo_DallE2_layout.png)

[🌐 **Homepage**](https://github.com/holl-/UnifyML)
&nbsp;&nbsp;&nbsp; [🔗 **API**](unifyml)
&nbsp;&nbsp;&nbsp; [**▶ YouTube Tutorials**]()
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/UnifyML/blob/main/docs/Introduction.ipynb) [**Introduction**](https://holl-.github.io/UnifyML/Introduction.html)
&nbsp; • &nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16>](https://colab.research.google.com/github/holl-/UnifyML/blob/main/docs/Examples.ipynb) [**Examples**](https://holl-.github.io/UnifyML/Examples.html)


We recommend starting with the [introduction notebook]() which walks you through the installation and introduces the general structure of UnifyML.
Check out the corresponding introductory [▶ tutorial video]() as well!

## Math Documentation

Getting started

* [Shapes](Shapes.html)
* [UnifyML's `Tensor`](Tensors.html)
* [Data types](Data_Types.html)


Advanced topics

* [Writing *n*-dimensional code](N_Dimensional.html)
* [Matrices and automatic matrix generation](Matrices.html)
* [Linear solves](Linear_Solves.html)
* [Non-uniform tensors](Non_Uniform.html)


## Neural Networks

* [Training neural networks](Networks.html)
* [Neural Network API](unifyml/nn/index.html)



## API Documentation

The [🔗 API documentation](unifyml) is generated using [pdoc3](https://pdoc3.github.io/pdoc/).
To manually generate the documentation, add the PhiFlow directory to your Python path and run
```bash
$ pdoc --html --output-dir docs --force unifyml
```
This requires PyTorch, TensorFlow and Jax to be installed, in addition to the standard Φ<sub>Flow</sub> requirements.


## Contributing to UnifyML

Contributions are welcome!

If you find a bug, feel free to open a GitHub issue or get in touch with the developers.
If you have changes to be merged, check out our [style guide](https://github.com/holl-/UnifyML/blob/main/CONTRIBUTING.md) before opening a pull request.
