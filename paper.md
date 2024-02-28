---
title: '$\Phi_\textrm{ML}$: Intuitive Scientific Computing with Dimension Types for Jax, PyTorch, TensorFlow & NumPy'
tags:
  - Python
  - Machine Learning
  - Jax
  - TensorFlow
  - PyTorch
  - NumPy
  - Differentiable simulations
  - Sparse linear systems
  - Preconditioners
authors:
  - name: Philipp Holl
    orcid: 0000-0001-9246-5195
    affiliation: 1
  - name: Nils Thuerey
    orcid: 0000-0001-6647-8910
    affiliation: 1
affiliations:
  - name: School of Computation, Information and Technology, Technical University of Munich, Germany
    index: 1
date: 01 August 2023
bibliography: paper.bib
---

# Summary

$\Phi_\textrm{ML}$ is a math and neural network library designed for science applications.
It enables users to quickly evaluate many network architectures on their data sets, perform (sparse) linear and non-linear optimization, and write differentiable simulations that scale to *n* dimensions.
$\Phi_\textrm{ML}$ is compatible with Jax, PyTorch, TensorFlow and NumPy, and user code can be executed on all of these backends.
The project is hosted at [https://github.com/tum-pbs/PhiML](https://github.com/tum-pbs/PhiML) under the MIT license.

# Statement of need

Machine learning (ML) has become an essential tool for scientific research. In recent years, ML has been used to make significant advances in a wide range of scientific fields, including
chemistry [@Molecular2018],
materials science [@Materials2019],
weather and climate prediction [@Weather2022; @Climate2022],
computational fluid dynamics [@CFD2020],
drug discovery [@DrugDiscovery2019; @AlphaFold2021],
astrophysics [@CMB2020; @Galaxy2004; @Galaxy2015],
geology [@Mineral2015], and many more.
The use of ML for scientific applications is still in its early stages, but it has the potential to revolutionize the way that science is done. ML can help researchers to make new discoveries and insights that were previously impossible.

The availability of domain knowledge sets science applications apart from other ML fields like computer vision or language modelling.
Domain knowledge often allows for explicit modelling of known dynamics by simulating them with handwritten algorithms, which has been shown to improve results when training ML models [@SolverInTheLoop2020; @PINN2019].
Implementing differentiable simulations into ML frameworks requires different functions and concepts than classical ML tasks.
The major differences are:

* Data typically represent objects or signals that exist in space and time. Data dimensions are interpretable, e.g. vector components, time series, *n*-dimensional lattices.
* Information transfer is usually local, resulting in sparsity in the dependency matrix between objects (particles, elements or cells).
* A high numerical accuracy is desirable for some operations, often requiring 64-bit and 32-bit floating point calculations.

However, current machine learning frameworks have been designed for the core ML tasks which reflects in their priorities and design choices.
This can result in overly verbose code when implementing scientific applications and may require implementing custom operators, since many 
common functions like sparse-sparse matrix multiplication, periodic padding or sparse linear solvers are not available in all libraries.

$\Phi_\textrm{ML}$ is a scientific computing library based on Python 3 [@Python3] targeting scientific applications that use machine learning methods.
Its main goals are:

* **Reusability.** Code based on $\Phi_\textrm{ML}$ should be able to run in many settings without modification. It should be agnostic towards the dimensionality of simulated systems and the employed discretization. All code should be trivially vectorizable.
* **Compatibility.** Users should be free to choose whatever ML or third-party library they desire without modifying their simulation code. $\Phi_\textrm{ML}$ should support Linux, Windows and Mac.
* **Usability.** $\Phi_\textrm{ML}$ should be easy to learn and use, matching existing APIs where possible. It should encourage users to write concise and expressive code.
* **Maintainability.** All high-level source code of $\Phi_\textrm{ML}$ should be easy to understand. Continuous testing should be used to ensure that future updates do not break existing code.
* **Performance.** $\Phi_\textrm{ML}$ should be able to make use of hardware accelerators, such as GPUs and TPUs, where possible. During development, we prioritize rapid code iterations over execution speed but the completed code should run as fast as if written directly against the chosen ML library.

In the following, we explain the architecture and major features that help $\Phi_\textrm{Flow}$ reach these goals.
$\Phi_\textrm{ML}$ consists of a high-level NumPy-like API geared towards writing easy-to-read and scalable simulation code, as well as a neural network API designed to allow users to quickly iterate over many network architectures and hyperparameter settings.
Similar to eagerpy [@rauber2020eagerpy], $\Phi_\textrm{ML}$ integrates with Jax [@Jax2018], PyTorch [@PyTorch2019], TensorFlow [@TensorFlow2016] and NumPy [@NumPy2020] and provides a custom `Tensor` class.
However, $\Phi_\textrm{ML}$ adds additional functionality.

* **Dimension names.** Tensor dimensions are always referenced by their user-defined name, not their index. We support the syntax `tensor.dim` for operations like indexing or unstacking to make using dimension names as simple as possible.
* **Automatic reshaping.** $\Phi_\textrm{ML}$ automatically transposes tensors and inserts singleton dimensions to match arguments. Consequently, user code is agnostic to the dimension order by default.
* **Element names.** Slices or *items* along dimensions can be named as well, e.g. allowing users to specify that a dimension lists the values `(x,y,z)` or `(r,g,b)`. These names can be used in slicing, gathering and scattering operations.
* **Dimension types.** Tensor dimensions are grouped into five different types: *batch*, *spatial*, *instance*, *channel*, and *dual*. This allows tensor-related functions to automatically select dimensions to operate on, without requiring the user to specify individual dimensions.
* **Non-uniform tensors.** Stacking tensors with different dimension sizes yields non-uniform tensors. $\Phi_\textrm{ML}$ keeps track of the resulting shape, allowing users to operate on non-uniform tensors the same way as uniform ones.
* **Floating-point precision by context.** All tensor operations determine the desired floating point precision from the operation context, not the data types of its inputs. This is much simpler and more predictable than the systems used by other libraries.
* **Lazy stacking.** New memory is only allocated once stacked data is required as a block. Consequently, functions can unstack the components, operate on them individually, and restack them, without worrying about unnecessary memory allocations.
* **Sparse matrices from linear functions.** $\Phi_\textrm{ML}$ can transform linear functions into their corresponding sparse matrix representation. This makes solving linear systems of equations more performant and enables computation of preconditioners.
* **Compute device from Inputs.** Tensor operations execute on the device on which the tensors reside. This prevents unintentional copies and transfers, as users have to explicitly declare them.
* **Custom CUDA Operatorions.** $\Phi_\textrm{ML}$ provides custom CUDA kernels for specific operations that could bottleneck simulations, such as grid sampling for TensorFlow or linear solves.



# Research Projects

$\Phi_\textrm{ML}$ has been in development since 2019 as part of the [$\Phi_\textrm{Flow}$](https://github.com/tum-pbs/PhiFlow) project where it originated as a unified API for TensorFlow and NumPy, used to run differentiable fluid simulations.
$\Phi_\textrm{Flow}$ includes geometry, physics, and visualization modules, all of which use the `math` API of $\Phi_\textrm{ML}$ to benefit from its reusability, compatibility, and performance.

It was first used to show that differentiable PDE simulations can be used to train neural networks that steer the dynamics towards desired outcomes [@phiflow].
Differentiable PDEs, implemented against $\Phi_\textrm{ML}$'s API, were later shown to benefit learning corrections for low-resolution or incomplete physics models [@SolverInTheLoop2020].
These findings were summarized and formalized in @PBDL2021, along with many additional examples.

The library was also used in network optimization publications, such as showing that inverted simulations can be used to train networks [@ScaleInvariant2022] and that gradient inversion benefits learning the solutions to inverse problems [@HalfInverse2022].

Simulations powered by $\Phi_\textrm{ML}$ have since been used in open data sets [@PDEBench; @PDEArena] and in publications from various research groups [@brandstetter2021message; @wandel2021teaching; @brandstetter2023clifford; @wandel2020learning; @sengar2021multi; @parekh1993sex; @ramos2022control; @wang2022approximately; @wang2022meta; @wang2023applications; @wu2022learning; @li2023latent].



# Acknowledgements

We would like to thank Robin Greif, Kartik Bali, Elias Djossou and Brener Ramos for their contributions, as well as everyone who contributed to the project on GitHub.


# References


