---
title: '$\Phi_\textrm{ML}$: A Science-oriented Math and Neural Network Library for Jax, PyTorch, TensorFlow & NumPy'
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
    affiliation: 1
affiliations:
  - name: Technical University of Munich
    index: 1
date: 01 August 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
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
* A high numerical accuracy is desirable, often requiring 64-bit floating point calculations.

However, current machine learning frameworks have been designed for the core ML tasks which reflects in their priorities and design choices.
This can result in overly verbose code when implementing scientific applications and may require implementing custom operators, since many 
common functions like sparse-sparse matrix multiplication, periodic padding or triangular solves are not available in all libraries.

$\Phi_\textrm{ML}$ is a scientific computing library based on Python 3 [@Python3] that aims to address these issues and simplify scientific code in the process.
It consists of a high-level NumPy-like API geared towards writing easy-to-read and scalable simulation code, as well as a neural network API designed to allow users to quickly iterate over many network architectures and hyperparameter settings.
Similar to eagerpy [@rauber2020eagerpy], $\Phi_\textrm{ML}$ integrates with Jax [@Jax2018], PyTorch [@PyTorch2019], TensorFlow [@TensorFlow2016] and NumPy [@NumPy2020], providing a custom Tensor class.
However, unlike eagerpy, $\Phi_\textrm{ML}$'s `Tensor` adds additional functionality to make user code more concise and easier to read.

$\Phi_\textrm{ML}$ has been in development since 2019 as part of the [$\Phi_\textrm{Flow}$](https://github.com/tum-pbs/PhiFlow) [@phiflow] project where it originated as a unified API for TensorFlow and NumPy, used to run differentiable fluid simulations.
With $\Phi_\textrm{Flow}$ version 2.0 and consecutive releases, $\Phi_\textrm{ML}$ underwent a drastic overhaul.
A major issue with the previous API, and in fact all popular ML APIs, is the need for reshaping, which can quickly get out of hand for physical simulations.
The work towards automatic reshaping sparked most of the changes that have been made to the library since.

We will first explain the design principles underlying $\Phi_\textrm{ML}$'s development, before detailing the major design decisions and resulting architecture.
For a list of supported features, see the [GitHub homepage](https://github.com/tum-pbs/PhiML).


# Design Principles
Here, we lay out our goals in developing $\Phi_\textrm{ML}$, which serve as the foundation for the design.

### Reusability
Simulation code based on $\Phi_\textrm{ML}$ should be able to run in many settings without modification.
The dynamics of a system, e.g. governed by a partial differential equations, are often formulated in a dimension-agnostic manner.
Simulation code implementing these dynamics should also exhibit that property.
Most simulations use some form of discretization, such as particles or grids.
Simulation code written for one such discretization should be easy to port to another appropriate one.

### Compatibility
There are many toolkits and libraries extending ML frameworks with specialized functionality.
These are generally only available for a certain framework, be it TensorFlow, PyTorch or Jax.
$\Phi_\textrm{ML}$ users should be free to choose whatever framework they desire without modifying their simulation code.
Additionally, simulations should be able to run on GPUs and CPUs and be vectorizable without modification.
$\Phi_\textrm{ML}$ should support Linux, Windows and Mac.

### Usability
$\Phi_\textrm{ML}$ should be easy to learn and use.
To achieve this, the API should be intuitive with expressively named functions matching existing frameworks where possible.
User code as well as built-in simulation functionality should be easy to read, i.e. concise and expressive.
We give a more detailed explanation of easy-to-read code below.

### Maintainability
Users should be able to read and understand all high-level source code of $\Phi_\textrm{ML}$.
All relevant framework functions should undergo continuous testing to ensure patches do not break existing code.
When installing $\Phi_\textrm{ML}$, users should be able to check the installation status and get hints as to how to solve potential issues.

### Performance
Code using $\Phi_\textrm{ML}$ should be able to make use of hardware accelerators (GPUs, TPUs) where possible.
During development, we prioritize rapid code iterations over execution speed but the completed code should run as fast as if written directly against the chosen ML library.


# Major Design Decisions

### Support for Jax, PyTorch, TensorFlow & NumPy
A large fraction of scientific code is re-written one or multiple times due to different preferences in programing languages and libraries.
To avoid this as much as possible and reach a large audience, we decided to make $\Phi_\textrm{ML}$ compatible with all major Python-based ML libraries as well as NumPy, which they all integrate with.
To realize this, we employ the adapter pattern [@HFDPatterns2004], creating an abstract `Backend` class with adapter subclasses for NumPy, TensorFlow, PyTorch and Jax.
This API operates directly on backend-specific tensors, and we use it to implement low-level functions, such as linear algebra routines and neighborhood search.
However, writing code that actually runs with all backends requires advanced knowledge of all backends due to the subtle differences between them.
PyTorch, for example, does not allow negative steps in tensor slices and TensorFlow does not support assigning values to slices.


### Custom `Tensor` class
The differences between the backends motivate us to provide a `Tensor` class that handles consistently across all backends.
It also enables most of the additional functionality described below, making it easier to write reusable code.
A $\Phi_\textrm{ML}$ tensor wraps and extends a tensor from one of the supported backends.
To operate efficiently on $\Phi_\textrm{ML}$ tensors, we include a NumPy-like public API while relegating the `Backend` API to internal use.
The public API takes in $\Phi_\textrm{ML}$ tenors, determines the appropriate `Backend`, and calls the corresponding low-level function.
Since all backend-specific tensors are represented by the same `Tensor` class in $\Phi_\textrm{ML}$, code written against $\Phi_\textrm{ML}$'s public API is backend-agnostic.
Data can also be passed between backends, internally using the tensor sharing functionality of DLPack [@DLPack2017] when possible.
This way, an easy-to-use PyTorch network can interact with a Jax simulation for performance but also with an identical PyTorch simulation to facilitate debugging.


### Named dimensions
In $\Phi_\textrm{ML}$, dimensions are not referenced by their index but by name instead.
We make dimension names mandatory for all dimensions, forcing users to explicitly document the meaning of each dimension upon creation.
The name information gets preserved by tensor manipulations and can be inspected at any later point, e.g. by printing it or using a debugger.
Named dimensions are also present in other numerics libraries, such as pandas [@Pandas2010], xarray [@xarray2017], einops [@Einops2018],
and are available for PyTorch as an add-on [@NamedTensor].
However, these libraries make dimension names optional and, consequently, cannot support them to the same extent that $\Phi_\textrm{ML}$ can,
preventing mainstream adoption.
In $\Phi_\textrm{ML}$, dimension names are one part of a carefully-designed set of tools, making them more intuitive and useful than in previous libraries.
For instance, $\Phi_\textrm{ML}$ introduces the convenience slicing syntax `tensor.dim_name[start:stop:step]`, replacing the less readable slices `tensor[..., start:stop:step, :]`, and supports dimension names in all functions as first-class citizens.
While naming dimensions adds a small amount of additional code, this is easily outweighed by the gains in readability and ease of debugging.
Furthermore, dimension names enable automatic reshaping, which eliminates the need for reshaping operations in user code, often significantly reducing the amount of required boilerplate code.


### Automatic reshaping
Named dimensions make it possible to perform reshaping, transposing, squeezing and un-squeezing operations completely under-the-hood.
$\Phi_\textrm{ML}$ realizes this by aligning equally-named dimensions.
Take the operation `a + b` where `a` has dimensions (x, y) and `b` has (y, z).
Then $\Phi_\textrm{ML}$ will expand `a` by z and `b` by x so that both arguments have the common shape (x,y,z) before adding them.
This automatic reshaping eliminates the vast majority of shape-related errors as user code is agnostic to the dimension order by default.


### Element names along dimensions
In addition to naming dimensions, $\Phi_\textrm{ML}$ also supports naming slices or *items* along dimensions.
This is optional but highly recommended for dimensions that enumerate interpretable quantities, such as vector components (x, y, z).
UnifyML can then check at runtime that the component order is consistent, i.e. that no vector (z, y, x) is added to an (x, y, z)-ordered quantity.
Additionally, the slicing syntax becomes more readable when using item names, e.g. `tensor.vector['x']` instead of the traditional `tensor[:, 0, ...]` or `tensor[:, -1, ...]` (PyTorch dimension order).


### Non-uniform tensors
With some data structures, such as staggered grids, the number of elements along one or multiple dimensions can be variable.
We will refer to tensors holding such data as *non-uniform* tensors, but they are also known as *ragged* or *nested* tensors.
Users will often pad the missing elements with zeros to make the data easier to handle but this can lead to problems down the line.
Instead, $\Phi_\textrm{ML}$ automatically creates non-uniform tensors when stacking tensors with non-matching shapes.
The `shape` attribute of a non-uniform tensor stores its exact layout, allowing users to operate on non-uniform shapes like on regular shapes, e.g. allocating new memory with `zeros(non_uniform_shape)`.


### Unified functional math
For differentiation, just-in-time compilation and iterative solves, we adopt a function-based approach similar to Jax.
This is different from TensorFlow, where gradients are tracked via Python context managers, and PyTorch, where gradients are attached to tensors.
$\Phi_\textrm{ML}$ unifies these different paradigms, providing unified function operations that run with all backends.
For example, `math.functional_gradient(f)` returns a function that computes the gradient of `f` and, to solve a sparse system of linear equations, users simply supply a Python function and the desired output of that function.


### Dimension types
In all backend libraries, tensor operations act only on certain dimensions, determined by the dimension index in the shape.
This behaviour is generally not consistent for all backend libraries.
Consider the response to extra leading dimensions in PyTorch:

* Most functions treat leading dimensions as batch dimensions, i.e. they deal with slices independently, either sequentially or in parallel.
* Some functions like `bincount` do not allow extra dimensions.
* Reduction functions like `sum` or `prod` reduce leading dimensions by default.
* Some functions, such as `histogram`, flatten all input dimensions.
* Some functions, such as `pad`, only allow a certain number of leading dimensions.

$\Phi_\textrm{ML}$ solves these issues by assigning a type to each dimension.
Each of the five allowed types, *batch*, *spatial*, *instance*, *channel*, and *dual*, determines how math functions act on dimensions of that type.
Spatial operations like `fft` only act on spatial dimensions and
*all* functions accept tensors with any number of batch dimensions which are always preserved in the operation.
Importantly, the order of dimensions is irrelevant to all math functions, only the types matter.
For an explanation of all dimension types and further advantages of this system, see the online documentation.


### Floating-point precision by context
Specifying the floating point precision can be a major headache in computing libraries.
NumPy automatically up-casts data types (`bool` → `int` → `float` → `complex`) and floating point precision (16 bit → 32 bit → 64 bit).
This can cause unintentional data type conversions when trying to run code with a different precision, as new arrays are FP64 by default.
To avoid these issues, TensorFlow has completely disabled automatic type conversion and Jax has disabled FP64 by default.
$\Phi_\textrm{ML}$ solves the data type problem by enabling automatic casting but determining the desired floating point precision from the operation context rather than the data types of its inputs.
The precision can be set globally or specified locally via context managers.
All operations automatically convert tensors of non-matching data types.
This avoids data-type-related problems and errors, as well as making user code more concise and cohesive.


### Lazy stacking
Simulations often perform component-wise operations separately if there is no function achieving the desired effect with a single call, like computing the x, y and z-component of a velocity field in three lines.
This often leads users to declare separate variables for the components to avoid repeated tensor stacking and slicing.
However, this clutters the code and prevents it from being dimension-agnostic.
Instead, $\Phi_\textrm{ML}$ performs lazy stacking by default, i.e. memory is only allocated once the stacked data is required as a block.
Consequently, functions can unstack the components, operate on them individually, and restack them, without worrying about unnecessary memory allocations.
This system also facilitates stacking tracer tensors, which cannot be done eagerly.


### Just-in-time compilation
While the previous features allow for concise, expressive and flexible code, the added abstraction layer and shape tracking induces an additional performance overhead.
To avoid this in production, $\Phi_\textrm{ML}$ supports just-in-time (JIT) compilation for PyTorch, TensorFlow and Jax.
Once compiled, only the tensor operations are executed, eliminating all Python-based overhead.


### Sparse matrices from linear functions
Solving linear systems of equations is a key requirement in both particle and grid-based simulations.
Since the physical influence is typically limited to neighboring sample points or particles, the resulting linear systems are often sparse.
Constructing such sparse matrices by hand yields code that is hard to understand and debug as well as limited to specific boundary conditions.
Instead, $\Phi_\textrm{ML}$ lets users specify linear systems with a linear Python function, like with matrix-free solvers.
However, these functions often consist of many individual operations, which makes it inefficient to call them at each solver iteration.
To avoid this overhead, $\Phi_\textrm{ML}$ can convert most linear and affine functions to sparse matrices so that solvers can perform the matrix multiplication in a single operation.
When JIT-compiling a simulation that includes a linear solve, the matrix generation will be performed during the initial tracing of the function, assuming the sparsity pattern is constant.


### Compute device from Inputs
Like PyTorch, $\Phi_\textrm{ML}$ executes operations on the device where the tensors are allocated.
This prevents unintentional copies of tensors as users have to explicitly declare transfer operations.
This is unlike TensorFlow, where context managers can be used to specify the target device for code blocks.


### Custom CUDA Operatorions
$\Phi_\textrm{ML}$ provides custom CUDA kernels for specific operations that could bottleneck simulations, such as grid sampling for TensorFlow or linear solves.
If available, these will be used automatically in place of the fallback Python implementation.


# Acknowledgements

We would like to thank Robin Greif, Kartik Bali, Elias Djossou and Brener Ramos for their contributions, as well as everyone who contributed to the project on GitHub.


# References


