"""
This package contains utilities for running PhiML code in parallel.
To use it, you need to declare the code to be parallelized as properties of a dataclass.
A tutorial can be found [here](https://tum-pbs.github.io/PhiML/Parallel_Compute.html).
The following is a simple example that runs a batch of FFTs in parallel:

>>> from dataclasses import dataclass
>>> from functools import cached_property
>>> from phiml.math import Tensor, spatial, fft, batch
>>> from phiml.parallel import parallel_property, parallel_compute, INFER, MIXED
>>>
>>> @dataclass(frozen=True)
>>> class ParallelComputation:
>>>     data: Tensor
>>>
>>>     @parallel_property(requires=spatial, out=INFER, on_direct_eval='raise')
>>>     def result(self) -> Tensor:
>>>         return fft(self.data)
>>>
>>> data = ...
>>> computation = ParallelComputation(data)
>>> parallel_compute(computation, [ParallelComputation.result], parallel_dims=batch, max_workers=4)
>>> computation.result

Properties declared as `@cached_property` behave as `@parallel_property(requires=INFER, out=INFER, on_direct_eval='host-compute')`.

## Computational Graph

PhiML performs static code analysis (source code in 1.14, bytecode from 1.15 onwards) to determine dependencies between
all properties declared as either `@parallel_property` or `@cached_property`.
Static analysis traces into methods and properties defined in the same class, but does not trace functions outside.
**Do not pass a reference `self` to external calls**, as this could lead to dependencies not being captured properly.

The resulting computational graph is split into computation stages depending on the `requires` values of the involved properties.
The `requires` property declares dims that must be present on the data in order to compute the result.
Properties that have no required dims can be parallelized across all dims specified in `parallel_compute`.
Properties that cannot be parallelized at all (because all parallel dims are marked as `requires`) are computed on the host process.

## INFER via Dynamic Traces

When either `requires` or `out` is set as `INFER`, PhiML performs a dynamic trace to infer their values.
These properties may only use PhiML functions and not all functions are supported as of yet.
Use `phiml.set_loggin_level` to catch failed traces.

## MIXED Parallelization

For properties that only use supported PhiML calls, you can set `requires=MIXED`, which allows PhiML to split the computation of the proeprty into multiple stages.
For example, the expression `math.sum(batched_data * 2, 'example,x,y')` would be split into three parts if `example` is parallelized over:

* Multiplication (parallel)
* Sum over `x,y` (parallel)
* Sum over `example` (on host)

Data transfers between workers and host are performed as needed.

Currently, tensor operators (`a+b`), simple one-tensor functions (`abs,exp,sin,round,is_nan,...`) and reduction functions (`sum,prod,max,finite_mean,...`) are supported.
Do not enable `MIXED` on functions that use unsupported functions. While direct calls on tracers will fail, there can still be undesirable effects.

## Disk Caching

By configuring `parallel_compute`, you can have workers write results onto disk instead of serializing the full result in transfers.
This should be used if system memory is limited or data needs to be passed between processes many times.
See [this example](https://tum-pbs.github.io/PhiML/Cached_Parallel_Example.html) for a demonstration.
"""

from ._parallel import parallel_compute, parallel_property, INFER, MIXED

from ._tensor_cache import on_load_into_memory, get_cache_files, set_cache_ttl, load_cache_as

__all__ = [key for key in globals().keys() if not key.startswith('_')]
