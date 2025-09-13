import os
from dataclasses import dataclass
from functools import cached_property

from phiml import Tensor, tensor, mean, batch
from phiml.dataclasses import parallel_property


def expensive_to_compute(x: float) -> float:
    import time
    time.sleep(1)  # Simulate an expensive computation
    return 2 * x


@dataclass(frozen=True)
class ParallelComputation:
    data: Tensor

    @parallel_property
    def result(self) -> Tensor | float:
        return expensive_to_compute(float(self.data))


@dataclass(frozen=True)
class ParallelDepComputation:
    data: Tensor

    @cached_property
    def tmp_result(self) -> Tensor | float:
        return expensive_to_compute(float(self.data))

    @cached_property
    def result(self) -> Tensor | float:
        return self.tmp_result + 1


@dataclass(frozen=True)
class ParallelMeanComputation:
    data: Tensor

    @parallel_property
    def individual_result(self) -> Tensor | float:
        print(f"Computing individual_result pid={os.getpid()}")
        return expensive_to_compute(float(self.data))

    @parallel_property(requires=batch)
    def mean(self) -> Tensor | float:
        print(f"Computing mean, pid={os.getpid()}")
        return mean(self.individual_result, batch)


@dataclass(frozen=True)
class ParallelNormComputation:
    data: Tensor

    @parallel_property
    def individual_result(self) -> Tensor | float:
        print(f"Computing individual_result pid={os.getpid()}")
        return expensive_to_compute(float(self.data))

    @parallel_property(requires=batch)
    def mean(self) -> Tensor | float:
        print(f"Computing mean, pid={os.getpid()}")
        return mean(self.individual_result, batch)

    @parallel_property
    def normalized_result(self) -> Tensor | float:
        print(f"Computing normalized_result pid={os.getpid()}")
        return self.individual_result - self.mean