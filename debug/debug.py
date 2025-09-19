from dataclasses import dataclass

from phiml import *
from phiml.parallel import parallel_property, parallel_compute, MIXED

set_logging_level()


@dataclass
class ExampleClass:
    file: Tensor

    @parallel_property(out=instance('points'))
    def disk_data(self) -> Tensor:  # not traceable, but we know that parallel dims are in the output
        assert self.file.shape.volume == 1
        file = self.file.item()
        data = randn(instance(points=len(file)))
        # data -= data.mean
        # data /= data.std
        return data

    @parallel_property(requires=MIXED)
    def normalized(self):
        # return self.disk_data.finite_std
        offset = self.disk_data.finite_mean
        norm = self.disk_data.finite_std
        return (self.disk_data - offset) / norm

# result = split_property_into_methods(ExampleClass.complex_calculation)
# print(result)

if __name__ == '__main__':
    # code = class_to_string(ExampleClass)
    # exec(code, namespace)

    data = ExampleClass(-f-f"data_{arange(batch(b=2))}.txt")
    parallel_compute(data, [ExampleClass.normalized], max_workers=1)
    print(data.__dict__)

# trace_result = trace_shapes(data, batch, ['complex_calculation'])
# print(trace_result['shapes'])
