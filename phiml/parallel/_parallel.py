import multiprocessing
import os.path
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Sequence, Optional, Dict, Union, Any, Tuple

import h5py
import numpy as np

from ..dataclasses import data_fields
from ._cls_serialize import is_class_from_notebook, class_to_string
from ..dataclasses._dep import cache_deps, field_deps
from ._pgraph import PGraphNode, build_stages
from ._tensor_cache import _WORKER_LOAD_AS, _LOAD_AS, H5Source, write_to_h5
from ..backend import ML_LOGGER
from ..backend._backend import get_backend
from ..backend._dtype import FLOAT32
from ..math import unstack, stack, batch, dual, instance, spatial, channel
from ..math import DimFilter, shape, Shape, EMPTY_SHAPE, merge_shapes
from ..math._magic_ops import all_attributes
from ..math._tensors import equality_by_shape_and_value
from ..math._tree import disassemble_tree, assemble_tree, get_value_by_path
from ..math._trace import to_tracers, Trace, TracedOp, Tracer, expand_tracers


def parallel_compute(instance, properties: Sequence, parallel_dims=batch,
                     max_workers=multiprocessing.cpu_count(), memory_limit: Optional[float] = None, cache_dir: str = None, keep_intermediate=False):
    """
    Compute the values of properties decorated with `@cached_property` or `@parallel_property` of a dataclass instance in parallel.

    **Multiple stages via `requires=...`**
    If `@parallel_property` are computed whose `requires` overlaps with `parallel_dims`, a separate computation stage is set up to compute these properties with fewer parallel workers.
    In the presence of different `requires`, the computation is split into different stages in accordance with the property dependency graph.
    Properties that cannot be parallelized (because it requires all `parallel_dims`) are computed on the host process.

    **Caching tensors to disk**
    When `memory_limit` and `cache_dir` are set, the evaluation will try to adhere to the given memory limits by moving tensors out of memory onto disk.
    This is only applied to the outputs of `@cached_property` and `@parallel_property` calls, not to intermediate values used in their computation.
    The per-process memory limit is calculated per stage, dividing the total memory by the active worker count.
    Cached tensors behave like regular tensors and are temporarily loaded back into memory when accessing their values or using them in a computation.
    When parallelizing, the full result is assembled by stacking multiple disk-backed tensors from different files created by different processes.
    These composite tensors will reference multiple binary files and can be pickled/unpickled safely without loading the data into memory.
    This enables passing large data references to different processes or saving the structure to a file without the data content.

    See Also:
        `cached_property`, `parallel_property`, `get_cache_files()`, `on_load_into_memory()`.

    Warnings:
        `parallel_compute` breaks automatic differentiation.

    Args:
        instance: Dataclass instance for which to compute the values of `@cached_property` or `@parallel_property` fields.
        properties: References to the unbound properties. These must be `cached_property` or `parallel_property`.
        parallel_dims: Dimensions to parallelize over.
        max_workers: Number of processes to spawn.
        memory_limit: Limit to the total memory consumption from `Tensor` instances on property outputs.
        cache_dir: Directory path to store cached tensors in if `memory_limit` is set.
        keep_intermediate: Whether the outputs of cached properties required to compute `properties` but not contained in `properties` should be kept in memory.
            If `False`, these values will not be cached on `instance` after this call.
    """
    assert hasattr(instance, '__dict__'), f"parallel_compute requires instance to have __dict__. Slots are not supported."
    if memory_limit is not None:
        assert cache_dir is not None, "cache_dir must be specified if memory_limit is set"
    dims = shape(instance).only(parallel_dims)
    # --- Build graph of relevant properties ---
    cls = type(instance)
    if is_class_from_notebook(cls):
        class_example = (cls.__name__, class_to_string(cls))
    else:
        class_example = cls.__new__(cls)
    nodes: Dict[str, PGraphNode] = {}
    output_user = PGraphNode('<output>', EMPTY_SHAPE, parallel_dims, None, False, set(), None, False, [], 999999)
    for p in properties:
        recursive_add_node(instance, cls, property_name(p), p, dims, nodes).users.append(output_user)
    # nodes = merge_duplicate_nodes(nodes.values())
    for node in nodes.values():
        if node.name in instance.__dict__:
            node.done = True
            node.dependencies = []
    stages = build_stages(nodes)
    ML_LOGGER.debug(f"Assembled {len(stages)} stages containing {sum(len(ns) for ns in stages)} properties for parallel computation.")
    # --- Execute stages ---
    any_parallel = any(stage_nodes[0].distributed for stage_nodes in stages) and max_workers > 0
    max_workers = min(max_workers, max(stage_nodes[0].distributed.volume for stage_nodes in stages))
    init_args = (_WORKER_LOAD_AS[-1].name if _WORKER_LOAD_AS else None,)
    with ProcessPoolExecutor(initializer=init_worker, initargs=init_args, max_workers=max_workers) if any_parallel else nullcontext() as pool:
        for stage_idx, stage_nodes in enumerate(stages):
            parallel_dims = stage_nodes[0].distributed
            if parallel_dims and any_parallel:
                ML_LOGGER.debug(f"Parallel | {parallel_dims} | {[n.name for n in stage_nodes]}")
                property_names = [n.name for n in stage_nodes if n.is_used_later]
                programs = {n.name: n.program for n in stage_nodes if n.program is not None}
                required_caches = set.union(*[n.prior_dep_names for n in stage_nodes])
                required_fields = set.union(*[n.field_dep_names for n in stage_nodes])
                # --- Split data ---
                instances = unstack(instance, parallel_dims)
                n = len(instances)
                caches = [unstack(instance.__dict__[c], parallel_dims, expand=True) for c in required_caches]
                data = []
                for i, inst_i in enumerate(instances):
                    data_i = {}
                    data_i.update(**{f_name: inst_i.__dict__[f_name] for f_name in required_fields})
                    if caches:
                        data_i.update(**{c: caches[j][i] for j, c in enumerate(required_caches)})
                    data.append(data_i)
                keep_intermediate or delete_intermediate_caches(instance, stages, stage_idx)
                # --- Submit to pool ---
                mem_per_item = memory_limit / min(max_workers, len(instances)) if memory_limit is not None else None
                cache_dir is not None and os.makedirs(cache_dir, exist_ok=True)
                cache_file_suggestions = [os.path.join(cache_dir, f"s{stage_idx}_i{i}") for i in range(len(instances))] if cache_dir is not None else [None] * len(instances)
                results = list(pool.map(_evaluate_properties, [class_example]*n, [property_names]*n, [programs]*n, data, [mem_per_item]*n, cache_file_suggestions))
                for name, *outputs in zip(property_names, *results):
                    output = stack(outputs, parallel_dims)
                    instance.__dict__[name] = output
            else:  # No parallelization in this stage
                ML_LOGGER.debug(f"Host | {[n.name for n in stage_nodes]}")
                _EXECUTION_STATUS.append('host')
                for n in stage_nodes:
                    get_property_value(instance, n.name, n.program)
                assert _EXECUTION_STATUS.pop() == 'host', "Host execution status mismatch. (internal error)"
                keep_intermediate or delete_intermediate_caches(instance, stages, stage_idx)


def delete_intermediate_caches(instance, stages: list, stage_idx: int):
    for node in sum(stages[:stage_idx+1], []):
        if not node.persistent and not node.has_users_after(stage_idx) and node.name in instance.__dict__:
            ML_LOGGER.debug(f"Deleting host cache of {node.name} as it's not needed after stage {stage_idx}")
            del instance.__dict__[node.name]


def init_worker(load_as: Optional[str]):  # called on workers before _evaluate_properties
    _WORKER_LOAD_AS.clear()
    _LOAD_AS.clear()
    if load_as is not None:
        _LOAD_AS.append(get_backend(load_as))
    ML_LOGGER.debug(f"Worker initialized. load_as={load_as}. pid={os.getpid()}")


def _evaluate_properties(data_class_example: Any, properties, programs, cache: dict, mem_limit, cache_file_base: Optional[str]):  # called on workers
    if isinstance(data_class_example, tuple):  # (name, source)
        namespace = {}
        exec(data_class_example[1], namespace)
        base_cls = namespace[data_class_example[0]]
    else:
        base_cls = type(data_class_example)
    cls = instantiate_with_programs(base_cls, programs)
    instance = cls.__new__(cls)
    instance.__dict__.update(cache)
    # print(f"Evaluating {properties} on {type(instance).__name__} with values {instance.__dict__}")
    _EXECUTION_STATUS.append('worker')
    values = [get_property_value(instance, prop, None) for prop in properties]
    assert _EXECUTION_STATUS.pop() == 'worker', "Worker execution status mismatch. (internal error)"
    if mem_limit is not None:
        tree, tensors = disassemble_tree(values, False, attr_type=all_attributes)
        sizes = [sum([t.backend.sizeof(nat) for nat in t._natives()]) for t in tensors]
        if sum(sizes) > mem_limit:
            sorted_pairs = sorted(zip(sizes, range(len(tensors))))
            sizes, indices = zip(*sorted_pairs)
            total_sizes = np.cumsum(sizes)
            cutoff = np.argmax(total_sizes > mem_limit)
            file_counter = 0
            while True:
                cache_file = f"{cache_file_base}_{file_counter}.h5"
                try:
                    with h5py.File(cache_file, 'x') as f:  # fail if exists
                        ML_LOGGER.debug(f"Result is too large for memory limit. Tensor sizes (bytes): {sizes[::-1]}. Caching {len(tensors) - cutoff} tensors to {cache_file}.")
                        h5_ref = H5Source(cache_file)
                        for i in range(len(tensors) - cutoff - 1, len(tensors)):
                            tensors[i] = write_to_h5(tensors[i], f't{i}', f, h5_ref)
                        break
                except FileExistsError:
                    file_counter += 1
            values = assemble_tree(tree, tensors, attr_type=all_attributes)
    # return 1
    return values


def instantiate_with_programs(data_class: type, programs: Dict[str, Any]):
    if not programs:
        return data_class
    class_name = data_class.__name__
    # we may support compiling graphs in the future
    class_code = f"""
from dataclasses import dataclass
from functools import cached_property

@dataclass(frozen={data_class.__dataclass_params__.frozen})
class {class_name}_Derived({class_name}):
"""
    for prop, prog in programs.items():
        class_code += f"""
    @cached_property
    def {prop}(self):
        result = GRAPH_{prop}.run(self)
        return result
"""
    namespace = {f'GRAPH_{prop}': prog for prop, prog in programs.items()}
    namespace[class_name] = data_class
    exec(class_code, namespace)
    cls = namespace[f'{class_name}_Derived']
    return cls


_EXECUTION_STATUS = []  # 'host', 'worker'


def recursive_add_node(obj, cls, name: str, prop: Optional, dims: Shape, nodes: Dict[str, PGraphNode]) -> PGraphNode:
    if name in nodes:
        return nodes[name]
    prop = getattr(cls, name) if prop is None else prop
    persistent = isinstance(prop, ParallelProperty) and prop.persistent
    dep_names = cache_deps(cls, prop)
    dependencies = [recursive_add_node(obj, cls, n, None, dims, nodes) for n in dep_names]
    # --- Determine shape ---
    spec_out = prop.out if isinstance(prop, ParallelProperty) else INFER
    needs_trace = spec_out is INFER or (isinstance(prop, ParallelProperty) and prop.requires in {MIXED, INFER})
    already_computed = name in obj.__dict__
    if already_computed:
        precomputed_value = obj.__dict__[name]
        node = nodes[name] = PGraphNode(name, precomputed_value, EMPTY_SHAPE, None, True, set(), [], True, stage=-1)
        return node
    if needs_trace:
        out, trace = trace_cached_property(obj, cls, name, prop, dims, {d.name: (d.out, d.distributed) for d in dependencies})
        if isinstance(prop, ParallelProperty):
            if prop.requires is MIXED:
                last_node = split_mixed_prop(name, out, trace, dims, persistent, nodes)
                if last_node is not NotImplemented:
                    return last_node
                prop.requires = INFER
            if prop.requires is INFER:
                requires = merge_shapes(*[op.req_dims for op in trace.all_ops])
                ML_LOGGER.debug(f"Inferred {cls.__name__}.{name}.requires={requires}")
            else:
                requires = prop.requires
        else:  # @cached_property
            requires = merge_shapes(*[op.req_dims for op in trace.all_ops])
        if spec_out is not INFER:
            out = spec_out
        else:
            ML_LOGGER.debug(f"Inferred {cls.__name__}.{name}.out={out}")
    else:
        assert isinstance(prop, ParallelProperty)
        out = spec_out
        requires = prop.requires
    # --- Add node ---
    distributed = dims - requires  # ToDo does not take input shape into account
    field_dep_names = field_deps(cls, prop)
    node = nodes[name] = PGraphNode(name, out, distributed, None, persistent, field_dep_names, dependencies)
    for dep in node.dependencies:
        dep.users.append(node)
    return node


ALL_UNKNOWN = batch('__b__') + dual('__d__') + instance('__i__') + spatial('__s__') + channel('__c__')


def trace_cached_property(obj, cls, p_name: str, prop: cached_property, distributed: Shape, known_out: Dict[str, Tuple[Tracer, Shape]]):
    trace = Trace(f"{cls.__name__}.{p_name}", shape(obj), distributed)
    traced = cls.__new__(cls)
    f_deps = field_deps(cls, prop)
    traced.__dict__.update({f.name: trace.add_inputs(f.name, getattr(obj, f.name)) for f in data_fields(obj) if f.name in f_deps})
    for name, (tracers, distributed) in known_out.items():
        new_tracers = trace.add_tracers_as_input(name, tracers, distributed, use_label=True)
        traced.__dict__[name] = new_tracers
    with trace:
        try:
            trace_out = prop.func(traced)
            return trace_out, trace
        except Exception as exc:
            raise exc
            out_tracer = Tracer(trace, distributed + ALL_UNKNOWN, FLOAT32, {}, EMPTY_SHAPE, None)
            ML_LOGGER.warning(f"Failed to trace property {cls.__name__}.{p_name}: {type(exc)} {exc}. Assuming default shape {out_tracer.shape}")
            return out_tracer, trace


def split_mixed_prop(name: str, out: Any, trace: Trace, parallel_dims: Shape, persistent: bool, nodes: Dict[str, PGraphNode]):
    if len(trace.all_ops) == 0:
        return NotImplemented
    for i, op in enumerate(trace.all_ops):
        op.label = f"{name}_{i}_{op.name}"
    if isinstance(out, Tracer):
        out._op[0].label = name
    else:  # add assemble operation
        out_tree, all_out_tensors = disassemble_tree(out, False, all_attributes)
        trace.add_op('special', 'assemble_tree', (out_tree, all_out_tensors, all_attributes), {}, EMPTY_SHAPE, check_active=False, label=name)
    ML_LOGGER.debug(f"Splitting {name} into {[op.label for op in trace.all_ops]}")
    for op in trace.all_ops:
        distributed = parallel_dims.only(op.input_shape) - op.req_dims
        # --- Gather dependencies ---
        in_tracers = op.input_tracers
        ext_dep_tracers = [t for t in in_tracers if t._op is None]
        ext_deps = {nodes[t._name]: t for t in ext_dep_tracers if t._name in nodes}
        ext_fields = {t: t._name for t in ext_dep_tracers if t._name not in nodes}
        ext_field_base = {v.split('[', 1)[0].split('.', 1)[0] for v in ext_fields.values()}
        int_dep_tracers = [t for t in in_tracers if t._op is not None]
        int_deps = [nodes[t._op[0].label] for t in int_dep_tracers]
        # --- Python code (print only) ---
        dep_expr = {}
        for node, tracer in ext_deps.items():
            dep_expr[tracer] = f"self.{node.name}"
        for tracer in int_dep_tracers:
            dep_expr[tracer] = f"self.{tracer._op[0].label}[{tracer._op[1]}]" if tracer._op[1] is not None else f"self.{tracer._op[0].label}"
        for tracer, dep_field in ext_fields.items():
            dep_expr[tracer] = f"self.{dep_field}"
        code = f"""
@parallel_property(requires='{','.join(parallel_dims.only(op.req_dims).names)}')
def {op.label}(self):
    {op.python_imports}
    return {op.python_expression(dep_expr)}
"""
        ML_LOGGER.debug(code)
        # --- Create Program & Node ---
        op_trace = Trace(op.label, ..., distributed)
        replacement = {}
        for t in in_tracers:
            replacement[t] = op_trace.add_input(dep_expr[t][5:], t)
        single_op = op.replace_input_tracers(op_trace, replacement)
        program = TraceProgram(single_op)
        is_output_node = op.name == 'assemble_tree' or (isinstance(out, Tracer) and op.label == name)
        out_tracers = out if is_output_node else op.outputs
        node = PGraphNode(op.label, out_tracers, distributed, program, persistent and is_output_node, ext_field_base, list(ext_deps) + int_deps)
        nodes[op.label] = node
        for dep in node.dependencies:
            dep.users.append(node)
    return node


@dataclass
class TraceProgram:
    op: TracedOp

    def run(self, obj):
        data = {t: get_value_by_path(f'.{t._name}', obj) for t in self.op.input_tracers}
        return self.op.run(data)

    def __eq__(self, other):
        with equality_by_shape_and_value(0, 0, True):
            return self.op.name == other.op.name and self.op.op_type == other.op.op_type and self.op.args == other.op.args

    def __hash__(self):
        return hash(self.op.name)


MIXED = object()  # requires=MIXED causes function trace and split into multiple stages
INFER = object()  # requires=INFER causes function trace


def parallel_property(func: Callable = None, /,
                      requires: Union[DimFilter, object] = None,
                      out: Any = INFER,
                      persistent: bool = False,
                      on_direct_eval='raise'):
    """
    Similar to `@cached_property` but with additional controls over parallelization.

    See Also:
        `parallel_compute()`.

    Args:
        func: Method to wrap.
        requires: Dimensions which must be present within one process. These cannot be parallelized when computing this property.
        out: Declare output shapes and dtypes in the same tree structure as the output of `func`.
            Placeholders for `shape` and `dtype` can be created using `shape * dtype`.
            `Shape` instances will be assumed to be of floating-point type.
        persistent: If `True` the output of this property will be available after `parallel_compute` even if it was not specified as a property to be computed,
            as long as its computation is necessary to compute any of the requested properties.
        on_direct_eval: What to do when the property is accessed normally (outside `parallel_compute`) before it has been computed.
            Option:

            * `'raise'`: Raise an error.
            * `'host-compute'`: Compute the property directly, without using multi-threading.
    """
    assert on_direct_eval in {'host-compute', 'raise'}
    if out is not INFER:
        out = to_tracers(out)
    if func is None:
        return partial(parallel_property, requires=requires, out=out, persistent=persistent, on_direct_eval=on_direct_eval)
    return ParallelProperty(func, requires, out, persistent, on_direct_eval)


_NOT_CACHED = object()


class ParallelProperty(cached_property):
    def __init__(self, func, requires: DimFilter, out: Any, persistent: bool, on_direct_eval: str):
        super().__init__(func)
        self.requires = requires
        self.out = out
        self.persistent = persistent
        self.on_direct_eval = on_direct_eval

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)

    def __repr__(self):
        return self.attrname

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError("Cannot use cached_property instance without calling __set_name__ on it.")
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            raise TypeError(f"No '__dict__' attribute on {type(instance).__name__!r} instance to cache {self.attrname!r} property.") from None
        val = cache.get(self.attrname, _NOT_CACHED)
        if val is _NOT_CACHED:
            if _EXECUTION_STATUS:
                val = super().__get__(instance, owner=owner)
            elif self.on_direct_eval == 'host-compute':
                val = super().__get__(instance, owner=owner)
            elif self.on_direct_eval == 'raise':
                raise NonParallelAccess(f"@parallel_property '{self.attrname}' can only be accessed after evaluation by parallel_compute()")
        return val

    def get_on_host(self, instance):
        return super().__get__(instance, owner=None)

    # __class_getitem__ = classmethod(GenericAlias)  # inherited from cached_property


class NonParallelAccess(Exception):
    pass


def get_property_value(obj, name: str, program: Optional[TraceProgram]):
    if program:
        result = program.run(obj)
        obj.__dict__[name] = result
        return result
    prop = getattr(type(obj), name)
    if isinstance(prop, ParallelProperty):
        return prop.get_on_host(obj)
    else:
        return prop.__get__(obj)


def property_name(p):
    if isinstance(p, property):
        return p.fget.__name__
    elif isinstance(p, cached_property):
        return p.func.__name__
    elif hasattr(p, '__qualname__'):
        return p.__qualname__
    else:
        raise ValueError(f"Not a property: {p}")
