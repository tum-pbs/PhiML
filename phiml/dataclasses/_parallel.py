import ast
import dataclasses
import inspect
import multiprocessing
import os.path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Sequence, Set, Optional, Dict, FrozenSet, List

import h5py
import numpy as np

from .. import unstack, stack
from ..backend import ML_LOGGER
from ..dataclasses._dep import MemberVariableAnalyzer
from ..dataclasses._tensor_cache import DiskTensor, H5Source
from ..math import DimFilter, shape, Shape, EMPTY_SHAPE
from ..math._magic_ops import all_attributes
from ..math._tensors import disassemble_tree, assemble_tree, Dense


def parallel_compute(instance, properties: Sequence, parallel_dims=shape,
                     max_workers=multiprocessing.cpu_count(), memory_limit: Optional[float] = None, cache_dir: str = None):
    """
    Compute the values of properties of a dataclass instance in parallel.

    If `@parallel_property` are computed whose `requires` overlaps with `parallel_dims`, a separate computation stage is set up to compute these properties with potentially less parallel workers.
    In the presence of different `requires`, the computation is split into different stages in accordance with the property dependency graph.

    Warnings:
        This breaks automatic differentiation as values may be cached on disk.

    See Also:
        `cached_property`.

    Args:
        instance: Dataclass instance for which to compute the values of `@cached_property` or `@parallel_property` fields.
        properties: References to the unbound properties. These must be `cached_property` or `parallel_property`.
        parallel_dims: Dimensions to parallelize over.
        max_workers: Number of processes to spawn.
        memory_limit: Set a limit to the total memory consumption from `Tensor` instances. If more than the allotted memory would be used, tensors are cached on disk.
        cache_dir: If `memory_limit` is set, cache files are stored in this directory.
    """
    # ToDo: Maybe add option to convert to NumPy? Or auto-convert tensors to the original backend on load from disk.
    if memory_limit is not None:
        assert cache_dir is not None, "cache_dir must be specified if memory_limit is set"
    dims = shape(instance).only(parallel_dims)
    # --- Build graph of relevant properties ---
    cls = type(instance)
    nodes: Dict[str, PGraphNode] = {}
    output_user = PGraphNode('<output>', None, None, None, False, [], -1)
    for p in properties:
        recursive_add_node(cls, property_name(p), p, dims, nodes).users.append(output_user)
    for node in nodes.values():
        if node.name in instance.__dict__:
            node.done = True
            node.dependencies = []
    stages = build_stages(nodes)
    any_parallel = any(dims - tuple(stage_nodes[0].requires) for stage_nodes in stages)
    if any_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            _execute_stages(instance, stages, dims, pool, memory_limit, cache_dir)
    else:
        _execute_stages(instance, stages, dims, None, memory_limit, cache_dir)


def _execute_stages(instance, stages: List[List['PGraphNode']], dims, pool, memory_limit, cache_dir):
    for stage_idx, stage_nodes in enumerate(stages):
        parallel_dims = dims - tuple(stage_nodes[0].requires)
        if parallel_dims:
            ML_LOGGER.debug(f"Parallel | {parallel_dims} | {[n.name for n in stage_nodes]}")
            property_names = [n.name for n in stage_nodes if n.is_used_later]
            required_caches = set.union(*[n.all_dep_names for n in stage_nodes])
            # --- Submit to pool ---
            instances = unstack(instance, parallel_dims)
            caches = [unstack(instance.__dict__[c], parallel_dims, expand=True) for c in required_caches]
            caches = [{c: caches[j][i] for j, c in enumerate(required_caches)} for i in range(len(instances))] if caches else [{}] * len(instances)
            inst_props = [property_names] * len(instances)
            mem_per_item = memory_limit / len(instances) if memory_limit is not None else None
            os.makedirs(cache_dir, exist_ok=True)
            cache_files = [os.path.join(cache_dir, f"s{stage_idx}_i{i}.h5") for i in range(len(instances))]
            results = list(pool.map(_evaluate_properties, instances, inst_props, caches, [mem_per_item]*len(instances), cache_files))
            for name, *outputs in zip(property_names, *results):
                output = stack(outputs, parallel_dims)
                instance.__dict__[name] = output
        else:  # No parallelization in this stage
            ML_LOGGER.debug(f"Host | {[n.name for n in stage_nodes]}")
            for n in stage_nodes:
                get_property_value(instance, n.name)


def _evaluate_properties(instance, properties, cache: dict, mem_limit, cache_file: str):  # called on workers
    instance.__dict__.update(cache)
    # print(f"Evaluating {properties} on {type(instance).__name__} with values {instance.__dict__}")
    values = [get_property_value(instance, p) for p in properties]
    if mem_limit is not None:
        tree, tensors = disassemble_tree(values, False, attr_type=all_attributes)
        sizes = [sum([t.backend.sizeof(nat) for nat in t._natives()]) for t in tensors]
        if sum(sizes) > mem_limit:
            sorted_pairs = sorted(zip(sizes, range(len(tensors))))
            sizes, indices = zip(*sorted_pairs)
            total_sizes = np.cumsum(sizes)
            cutoff = np.argmax(total_sizes > mem_limit)
            ML_LOGGER.debug(f"Result is too large for memory limit. Tensor sizes (bytes): {sizes[::-1]}. Caching {len(tensors) - cutoff} tensors to {cache_file}.")
            with h5py.File(cache_file, 'w') as f:
                h5_ref = H5Source(cache_file)
                for i in range(len(tensors) - cutoff - 1, len(tensors)):
                    t = tensors[i]
                    if isinstance(t, Dense):
                        f.create_dataset(f'array_{i}', data=t.backend.numpy(t._native))
                        disk_tensor = DiskTensor(h5_ref, f'array_{i}', {}, t._names, t._shape, t._backend, t.dtype)
                        tensors[i] = disk_tensor
                    else:
                        raise NotImplementedError
            values = assemble_tree(tree, tensors, attr_type=all_attributes)
    return values


def get_dependencies(cls: type, cls_property) -> Set[str]:
    if hasattr(cls, '__phiml_parallel_dep__') and cls_property in cls.__phiml_parallel_dep__:
        return cls.__phiml_parallel_dep__[cls_property]
    if isinstance(cls_property, cached_property):
        method = cls_property.func
    elif isinstance(cls_property, property):
        method = cls_property.fget
    else:
        assert callable(cls_property) and hasattr(cls_property, '__qualname__'), f"Dependency resolver failed on {cls_property} of {cls.__name__}"
        method = cls_property
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__}")
    source_code = inspect.getsource(method)
    indent0 = len(source_code) - len(source_code.lstrip())
    source_code_top = "\n".join([line[indent0:] for line in source_code.split("\n")])
    tree = ast.parse(source_code_top)
    analyzer = MemberVariableAnalyzer()
    analyzer.visit(tree)
    direct_deps = analyzer.member_vars
    fields = set([f.name for f in dataclasses.fields(cls)])
    all_caches = set([m for m in dir(cls) if isinstance(getattr(cls, m), cached_property)])
    prop_deps = all_caches & direct_deps
    # --- Indirect dependencies via method/property calls ---
    for method_dep in direct_deps - fields - prop_deps:
        if not hasattr(cls, method_dep):
            continue  # may be a dynamic dim, such as vector
        dep = getattr(cls, method_dep)
        if callable(dep) or isinstance(dep, property):
            prop_deps.update(get_dependencies(cls, dep))
    # --- Cache result ---
    if not hasattr(cls, '__phiml_parallel_dep__'):
        cls.__phiml_parallel_dep__ = {cls_property: prop_deps}
    else:
        cls.__phiml_parallel_dep__[cls_property] = prop_deps
    return prop_deps


@dataclass
class PGraphNode:
    name: str
    prop: property
    requires: FrozenSet[str]
    dependencies: Sequence['PGraphNode'] = None
    done: bool = False
    users: List['PGraphNode'] = dataclasses.field(default_factory=lambda: [])
    stage: int = None

    def __repr__(self):
        return f"{self.name} ({f'done in stage {self.stage}' if self.done else 'pending'}) depending on {[n.name for n in self.dependencies]} with {len(self.users)} users"

    @property
    def can_run_now(self):
        return not self.done and all(dep.done or (dep.can_run_now and dep.requires == self.requires) for dep in self.dependencies)

    @property
    def is_used_later(self):  # requires caching
        return any(u.stage != self.stage for u in self.users)  # output has stage -1

    @property
    def all_dep_names(self) -> Set[str]:
        return set.union(*[{dep.name, *dep.all_dep_names} for dep in self.dependencies]) if self.dependencies else set()


def recursive_add_node(cls, name: str, prop: Optional, dims: Shape, nodes: Dict[str, PGraphNode]) -> PGraphNode:
    if name in nodes:
        return nodes[name]
    prop = getattr(cls, name) if prop is None else prop
    requires = get_required_dims(prop, dims)
    node = nodes[name] = PGraphNode(name, prop, frozenset(requires.names))
    dep_names = get_dependencies(cls, prop)
    node.dependencies = [recursive_add_node(cls, n, None, dims, nodes) for n in dep_names]
    for dep in node.dependencies:
        dep.users.append(node)
    return node


def build_stages(nodes: Dict[str, PGraphNode]) -> List[List[PGraphNode]]:
    """ Groups nodes by same `requires`, taking dependencies into account. """
    stages = []
    while any(not n.done for n in nodes.values()):
        candidates = [n for n in nodes.values() if n.can_run_now]
        candidate_req = set([cn.requires for cn in candidates])
        stage_req = next(iter(candidate_req))
        stage = []
        for n in candidates:
            if n.requires == stage_req:
                n.done = True
                n.stage = len(stages)
                stage.append(n)
        stages.append(stage)
    return stages


def parallel_property(func: Callable = None, /, requires: DimFilter = None, allow_sequential=False):
    """
    Similar to `@cached_property` but with additional controls over parallelization.

    See Also:
        `parallel_compute()`.

    Args:
        func: Method to wrap.
        requires: Dimensions which must be present within one process. These cannot be parallelized when computing this property.
        allow_sequential: Whether calling the property normally (outside `parallel_compute`) is allowed for non-parallelized computation. If `False`, raises an error on invocation.
    """
    if func is None:
        return partial(parallel_property, requires=requires, allow_sequential=allow_sequential)
    return ParallelProperty(func, requires=requires, allow_sequential=allow_sequential)


_NOT_CACHED = object()


class ParallelProperty(cached_property):
    def __init__(self, func, requires: DimFilter, allow_sequential: bool):
        super().__init__(func)
        self.requires = requires
        self.allow_sequential = allow_sequential

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)

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
            if self.allow_sequential:
                val = super().__get__(instance, owner=owner)
            else:
                raise NonParallelAccess(f"@parallel_property '{self.attrname}' can only be accessed after evaluation by parallel_compute()")
        return val

    def get_on_host(self, instance):
        return super().__get__(instance, owner=None)

    # __class_getitem__ = classmethod(GenericAlias)  # inherited from cached_property


class NonParallelAccess(Exception):
    pass


def get_property_value(instance, name: str):
    prop = getattr(type(instance), name)
    if isinstance(prop, ParallelProperty):
        return prop.get_on_host(instance)
    else:
        return prop.__get__(instance)


def property_name(p):
    if isinstance(p, property):
        return p.fget.__name__
    elif isinstance(p, cached_property):
        return p.func.__name__
    elif hasattr(p, '__qualname__'):
        return p.__qualname__
    else:
        raise ValueError(f"Not a property: {p}")


def get_required_dims(p, dims: Shape) -> Shape:
    if isinstance(p, ParallelProperty):
        return dims.only(p.requires)
    return EMPTY_SHAPE
