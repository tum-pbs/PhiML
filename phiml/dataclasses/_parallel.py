import ast
import dataclasses
import inspect
import multiprocessing
import os.path
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Sequence, Set, Optional, Dict, FrozenSet, List

import h5py
import numpy as np

from .. import unstack, stack, batch
from ..backend import ML_LOGGER
from ..dataclasses._dep import MemberVariableAnalyzer
from ..dataclasses._tensor_cache import H5Source, write_to_h5
from ..math import DimFilter, shape, Shape, EMPTY_SHAPE
from ..math._magic_ops import all_attributes
from ..math._tensors import disassemble_tree, assemble_tree


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
    if memory_limit is not None:
        assert cache_dir is not None, "cache_dir must be specified if memory_limit is set"
    dims = shape(instance).only(parallel_dims)
    # --- Build graph of relevant properties ---
    cls = type(instance)
    nodes: Dict[str, PGraphNode] = {}
    output_user = PGraphNode('<output>', None, None, None, False, [], 999999)
    for p in properties:
        recursive_add_node(cls, property_name(p), p, dims, nodes).users.append(output_user)
    for node in nodes.values():
        if node.name in instance.__dict__:
            node.done = True
            node.dependencies = []
    stages = build_stages(nodes)
    ML_LOGGER.debug(f"Assembled {len(stages)} stages containing {sum(len(ns) for ns in stages)} properties for parallel computation.")
    # --- Execute stages ---
    any_parallel = any(dims - tuple(stage_nodes[0].requires) for stage_nodes in stages)
    max_workers = min(max_workers, max((dims - tuple(stage_nodes[0].requires)).volume for stage_nodes in stages))
    with ProcessPoolExecutor(max_workers=max_workers) if any_parallel else nullcontext() as pool:
        for stage_idx, stage_nodes in enumerate(stages):
            parallel_dims = dims - tuple(stage_nodes[0].requires)
            if parallel_dims:
                ML_LOGGER.debug(f"Parallel | {parallel_dims} | {[n.name for n in stage_nodes]}")
                property_names = [n.name for n in stage_nodes if n.is_used_later]
                required_caches = set.union(*[n.prior_dep_names for n in stage_nodes])
                # --- Submit to pool ---
                instances = unstack(instance, parallel_dims)
                caches = [unstack(instance.__dict__[c], parallel_dims, expand=True) for c in required_caches]
                caches = [{c: caches[j][i] for j, c in enumerate(required_caches)} for i in range(len(instances))] if caches else [{}] * len(instances)
                keep_intermediate or delete_intermediate_caches(instance, stages, stage_idx)
                inst_props = [property_names] * len(instances)
                mem_per_item = memory_limit / min(max_workers, len(instances)) if memory_limit is not None else None
                cache_dir is not None and os.makedirs(cache_dir, exist_ok=True)
                cache_file_suggestions = [os.path.join(cache_dir, f"s{stage_idx}_i{i}") for i in range(len(instances))] if cache_dir is not None else [None] * len(instances)
                results = list(pool.map(_evaluate_properties, instances, inst_props, caches, [mem_per_item]*len(instances), cache_file_suggestions))
                for name, *outputs in zip(property_names, *results):
                    output = stack(outputs, parallel_dims)
                    instance.__dict__[name] = output
            else:  # No parallelization in this stage
                ML_LOGGER.debug(f"Host | {[n.name for n in stage_nodes]}")
                _EXECUTION_STATUS.append('host')
                for n in stage_nodes:
                    get_property_value(instance, n.name)
                assert _EXECUTION_STATUS.pop() == 'host', "Host execution status mismatch. (internal error)"
                keep_intermediate or delete_intermediate_caches(instance, stages, stage_idx)


def delete_intermediate_caches(instance, stages: list, stage_idx: int):
    for node in sum(stages[:stage_idx+1], []):
        if not node.has_users_after(stage_idx) and node.name in instance.__dict__:
            ML_LOGGER.debug(f"Host | Removing cache of {node.name} as it's not needed after stage {stage_idx}")
            del instance.__dict__[node.name]


def _evaluate_properties(instance, properties, cache: dict, mem_limit, cache_file_base: Optional[str]):  # called on workers
    instance.__dict__.update(cache)
    # print(f"Evaluating {properties} on {type(instance).__name__} with values {instance.__dict__}")
    _EXECUTION_STATUS.append('worker')
    values = [get_property_value(instance, p) for p in properties]
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
    return values


_EXECUTION_STATUS = []  # 'host', 'worker'


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
        return f"{self.name}{f'@{self.stage}' if self.done else ' (pending)'}->{[n.name for n in self.dependencies]}<-{len(self.users)}"

    @property
    def can_run_now(self):
        return not self.done and all(dep.done or (dep.can_run_now and dep.requires == self.requires) for dep in self.dependencies)

    @property
    def is_used_later(self):  # requires caching
        return any(u.stage != self.stage for u in self.users)  # output has stage -1

    @property
    def all_dep_names(self) -> Set[str]:
        return set.union(*[{dep.name, *dep.all_dep_names} for dep in self.dependencies]) if self.dependencies else set()

    @property
    def prior_dep_names(self) -> Set[str]:
        prior_dependencies = [dep for dep in self.dependencies if dep.stage < self.stage]
        return {dep.name for dep in prior_dependencies}

    def has_users_after(self, stage: int):
        return any(u.stage > stage for u in self.users)


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


def parallel_property(func: Callable = None, /, requires: DimFilter = None, on_direct_eval='raise'):
    """
    Similar to `@cached_property` but with additional controls over parallelization.

    See Also:
        `parallel_compute()`.

    Args:
        func: Method to wrap.
        requires: Dimensions which must be present within one process. These cannot be parallelized when computing this property.
        on_direct_eval: What to do when the property is accessed normally (outside `parallel_compute`) before it has been computed.
            Option:

            * `'raise'`: Raise an error.
            * `'host-compute'`: Compute the property directly, without using multi-threading.
    """
    assert on_direct_eval in {'host-compute', 'raise'}
    if func is None:
        return partial(parallel_property, requires=requires, on_direct_eval=on_direct_eval)
    return ParallelProperty(func, requires=requires, on_direct_eval=on_direct_eval)


_NOT_CACHED = object()


class ParallelProperty(cached_property):
    def __init__(self, func, requires: DimFilter, on_direct_eval: str):
        super().__init__(func)
        self.requires = requires
        self.on_direct_eval = on_direct_eval

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
