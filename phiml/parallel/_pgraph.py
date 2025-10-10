import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Set, Dict, List, Any, Union, Optional

from ..math import Shape
from ..math._trace import Tracer


@dataclass
class PGraphNode:
    name: str
    out: Union[Tracer, Any]
    distributed: Shape
    program: Optional[Any]  # code as str or Tracer objects?
    persistent: bool
    field_dep_names: Set[str]
    dependencies: Sequence['PGraphNode'] = None
    done: bool = False
    users: List['PGraphNode'] = dataclasses.field(default_factory=lambda: [])
    stage: int = None

    def __repr__(self):
        return f"{self.name}{f'@{self.stage}' if self.done else ' (pending)'}->{[n.name for n in self.dependencies]}<-{len(self.users)}"

    @cached_property
    def dist_names(self):
        return frozenset(self.distributed.names)

    @property
    def can_run_now(self):
        return not self.done and all(dep.done or (dep.can_run_now and dep.dist_names == self.dist_names) for dep in self.dependencies)

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

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


def build_stages(nodes: Dict[str, PGraphNode]) -> List[List[PGraphNode]]:
    """ Groups nodes by same `requires`, taking dependencies into account. """
    # ToDo check for cycles
    stages = []
    while any(not n.done for n in nodes.values()):
        candidates = [n for n in nodes.values() if n.can_run_now]
        candidate_dist = set([cn.dist_names for cn in candidates])
        stage_dist = next(iter(candidate_dist))
        stage = []
        for n in candidates:
            if n.dist_names == stage_dist:
                n.done = True
                n.stage = len(stages)
                stage.append(n)
        stages.append(stage)
    return stages
