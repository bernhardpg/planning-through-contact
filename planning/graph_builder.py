from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet
from pydrake.math import eq

from geometry.contact import (
    CollisionPair,
    ContactModeType,
    RigidBody,
    calc_intersection_of_contact_modes,
)
from geometry.polyhedron import PolyhedronFormulator

# flake8: noqa


@dataclass
class ModeConfig:
    modes: Dict[str, ContactModeType]
    additional_constraints: Optional[npt.NDArray[sym.Formula]] = None

    def calculate_match(self, other) -> int:
        modes_self = list(self.modes.values())
        modes_other = list(other.modes.values())

        num_not_equal = sum([m1 != m2 for m1, m2 in zip(modes_self, modes_other)])
        return num_not_equal


@dataclass(order=True)
class PrioritizedModeConfig:
    priority: int
    item: ModeConfig = field(compare=False)


@dataclass
class GraphVertex:
    name: str
    config: ModeConfig
    convex_set: ConvexSet


@dataclass
class GraphEdge:
    u: GraphVertex
    v: GraphVertex


# TODO Find a better way to pass all these objects
class Graph:
    def __init__(
        self,
        collision_pairs: List[CollisionPair],
        all_decision_vars: npt.NDArray[sym.Variable],
    ):
        self.collision_pairs = {p.name: p for p in collision_pairs}
        self.edges = []
        self.vertices = []
        self.all_decision_vars = all_decision_vars

    # TODO not sure if this belongs here
    def create_new_vertex(
        self, config: ModeConfig, name: Optional[str] = None
    ) -> Optional[GraphVertex]:
        contact_modes = [
            self.collision_pairs[pair].contact_modes[mode]
            for pair, mode in config.modes.items()
        ]
        intersects, (
            calculated_name,
            intersection,
        ) = calc_intersection_of_contact_modes(contact_modes)
        if not intersects:
            return None

        if config.additional_constraints is not None:
            additional_set = PolyhedronFormulator(
                config.additional_constraints
            ).formulate_polyhedron(self.all_decision_vars)
            intersects = intersects and intersection.IntersectsWith(additional_set)
            if not intersects:
                return None

            intersection = intersection.Intersection(additional_set)

        name = f"{name}: {calculated_name}" if name is not None else calculated_name
        vertex = GraphVertex(name, config, intersection)
        return vertex

    def add_edge(self, u: GraphVertex, v: GraphVertex) -> None:
        e = GraphEdge(u, v)
        self.edges.append(e)

    def add_vertex(self, u: GraphVertex) -> None:
        self.vertices.append(u)

    def add_source(self, source: GraphVertex) -> None:
        self.source = source
        self.add_vertex(source)

    def add_target(self, target: GraphVertex) -> None:
        self.target = target
        self.add_vertex(target)

    def contains_vertex(self, v: GraphVertex) -> bool:
        # TODO clean up
        vertices_names = [v.name for v in self.vertices]
        return v.name in vertices_names


class GraphBuilder:
    def __init__(
        self,
        all_decision_vars: List[sym.Variable],  # TODO this should not be here
        rigid_bodies: List[RigidBody],
        collision_pairs: List[CollisionPair],  # TODO Will be removed
        external_forces: List[sym.Expression],
        additional_constraints: Optional[List[sym.Formula]],
        allow_sliding: bool = False,
    ) -> None:

        self.all_decision_vars = all_decision_vars
        self.rigid_bodies = rigid_bodies
        self.collision_pairs = collision_pairs
        unactuated_dofs = self._get_unactuated_dofs(
            self.rigid_bodies, self.position_dim
        )

        # TODO maybe I should have a module that deals with CollisionPairs? CollisionPairHandler?
        force_balance_constraints = self.construct_force_balance(
            collision_pairs,
            self.rigid_bodies,
            external_forces,
            unactuated_dofs,
        )
        for p in self.collision_pairs:
            p.add_force_balance(force_balance_constraints)
        for p in self.collision_pairs:
            p.add_constraint_to_all_modes(additional_constraints)

        for p in self.collision_pairs:
            p.formulate_contact_modes(self.all_decision_vars, allow_sliding)

        # 1. Build source and target nodes (with force balance)
        # 2. Start with source node:
        #   - change one contact mode at a time to obtain the frontier.
        # 3. Add node edge if it is reachable
        # 4. Repeat until we 'hit' target node:
        #   - Repeat until we can actually make an edge to the target node
        #
        # How to deal with repeated visits to a node? For now we just make graph repeated after building it

    # TODO feels like I should be able to remove this too
    @property
    def position_dim(self) -> int:
        return self.rigid_bodies[0].dim

    def _get_unactuated_dofs(
        self, rigid_bodies: List[RigidBody], dim: int
    ) -> npt.NDArray[np.int32]:
        unactuated_idxs = [i for i, b in enumerate(rigid_bodies) if not b.actuated]
        unactuated_dofs = np.concatenate(
            [np.arange(idx * dim, (idx + 1) * dim) for idx in unactuated_idxs]
        )
        return unactuated_dofs

    # TODO move to contact or dynamics module
    @staticmethod
    def construct_force_balance(
        collision_pairs: List[CollisionPair],
        bodies: List[RigidBody],
        external_forces: npt.NDArray[sym.Expression],
        unactuated_dofs: npt.NDArray[np.int64],
    ) -> List[sym.Formula]:
        normal_jacobians = np.vstack(
            [p.get_normal_jacobian_for_bodies(bodies) for p in collision_pairs]
        )
        tangential_jacobians = np.vstack(
            [p.get_tangential_jacobian_for_bodies(bodies) for p in collision_pairs]
        )

        normal_forces = np.concatenate([p.lam_n for p in collision_pairs])
        friction_forces = np.concatenate([p.lam_f for p in collision_pairs])

        all_force_balances = eq(
            normal_jacobians.T.dot(normal_forces)
            + tangential_jacobians.T.dot(friction_forces)
            + external_forces,
            0,
        )
        force_balance = all_force_balances[unactuated_dofs, :]
        return force_balance

    def add_source_config(self, mc: ModeConfig) -> None:
        self.source_config = mc

    def add_target_config(self, mc: ModeConfig) -> None:
        self.target_config = mc

    # TODO move and clean up
    def switch_mode(
        self, pair_name: str, old_modes: Dict[str, ContactModeType]
    ) -> Dict[str, ContactModeType]:
        new_modes = dict(old_modes)
        if old_modes[pair_name] == ContactModeType.NO_CONTACT:
            new_modes[pair_name] = ContactModeType.ROLLING
        elif old_modes[pair_name] == ContactModeType.ROLLING:
            new_modes[pair_name] = ContactModeType.NO_CONTACT
        return new_modes

    # TODO should not take in graph
    # TODO move and clean up
    def find_adjacent_mode_configs(
        self, curr_vertex: GraphVertex, graph: Graph
    ) -> List[ModeConfig]:
        if curr_vertex is None:
            breakpoint()

        current_modes = curr_vertex.config.modes
        if curr_vertex.config.additional_constraints is not None:
            # if we have additional constraints, first explore removing these
            return [ModeConfig(current_modes)]
        else:
            new_modes = [
                ModeConfig(self.switch_mode(pair, current_modes))
                for pair in current_modes.keys()
            ]
            return new_modes

    def build_graph(self, algorithm: Literal["BFS, DFS"] = "BFS") -> Graph:
        # TODO change name
        if algorithm == "DFS":
            INDEX_TO_POP = -1
        elif algorithm == "BFS":
            INDEX_TO_POP = 0

        graph = Graph(
            self.collision_pairs,
            self.all_decision_vars,
        )
        source = graph.create_new_vertex(self.source_config, "source")
        target = graph.create_new_vertex(self.target_config, "target")

        graph.add_source(source)
        graph.add_target(target)

        u = source
        frontier = PriorityQueue()
        # TODO refactor this into a function
        new_modes = self.find_adjacent_mode_configs(u, graph)
        priorities = [m.calculate_match(self.target_config) for m in new_modes]
        for (pri, m) in zip(priorities, new_modes):
            frontier.put(PrioritizedModeConfig(pri, m))

        TIMEOUT_LIMIT = 100
        counter = 0
        while True:
            if frontier.empty():
                raise RuntimeError("Frontier empty, but target not found")
            m = frontier.get().item
            counter += 1
            v = graph.create_new_vertex(m)

            if u.convex_set.IntersectsWith(v.convex_set):
                # TODO clean up
                if not graph.contains_vertex(v):
                    graph.add_vertex(v)
                graph.add_edge(u, v)

                new_modes = self.find_adjacent_mode_configs(v, graph)
                priorities = [m.calculate_match(self.target_config) for m in new_modes]
                for (pri, m) in zip(priorities, new_modes):
                    frontier.put(PrioritizedModeConfig(pri, m))

                found_target = v.convex_set.IntersectsWith(target.convex_set)
                if found_target:
                    graph.add_edge(v, target)
                    break
                u = v
            if counter == TIMEOUT_LIMIT:
                print("Timed out after {TIMEOUT_LIMIT} node expansions")
                break

        return graph
