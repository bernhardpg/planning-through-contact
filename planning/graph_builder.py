from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet
from pydrake.math import eq

from geometry.contact import (
    CollisionPair,
    ContactModeType,
    calc_intersection_of_contact_modes,
)
from geometry.polyhedron import PolyhedronFormulator

# flake8: noqa


@dataclass
class ModeConfig:
    modes: Dict[str, ContactModeType]
    additional_constraints: Optional[npt.NDArray[sym.Formula]] = None


@dataclass
class GraphVertex:
    name: str
    config: ModeConfig
    convex_set: ConvexSet


@dataclass
class GraphEdge:
    u: GraphVertex
    v: GraphVertex


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
    def create_new_vertex(self, config: ModeConfig) -> Optional[GraphVertex]:
        contact_modes = [
            self.collision_pairs[pair].contact_modes[mode]
            for pair, mode in config.modes.items()
        ]
        intersects, (name, intersection) = calc_intersection_of_contact_modes(
            contact_modes
        )
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

        vertex = GraphVertex(name, config, intersection)
        return vertex

    def add_edge(self, u: GraphVertex, v: GraphVertex) -> None:
        e = GraphEdge(u, v)
        self.edges.append(e)

    def add_vertex(self, u: GraphVertex) -> None:
        self.vertices.append(u)


class GraphBuilder:
    def __init__(
        self,
        collision_pairs: List[CollisionPair],  # TODO for now I define this manually
        unactuated_bodies: List[str],  # TODO make part of RigidBody
        external_forces: List[sym.Expression],
        additional_constraints: Optional[List[sym.Formula]],
        allow_sliding: bool = False,
    ) -> None:

        self.collision_pairs = collision_pairs
        self.all_bodies = self._collect_all_rigid_bodies(collision_pairs)
        unactuated_dofs = self._get_unactuated_dofs(
            unactuated_bodies, self.all_bodies, self.dim
        )
        force_balance_constraints = self.construct_force_balance(
            collision_pairs,
            self.all_bodies,
            external_forces,
            unactuated_dofs,
        )
        for p in self.collision_pairs:
            p.add_force_balance(force_balance_constraints)
        for p in self.collision_pairs:
            p.add_constraint_to_all_modes(additional_constraints)

        self.all_decision_vars = self._collect_all_decision_vars(self.collision_pairs)
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

    @property
    def dim(self) -> int:
        return self.collision_pairs[0].body_a.dim

    @staticmethod
    def _collect_all_rigid_bodies(pairs: List[CollisionPair]) -> List[str]:
        all_body_names = sorted(
            list(set(sum([[p.body_a.name, p.body_b.name] for p in pairs], [])))
        )
        return all_body_names

    def _collect_all_pos_vars(
        self, pairs: List[CollisionPair]
    ) -> npt.NDArray[sym.Variable]:
        all_pos_vars = np.concatenate(
            [np.concatenate((p.body_a.pos.x, p.body_b.pos.x)).flatten() for p in pairs]
        )
        unique_pos_vars = set(all_pos_vars)
        sorted_unique_pos_vars = np.array(
            sorted(list(unique_pos_vars), key=lambda x: x.get_name())
        )
        return sorted_unique_pos_vars

    def _collect_all_decision_vars(
        self, pairs: List[CollisionPair]
    ) -> npt.NDArray[sym.Variable]:
        all_pos_vars = self._collect_all_pos_vars(pairs)
        all_normal_force_vars = np.concatenate([p.lam_n for p in pairs]).flatten()
        all_friction_force_vars = np.concatenate([p.lam_f for p in pairs]).flatten()
        all_vars = np.concatenate(
            [all_pos_vars, all_normal_force_vars, all_friction_force_vars]
        )
        return all_vars

    def _get_unactuated_dofs(
        self, unactuated_bodies: List[str], all_bodies: List[str], dim: int
    ) -> npt.NDArray[np.int32]:
        unactuated_idxs = [all_bodies.index(b) * dim for b in unactuated_bodies]
        unactuated_dofs = np.concatenate(
            [np.arange(idx, idx + dim) for idx in unactuated_idxs]
        )
        return unactuated_dofs

    # TODO move to contact module
    @staticmethod
    def construct_force_balance(
        collision_pairs: List[CollisionPair],
        bodies: List[str],
        external_forces: npt.NDArray[sym.Expression],
        unactuated_dofs: npt.NDArray[np.int32],
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

    def add_source(self, mode: ModeConfig) -> None:
        self.source = mode

    def add_target(self, mode: ModeConfig) -> None:
        self.target = mode

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
    def find_adjacent_vertices(
        self, curr_vertex: GraphVertex, graph: Graph
    ) -> List[GraphVertex]:
        current_modes = curr_vertex.config.modes
        new_modes = (
            ModeConfig(self.switch_mode(pair, current_modes))
            for pair in current_modes.keys()
        )
        new_vertices = [graph.create_new_vertex(m) for m in new_modes]
        return new_vertices

    def build_graph(self, algorithm: Literal["BFS, DFS"] = "BFS") -> None:
        if algorithm == "DFS":
            INDEX_TO_POP = -1
        elif algorithm == "BFS":
            INDEX_TO_POP = 0

        graph = Graph(self.collision_pairs, self.all_decision_vars)
        source = graph.create_new_vertex(self.source)
        target = graph.create_new_vertex(self.target)

        u = source
        frontier = []
        frontier.extend(self.find_adjacent_vertices(u, graph))
        while True:
            v = frontier.pop(INDEX_TO_POP)
            if u.convex_set.IntersectsWith(v.convex_set):
                graph.add_vertex(v)
                graph.add_edge(u, v)
                frontier.extend(self.find_adjacent_vertices(v, graph))
                found_target = v.convex_set.IntersectsWith(target.convex_set)
                if found_target:
                    break
                u = v

        breakpoint()
