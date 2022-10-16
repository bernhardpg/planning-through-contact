from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Dict, List, Optional

import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet

from geometry.contact import (
    CollisionPair,
    CollisionPairHandler,
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

    @staticmethod
    def create_vertex_from_mode_config(
        config: "ModeConfig",
        collision_pairs: List[CollisionPair],
        all_decision_vars: List[sym.Variable],
        name: Optional[str] = None,
    ) -> Optional["GraphVertex"]:
        contact_modes = [
            collision_pairs[pair].contact_modes[mode]
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
            ).formulate_polyhedron(all_decision_vars)
            intersects = intersects and intersection.IntersectsWith(additional_set)
            if not intersects:
                return None

            intersection = intersection.Intersection(additional_set)

        name = f"{name}: {calculated_name}" if name is not None else calculated_name
        vertex = GraphVertex(name, config, intersection)
        return vertex


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


class Graph:
    def __init__(
        self,
    ):
        self.edges = []
        self.vertices = []

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
    def __init__(self, collision_pair_handler: CollisionPairHandler) -> None:
        self.collision_pair_handler = collision_pair_handler

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
    def find_adjacent_contact_modes(
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

    def build_graph(self, prune: bool = False) -> Graph:
        if prune:  # NOTE: Will be expanded with more advanced graph search algorithms
            graph = self.prioritized_search_from_source(
                self.collision_pair_handler.collision_pairs_by_name,
                self.collision_pair_handler.all_decision_vars,
                self.source_config,
                self.target_config,
            )
        else:  # naive graph building
            graph = self.all_possible_permutations(
                self.collision_pair_handler.collision_pairs,
                self.collision_pair_handler.all_decision_vars,
                self.source_config,
                self.target_config,
            )

        return graph

    def prioritized_search_from_source(
        self,
        collision_pairs_by_name: Dict[str, CollisionPair],
        all_decision_vars: List[sym.Variable],
        source_config: ModeConfig,
        target_config: ModeConfig,
    ):
        graph = Graph()

        source = ModeConfig.create_vertex_from_mode_config(
            source_config, collision_pairs_by_name, all_decision_vars, "source"
        )
        target = ModeConfig.create_vertex_from_mode_config(
            target_config, collision_pairs_by_name, all_decision_vars, "target"
        )

        graph.add_source(source)
        graph.add_target(target)

        u = source
        frontier = PriorityQueue()
        new_modes = self.find_adjacent_contact_modes(u, graph)
        priorities = [m.calculate_match(target_config) for m in new_modes]
        for (pri, m) in zip(priorities, new_modes):
            frontier.put(PrioritizedModeConfig(pri, m))

        TIMEOUT_LIMIT = 100
        counter = 0
        while True:
            if frontier.empty():
                raise RuntimeError("Frontier empty, but target not found")
            m = frontier.get().item
            counter += 1
            v = ModeConfig.create_vertex_from_mode_config(
                m, collision_pairs_by_name, all_decision_vars
            )

            if u.convex_set.IntersectsWith(v.convex_set):
                # TODO clean up
                if not graph.contains_vertex(v):
                    graph.add_vertex(v)
                graph.add_edge(u, v)

                new_modes = self.find_adjacent_contact_modes(v, graph)
                priorities = [m.calculate_match(target_config) for m in new_modes]
                for (pri, m) in zip(priorities, new_modes):
                    frontier.put(PrioritizedModeConfig(pri, m))

                found_target = v.convex_set.IntersectsWith(target.convex_set)
                if found_target:
                    graph.add_edge(v, target)
                    break
                u = v
            if counter == TIMEOUT_LIMIT:
                print("Timed out after {TIMEOUT_LIMIT} node expansions")

        return graph

    def all_possible_permutations(
        self,
        collision_pairs: List[CollisionPair],
        all_decision_vars: List[sym.Variable],
        source_config: ModeConfig,
        target_config: ModeConfig,
    ):

        # [(n_m), (n_m), ... (n_m)] n_p times --> n_m * n_p
        # TODO: This is outdated: We now use dicts, but this assumes list of contact modes.
        breakpoint()


#        contact_pairs = [list(p.contact_modes.values()) for p in pairs]
#        # Cartesian product:
#        # S = P_1 X P_2 X ... X P_n_p
#        # |S| = |P_1| * |P_2| * ... * |P_n_p|
#        #     = n_m * n_m * ... * n_m
#        #     = n_m^n_p
#        possible_contact_permutations = list(itertools.product(*contact_pairs))
#
#        print("Building convex sets...")
#        intersects, intersections = zip(
#            *[
#                calc_intersection_of_contact_modes(perm)
#                for perm in tqdm(possible_contact_permutations)
#            ]
#        )
#
#        convex_sets = {
#            name: intersection
#            for intersects, (name, intersection) in zip(intersects, intersections)
#            if intersects
#        }
#        print(f"Number of feasible sets: {len(convex_sets)}")
#
#        return convex_sets
