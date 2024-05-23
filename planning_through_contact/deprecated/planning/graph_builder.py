import warnings
from dataclasses import dataclass
from itertools import permutations
from queue import PriorityQueue
from typing import Optional, Tuple

from deprecated.geometry.collision_pair import CollisionPairHandler
from deprecated.geometry.contact_mode import (
    ContactModeConfig,
    PrioritizedContactModeConfig,
)
from pydrake.geometry.optimization import ConvexSet
from tqdm import tqdm


@dataclass(eq=True)
class GraphVertex:
    name: str
    config: ContactModeConfig
    convex_set: ConvexSet

    @classmethod
    def create_from_mode_config(
        cls,
        config: ContactModeConfig,
        collision_pair_handler: CollisionPairHandler,
        name: Optional[str] = None,
        assert_nonempty: bool = True,
    ) -> Optional["GraphVertex"]:
        result = collision_pair_handler.create_convex_set_from_mode_config(config, name)
        # TODO is this bad code style?
        if result is None:
            if assert_nonempty:
                raise RuntimeError(
                    f"Convex set is empty for specified mode_config with name '{name}'"
                )
            else:
                return None
        convex_set, name = result
        return cls(name, config, convex_set)


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

    def contains_mode_cfg(
        self, mc: ContactModeConfig
    ) -> Tuple[bool, Optional[GraphVertex]]:
        found_vertex = next((v for v in self.vertices if v.config == mc), None)
        if found_vertex is not None:
            return True, found_vertex
        else:
            return False, None


class GraphBuilder:
    def __init__(self, collision_pair_handler: CollisionPairHandler) -> None:
        self.collision_pair_handler = collision_pair_handler

    def add_source_config(self, mc: ContactModeConfig) -> None:
        self.source_config = mc

    def add_target_config(self, mc: ContactModeConfig) -> None:
        self.target_config = mc

    def build_graph(self, prune: bool = False) -> Graph:
        if prune:
            warnings.warn(
                "The prune functionality is very experimental and should not be used."
            )
            graph = self.prioritized_search_from_source(
                self.collision_pair_handler,
                self.source_config,
                self.target_config,
            )
        else:  # naive graph building
            graph = self.all_possible_contact_modes(
                self.collision_pair_handler,
                self.source_config,
                self.target_config,
            )

        return graph

    def prioritized_search_from_source(
        self,
        collision_pair_handler: CollisionPairHandler,
        source_config: ContactModeConfig,
        target_config: ContactModeConfig,
    ):
        graph = Graph()

        source = GraphVertex.create_from_mode_config(
            source_config, collision_pair_handler, "source"
        )
        target = GraphVertex.create_from_mode_config(
            target_config, collision_pair_handler, "target"
        )

        graph.add_source(source)
        graph.add_target(target)

        u = source
        frontier = PriorityQueue()
        new_mode_cfgs = ContactModeConfig.create_all_adjacent_modes(u.config)
        priorities = [m.calculate_match(target_config) for m in new_mode_cfgs]
        for pri, m in zip(priorities, new_mode_cfgs):
            frontier.put(PrioritizedContactModeConfig(pri, m))

        TIMEOUT_LIMIT = 100
        counter = 0
        while True:
            if frontier.empty():
                raise RuntimeError("Frontier empty, but target not found")
            counter += 1

            mode_cfg = frontier.get().item
            mode_cfg_already_in_graph, found_vertex = graph.contains_mode_cfg(mode_cfg)
            if mode_cfg_already_in_graph:
                v = found_vertex
            else:
                v = GraphVertex.create_from_mode_config(
                    mode_cfg, collision_pair_handler
                )
                graph.add_vertex(v)

            if u.convex_set.IntersectsWith(v.convex_set):
                graph.add_edge(u, v)

                new_mode_cfgs = ContactModeConfig.create_all_adjacent_modes(v.config)
                priorities = [m.calculate_match(target_config) for m in new_mode_cfgs]
                for pri, m in zip(priorities, new_mode_cfgs):
                    frontier.put(PrioritizedContactModeConfig(pri, m))

                found_target = v.convex_set.IntersectsWith(target.convex_set)
                if found_target:
                    graph.add_edge(v, target)
                    break
                u = v
            if counter == TIMEOUT_LIMIT:
                print("Timed out after {TIMEOUT_LIMIT} node expansions")

        return graph

    def all_possible_contact_modes(
        self,
        collision_pair_handler: CollisionPairHandler,
        source_config: ContactModeConfig,
        target_config: ContactModeConfig,
    ) -> Graph:
        graph = Graph()
        source = GraphVertex.create_from_mode_config(
            source_config, collision_pair_handler, "source"
        )
        target = GraphVertex.create_from_mode_config(
            target_config, collision_pair_handler, "target"
        )
        graph.add_source(source)
        graph.add_target(target)

        contact_mode_cfgs = collision_pair_handler.all_possible_contact_cfg_perms()
        print("Creating a convex set for each possible contact mode permutation...")
        vertices = list(
            filter(
                lambda x: x is not None,
                [
                    GraphVertex.create_from_mode_config(
                        cfg, collision_pair_handler, assert_nonempty=False
                    )
                    for cfg in tqdm(contact_mode_cfgs)
                ],
            )
        )
        graph.vertices.extend(vertices)

        print("Creating edges between all overlapping sets...")
        edges = [
            GraphEdge(u, v)
            for u, v in tqdm(list(permutations(graph.vertices, 2)))
            if u.convex_set.IntersectsWith(v.convex_set)
        ]
        graph.edges.extend(edges)

        return graph
