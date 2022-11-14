from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import (
    CartesianProduct,
    ConvexSet,
    GraphOfConvexSets,
    VPolytope,
)
from pydrake.math import eq
from pydrake.solvers import Binding, Cost, L2NormCost, MathematicalProgramResult


def create_test_polytopes() -> List[VPolytope]:
    vertices = [
        np.array([[-1, 0], [0, 0], [0, 3], [-1, 3]]),
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        np.array([[0, 1], [1, 1], [1, 2], [0, 2]]),
        np.array([[0, 2], [1, 2], [1, 3], [0, 3]]),
        np.array([[1, 0], [2, 0], [2, 1], [1, 1]]),
        np.array([[1, 1], [2, 1], [2, 2], [1, 2]]),
        np.array([[1, 2], [2, 2], [2, 3], [1, 3]]),
        np.array([[2, 0], [3, 0], [3, 1], [2, 1]]),
        np.array([[2, 1], [3, 1], [3, 2], [2, 2]]),
        np.array([[2, 2], [3, 2], [3, 3], [2, 3]]),
        np.array([[3, 1.3], [4, 1.3], [4, 1.7], [3, 1.7]]),
    ]
    polytopes = [VPolytope(vs.T) for vs in vertices]
    return polytopes


# TODO move to visualization library
def visualize_polytopes(polytopes: List[VPolytope], show=False) -> None:
    temp = [p.vertices().T for p in polytopes]
    vertices = [np.concatenate([t, t[0:1, :]]) for t in temp]
    for vs in vertices:
        plt.plot(vs[:, 0], vs[:, 1])
    if show:
        plt.show()


NodeId = TypeVar("NodeId")
Edge = Tuple[NodeId, NodeId]


class ConvexSetNode:
    def __init__(self, convex_set: ConvexSet, id: NodeId) -> None:
        self.convex_set = convex_set
        self.id = id

    def connected_to(self, other: "ConvexSetNode") -> bool:
        return self.convex_set.IntersectsWith(other.convex_set)


Node = TypeVar("Node")


@dataclass(order=True)
class PrioritizedNode:
    priority: float
    item: Node = field(compare=False)


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges = edges

    def add_node(self, n: Node) -> None:
        self.nodes.append(n)

    def add_edge(self, e: Edge) -> None:
        self.edges.append(e)

    def get_neighbours(self, node: Node) -> List[Node]:
        connected_nodes = [
            n for n in self.nodes if n.connected_to(node) and not n.id == node.id
        ]
        return connected_nodes


# TODO Refactor to own class
# TODO Should be used by other parts of the project too!
class Gcs:
    def __init__(self, graph: Graph, num_ctrl_points: int = 3) -> None:
        self.NUM_DIMS = 2  # TODO should not be hardcoded
        self.NUM_CTRL_POINTS = num_ctrl_points
        self.gcs = GraphOfConvexSets()

        self.vertices = {
            n.id: self.gcs.AddVertex(
                # Cartesian power is used to make dimension of
                # convex set reflect number of control points
                self.cartesian_power(n.convex_set, num_ctrl_points),
                n.id,
            )
            for n in graph.nodes
        }
        self.edges = [
            self.gcs.AddEdge(self.vertices[u_id], self.vertices[v_id])
            for u_id, v_id in graph.edges
        ]

        for e in self.edges:
            self.add_continuity_constraints(e)
            self.add_path_length_cost(e)

    @staticmethod
    def cartesian_power(convex_set: ConvexSet, power: int) -> CartesianProduct:
        convex_set_n_times = [convex_set] * power
        return CartesianProduct(convex_set_n_times)

    @property
    def decision_var_vector(self) -> npt.NDArray[sym.Variable]:
        _, v = next(iter(self.vertices.items()))
        return v.x()

    @property
    def decision_var_matrix(self) -> npt.NDArray[sym.Variable]:
        return self.decision_var_vector.reshape((self.NUM_DIMS, -1), order="F")

    @property
    def path_length_cost(self) -> L2NormCost:
        diffs = np.diff(self.decision_var_matrix)
        A = sym.DecomposeLinearExpressions(diffs.flatten(), self.decision_var_vector)
        b = np.zeros((A.shape[0], 1))
        return L2NormCost(A, b)

    @property
    def continuity_constraint_matrices(
        self,
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        A_first = sym.DecomposeLinearExpressions(
            self.decision_var_matrix[:, 0], self.decision_var_vector
        )
        A_last = sym.DecomposeLinearExpressions(
            self.decision_var_matrix[:, -1], self.decision_var_vector
        )
        return A_first, A_last

    def set_source(self, source_id: NodeId) -> None:
        self.source_id = source_id

    def set_target(self, target_id: NodeId) -> None:
        self.target_id = target_id

    def add_vertex(self, node: Node) -> GraphOfConvexSets.Vertex:
        vertex = self.gcs.AddVertex(
            self.cartesian_power(node.convex_set, self.NUM_CTRL_POINTS), node.id
        )
        self.vertices[node.id] = vertex
        return vertex

    def add_edge(self, u_id: NodeId, v_id: NodeId) -> GraphOfConvexSets.Edge:
        edge = self.gcs.AddEdge(self.vertices[u_id], self.vertices[v_id])
        self.edges.append(edge)
        return edge

    def add_path_length_cost(self, edge: GraphOfConvexSets.Edge) -> None:
        # cost = Binding[Cost](self.path_length_cost, edge.xu())
        # TODO I need to correctly penalize all edges!
        cost = Binding[Cost](self.path_length_cost, edge.xv())
        edge.AddCost(cost)

    def add_continuity_constraints(self, edge: GraphOfConvexSets.Edge) -> None:
        A_first, A_last = self.continuity_constraint_matrices
        xu, xv = edge.xu(), edge.xv()
        constraints = eq(A_last.dot(xu), A_first.dot(xv))
        for c in constraints:
            edge.AddConstraint(c)

    def solve(self, use_convex_relaxation: bool = False) -> MathematicalProgramResult:
        assert self.source_id is not None and self.target_id is not None
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = use_convex_relaxation
        if use_convex_relaxation is True:
            options.preprocessing = True  # TODO Do I need to deal with this?
            options.max_rounded_paths = 10

        result = self.gcs.SolveShortestPath(
            self.vertices[self.source_id], self.vertices[self.target_id], options
        )
        assert result.is_success()
        return result

    def save_graph_diagram(
        self, filename: str, result: Optional[MathematicalProgramResult] = None
    ) -> None:
        if result is not None:
            graphviz = self.gcs.GetGraphvizString(result, False, precision=1)
        else:
            graphviz = self.gcs.GetGraphvizString()
        data = pydot.graph_from_dot_data(graphviz)[0]
        data.write_svg(filename)

    def get_ctrl_points(
        self, result: MathematicalProgramResult
    ) -> npt.NDArray[np.float64]:
        flow_variables = [e.phi() for e in self.edges]
        flow_results = [result.GetSolution(p) for p in flow_variables]
        active_edges = [
            edge for edge, flow in zip(self.edges, flow_results) if flow >= 0.99
        ]
        path = self._find_path_to_target(
            active_edges, self.vertices[self.target_id], self.vertices[self.source_id]
        )
        vertex_values = np.hstack(
            [
                result.GetSolution(v.x()).reshape((self.NUM_DIMS, -1), order="F")
                for v in path
            ]
        )
        return vertex_values

    def _find_path_to_target(
        self,
        edges: List[GraphOfConvexSets.Edge],
        target: GraphOfConvexSets.Vertex,
        u: GraphOfConvexSets.Vertex,
    ) -> List[GraphOfConvexSets.Vertex]:
        current_edge = next(e for e in edges if e.u() == u)
        v = current_edge.v()
        target_reached = v == target
        if target_reached:
            return [u] + [v]
        else:
            return [u] + self._find_path_to_target(edges, target, v)


# Algorithm
# 1. Start with a graph and a recently added node n
# 2. For each neighbour nb of n: Compute GCS cost
# 2a. Formulate GCS graph with nb added (includes all edges to nb)
# 2b. Solve GCS with nb added with edge from nb to target (without continuouty constraint)
# 2c. Remove nb / formulate new problem
# 3.  Add all nodes and weights to frontier (update the ones that are already there)
# 4. Pick the best node in the frontier and add it to the graph.
# 5. Repeat

# Initialize:
# 1. Start only with source node in graph


class GraphBuilder:
    def __init__(self, nodes: List[Node]):
        self.all_nodes = {n.id: n for n in nodes}

    def get_node(self, id: NodeId) -> Node:
        return self.all_nodes[id]

    def build_graph(self, source_id: NodeId, target_id: NodeId):
        graph = Graph([self.get_node(source_id), self.get_node(target_id)], [])
        frontier = PriorityQueue()
        current_node = self.get_node(source_id)

        target_reached = False
        while not target_reached:
            new_nodes = [
                node
                for node in self.get_neighbours(current_node.id)
                if node not in graph.nodes
            ]
            prioritized_neighbours = [
                PrioritizedNode(
                    self.calculate_relaxed_problem_cost(
                        graph, current_node, new_node, source_id, target_id
                    ),
                    new_node,
                )
                for new_node in new_nodes
            ]
            for node in prioritized_neighbours:
                frontier.put(node)

            current_node = frontier.get().item
            while current_node in graph.nodes:
                current_node = frontier.get().item

            neighbours_in_graph = graph.get_neighbours(current_node)
            edges = [(current_node.id, v.id) for v in neighbours_in_graph] + [
                (u.id, current_node.id) for u in neighbours_in_graph
            ]
            graph.add_node(current_node)
            print(f"Adding node {current_node.id}")
            for edge in edges:
                graph.add_edge(edge)

    def get_neighbours(self, node_id: NodeId) -> List[Node]:
        # NOTE: This is where new nodes should be generated "on-the-go"
        connected_nodes = [
            n
            for n in self.all_nodes.values()
            if n.connected_to(self.get_node(node_id)) and not n.id == node_id
        ]
        return connected_nodes

    @staticmethod
    def calculate_relaxed_problem_cost(
        base_graph: Graph,
        current_node: Node,
        new_node: Node,
        source_id: NodeId,
        target_id: NodeId,
    ) -> float:
        gcs = Gcs(base_graph)

        gcs.add_vertex(new_node)

        # We do not enforce continuity on the relaxed edge
        relaxed_edge = gcs.add_edge(new_node.id, target_id)
        gcs.add_path_length_cost(relaxed_edge)

        new_edge = gcs.add_edge(current_node.id, new_node.id)
        gcs.add_continuity_constraints(new_edge)
        gcs.add_path_length_cost(new_edge)

        gcs.set_source(source_id)
        gcs.set_target(target_id)
        gcs.save_graph_diagram("output/latest_graph_unsolved.svg")
        result = gcs.solve()

        ctrl_points = gcs.get_ctrl_points(result)
        gcs.save_graph_diagram("output/latest_graph_solved.svg", result)
        print(f"Cost for node {new_node.id}: {result.get_optimal_cost()}")

        # TODO remove
        visualize_polytopes(create_test_polytopes())
        plt.plot(ctrl_points[0, :], ctrl_points[1, :], marker="o")
        plt.show()

        return result.get_optimal_cost()


def gcs_a_star():
    print("Running GCS A* demo")

    polytopes = create_test_polytopes()
    visualize_polytopes(polytopes)
    all_nodes = [ConvexSetNode(poly, str(idx)) for idx, poly in enumerate(polytopes)]
    builder = GraphBuilder(all_nodes)
    builder.build_graph(all_nodes[0].id, all_nodes[-1].id)
