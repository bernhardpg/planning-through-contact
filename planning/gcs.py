import matplotlib.pyplot as plt
import cdd

from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from typing import List, Literal, Union, Optional

import math
from pydrake.math import le, ge, eq
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt

from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.solvers import (
    MathematicalProgram,
    Solve,
    MathematicalProgramResult,
    L1NormCost,
    Cost,
    Binding,
    Constraint,
)

# Need some way of automatically generating convex sets


@dataclass
class BezierCtrlPoints:
    dim: int
    order: int  # Bezier curve order
    x: Optional[npt.NDArray[sym.Expression]] = None  # TODO rename to ctrl points
    name: Optional[str] = None

    def __post_init__(self):
        self.n_vars = self.order + 1
        if self.x is None:
            self.x = sym.MakeMatrixContinuousVariable(
                self.dim, self.order + 1, self.name
            )

    def get_derivative(self) -> "BezierCtrlPoints":
        der_ctrl_points = self.order * (self.x[:, 1:] - self.x[:, 0:-1])
        derivative = BezierCtrlPoints(self.dim, self.order - 1, der_ctrl_points)
        return derivative

    def __add__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        if type(other) == BezierCtrlPoints:
            assert other.dim == self.dim
            assert other.order == self.order
            new_ctrl_points = self.x + other.x
        else:
            new_ctrl_points = self.x + other
        return BezierCtrlPoints(self.dim, self.order, new_ctrl_points)

    def __radd__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        return self + other

    def __sub__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        if type(other) == BezierCtrlPoints:
            assert other.dim == self.dim
            assert other.order == self.order
            new_ctrl_points = self.x - other.x
        else:
            new_ctrl_points = self.x - other
        return BezierCtrlPoints(self.dim, self.order, new_ctrl_points)

    def __rsub__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        return self - other

    def __mul__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        if type(other) == BezierCtrlPoints:
            assert other.dim == self.dim
            assert other.order == self.order
            new_ctrl_points = self.x * other.x
        else:
            new_ctrl_points = self.x * other
        return BezierCtrlPoints(self.dim, self.order, new_ctrl_points)

    def __rmul__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        return self * other

    def __le__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        if type(other) == BezierCtrlPoints:
            assert other.dim == self.dim
            assert other.order == self.order
            return le(self.x, other.x)
        else:
            return le(self.x, other)

    def __ge__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        if type(other) == BezierCtrlPoints:
            assert other.dim == self.dim
            assert other.order == self.order
            return ge(self.x, other.x)
        else:
            return ge(self.x, other)

    def __eq__(
        self, other: Union["BezierCtrlPoints", npt.NDArray[np.float64], float, int]
    ) -> "BezierCtrlPoints":
        if type(other) == BezierCtrlPoints:
            assert other.dim == self.dim
            assert other.order == self.order
            return eq(self.x, other.x)
        else:
            return eq(self.x, other)


class BezierConvexSet:
    def __init__(self, constraints: List[sym.Expression]):
        flattened_constraints = [c.flatten() for c in constraints]
        self.constraints = np.concatenate(flattened_constraints)

    def formulate_polyhedron(
        self, variables: npt.NDArray[sym.Variable]
    ) -> opt.HPolyhedron:
        expressions = []
        for formula in self.constraints:
            kind = formula.get_kind()
            lhs, rhs = formula.Unapply()[1]
            if kind == sym.FormulaKind.Eq:
                # Eq constraint ax = b is
                # implemented as ax <= b, -ax <= -b
                expressions.append(lhs - rhs)
                expressions.append(rhs - lhs)
            elif kind == sym.FormulaKind.Geq:
                # lhs >= rhs
                # ==> rhs - lhs <= 0
                expressions.append(rhs - lhs)
            elif kind == sym.FormulaKind.Leq:
                # lhs <= rhs
                # ==> lhs - rhs <= 0
                expressions.append(lhs - rhs)

        # We now have expr <= 0 for all expressions
        # ==> we get Ax - b <= 0
        A, b_neg = sym.DecomposeAffineExpressions(expressions, variables)

        # Polyhedrons are of the form: Ax <= b
        b = -b_neg
        convex_set_as_polyhedron = opt.HPolyhedron(A, b)
        return convex_set_as_polyhedron


@dataclass
class ContactMode:
    # TODO: also needs to take in contact pairs!
    position_vars: npt.NDArray[BezierCtrlPoints]
    position_constraints: npt.NDArray[npt.NDArray[sym.Formula]]
    normal_force_vars: npt.NDArray[BezierCtrlPoints]
    friction_force_vars: npt.NDArray[BezierCtrlPoints]
    mode: Literal["no_contact", "rolling_contact", "sliding_contact"]
    friction_coeff: float
    normal_jacobian: npt.NDArray[np.float64]
    tangential_jacobian: npt.NDArray[np.float64]

    @property
    def x(self) -> npt.NDArray[sym.Variable]:
        return self.all_vars_flattened

    @property
    def A(self) -> npt.NDArray[np.float64]:
        return self.convex_set.A()

    @property
    def b(self) -> npt.NDArray[np.float64]:
        return self.convex_set.b()

    @property
    def all_vars_flattened(self) -> npt.NDArray[sym.Variable]:
        all_vars = np.concatenate(
            (
                self.position_vars,
                self.normal_force_vars,
                self.friction_force_vars,
                [self.slack_vars],  # TODO bad code
            )
        )
        return np.concatenate([var.x.flatten() for var in all_vars])

    @property
    def velocity_vars(self) -> npt.NDArray[sym.Variable]:
        return np.array([pos.get_derivative() for pos in self.position_vars])

    @property
    def rel_sliding_vel(self) -> npt.NDArray[sym.Expression]:
        # TODO add comment
        # must get the zero-th element, as the result will be contained in a numpy array, and we just want the BezierCtrlPoints object
        return self.tangential_jacobian.dot(self.velocity_vars)[0]

    def __post_init__(self):
        # TODO Will need one slack variable per contact point
        self.slack_vars = BezierCtrlPoints(
            dim=self.position_vars[0].dim,
            order=self.velocity_vars[0].order,
            name="gamma",
        )

        constraints = [self.position_constraints]
        if self.mode == "no_contact":
            constraints.append(self.create_zero_contact_force_constraints())
        elif self.mode == "rolling_contact":
            constraints.append(self.create_nonzero_contact_force_constraints())
            # There are multiple friction cone constraint in a list, so here we merge the lists
            constraints = sum(
                (constraints, self.create_inside_friction_cone_constraints()), []
            )
        else:
            raise NotImplementedError  # TODO
        constraints.append(self.create_force_balance_constraints())
        self.convex_set = BezierConvexSet(constraints).formulate_polyhedron(
            self.all_vars_flattened
        )

    def create_zero_contact_force_constraints(self) -> npt.NDArray[sym.Formula]:
        # TODO: Must also deal with contact pairs here when we have more than one!
        lam_n = self.normal_force_vars[0]
        constraint = lam_n == 0
        return constraint

    # TODO these two functions can be merged
    def create_nonzero_contact_force_constraints(
        self,
    ) -> npt.NDArray[sym.Formula]:
        # TODO: Is this a sensible way of enforcing nonzero contact foprce?
        EPS = 1e-5
        lam_n = self.normal_force_vars[0]
        constraint = lam_n >= EPS
        return constraint

    def create_force_balance_constraints(self) -> npt.NDArray[sym.Formula]:
        # TODO must generalize to jacobian
        lam_f = self.friction_force_vars[0]
        lam_n = self.normal_force_vars[0]
        constraint = lam_f == lam_n
        return constraint

    def create_inside_friction_cone_constraints(
        self,
    ) -> List[npt.NDArray[sym.Formula]]:
        EPS = 1e-5  # TODO
        LAM_G = 9.81  # TODO: should not be hardcoded
        # TODO: Need to figure out how to handle contact pairs!

        lam_f = self.friction_force_vars[0]
        fc_constraint = lam_f <= self.friction_coeff * LAM_G
        slack_constraint = self.slack_vars == 0
        nonzero_friction_force_constraint = lam_f >= EPS
        rel_vel_constraint = self.slack_vars + self.rel_sliding_vel == 0
        return [
            fc_constraint,
            slack_constraint,
            nonzero_friction_force_constraint,
            rel_vel_constraint,
        ]


@dataclass
class GcsPlanner:
    order: int
    convex_sets: List[opt.ConvexSet]
    gcs: opt.GraphOfConvexSets = GraphOfConvexSets()

    def __post_init__(self):
        self.n_ctrl_points = self.order + 1
        self.dim = self.convex_sets[0].A().shape[1]

        for i, s in enumerate(self.convex_sets):
            self._create_vertex_from_set(s, f"v_{i}")

        self._create_edge_between_overlapping_sets()
        for e in self.gcs.Edges():
            print(f"Edge name: {e.name()}")

        for e in self.gcs.Edges():
            self._add_continuity_constraint(e)
            self._add_path_length_cost(e)

    def _create_vertex_from_set(self, s: opt.ConvexSet, name: str) -> None:
        # We need (order + 1) control variables within each set,
        # solve this by taking the Cartesian product of the set
        # with itself (order + 1) times:
        # A (x) A = [A 0;
        #            0 A]
        one_set_per_decision_var = s.CartesianPower(self.n_ctrl_points)
        # This creates ((order + 1) * dim) decision variables per vertex
        # which for 2D with order=2 will be ordered (x1, y1, x2, y2, x3, y3)
        self.gcs.AddVertex(one_set_per_decision_var, name)

    def _create_edge_between_overlapping_sets(self) -> None:
        for u, set_u in zip(self.gcs.Vertices(), self.convex_sets):
            for v, set_v in zip(self.gcs.Vertices(), self.convex_sets):
                # TODO this can be speed up as we dont need to check for overlap both ways
                if u == v:
                    continue
                sets_are_overlapping = set_u.IntersectsWith(set_v)
                if sets_are_overlapping:
                    self.gcs.AddEdge(u.id(), v.id(), f"({u.name()}, {v.name()})")

    def _add_continuity_constraint(self, edge: GraphOfConvexSets.Edge) -> None:
        u = edge.xu()  # (order + 1, dim)
        v = edge.xv()
        u_last_ctrl_point = u[-self.dim :]
        v_first_ctrl_point = v[: self.dim]
        # TODO: if we also want continuity to a higher degree, this needs to be enforced here!
        continuity_constraints = eq(u_last_ctrl_point, v_first_ctrl_point)
        for c in continuity_constraints:
            edge.AddConstraint(c)

    def _reshape_ctrl_points_to_matrix(
        self, vec: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        matr = vec.reshape((self.dim, self.n_ctrl_points), order="F")
        return matr

    def _add_path_length_cost(self, edge: GraphOfConvexSets.Edge) -> None:
        # Minimize euclidean distance between subsequent control points
        # NOTE: we only minimize squared distance right now (which is the same)
        ctrl_points = self._reshape_ctrl_points_to_matrix(edge.xu())
        differences = np.array(
            [ctrl_points[:, i + 1] - ctrl_points[:, i] for i in range(len(ctrl_points))]
        )
        A = sym.DecomposeLinearExpressions(differences.flatten(), edge.xu())
        b = np.zeros((A.shape[0], 1))
        l1_norm_cost = L1NormCost(A, b)  # TODO: This is just to have some cost
        edge.AddCost(Binding[Cost](l1_norm_cost, edge.xu()))
        # TODO: no cost for source vertex

    def add_point_vertex(
        self,
        p: npt.NDArray[np.float64],
        name: str,
        flow_direction: Literal["in", "out"],
    ) -> GraphOfConvexSets.Vertex:
        singleton_set = opt.Point(p)
        point_vertex = self.gcs.AddVertex(singleton_set, name)

        for v, set_v in zip(self.gcs.Vertices(), self.convex_sets):
            sets_are_overlapping = singleton_set.IntersectsWith(set_v)
            if sets_are_overlapping:
                if flow_direction == "out":
                    edge = self.gcs.AddEdge(
                        point_vertex.id(),
                        v.id(),
                        f"({point_vertex.name()}, {v.name()})",
                    )
                elif flow_direction == "in":
                    edge = self.gcs.AddEdge(
                        v.id(),
                        point_vertex.id(),
                        f"({v.name()}, {point_vertex.name()})",
                    )
                self._add_continuity_constraint(edge)
        return point_vertex

    def _solve(
        self, source: GraphOfConvexSets.Vertex, target: GraphOfConvexSets.Vertex
    ) -> MathematicalProgramResult:
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = True  # TODO implement rounding
        options.max_rounded_paths = 10  # Must be >0 to actually do proper rounding

        result = self.gcs.SolveShortestPath(source, target, options)
        assert result.is_success
        return result

    def _reconstruct_path(
        self, result: MathematicalProgramResult
    ) -> List[npt.NDArray[np.float64]]:
        edges = self.gcs.Edges()
        flow_variables = [e.phi() for e in edges]
        flow_results = [result.GetSolution(p) for p in flow_variables]
        active_edges = [edge for edge, flow in zip(edges, flow_results) if flow == 1.00]
        # Observe that we only need the first vertex in every edge to reconstruct the entire graph
        vertices_in_path = [edge.xu() for edge in active_edges]
        vertex_values = [result.GetSolution(v) for v in vertices_in_path]

        return vertex_values

    def calculate_path(
        self, source: GraphOfConvexSets.Vertex, target: GraphOfConvexSets.Vertex
    ) -> List[npt.NDArray[np.float64]]:
        result = self._solve(source, target)
        vertex_values = self._reconstruct_path(result)
        return vertex_values
