import matplotlib.pyplot as plt
import cdd

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import List, Literal

import math
from pydrake.math import le, ge, eq
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt

from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.solvers import MathematicalProgram, Solve, MathematicalProgramResult

# TODO: Plan
# 1. Implement Bezier Curves
# 2. Implement vertices
# 3. Implement edges
# 4. Implement edge lengths
# 5. Formulate the math prog
# ...


# TODO: Replace with VPolytope
class Polyhedron(opt.HPolyhedron):
    def get_vertices(self) -> npt.NDArray[np.float64]:  # [N, 2]
        # NOTE: Use cdd to calculate vertices
        # cdd expects: [b -A], where A'x <= b
        # We have A'x >= b ==> -A'x <= -b
        # Hence we need [-b A]
        A = self.A()
        b = self.b().reshape((-1, 1))
        dim = A.shape[0]
        cdd_matrix = cdd.Matrix(np.hstack((b, -A)))
        cdd_matrix.rep_type = cdd.RepType.INEQUALITY
        cdd_poly = cdd.Polyhedron(cdd_matrix)
        generators = np.array(cdd_poly.get_generators())
        # cdd specific, see https://pycddlib.readthedocs.io/en/latest/polyhedron.html
        vertices = generators[:, 1 : 1 + dim]
        return vertices


@dataclass
class BezierCurve:
    deg: int
    dim: int

    def __post_init__(self):  # TODO add initial conditions here
        self.prog = MathematicalProgram()

        self.num_ctrl_points = self.deg + 1
        self.ctrl_points = self.prog.NewContinuousVariables(
            self.dim, self.num_ctrl_points, "gamma"
        )

        self.coeffs = [
            BernsteinPolynomial(deg=self.deg, k=k) for k in range(0, self.deg + 1)
        ]

    def constrain_to_polyhedron(self, poly: opt.HPolyhedron):
        A = poly.A()
        b = poly.b().reshape((-1, 1))
        # NOTE: Would like to do:
        # self.prog.AddLinearConstraint(A.dot(self.ctrl_points) <= b)
        # But this doesnt work, see https://github.com/RobotLocomotion/drake/issues/16025
        constraint = le(A.dot(self.ctrl_points), b)
        self.prog.AddLinearConstraint(constraint)

    def constrain_start_pos(self, x0: npt.NDArray[np.float64]) -> None:
        constraint = eq(self.ctrl_points[:, 0:1], x0)
        self.prog.AddLinearConstraint(constraint)

    def constrain_end_pos(self, xf: npt.NDArray[np.float64]) -> None:
        constraint = eq(self.ctrl_points[:, -1:], xf)
        self.prog.AddLinearConstraint(constraint)

    def solve(self):
        self.result = Solve(self.prog)
        assert self.result.is_success

    def calc_ctrl_points(self):
        self.solve()
        assert self.result is not None
        self.ctrl_point_values = self.result.GetSolution(self.ctrl_points)

    def eval_coeffs(self, at_s: float) -> npt.NDArray[np.float64]:
        evaluated_coeffs = np.array(
            [coeff.eval(at_s) for coeff in self.coeffs]
        ).reshape((-1, 1))
        return evaluated_coeffs

    def eval(self, at_s: float) -> npt.NDArray[np.float64]:
        assert self.ctrl_point_values is not None
        evaluated_coeffs = self.eval_coeffs(at_s)
        path_value = self.ctrl_point_values.dot(evaluated_coeffs)
        return path_value


@dataclass
class BernsteinPolynomial:
    deg: int
    k: int

    def __post_init__(self) -> sym.Expression:
        self.s = sym.Variable("s")
        self.poly = (
            math.comb(self.deg, self.k)
            * np.power(self.s, self.k)
            * np.power((1 - self.s), (self.deg - self.k))
        )

    def eval(self, at_s: float) -> float:
        assert 0 <= at_s and at_s <= 1.0
        env = {self.s: at_s}
        value_at_s = self.poly.Evaluate(env)
        return value_at_s


@dataclass
class BezierGCS:
    order: int
    convex_sets: List[opt.ConvexSet]
    gcs: opt.GraphOfConvexSets = GraphOfConvexSets()

    def __post_init__(self):
        self.n_ctrl_points = self.order + 1
        self.dim = self.convex_sets[0].A().shape[1]

        self._create_vertices_from_convex_sets()
        self._create_edge_between_overlapping_sets()
        for e in self.gcs.Edges():
            self._add_continuity_constraint(e)

    def _create_vertices_from_convex_sets(self):
        for i, s in enumerate(self.convex_sets):
            # We need (order + 1) control variables within each set,
            # solve this by taking the Cartesian product of the set
            # with itself (order + 1) times:
            # A (x) A = [A 0;
            #            0 A]
            one_set_per_decision_var = s.CartesianPower(self.n_ctrl_points)
            # This creates ((order + 1) * dim) decision variables per vertex
            # which for 2D with order=2 will be ordered (x1, y1, x2, y2, x3, y3)
            self.gcs.AddVertex(one_set_per_decision_var, name=f"v_{i}")

    def _create_edge_between_overlapping_sets(self):
        for u, set_u in zip(self.gcs.Vertices(), self.convex_sets):
            for v, set_v in zip(self.gcs.Vertices(), self.convex_sets):
                # TODO this can be speed up as we dont need to check for overlap both ways
                if u == v:
                    continue
                sets_are_overlapping = set_u.IntersectsWith(set_v)
                if sets_are_overlapping:
                    self.gcs.AddEdge(u.id(), v.id(), f"({u.name()}, {v.name()})")

    def _add_continuity_constraint(self, edge: GraphOfConvexSets.Edge):
        u = edge.xu()  # (order + 1, dim)
        v = edge.xv()
        u_last_ctrl_point = u[-self.dim :]
        v_first_ctrl_point = v[: self.dim]
        # TODO: if we also want continuity to a higher degree, this needs to be enforced here!
        continuity_constraints = eq(u_last_ctrl_point, v_first_ctrl_point)
        for c in continuity_constraints:
            edge.AddConstraint(c)

    def add_point_vertex(
        self,
        p: npt.NDArray[np.float64],
        name: str,
        flow_direction: Literal["in", "out"],
    ) -> None:
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
        return


def create_test_polyhedron_1() -> Polyhedron:
    # NOTE: In my example I used the other halfspace notation, hence the sign flips
    A = -np.array([[1, -1], [-1, -1], [0, 1], [1, 0]], dtype=np.float64)
    b = -np.array([-1, -5, 0, 0], dtype=np.float64).reshape((-1, 1))

    poly = Polyhedron(A, b)
    return poly


def create_test_polyhedron_2() -> Polyhedron:
    # NOTE: In my example I used the other halfspace notation, hence the sign flips
    A = -np.array([[1, -3], [-1, -0.7], [0, 1], [1, 0]], dtype=np.float64)
    b = -np.array([0.2, -9, 1, 3.5], dtype=np.float64).reshape((-1, 1))

    poly = Polyhedron(A, b)
    return poly


def create_test_polyhedrons() -> List[Polyhedron]:
    return [create_test_polyhedron_1(), create_test_polyhedron_2()]


def test_bezier_curve() -> None:
    deg = 2
    dim = 2

    poly = create_test_polyhedron_1()
    vertices = poly.get_vertices()
    plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3)

    x0 = np.array([0, 0.5]).reshape((-1, 1))
    xf = np.array([4, 1]).reshape((-1, 1))

    bezier_curve = BezierCurve(deg, dim)
    bezier_curve.constrain_to_polyhedron(poly)
    bezier_curve.constrain_start_pos(x0)
    bezier_curve.constrain_end_pos(xf)
    bezier_curve.calc_ctrl_points()
    path = np.concatenate(
        [bezier_curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
    ).T

    plt.plot(path[:, 0], path[:, 1])
    plt.scatter(x0[0], x0[1])
    plt.scatter(xf[0], xf[1])

    plt.show()


def plot_polyhedrons(polys: List[Polyhedron]) -> None:
    for poly in polys:
        vertices = poly.get_vertices()
        plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3)
    plt.show()


def test_gcs() -> None:
    order = 2

    polys = create_test_polyhedrons()

    path = BezierGCS(order, polys)
    # plot_polyhedrons(polys)

    x0 = np.array([0, 0.5]).reshape((-1, 1))
    xf = np.array([4, 1]).reshape((-1, 1))

    path.add_point_vertex(x0, "source", "out")
    path.add_point_vertex(xf, "target", "in")
    breakpoint()

    return


def test_bernstein_polynomial() -> None:
    deg = 2
    k = 0

    bp = BernsteinPolynomial(deg, k)


def main():
    # test_bezier_curve()
    test_gcs()

    return 0


if __name__ == "__main__":
    main()
